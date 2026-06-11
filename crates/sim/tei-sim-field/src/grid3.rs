//! 3D Yee grid — full (Ex, Ey, Ez, Hx, Hy, Hz) leapfrog with CPML on all
//! six faces and a deterministic rayon slab decomposition (F3).
//!
//! ## Yee discretization (Yee 1966; Taflove & Hagness 2005, ch. 3)
//!
//! Normalized units ε₀ = μ₀ = c = 1, Δx = Δy = Δz = Δ = 1, non-magnetic
//! isotropic dielectric media ε_r(x, y, z) (dispersive Drude/Lorentz media
//! are layered on by [`crate::medium3`] via the ADE method):
//!
//! ```text
//!   ∂E/∂t = (1/ε_r)·∇×H        ∂H/∂t = −∇×E
//! ```
//!
//! Component positions on the cell (i, j, k):
//!
//! ```text
//!   Ex(i+½, j, k)    Ey(i, j+½, k)    Ez(i, j, k+½)
//!   Hx(i, j+½, k+½)  Hy(i+½, j, k+½)  Hz(i+½, j+½, k)
//! ```
//!
//! E lives at integer time steps n·Δt, H at half steps; centered
//! differences give the second-order leapfrog. Storage is row-major with z
//! (k) contiguous: `f[(i·dⱼ + j)·dₖ + k]` where (dᵢ, dⱼ, dₖ) are that
//! component's dimensions ((nx−1, ny, nz) for Ex, …).
//!
//! **Stability (CFL)**: in 3D the leapfrog is stable iff
//! c·Δt ≤ Δ/√3 (Taflove & Hagness §4.7). We parameterize Δt = S·Δ/√3 with
//! Courant number S < 1 (default 0.5 — the same safety margin convention as
//! the 2D grid's Δt = S·Δ/√2).
//!
//! **Numerical dispersion** (Taflove & Hagness eq. 4.15): the 3D relation
//!
//! ```text
//!   [1/(cΔt)·sin(ωΔt/2)]² = Σ_w [1/Δ·sin(k_wΔ/2)]²,  w ∈ {x, y, z}
//! ```
//!
//! reduces for axis-aligned propagation to the same closed form the 2D
//! grid validates against — [`crate::grid::yee_axis_wavenumber`] applies
//! unchanged (it only depends on ω and Δt).
//!
//! ## CPML on six faces (Roden & Gedney 2000)
//!
//! The convolutional PML machinery is shared with the 2D grid
//! ([`crate::grid::cpml_coeffs`]): per-axis polynomially graded (σ, κ, α)
//! profiles evaluated at each staggered component's true position, one ψ
//! memory variable per (component, PML-normal derivative) pair. In 3D each
//! field component has curl derivatives along the two axes orthogonal to
//! it, so there are 12 ψ arrays. They are allocated full-size (zero and
//! untouched outside the strips) — at F3 scales (≤ 10⁷ cells) the memory
//! is cheap and the layout keeps the slab decomposition trivial; strip
//! packing is an F4 (GPU) concern. The outermost tangential-E samples are
//! never updated (PEC ring); with `npml = 0` the grid is therefore an
//! exact PEC box cavity — the F3 anchor validation.
//!
//! ## Rayon slab decomposition + determinism
//!
//! Each of the six component updates parallelizes over **x-slabs** (planes
//! of constant i, the slowest-varying index, so every slab is one
//! contiguous memory chunk via `par_chunks_mut`). Writes go only to the
//! slab's own component plane and its ψ planes; reads of the other field
//! touch planes i−1/i/i+1 through shared references — Yee staggering makes
//! the write sets disjoint by construction, no halo exchange or buffering
//! is needed. Every cell's update is one fixed arithmetic expression with
//! no reductions and no accumulation order to vary, so the floating-point
//! result is **bit-identical run-to-run and at any thread count** (the
//! validation suite asserts both). Grids below [`PAR_MIN_CELLS`] cells run
//! the identical loop bodies serially to skip the fork-join overhead —
//! same arithmetic, same bits.

use crate::grid::{CpmlParams, cpml_coeffs};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

/// Below this many Yee cells (nx·ny·nz) the update loops run serially —
/// the per-step fork-join overhead dominates tiny grids. The arithmetic is
/// identical either way, so results do not depend on which path runs.
pub const PAR_MIN_CELLS: usize = 32_768;

/// Target Yee cells per rayon task: consecutive x-planes are grouped into
/// slabs of at least this many samples so the per-task work amortizes the
/// scheduling overhead (one plane of a small grid is only a few thousand
/// floats). Grouping only changes which thread runs which plane, never
/// the per-cell arithmetic — determinism is unaffected.
const SLAB_TARGET_CELLS: usize = 32_768;

/// 3D grid construction parameters.
#[derive(Debug, Clone)]
pub struct Grid3Spec {
    /// Integer grid points along x.
    pub nx: usize,
    /// Integer grid points along y.
    pub ny: usize,
    /// Integer grid points along z.
    pub nz: usize,
    /// Courant number S; Δt = S·Δ/√3. Stable for S < 1.
    pub courant: f64,
    /// CPML thickness in cells on each of the six faces (0 = PEC box).
    pub npml: usize,
    pub cpml: CpmlParams,
}

pub(crate) fn default_courant3() -> f64 {
    0.5
}

/// Axis selector (source polarization, snapshot slice normal).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Axis {
    X,
    Y,
    Z,
}

/// Electric-field component selector (probes, DFT monitors).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Comp {
    Ex,
    Ey,
    Ez,
}

impl Axis {
    /// The E component polarized along this axis.
    pub fn e_comp(self) -> Comp {
        match self {
            Axis::X => Comp::Ex,
            Axis::Y => Comp::Ey,
            Axis::Z => Comp::Ez,
        }
    }
}

/// What a snapshot slice samples.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SliceField {
    /// Raw Ez samples (staggered at k+½; the slice index is clamped to the
    /// last Ez sample along its axis).
    #[default]
    Ez,
    /// |E| = √(Ex² + Ey² + Ez²) with each component averaged onto the
    /// integer grid node (single-sided at the walls).
    EMag,
}

/// PML strip index ranges for a half-position axis with `n_h` samples.
#[inline]
fn strips_h(n_h: usize, npml: usize) -> impl Iterator<Item = usize> {
    (0..npml).chain(n_h - npml..n_h)
}

/// PML strip index ranges for an integer-position axis with `n` samples
/// (the outermost samples 0 and n−1 are PEC and never updated).
#[inline]
fn strips_e(n: usize, npml: usize) -> impl Iterator<Item = usize> {
    (1..npml).chain(n - npml..n - 1)
}

#[inline]
fn in_strip_h(i: usize, n_h: usize, npml: usize) -> bool {
    i < npml || i >= n_h - npml
}

#[inline]
fn in_strip_e(i: usize, n: usize, npml: usize) -> bool {
    (i >= 1 && i < npml) || (i + npml >= n && i + 1 < n)
}

/// Run `f(i, a_plane, b_plane, c_plane)` for every x-plane of three
/// equally-shaped arrays — in parallel via rayon when `par`, serially
/// otherwise. Parallel tasks are slabs of consecutive planes grouped to
/// ≥ [`SLAB_TARGET_CELLS`] samples. The per-plane body is the same
/// closure in both paths, so the arithmetic (and therefore the result
/// bits) cannot differ across paths, runs, or thread counts.
fn for_planes<F>(par: bool, a: &mut [f64], b: &mut [f64], c: &mut [f64], plane: usize, f: F)
where
    F: Fn(usize, &mut [f64], &mut [f64], &mut [f64]) + Sync + Send,
{
    if par {
        let pps = (SLAB_TARGET_CELLS / plane).max(1); // planes per slab
        let chunk = pps * plane;
        a.par_chunks_mut(chunk)
            .zip_eq(b.par_chunks_mut(chunk))
            .zip_eq(c.par_chunks_mut(chunk))
            .enumerate()
            .for_each(|(s, ((sa, sb), sc))| {
                let mut bs = sb.chunks_mut(plane);
                let mut cs = sc.chunks_mut(plane);
                for (p, pa) in sa.chunks_mut(plane).enumerate() {
                    f(s * pps + p, pa, bs.next().unwrap(), cs.next().unwrap());
                }
            });
    } else {
        let mut bs = b.chunks_mut(plane);
        let mut cs = c.chunks_mut(plane);
        for (i, pa) in a.chunks_mut(plane).enumerate() {
            f(i, pa, bs.next().unwrap(), cs.next().unwrap());
        }
    }
}

fn unzip3(v: Vec<(f64, f64, f64)>) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut a = Vec::with_capacity(v.len());
    let mut b = Vec::with_capacity(v.len());
    let mut c = Vec::with_capacity(v.len());
    for (x, y, z) in v {
        a.push(x);
        b.push(y);
        c.push(z);
    }
    (a, b, c)
}

/// 3D Yee grid with CPML on all six faces. Field arrays are public for
/// monitors and the ADE media pass; layouts are documented on each field.
#[derive(Debug, Clone)]
pub struct Grid3d {
    pub nx: usize,
    pub ny: usize,
    pub nz: usize,
    pub npml: usize,
    pub dt: f64,
    /// Ex at (i+½, j, k), dims (nx−1, ny, nz).
    pub ex: Vec<f64>,
    /// Ey at (i, j+½, k), dims (nx, ny−1, nz).
    pub ey: Vec<f64>,
    /// Ez at (i, j, k+½), dims (nx, ny, nz−1).
    pub ez: Vec<f64>,
    /// Hx at (i, j+½, k+½), dims (nx, ny−1, nz−1).
    pub hx: Vec<f64>,
    /// Hy at (i+½, j, k+½), dims (nx−1, ny, nz−1).
    pub hy: Vec<f64>,
    /// Hz at (i+½, j+½, k), dims (nx−1, ny−1, nz).
    pub hz: Vec<f64>,
    // Update coefficients Δt/ε_r at each E component's own position.
    ce_x: Vec<f64>,
    ce_y: Vec<f64>,
    ce_z: Vec<f64>,
    // CPML tables per axis; *_e at integer positions, *_h at half positions.
    inv_kx_e: Vec<f64>,
    b_x_e: Vec<f64>,
    c_x_e: Vec<f64>,
    inv_kx_h: Vec<f64>,
    b_x_h: Vec<f64>,
    c_x_h: Vec<f64>,
    inv_ky_e: Vec<f64>,
    b_y_e: Vec<f64>,
    c_y_e: Vec<f64>,
    inv_ky_h: Vec<f64>,
    b_y_h: Vec<f64>,
    c_y_h: Vec<f64>,
    inv_kz_e: Vec<f64>,
    b_z_e: Vec<f64>,
    c_z_e: Vec<f64>,
    inv_kz_h: Vec<f64>,
    b_z_h: Vec<f64>,
    c_z_h: Vec<f64>,
    // CPML ψ memory variables: one per (component, PML-normal derivative),
    // each shaped like its component (zero outside the strips).
    psi_ex_y: Vec<f64>,
    psi_ex_z: Vec<f64>,
    psi_ey_z: Vec<f64>,
    psi_ey_x: Vec<f64>,
    psi_ez_x: Vec<f64>,
    psi_ez_y: Vec<f64>,
    psi_hx_y: Vec<f64>,
    psi_hx_z: Vec<f64>,
    psi_hy_z: Vec<f64>,
    psi_hy_x: Vec<f64>,
    psi_hz_x: Vec<f64>,
    psi_hz_y: Vec<f64>,
    /// Parallel slab decomposition engaged (grid ≥ [`PAR_MIN_CELLS`]).
    par: bool,
}

impl Grid3d {
    /// Build a grid. `eps` holds the per-sample relative permittivity at
    /// each E component's own staggered position: `eps[0]` shaped like Ex,
    /// `eps[1]` like Ey, `eps[2]` like Ez (see [`crate::EpsSpec3::build`]).
    pub fn new(spec: &Grid3Spec, eps: [Vec<f64>; 3]) -> Self {
        let (nx, ny, nz, npml) = (spec.nx, spec.ny, spec.nz, spec.npml);
        assert!(
            nx > 2 * npml + 2 && ny > 2 * npml + 2 && nz > 2 * npml + 2,
            "grid must leave interior cells beyond the CPML"
        );
        assert!(spec.courant > 0.0, "courant must be positive");
        let dt = spec.courant / 3f64.sqrt();
        let [eps_x, eps_y, eps_z] = eps;
        assert_eq!(eps_x.len(), (nx - 1) * ny * nz, "eps[0] must be Ex-shaped");
        assert_eq!(eps_y.len(), nx * (ny - 1) * nz, "eps[1] must be Ey-shaped");
        assert_eq!(eps_z.len(), nx * ny * (nz - 1), "eps[2] must be Ez-shaped");
        for e in eps_x.iter().chain(&eps_y).chain(&eps_z) {
            assert!(*e > 0.0, "eps_r must be positive");
        }
        let ce = |v: &[f64]| v.iter().map(|&e| dt / e).collect::<Vec<f64>>();

        let p = &spec.cpml;
        let at = |pos: f64, n: usize| cpml_coeffs(pos, n, npml, dt, p);
        let (inv_kx_e, b_x_e, c_x_e) = unzip3((0..nx).map(|i| at(i as f64, nx)).collect());
        let (inv_kx_h, b_x_h, c_x_h) =
            unzip3((0..nx - 1).map(|i| at(i as f64 + 0.5, nx)).collect());
        let (inv_ky_e, b_y_e, c_y_e) = unzip3((0..ny).map(|j| at(j as f64, ny)).collect());
        let (inv_ky_h, b_y_h, c_y_h) =
            unzip3((0..ny - 1).map(|j| at(j as f64 + 0.5, ny)).collect());
        let (inv_kz_e, b_z_e, c_z_e) = unzip3((0..nz).map(|k| at(k as f64, nz)).collect());
        let (inv_kz_h, b_z_h, c_z_h) =
            unzip3((0..nz - 1).map(|k| at(k as f64 + 0.5, nz)).collect());

        Self {
            nx,
            ny,
            nz,
            npml,
            dt,
            ex: vec![0.0; (nx - 1) * ny * nz],
            ey: vec![0.0; nx * (ny - 1) * nz],
            ez: vec![0.0; nx * ny * (nz - 1)],
            hx: vec![0.0; nx * (ny - 1) * (nz - 1)],
            hy: vec![0.0; (nx - 1) * ny * (nz - 1)],
            hz: vec![0.0; (nx - 1) * (ny - 1) * nz],
            ce_x: ce(&eps_x),
            ce_y: ce(&eps_y),
            ce_z: ce(&eps_z),
            inv_kx_e,
            b_x_e,
            c_x_e,
            inv_kx_h,
            b_x_h,
            c_x_h,
            inv_ky_e,
            b_y_e,
            c_y_e,
            inv_ky_h,
            b_y_h,
            c_y_h,
            inv_kz_e,
            b_z_e,
            c_z_e,
            inv_kz_h,
            b_z_h,
            c_z_h,
            psi_ex_y: vec![0.0; (nx - 1) * ny * nz],
            psi_ex_z: vec![0.0; (nx - 1) * ny * nz],
            psi_ey_z: vec![0.0; nx * (ny - 1) * nz],
            psi_ey_x: vec![0.0; nx * (ny - 1) * nz],
            psi_ez_x: vec![0.0; nx * ny * (nz - 1)],
            psi_ez_y: vec![0.0; nx * ny * (nz - 1)],
            psi_hx_y: vec![0.0; nx * (ny - 1) * (nz - 1)],
            psi_hx_z: vec![0.0; nx * (ny - 1) * (nz - 1)],
            psi_hy_z: vec![0.0; (nx - 1) * ny * (nz - 1)],
            psi_hy_x: vec![0.0; (nx - 1) * ny * (nz - 1)],
            psi_hz_x: vec![0.0; (nx - 1) * (ny - 1) * nz],
            psi_hz_y: vec![0.0; (nx - 1) * (ny - 1) * nz],
            par: nx * ny * nz >= PAR_MIN_CELLS,
        }
    }

    /// One leapfrog step (non-dispersive media): H to (n+½)Δt, E to
    /// (n+1)Δt. Dispersive runs interleave the ADE passes of
    /// [`crate::medium3::DispMedia`] between [`Self::update_h`] and after
    /// [`Self::update_e`].
    pub fn step(&mut self) {
        self.update_h();
        self.update_e();
    }

    #[inline]
    fn idx(&self, comp: Comp, i: usize, j: usize, k: usize) -> usize {
        match comp {
            Comp::Ex => (i * self.ny + j) * self.nz + k,
            Comp::Ey => (i * (self.ny - 1) + j) * self.nz + k,
            Comp::Ez => (i * self.ny + j) * (self.nz - 1) + k,
        }
    }

    /// Sample dimensions of an E component: (along x, along y, along z).
    pub fn comp_dims(&self, comp: Comp) -> (usize, usize, usize) {
        match comp {
            Comp::Ex => (self.nx - 1, self.ny, self.nz),
            Comp::Ey => (self.nx, self.ny - 1, self.nz),
            Comp::Ez => (self.nx, self.ny, self.nz - 1),
        }
    }

    /// Read an E component at its sample indices (i, j, k).
    #[inline]
    pub fn e_at(&self, comp: Comp, i: usize, j: usize, k: usize) -> f64 {
        let idx = self.idx(comp, i, j, k);
        match comp {
            Comp::Ex => self.ex[idx],
            Comp::Ey => self.ey[idx],
            Comp::Ez => self.ez[idx],
        }
    }

    /// Add `v` onto an E component sample (soft sources).
    #[inline]
    pub fn add_e(&mut self, comp: Comp, i: usize, j: usize, k: usize, v: f64) {
        let idx = self.idx(comp, i, j, k);
        match comp {
            Comp::Ex => self.ex[idx] += v,
            Comp::Ey => self.ey[idx] += v,
            Comp::Ez => self.ez[idx] += v,
        }
    }

    /// Max |E| over all three component arrays (NaN-safe: NaN counts +∞).
    pub fn max_abs_e(&self) -> f64 {
        self.ex
            .iter()
            .chain(&self.ey)
            .chain(&self.ez)
            .map(|&v| v.abs())
            .fold(
                0.0,
                |a, b| if b.is_nan() { f64::INFINITY } else { a.max(b) },
            )
    }

    fn e_node(&self, i: usize, j: usize, k: usize) -> (f64, f64, f64) {
        let avg = |f: &[f64], lo: usize, hi: usize| 0.5 * (f[lo] + f[hi]);
        let exn = match i {
            0 => self.ex[self.idx(Comp::Ex, 0, j, k)],
            i if i + 1 >= self.nx => self.ex[self.idx(Comp::Ex, self.nx - 2, j, k)],
            i => avg(
                &self.ex,
                self.idx(Comp::Ex, i - 1, j, k),
                self.idx(Comp::Ex, i, j, k),
            ),
        };
        let eyn = match j {
            0 => self.ey[self.idx(Comp::Ey, i, 0, k)],
            j if j + 1 >= self.ny => self.ey[self.idx(Comp::Ey, i, self.ny - 2, k)],
            j => avg(
                &self.ey,
                self.idx(Comp::Ey, i, j - 1, k),
                self.idx(Comp::Ey, i, j, k),
            ),
        };
        let ezn = match k {
            0 => self.ez[self.idx(Comp::Ez, i, j, 0)],
            k if k + 1 >= self.nz => self.ez[self.idx(Comp::Ez, i, j, self.nz - 2)],
            k => avg(
                &self.ez,
                self.idx(Comp::Ez, i, j, k - 1),
                self.idx(Comp::Ez, i, j, k),
            ),
        };
        (exn, eyn, ezn)
    }

    /// Extract a 2D rendering slice normal to `axis` at sample `index`:
    /// returns `(n0, n1, data)` with `data[a·n1 + b]` over the two
    /// remaining axes in x→y→z order (X slice → (ny, nz), Y → (nx, nz),
    /// Z → (nx, ny)). [`SliceField::Ez`] samples the staggered Ez array
    /// directly (index clamped along z); [`SliceField::EMag`] averages all
    /// three components onto integer nodes.
    pub fn slice(&self, axis: Axis, index: usize, field: SliceField) -> (usize, usize, Vec<f64>) {
        let (nx, ny, nz) = (self.nx, self.ny, self.nz);
        let node = |i: usize, j: usize, k: usize| -> f64 {
            match field {
                SliceField::Ez => self.ez[self.idx(Comp::Ez, i, j, k.min(nz - 2))],
                SliceField::EMag => {
                    let (a, b, c) = self.e_node(i, j, k);
                    (a * a + b * b + c * c).sqrt()
                }
            }
        };
        match axis {
            Axis::X => {
                let i = index.min(nx - 1);
                let mut data = Vec::with_capacity(ny * nz);
                for j in 0..ny {
                    for k in 0..nz {
                        data.push(node(i, j, k));
                    }
                }
                (ny, nz, data)
            }
            Axis::Y => {
                let j = index.min(ny - 1);
                let mut data = Vec::with_capacity(nx * nz);
                for i in 0..nx {
                    for k in 0..nz {
                        data.push(node(i, j, k));
                    }
                }
                (nx, nz, data)
            }
            Axis::Z => {
                let k = index.min(nz - 1);
                let mut data = Vec::with_capacity(nx * ny);
                for i in 0..nx {
                    for j in 0..ny {
                        data.push(node(i, j, k));
                    }
                }
                (nx, ny, data)
            }
        }
    }

    /// H half-step: all three H components from the curl of E, plus the
    /// CPML ψ recursions. Parallel over x-slabs; see the module docs.
    pub fn update_h(&mut self) {
        let (nx, ny, nz, npml, dt, par) = (self.nx, self.ny, self.nz, self.npml, self.dt, self.par);
        let (ny1, nz1) = (ny - 1, nz - 1);

        // Hx(i, j+½, k+½) ← Hx − Δt·[(∂Ez/∂y)/κy − (∂Ey/∂z)/κz]
        {
            let (ez, ey) = (&self.ez, &self.ey);
            let (iky, ikz) = (&self.inv_ky_h, &self.inv_kz_h);
            let (by, cy) = (&self.b_y_h, &self.c_y_h);
            let (bz, cz) = (&self.b_z_h, &self.c_z_h);
            let plane = ny1 * nz1;
            for_planes(
                par,
                &mut self.hx,
                &mut self.psi_hx_y,
                &mut self.psi_hx_z,
                plane,
                |i, hxp, pyp, pzp| {
                    let ezp = &ez[i * ny * nz1..(i + 1) * ny * nz1];
                    let eyp = &ey[i * ny1 * nz..(i + 1) * ny1 * nz];
                    for j in 0..ny1 {
                        let h = &mut hxp[j * nz1..(j + 1) * nz1];
                        let ez0 = &ezp[j * nz1..(j + 1) * nz1];
                        let ez1 = &ezp[(j + 1) * nz1..(j + 2) * nz1];
                        let eyr = &eyp[j * nz..(j + 1) * nz];
                        let ikyj = iky[j];
                        for k in 0..nz1 {
                            h[k] -=
                                dt * ((ez1[k] - ez0[k]) * ikyj - (eyr[k + 1] - eyr[k]) * ikz[k]);
                        }
                    }
                    for j in strips_h(ny1, npml) {
                        let (b, c) = (by[j], cy[j]);
                        let h = &mut hxp[j * nz1..(j + 1) * nz1];
                        let p = &mut pyp[j * nz1..(j + 1) * nz1];
                        let ez0 = &ezp[j * nz1..(j + 1) * nz1];
                        let ez1 = &ezp[(j + 1) * nz1..(j + 2) * nz1];
                        for k in 0..nz1 {
                            p[k] = b * p[k] + c * (ez1[k] - ez0[k]);
                            h[k] -= dt * p[k];
                        }
                    }
                    for j in 0..ny1 {
                        let h = &mut hxp[j * nz1..(j + 1) * nz1];
                        let p = &mut pzp[j * nz1..(j + 1) * nz1];
                        let eyr = &eyp[j * nz..(j + 1) * nz];
                        for k in strips_h(nz1, npml) {
                            p[k] = bz[k] * p[k] + cz[k] * (eyr[k + 1] - eyr[k]);
                            h[k] += dt * p[k];
                        }
                    }
                },
            );
        }

        // Hy(i+½, j, k+½) ← Hy − Δt·[(∂Ex/∂z)/κz − (∂Ez/∂x)/κx]
        {
            let (ex, ez) = (&self.ex, &self.ez);
            let (ikx, ikz) = (&self.inv_kx_h, &self.inv_kz_h);
            let (bx, cx) = (&self.b_x_h, &self.c_x_h);
            let (bz, cz) = (&self.b_z_h, &self.c_z_h);
            let plane = ny * nz1;
            for_planes(
                par,
                &mut self.hy,
                &mut self.psi_hy_z,
                &mut self.psi_hy_x,
                plane,
                |i, hyp, pzp, pxp| {
                    let exp_ = &ex[i * ny * nz..(i + 1) * ny * nz];
                    let ez0p = &ez[i * ny * nz1..(i + 1) * ny * nz1];
                    let ez1p = &ez[(i + 1) * ny * nz1..(i + 2) * ny * nz1];
                    let ikxi = ikx[i];
                    for j in 0..ny {
                        let h = &mut hyp[j * nz1..(j + 1) * nz1];
                        let exr = &exp_[j * nz..(j + 1) * nz];
                        let ez0 = &ez0p[j * nz1..(j + 1) * nz1];
                        let ez1 = &ez1p[j * nz1..(j + 1) * nz1];
                        for k in 0..nz1 {
                            h[k] -=
                                dt * ((exr[k + 1] - exr[k]) * ikz[k] - (ez1[k] - ez0[k]) * ikxi);
                        }
                    }
                    for j in 0..ny {
                        let h = &mut hyp[j * nz1..(j + 1) * nz1];
                        let p = &mut pzp[j * nz1..(j + 1) * nz1];
                        let exr = &exp_[j * nz..(j + 1) * nz];
                        for k in strips_h(nz1, npml) {
                            p[k] = bz[k] * p[k] + cz[k] * (exr[k + 1] - exr[k]);
                            h[k] -= dt * p[k];
                        }
                    }
                    if in_strip_h(i, nx - 1, npml) {
                        let (b, c) = (bx[i], cx[i]);
                        for j in 0..ny {
                            let h = &mut hyp[j * nz1..(j + 1) * nz1];
                            let p = &mut pxp[j * nz1..(j + 1) * nz1];
                            let ez0 = &ez0p[j * nz1..(j + 1) * nz1];
                            let ez1 = &ez1p[j * nz1..(j + 1) * nz1];
                            for k in 0..nz1 {
                                p[k] = b * p[k] + c * (ez1[k] - ez0[k]);
                                h[k] += dt * p[k];
                            }
                        }
                    }
                },
            );
        }

        // Hz(i+½, j+½, k) ← Hz − Δt·[(∂Ey/∂x)/κx − (∂Ex/∂y)/κy]
        {
            let (ey, ex) = (&self.ey, &self.ex);
            let (ikx, iky) = (&self.inv_kx_h, &self.inv_ky_h);
            let (bx, cx) = (&self.b_x_h, &self.c_x_h);
            let (by, cy) = (&self.b_y_h, &self.c_y_h);
            let plane = ny1 * nz;
            for_planes(
                par,
                &mut self.hz,
                &mut self.psi_hz_x,
                &mut self.psi_hz_y,
                plane,
                |i, hzp, pxp, pyp| {
                    let ey0p = &ey[i * ny1 * nz..(i + 1) * ny1 * nz];
                    let ey1p = &ey[(i + 1) * ny1 * nz..(i + 2) * ny1 * nz];
                    let exp_ = &ex[i * ny * nz..(i + 1) * ny * nz];
                    let ikxi = ikx[i];
                    for j in 0..ny1 {
                        let h = &mut hzp[j * nz..(j + 1) * nz];
                        let ey0 = &ey0p[j * nz..(j + 1) * nz];
                        let ey1 = &ey1p[j * nz..(j + 1) * nz];
                        let ex0 = &exp_[j * nz..(j + 1) * nz];
                        let ex1 = &exp_[(j + 1) * nz..(j + 2) * nz];
                        let ikyj = iky[j];
                        for k in 0..nz {
                            h[k] -= dt * ((ey1[k] - ey0[k]) * ikxi - (ex1[k] - ex0[k]) * ikyj);
                        }
                    }
                    if in_strip_h(i, nx - 1, npml) {
                        let (b, c) = (bx[i], cx[i]);
                        for j in 0..ny1 {
                            let h = &mut hzp[j * nz..(j + 1) * nz];
                            let p = &mut pxp[j * nz..(j + 1) * nz];
                            let ey0 = &ey0p[j * nz..(j + 1) * nz];
                            let ey1 = &ey1p[j * nz..(j + 1) * nz];
                            for k in 0..nz {
                                p[k] = b * p[k] + c * (ey1[k] - ey0[k]);
                                h[k] -= dt * p[k];
                            }
                        }
                    }
                    for j in strips_h(ny1, npml) {
                        let (b, c) = (by[j], cy[j]);
                        let h = &mut hzp[j * nz..(j + 1) * nz];
                        let p = &mut pyp[j * nz..(j + 1) * nz];
                        let ex0 = &exp_[j * nz..(j + 1) * nz];
                        let ex1 = &exp_[(j + 1) * nz..(j + 2) * nz];
                        for k in 0..nz {
                            p[k] = b * p[k] + c * (ex1[k] - ex0[k]);
                            h[k] += dt * p[k];
                        }
                    }
                },
            );
        }
    }

    /// E full step: all three E components from the curl of H, plus the
    /// CPML ψ recursions. Tangential E on the outer boundary is PEC and
    /// never touched. Parallel over x-slabs; see the module docs.
    pub fn update_e(&mut self) {
        let (nx, ny, nz, npml, par) = (self.nx, self.ny, self.nz, self.npml, self.par);
        let (ny1, nz1) = (ny - 1, nz - 1);

        // Ex(i+½, j, k) ← Ex + (Δt/ε)·[(∂Hz/∂y)/κy − (∂Hy/∂z)/κz]
        {
            let (hz, hy, ce_x) = (&self.hz, &self.hy, &self.ce_x);
            let (iky, ikz) = (&self.inv_ky_e, &self.inv_kz_e);
            let (by, cy) = (&self.b_y_e, &self.c_y_e);
            let (bz, cz) = (&self.b_z_e, &self.c_z_e);
            let plane = ny * nz;
            for_planes(
                par,
                &mut self.ex,
                &mut self.psi_ex_y,
                &mut self.psi_ex_z,
                plane,
                |i, exp_, pyp, pzp| {
                    let hzp = &hz[i * ny1 * nz..(i + 1) * ny1 * nz];
                    let hyp = &hy[i * ny * nz1..(i + 1) * ny * nz1];
                    let cep = &ce_x[i * ny * nz..(i + 1) * ny * nz];
                    for j in 1..ny1 {
                        let e = &mut exp_[j * nz..(j + 1) * nz];
                        let ce = &cep[j * nz..(j + 1) * nz];
                        let hz0 = &hzp[(j - 1) * nz..j * nz];
                        let hz1 = &hzp[j * nz..(j + 1) * nz];
                        let hyr = &hyp[j * nz1..(j + 1) * nz1];
                        let ikyj = iky[j];
                        for k in 1..nz1 {
                            e[k] +=
                                ce[k] * ((hz1[k] - hz0[k]) * ikyj - (hyr[k] - hyr[k - 1]) * ikz[k]);
                        }
                    }
                    for j in strips_e(ny, npml) {
                        let (b, c) = (by[j], cy[j]);
                        let e = &mut exp_[j * nz..(j + 1) * nz];
                        let ce = &cep[j * nz..(j + 1) * nz];
                        let p = &mut pyp[j * nz..(j + 1) * nz];
                        let hz0 = &hzp[(j - 1) * nz..j * nz];
                        let hz1 = &hzp[j * nz..(j + 1) * nz];
                        for k in 1..nz1 {
                            p[k] = b * p[k] + c * (hz1[k] - hz0[k]);
                            e[k] += ce[k] * p[k];
                        }
                    }
                    for j in 1..ny1 {
                        let e = &mut exp_[j * nz..(j + 1) * nz];
                        let ce = &cep[j * nz..(j + 1) * nz];
                        let p = &mut pzp[j * nz..(j + 1) * nz];
                        let hyr = &hyp[j * nz1..(j + 1) * nz1];
                        for k in strips_e(nz, npml) {
                            p[k] = bz[k] * p[k] + cz[k] * (hyr[k] - hyr[k - 1]);
                            e[k] -= ce[k] * p[k];
                        }
                    }
                },
            );
        }

        // Ey(i, j+½, k) ← Ey + (Δt/ε)·[(∂Hx/∂z)/κz − (∂Hz/∂x)/κx]
        {
            let (hx, hz, ce_y) = (&self.hx, &self.hz, &self.ce_y);
            let (ikx, ikz) = (&self.inv_kx_e, &self.inv_kz_e);
            let (bx, cx) = (&self.b_x_e, &self.c_x_e);
            let (bz, cz) = (&self.b_z_e, &self.c_z_e);
            let plane = ny1 * nz;
            for_planes(
                par,
                &mut self.ey,
                &mut self.psi_ey_z,
                &mut self.psi_ey_x,
                plane,
                |i, eyp, pzp, pxp| {
                    if i == 0 || i == nx - 1 {
                        return; // tangential to the x walls: PEC
                    }
                    let hxp = &hx[i * ny1 * nz1..(i + 1) * ny1 * nz1];
                    let hz0p = &hz[(i - 1) * ny1 * nz..i * ny1 * nz];
                    let hz1p = &hz[i * ny1 * nz..(i + 1) * ny1 * nz];
                    let cep = &ce_y[i * ny1 * nz..(i + 1) * ny1 * nz];
                    let ikxi = ikx[i];
                    for j in 0..ny1 {
                        let e = &mut eyp[j * nz..(j + 1) * nz];
                        let ce = &cep[j * nz..(j + 1) * nz];
                        let hxr = &hxp[j * nz1..(j + 1) * nz1];
                        let hz0 = &hz0p[j * nz..(j + 1) * nz];
                        let hz1 = &hz1p[j * nz..(j + 1) * nz];
                        for k in 1..nz1 {
                            e[k] +=
                                ce[k] * ((hxr[k] - hxr[k - 1]) * ikz[k] - (hz1[k] - hz0[k]) * ikxi);
                        }
                    }
                    for j in 0..ny1 {
                        let e = &mut eyp[j * nz..(j + 1) * nz];
                        let ce = &cep[j * nz..(j + 1) * nz];
                        let p = &mut pzp[j * nz..(j + 1) * nz];
                        let hxr = &hxp[j * nz1..(j + 1) * nz1];
                        for k in strips_e(nz, npml) {
                            p[k] = bz[k] * p[k] + cz[k] * (hxr[k] - hxr[k - 1]);
                            e[k] += ce[k] * p[k];
                        }
                    }
                    if in_strip_e(i, nx, npml) {
                        let (b, c) = (bx[i], cx[i]);
                        for j in 0..ny1 {
                            let e = &mut eyp[j * nz..(j + 1) * nz];
                            let ce = &cep[j * nz..(j + 1) * nz];
                            let p = &mut pxp[j * nz..(j + 1) * nz];
                            let hz0 = &hz0p[j * nz..(j + 1) * nz];
                            let hz1 = &hz1p[j * nz..(j + 1) * nz];
                            for k in 1..nz1 {
                                p[k] = b * p[k] + c * (hz1[k] - hz0[k]);
                                e[k] -= ce[k] * p[k];
                            }
                        }
                    }
                },
            );
        }

        // Ez(i, j, k+½) ← Ez + (Δt/ε)·[(∂Hy/∂x)/κx − (∂Hx/∂y)/κy]
        {
            let (hy, hx, ce_z) = (&self.hy, &self.hx, &self.ce_z);
            let (ikx, iky) = (&self.inv_kx_e, &self.inv_ky_e);
            let (bx, cx) = (&self.b_x_e, &self.c_x_e);
            let (by, cy) = (&self.b_y_e, &self.c_y_e);
            let plane = ny * nz1;
            for_planes(
                par,
                &mut self.ez,
                &mut self.psi_ez_x,
                &mut self.psi_ez_y,
                plane,
                |i, ezp, pxp, pyp| {
                    if i == 0 || i == nx - 1 {
                        return; // tangential to the x walls: PEC
                    }
                    let hy0p = &hy[(i - 1) * ny * nz1..i * ny * nz1];
                    let hy1p = &hy[i * ny * nz1..(i + 1) * ny * nz1];
                    let hxp = &hx[i * ny1 * nz1..(i + 1) * ny1 * nz1];
                    let cep = &ce_z[i * ny * nz1..(i + 1) * ny * nz1];
                    let ikxi = ikx[i];
                    for j in 1..ny1 {
                        let e = &mut ezp[j * nz1..(j + 1) * nz1];
                        let ce = &cep[j * nz1..(j + 1) * nz1];
                        let hy0 = &hy0p[j * nz1..(j + 1) * nz1];
                        let hy1 = &hy1p[j * nz1..(j + 1) * nz1];
                        let hx0 = &hxp[(j - 1) * nz1..j * nz1];
                        let hx1 = &hxp[j * nz1..(j + 1) * nz1];
                        let ikyj = iky[j];
                        for k in 0..nz1 {
                            e[k] += ce[k] * ((hy1[k] - hy0[k]) * ikxi - (hx1[k] - hx0[k]) * ikyj);
                        }
                    }
                    if in_strip_e(i, nx, npml) {
                        let (b, c) = (bx[i], cx[i]);
                        for j in 1..ny1 {
                            let e = &mut ezp[j * nz1..(j + 1) * nz1];
                            let ce = &cep[j * nz1..(j + 1) * nz1];
                            let p = &mut pxp[j * nz1..(j + 1) * nz1];
                            let hy0 = &hy0p[j * nz1..(j + 1) * nz1];
                            let hy1 = &hy1p[j * nz1..(j + 1) * nz1];
                            for k in 0..nz1 {
                                p[k] = b * p[k] + c * (hy1[k] - hy0[k]);
                                e[k] += ce[k] * p[k];
                            }
                        }
                    }
                    for j in strips_e(ny, npml) {
                        let (b, c) = (by[j], cy[j]);
                        let e = &mut ezp[j * nz1..(j + 1) * nz1];
                        let ce = &cep[j * nz1..(j + 1) * nz1];
                        let p = &mut pyp[j * nz1..(j + 1) * nz1];
                        let hx0 = &hxp[(j - 1) * nz1..j * nz1];
                        let hx1 = &hxp[j * nz1..(j + 1) * nz1];
                        for k in 0..nz1 {
                            p[k] = b * p[k] + c * (hx1[k] - hx0[k]);
                            e[k] -= ce[k] * p[k];
                        }
                    }
                },
            );
        }
    }
}
