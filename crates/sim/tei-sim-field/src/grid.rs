//! 2D TEz Yee grid with CPML absorbing boundaries.
//!
//! ## Yee discretization (Yee 1966; Taflove & Hagness 2005, ch. 3)
//!
//! Fields are z-invariant; we evolve the (Ez, Hx, Hy) component set in
//! normalized units ε₀ = μ₀ = c = 1, Δx = Δy = Δ = 1, with non-magnetic,
//! lossless (σ = 0) dielectric media described by a per-cell relative
//! permittivity ε_r:
//!
//! ```text
//!   ∂Ez/∂t = (1/ε_r)(∂Hy/∂x − ∂Hx/∂y)
//!   ∂Hx/∂t = −∂Ez/∂y
//!   ∂Hy/∂t = +∂Ez/∂x
//! ```
//!
//! Spatial staggering (one Yee cell):
//!
//! ```text
//!        Ez(i,j+1) ── Hy(i+½,j+1) ── Ez(i+1,j+1)
//!           │                            │
//!        Hx(i,j+½)                  Hx(i+1,j+½)
//!           │                            │
//!        Ez(i,j) ──── Hy(i+½,j) ──── Ez(i+1,j)
//! ```
//!
//! E lives at integer time steps n·Δt, H at half steps (n+½)·Δt; centered
//! differences in both space and time give the second-order leapfrog:
//!
//! ```text
//!   Hx ← Hx − Δt·(Ez[i,j+1] − Ez[i,j])           (at i, j+½)
//!   Hy ← Hy + Δt·(Ez[i+1,j] − Ez[i,j])           (at i+½, j)
//!   Ez ← Ez + (Δt/ε_r)·[(Hy[i+½,j] − Hy[i−½,j]) − (Hx[i,j+½] − Hx[i,j−½])]
//! ```
//!
//! **Stability (CFL)**: in 2D the leapfrog is stable iff c·Δt ≤ Δ/√2
//! (Taflove & Hagness §4.7). We parameterize Δt = S·Δ/√2 with Courant
//! number S < 1.
//!
//! **Numerical dispersion** (Taflove & Hagness eq. 4.15, square grid):
//!
//! ```text
//!   [1/(cΔt) · sin(ωΔt/2)]² = [1/Δ · sin(kxΔ/2)]² + [1/Δ · sin(kyΔ/2)]²
//! ```
//!
//! For axis-aligned propagation (ky = 0) this inverts in closed form; see
//! [`yee_axis_wavenumber`]. The validation suite measures the on-grid
//! wavelength of a CW source and checks it against this relation.
//!
//! ## CPML absorbing boundaries (Roden & Gedney 2000)
//!
//! Complex-frequency-shifted PML in the stretched-coordinate formulation:
//! each PML-normal derivative is replaced by ∂/∂w → (1/s_w)·∂/∂w with
//!
//! ```text
//!   s_w(ω) = κ_w + σ_w / (α_w + iω)        (ε₀ = 1)
//! ```
//!
//! Roden & Gedney's convolutional implementation (CPML) realizes 1/s_w in
//! the time domain by recursive convolution: each PML-region derivative
//! gains a memory variable ψ updated per step as
//!
//! ```text
//!   ψⁿ = b·ψⁿ⁻¹ + c·(∂F/∂w)ⁿ
//!   b   = exp[−(σ/κ + α)·Δt]
//!   c   = σ·(b − 1) / [κ·(σ + κ·α)]        (c = 0 when σ = 0)
//! ```
//!
//! and the update uses (1/κ)·∂F/∂w + ψ in place of ∂F/∂w. Profiles are
//! polynomially graded with depth ρ ∈ [0, d] into the PML (d = thickness):
//!
//! ```text
//!   σ(ρ) = σ_max·(ρ/d)^m            m = 3 (default)
//!   κ(ρ) = 1 + (κ_max − 1)·(ρ/d)^m
//!   α(ρ) = α_max·(1 − ρ/d)          (linear, max at the interface)
//! ```
//!
//! with the standard near-optimal conductivity (Taflove & Hagness eq. 7.66,
//! η₀ = √(μ₀/ε₀) = 1 in normalized units, background ε_r = 1 at the walls):
//!
//! ```text
//!   σ_max = 0.8·(m + 1) / (η₀·Δ·√ε_r) = 0.8·(m + 1)
//! ```
//!
//! Coefficients are evaluated at each staggered component's true position
//! (integer for Ez, half-integer for the H component normal-offset), per
//! Roden & Gedney. The outermost grid ring is PEC (Ez ≡ 0); the CPML decays
//! the wave before it reaches that wall.
//!
//! ## Deliberately out at F1 (roadmap §3.7)
//!
//! Dispersive (Lorentz/Drude) and conductive media, 3D, waveguide mode
//! sources, port monitors / S-parameter extraction, far-field transforms.
//! Those are F2/F3.

use serde::{Deserialize, Serialize};

/// CPML grading parameters. Defaults follow the standard recipe discussed
/// in the module docs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CpmlParams {
    /// Polynomial grading order m (3 is the common choice).
    #[serde(default = "default_m")]
    pub m: f64,
    /// Multiplier on the optimal σ_max = 0.8(m+1)/(η₀Δ).
    #[serde(default = "default_sigma_scale")]
    pub sigma_scale: f64,
    /// κ grading maximum (κ ≥ 1; improves evanescent/grazing absorption).
    #[serde(default = "default_kappa_max")]
    pub kappa_max: f64,
    /// CFS α maximum (shifts the pole off ω = 0; helps late-time fields).
    #[serde(default = "default_alpha_max")]
    pub alpha_max: f64,
}

fn default_m() -> f64 {
    3.0
}
fn default_sigma_scale() -> f64 {
    1.0
}
fn default_kappa_max() -> f64 {
    3.0
}
fn default_alpha_max() -> f64 {
    0.05
}

impl Default for CpmlParams {
    fn default() -> Self {
        Self {
            m: default_m(),
            sigma_scale: default_sigma_scale(),
            kappa_max: default_kappa_max(),
            alpha_max: default_alpha_max(),
        }
    }
}

/// Grid construction parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridSpec {
    /// Ez points along x.
    pub nx: usize,
    /// Ez points along y.
    pub ny: usize,
    /// Courant number S; Δt = S·Δ/√2. Stable for S < 1.
    #[serde(default = "default_courant")]
    pub courant: f64,
    /// CPML thickness in cells on each of the four sides.
    #[serde(default = "default_npml")]
    pub npml: usize,
    #[serde(default)]
    pub cpml: CpmlParams,
}

pub(crate) fn default_courant() -> f64 {
    0.5
}
pub(crate) fn default_npml() -> usize {
    10
}

/// Solve the axis-aligned (ky = 0) Yee dispersion relation for k given ω:
/// sin(kΔ/2) = (Δ/(cΔt))·sin(ωΔt/2), in normalized units c = Δ = 1.
/// Returns NaN if ω is beyond the propagating band.
pub fn yee_axis_wavenumber(omega: f64, dt: f64) -> f64 {
    let arg = (omega * dt / 2.0).sin() / dt;
    if arg.abs() > 1.0 {
        return f64::NAN;
    }
    2.0 * arg.asin()
}

/// 2D TEz Yee grid (Ez, Hx, Hy) with CPML on all four sides.
///
/// Storage is row-major with y (j) contiguous: `ez[i*ny + j]`. Hx lives on
/// y-edges (`nx × (ny−1)`), Hy on x-edges (`(nx−1) × ny`).
#[derive(Debug, Clone)]
pub struct Grid2d {
    pub nx: usize,
    pub ny: usize,
    pub npml: usize,
    pub dt: f64,
    /// Ez at (i, j), size nx·ny.
    pub ez: Vec<f64>,
    /// Hx at (i, j+½), size nx·(ny−1).
    pub hx: Vec<f64>,
    /// Hy at (i+½, j), size (nx−1)·ny.
    pub hy: Vec<f64>,
    /// Per-cell relative permittivity (size nx·ny).
    eps_r: Vec<f64>,
    /// Precomputed Ez update coefficient Δt/ε_r.
    ce: Vec<f64>,
    // CPML coefficient tables, evaluated at each component's position.
    // *_e: integer positions (Ez derivatives); *_h: half-integer (H derivatives).
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
    // CPML memory variables ψ (zero outside PML strips).
    psi_ez_x: Vec<f64>,
    psi_ez_y: Vec<f64>,
    psi_hx_y: Vec<f64>,
    psi_hy_x: Vec<f64>,
}

/// (1/κ, b, c) at coordinate `pos` (cells) along an axis with `n` integer
/// grid points and PML thickness `npml`; interfaces sit at npml and
/// n−1−npml. Shared by the 2D (F1) and 3D (F3) grids — the Roden-Gedney
/// recursive-convolution coefficients are per-axis and dimension-agnostic.
pub(crate) fn cpml_coeffs(
    pos: f64,
    n: usize,
    npml: usize,
    dt: f64,
    p: &CpmlParams,
) -> (f64, f64, f64) {
    let d = npml as f64;
    if npml == 0 {
        return (1.0, 1.0, 0.0);
    }
    let depth = (d - pos).max(pos - (n as f64 - 1.0 - d)).max(0.0);
    if depth <= 0.0 {
        return (1.0, 1.0, 0.0);
    }
    let r = (depth / d).min(1.0);
    let sigma_max = p.sigma_scale * 0.8 * (p.m + 1.0); // η₀ = Δ = 1, ε_r(wall) = 1
    let sigma = sigma_max * r.powf(p.m);
    let kappa = 1.0 + (p.kappa_max - 1.0) * r.powf(p.m);
    let alpha = p.alpha_max * (1.0 - r);
    let b = (-(sigma / kappa + alpha) * dt).exp();
    let denom = kappa * (sigma + kappa * alpha);
    let c = if sigma > 0.0 && denom > 0.0 {
        sigma * (b - 1.0) / denom
    } else {
        0.0
    };
    (1.0 / kappa, b, c)
}

impl Grid2d {
    /// Build a grid. `eps_r` is the per-Ez-cell relative permittivity,
    /// length nx·ny, all values ≥ 1 expected (not enforced beyond > 0).
    pub fn new(spec: &GridSpec, eps_r: Vec<f64>) -> Self {
        let (nx, ny, npml) = (spec.nx, spec.ny, spec.npml);
        assert!(
            nx > 2 * npml + 2 && ny > 2 * npml + 2,
            "grid must leave interior cells beyond the CPML"
        );
        assert!(spec.courant > 0.0, "courant must be positive");
        assert_eq!(eps_r.len(), nx * ny, "eps_r must be nx*ny");
        assert!(eps_r.iter().all(|&e| e > 0.0), "eps_r must be positive");
        let dt = spec.courant / 2f64.sqrt();
        let ce: Vec<f64> = eps_r.iter().map(|&e| dt / e).collect();

        let p = &spec.cpml;
        let at = |pos: f64, n: usize| cpml_coeffs(pos, n, npml, dt, p);
        let unzip3 = |v: Vec<(f64, f64, f64)>| {
            let mut a = Vec::with_capacity(v.len());
            let mut b = Vec::with_capacity(v.len());
            let mut c = Vec::with_capacity(v.len());
            for (x, y, z) in v {
                a.push(x);
                b.push(y);
                c.push(z);
            }
            (a, b, c)
        };
        let (inv_kx_e, b_x_e, c_x_e) = unzip3((0..nx).map(|i| at(i as f64, nx)).collect());
        let (inv_kx_h, b_x_h, c_x_h) =
            unzip3((0..nx - 1).map(|i| at(i as f64 + 0.5, nx)).collect());
        let (inv_ky_e, b_y_e, c_y_e) = unzip3((0..ny).map(|j| at(j as f64, ny)).collect());
        let (inv_ky_h, b_y_h, c_y_h) =
            unzip3((0..ny - 1).map(|j| at(j as f64 + 0.5, ny)).collect());

        Self {
            nx,
            ny,
            npml,
            dt,
            ez: vec![0.0; nx * ny],
            hx: vec![0.0; nx * (ny - 1)],
            hy: vec![0.0; (nx - 1) * ny],
            eps_r,
            ce,
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
            psi_ez_x: vec![0.0; nx * ny],
            psi_ez_y: vec![0.0; nx * ny],
            psi_hx_y: vec![0.0; nx * (ny - 1)],
            psi_hy_x: vec![0.0; (nx - 1) * ny],
        }
    }

    #[inline]
    pub fn ez_at(&self, i: usize, j: usize) -> f64 {
        self.ez[i * self.ny + j]
    }

    #[inline]
    pub fn add_ez(&mut self, i: usize, j: usize, v: f64) {
        self.ez[i * self.ny + j] += v;
    }

    /// One leapfrog step: H to (n+½)Δt from Eⁿ, then E to (n+1)Δt.
    pub fn step(&mut self) {
        self.update_h();
        self.update_e();
    }

    fn update_h(&mut self) {
        let (nx, ny, dt, npml) = (self.nx, self.ny, self.dt, self.npml);
        let ny1 = ny - 1;

        // Hx ← Hx − Δt·(∂Ez/∂y)/κ_y   (main pass, whole grid)
        for (hx_row, ez_row) in self.hx.chunks_exact_mut(ny1).zip(self.ez.chunks_exact(ny)) {
            for ((h, w), ik) in hx_row.iter_mut().zip(ez_row.windows(2)).zip(&self.inv_ky_h) {
                *h -= dt * (w[1] - w[0]) * ik;
            }
        }
        // Hx CPML strips (y-normal): ψ ← b·ψ + c·∂Ez/∂y; Hx ← Hx − Δt·ψ
        for i in 0..nx {
            let ez_row = &self.ez[i * ny..(i + 1) * ny];
            let hx_row = &mut self.hx[i * ny1..(i + 1) * ny1];
            let psi_row = &mut self.psi_hx_y[i * ny1..(i + 1) * ny1];
            for j in (0..npml).chain(ny1 - npml..ny1) {
                let d = ez_row[j + 1] - ez_row[j];
                let psi = &mut psi_row[j];
                *psi = self.b_y_h[j] * *psi + self.c_y_h[j] * d;
                hx_row[j] -= dt * *psi;
            }
        }

        // Hy ← Hy + Δt·(∂Ez/∂x)/κ_x
        for (i, hy_row) in self.hy.chunks_exact_mut(ny).enumerate() {
            let e0 = &self.ez[i * ny..(i + 1) * ny];
            let e1 = &self.ez[(i + 1) * ny..(i + 2) * ny];
            let f = dt * self.inv_kx_h[i];
            for ((h, a), b) in hy_row.iter_mut().zip(e0).zip(e1) {
                *h += f * (b - a);
            }
        }
        // Hy CPML strips (x-normal)
        for i in (0..npml).chain(nx - 1 - npml..nx - 1) {
            let e0 = &self.ez[i * ny..(i + 1) * ny];
            let e1 = &self.ez[(i + 1) * ny..(i + 2) * ny];
            let hy_row = &mut self.hy[i * ny..(i + 1) * ny];
            let psi_row = &mut self.psi_hy_x[i * ny..(i + 1) * ny];
            let (b, c) = (self.b_x_h[i], self.c_x_h[i]);
            for (((h, psi), a0), a1) in hy_row.iter_mut().zip(psi_row.iter_mut()).zip(e0).zip(e1) {
                *psi = b * *psi + c * (a1 - a0);
                *h += dt * *psi;
            }
        }
    }

    fn update_e(&mut self) {
        let (nx, ny, npml) = (self.nx, self.ny, self.npml);
        let ny1 = ny - 1;

        // Ez ← Ez + (Δt/ε_r)·[(∂Hy/∂x)/κ_x − (∂Hx/∂y)/κ_y]  (interior; PEC ring)
        for i in 1..nx - 1 {
            let ikx = self.inv_kx_e[i];
            let hy_im1 = &self.hy[(i - 1) * ny..i * ny];
            let hy_i = &self.hy[i * ny..(i + 1) * ny];
            let hx_i = &self.hx[i * ny1..(i + 1) * ny1];
            let ce_row = &self.ce[i * ny..(i + 1) * ny];
            let ez_row = &mut self.ez[i * ny..(i + 1) * ny];
            let it = ez_row[1..ny1]
                .iter_mut()
                .zip(&ce_row[1..ny1])
                .zip(&self.inv_ky_e[1..ny1])
                .zip(&hy_i[1..ny1])
                .zip(&hy_im1[1..ny1])
                .zip(hx_i.windows(2));
            for (((((e, &ce), &iky), &hyc), &hyp), w) in it {
                *e += ce * ((hyc - hyp) * ikx - (w[1] - w[0]) * iky);
            }
        }

        // Ez CPML strips, x-normal: ψ ← b·ψ + c·∂Hy/∂x; Ez ← Ez + (Δt/ε)·ψ
        for i in (1..npml).chain(nx - npml..nx - 1) {
            let hy_im1 = &self.hy[(i - 1) * ny..i * ny];
            let hy_i = &self.hy[i * ny..(i + 1) * ny];
            let ce_row = &self.ce[i * ny..(i + 1) * ny];
            let ez_row = &mut self.ez[i * ny..(i + 1) * ny];
            let psi_row = &mut self.psi_ez_x[i * ny..(i + 1) * ny];
            let (b, c) = (self.b_x_e[i], self.c_x_e[i]);
            for j in 1..ny1 {
                let d = hy_i[j] - hy_im1[j];
                let psi = &mut psi_row[j];
                *psi = b * *psi + c * d;
                ez_row[j] += ce_row[j] * *psi;
            }
        }

        // Ez CPML strips, y-normal: ψ ← b·ψ + c·∂Hx/∂y; Ez ← Ez − (Δt/ε)·ψ
        for i in 1..nx - 1 {
            let hx_i = &self.hx[i * ny1..(i + 1) * ny1];
            let ce_row = &self.ce[i * ny..(i + 1) * ny];
            let ez_row = &mut self.ez[i * ny..(i + 1) * ny];
            let psi_row = &mut self.psi_ez_y[i * ny..(i + 1) * ny];
            for j in (1..npml).chain(ny - npml..ny - 1) {
                let d = hx_i[j] - hx_i[j - 1];
                let psi = &mut psi_row[j];
                *psi = self.b_y_e[j] * *psi + self.c_y_e[j] * d;
                ez_row[j] -= ce_row[j] * *psi;
            }
        }
    }

    /// Total field energy ½Σ(ε_r·Ez² + Hx² + Hy²) over the whole grid
    /// (including the CPML, where it is being dissipated). E and H are
    /// staggered by Δt/2, so this is a diagnostic, not an exact invariant.
    pub fn energy(&self) -> f64 {
        let e: f64 = self
            .ez
            .iter()
            .zip(&self.eps_r)
            .map(|(&f, &eps)| eps * f * f)
            .sum();
        let hx: f64 = self.hx.iter().map(|&f| f * f).sum();
        let hy: f64 = self.hy.iter().map(|&f| f * f).sum();
        0.5 * (e + hx + hy)
    }

    /// Max |Ez| over the grid (NaN-safe: NaN compares as +∞ here).
    pub fn ez_max_abs(&self) -> f64 {
        self.ez.iter().map(|&v| v.abs()).fold(0.0, |a, b| {
            if b.is_nan() { f64::INFINITY } else { a.max(b) }
        })
    }

    /// Flat copy of the Ez field (row-major, `ez[i*ny + j]`) for rendering.
    pub fn snapshot_ez(&self) -> Vec<f64> {
        self.ez.clone()
    }
}
