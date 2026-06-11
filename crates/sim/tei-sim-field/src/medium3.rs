//! Dispersive media for the 3D grid — Drude and single-pole Lorentz via
//! the auxiliary-differential-equation (ADE) method (F3).
//!
//! ## Models (e^{−iωt} convention, ε₀ = 1)
//!
//! ```text
//!   Drude:    ε(ω) = ε∞ − ωp² / (ω² + iγω)
//!   Lorentz:  ε(ω) = ε∞ + ωp² / (ω0² − ω² − iγω)
//! ```
//!
//! ## ADE discretization (Taflove & Hagness 2005, §9.4)
//!
//! **Drude** evolves the polarization current J (per E component):
//! dJ/dt + γJ = ωp²·E, centered at step n with J on half steps:
//!
//! ```text
//!   J^{n+½} = ka·J^{n−½} + kb·Eⁿ
//!   ka = (1 − γΔt/2)/(1 + γΔt/2)      kb = ωp²Δt/(1 + γΔt/2)
//! ```
//!
//! and Ampère gains the current term, E^{n+1} = Eⁿ + (Δt/ε∞)·(∇×H)^{n+½}
//! − (Δt/ε∞)·J^{n+½}. The curl part is the ordinary grid update with
//! Δt/ε∞ baked into the update coefficient; the J term is applied as a
//! post-update correction on the dispersive cells only.
//!
//! **Lorentz** evolves the polarization P:
//! d²P/dt² + γ·dP/dt + ω0²·P = ωp²·E, centered at n:
//!
//! ```text
//!   P^{n+1} = c1·Pⁿ + c2·P^{n−1} + c3·Eⁿ
//!   D = 1/Δt² + γ/(2Δt)
//!   c1 = (2/Δt² − ω0²)/D     c2 = −(1/Δt² − γ/(2Δt))/D     c3 = ωp²/D
//! ```
//!
//! with E^{n+1} = Eⁿ + (Δt/ε∞)·(∇×H)^{n+½} − (P^{n+1} − Pⁿ)/ε∞.
//!
//! Both recurrences read Eⁿ **before** the curl update (the `pre` pass)
//! and correct E **after** it (the `post` pass); the run loop interleaves
//! them as update_h → pre → update_e → post. In the ωp → 0 limit kb and
//! c3 vanish, the auxiliary state stays exactly zero, and the corrections
//! subtract literal 0.0 — a dispersive run degenerates to the vacuum run
//! **bit-exactly** (asserted by the validation suite).
//!
//! ## Region semantics
//!
//! A material occupies the half-open Yee-cell box [i0,i1)×[j0,j1)×[k0,k1):
//! an E sample belongs to the region iff its **integer** indices fall in
//! the box (the +½ stagger stays inside its cell), so a region boundary
//! sits within half a cell of the nominal plane on every face. Inside a
//! region the background ε_r is replaced by the model's ε∞. Regions must
//! not overlap (overlapping regions would superpose polarizations but the
//! last region's ε∞ wins — not a physical configuration).
//!
//! The per-cell auxiliary state is stored sparsely (only dispersive cells
//! carry J or P), and the pre/post passes are sequential: they touch only
//! the region cells, a small fraction of the grid, and keeping them
//! serial keeps the determinism argument trivial.

use serde::{Deserialize, Serialize};

fn one() -> f64 {
    1.0
}

/// Dispersive material model parameters (normalized units, rad per unit
/// time). See the module docs for the permittivity functions.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum MaterialModel {
    /// Drude metal: ε(ω) = ε∞ − ωp²/(ω² + iγω). γ = 0 is the lossless
    /// plasma (total reflection below ωp).
    Drude {
        omega_p: f64,
        #[serde(default)]
        gamma: f64,
        #[serde(default = "one")]
        eps_inf: f64,
    },
    /// Single-pole Lorentz: ε(ω) = ε∞ + ωp²/(ω0² − ω² − iγω).
    Lorentz {
        omega_p: f64,
        omega0: f64,
        #[serde(default)]
        gamma: f64,
        #[serde(default = "one")]
        eps_inf: f64,
    },
}

/// A dispersive material assigned to a Yee-cell box region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MaterialRegion {
    pub model: MaterialModel,
    pub i0: usize,
    pub i1: usize,
    pub j0: usize,
    pub j1: usize,
    pub k0: usize,
    pub k1: usize,
}

#[derive(Debug, Clone)]
struct DrudeCell {
    idx: usize,
    ka: f64,
    kb: f64,
    /// Δt/ε∞ — Ampère current-term coefficient.
    cj: f64,
}

#[derive(Debug, Clone)]
struct LorentzCell {
    idx: usize,
    c1: f64,
    c2: f64,
    c3: f64,
    inv_eps_inf: f64,
}

/// ADE state for one E component.
#[derive(Debug, Clone, Default)]
struct CompDisp {
    drude: Vec<DrudeCell>,
    /// J^{n−½} per Drude cell.
    j: Vec<f64>,
    lorentz: Vec<LorentzCell>,
    /// Pⁿ, P^{n−1}, and the step's P^{n+1} − Pⁿ per Lorentz cell.
    p: Vec<f64>,
    p_prev: Vec<f64>,
    dp: Vec<f64>,
}

impl CompDisp {
    fn pre(&mut self, e: &[f64]) {
        for (c, j) in self.drude.iter().zip(&mut self.j) {
            *j = c.ka * *j + c.kb * e[c.idx];
        }
        for (m, c) in self.lorentz.iter().enumerate() {
            let pn = c.c1 * self.p[m] + c.c2 * self.p_prev[m] + c.c3 * e[c.idx];
            self.dp[m] = pn - self.p[m];
            self.p_prev[m] = self.p[m];
            self.p[m] = pn;
        }
    }

    fn post(&self, e: &mut [f64]) {
        for (c, j) in self.drude.iter().zip(&self.j) {
            e[c.idx] -= c.cj * *j;
        }
        for (m, c) in self.lorentz.iter().enumerate() {
            e[c.idx] -= self.dp[m] * c.inv_eps_inf;
        }
    }

    fn is_empty(&self) -> bool {
        self.drude.is_empty() && self.lorentz.is_empty()
    }
}

/// All dispersive-media state of a 3D run: per-component sparse ADE cells.
#[derive(Debug, Clone, Default)]
pub struct DispMedia {
    x: CompDisp,
    y: CompDisp,
    z: CompDisp,
}

impl DispMedia {
    /// Materialize `regions` on an (nx, ny, nz) grid: bakes each region's
    /// ε∞ into the per-component permittivity arrays (`eps[0]` Ex-shaped,
    /// `eps[1]` Ey-shaped, `eps[2]` Ez-shaped — the arrays later handed to
    /// `Grid3d::new`) and builds the sparse ADE cell lists.
    pub fn build(
        regions: &[MaterialRegion],
        (nx, ny, nz): (usize, usize, usize),
        dt: f64,
        eps: &mut [Vec<f64>; 3],
    ) -> Result<DispMedia, String> {
        let mut media = DispMedia::default();
        // (sample dims, target CompDisp index) per E component.
        let dims = [(nx - 1, ny, nz), (nx, ny - 1, nz), (nx, ny, nz - 1)];
        for r in regions {
            let eps_inf = match r.model {
                MaterialModel::Drude {
                    omega_p,
                    gamma,
                    eps_inf,
                } => {
                    if omega_p < 0.0 || gamma < 0.0 || eps_inf <= 0.0 {
                        return Err(format!(
                            "drude requires omega_p >= 0, gamma >= 0, eps_inf > 0 (got {omega_p}, {gamma}, {eps_inf})"
                        ));
                    }
                    eps_inf
                }
                MaterialModel::Lorentz {
                    omega_p,
                    omega0,
                    gamma,
                    eps_inf,
                } => {
                    if omega_p < 0.0 || omega0 <= 0.0 || gamma < 0.0 || eps_inf <= 0.0 {
                        return Err(format!(
                            "lorentz requires omega_p >= 0, omega0 > 0, gamma >= 0, eps_inf > 0 (got {omega_p}, {omega0}, {gamma}, {eps_inf})"
                        ));
                    }
                    eps_inf
                }
            };
            for (c, &(di, dj, dk)) in dims.iter().enumerate() {
                let comp = match c {
                    0 => &mut media.x,
                    1 => &mut media.y,
                    _ => &mut media.z,
                };
                for i in r.i0..r.i1.min(di) {
                    for j in r.j0..r.j1.min(dj) {
                        for k in r.k0..r.k1.min(dk) {
                            let idx = (i * dj + j) * dk + k;
                            eps[c][idx] = eps_inf;
                            match r.model {
                                MaterialModel::Drude { omega_p, gamma, .. } => {
                                    let den = 1.0 + gamma * dt / 2.0;
                                    comp.drude.push(DrudeCell {
                                        idx,
                                        ka: (1.0 - gamma * dt / 2.0) / den,
                                        kb: omega_p * omega_p * dt / den,
                                        cj: dt / eps_inf,
                                    });
                                    comp.j.push(0.0);
                                }
                                MaterialModel::Lorentz {
                                    omega_p,
                                    omega0,
                                    gamma,
                                    ..
                                } => {
                                    let d = 1.0 / (dt * dt) + gamma / (2.0 * dt);
                                    comp.lorentz.push(LorentzCell {
                                        idx,
                                        c1: (2.0 / (dt * dt) - omega0 * omega0) / d,
                                        c2: -(1.0 / (dt * dt) - gamma / (2.0 * dt)) / d,
                                        c3: omega_p * omega_p / d,
                                        inv_eps_inf: 1.0 / eps_inf,
                                    });
                                    comp.p.push(0.0);
                                    comp.p_prev.push(0.0);
                                    comp.dp.push(0.0);
                                }
                            }
                        }
                    }
                }
            }
        }
        Ok(media)
    }

    /// True when no region contributed any ADE cell.
    pub fn is_empty(&self) -> bool {
        self.x.is_empty() && self.y.is_empty() && self.z.is_empty()
    }

    /// ADE recurrences from Eⁿ — call after `update_h`, before `update_e`.
    pub fn pre(&mut self, ex: &[f64], ey: &[f64], ez: &[f64]) {
        self.x.pre(ex);
        self.y.pre(ey);
        self.z.pre(ez);
    }

    /// Ampère current/polarization corrections — call after `update_e`.
    pub fn post(&self, ex: &mut [f64], ey: &mut [f64], ez: &mut [f64]) {
        self.x.post(ex);
        self.y.post(ey);
        self.z.post(ez);
    }
}
