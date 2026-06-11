//! Symmetric slab waveguide TE mode solver — the F2 mode-source profile
//! generator and the "slab-waveguide effective index" validation anchor
//! (docs/SIM-ROADMAP.md §3.7).
//!
//! ## Geometry and modes
//!
//! A slab of relative permittivity `eps_core` and full width `2a` (the core)
//! in a uniform `eps_clad` background, invariant along the propagation axis
//! x; the transverse coordinate is y (offset `s = y − y_center`). For the
//! TEz field set this crate evolves (Ez, Hx, Hy), guided modes are the
//! classic **TE slab modes**: Ez(y)·e^{i(βx − ωt)} with
//!
//! ```text
//!   Ez'' + (ε(y)·k₀² − β²)·Ez = 0,    k₀ = ω/c = ω  (c = 1)
//! ```
//!
//! Inside the core the solution oscillates with κ = √(ε_core·k₀² − β²);
//! in the cladding it decays with γ = √(β² − ε_clad·k₀²). Matching Ez and
//! Ez' at |s| = a gives the transcendental dispersion relation. With
//! u = κa, v = γa and the V-number V = k₀·a·√(ε_core − ε_clad)
//! (u² + v² = V²), mode TE_m satisfies (Marcuse, *Theory of Dielectric
//! Optical Waveguides*; Saleh & Teich ch. 8):
//!
//! ```text
//!   v = u·tan(u − m·π/2),     u ∈ (m·π/2, min((m+1)·π/2, V))
//! ```
//!
//! — even modes (m even) are `cos(κs)` in the core, odd modes `sin(κs)`.
//! TE_m is guided iff V > m·π/2 (TE₀ has no cutoff). The left side is
//! strictly increasing and the right side √(V² − u²) strictly decreasing on
//! the bracket, so the root is unique; we bisect it to machine precision.
//! The effective index is n_eff = β/k₀ ∈ (√ε_clad, √ε_core).
//!
//! **Closed-form anchors** (used by the validation suite): at u = π/4 the
//! relation gives v = u, hence V = π/(2√2) — choosing parameters that hit
//! this V makes κ, γ, β, n_eff exact in closed form (e.g. a = 1,
//! ε_core = 2, ε_clad = 1, ω = π/(2√2) ⇒ n_eff = √6/2). Likewise u = π/3
//! ⇒ v = √3·u, V = 2π/3.
//!
//! ## Units
//!
//! Everything is in the crate's normalized grid units: c = 1, lengths in
//! cells, ω in rad per cell-traversal time. `half_width` is the core
//! half-width a in cells.

use std::f64::consts::FRAC_PI_2;

/// A symmetric dielectric slab waveguide (transverse description only).
#[derive(Debug, Clone, Copy)]
pub struct SlabWaveguide {
    /// Core relative permittivity (must exceed `eps_clad`).
    pub eps_core: f64,
    /// Cladding relative permittivity.
    pub eps_clad: f64,
    /// Core half-width a, in cells.
    pub half_width: f64,
}

/// One solved guided TE mode of a [`SlabWaveguide`] at a fixed ω.
#[derive(Debug, Clone, Copy)]
pub struct SlabMode {
    /// Mode order m (0 = fundamental, even symmetry).
    pub m: usize,
    /// Angular frequency the mode was solved at.
    pub omega: f64,
    /// Effective index n_eff = β/k₀.
    pub n_eff: f64,
    /// Propagation constant β (rad/cell).
    pub beta: f64,
    /// Core transverse wavenumber κ (rad/cell).
    pub kappa: f64,
    /// Cladding decay constant γ (1/cell).
    pub gamma: f64,
    /// Core half-width a (cells), copied from the waveguide.
    pub half_width: f64,
}

impl SlabWaveguide {
    /// Normalized frequency V = k₀·a·√(ε_core − ε_clad).
    pub fn v_number(&self, omega: f64) -> f64 {
        omega * self.half_width * (self.eps_core - self.eps_clad).sqrt()
    }

    /// Number of guided TE modes at ω: TE_m exists iff V > m·π/2,
    /// so the count is ⌈2V/π⌉ (≥ 1 for any V > 0 — TE₀ has no cutoff).
    pub fn mode_count(&self, omega: f64) -> usize {
        (self.v_number(omega) / FRAC_PI_2).ceil() as usize
    }

    /// Solve TE_m at ω by bisecting the dispersion relation. Returns `None`
    /// if the mode is not guided (V ≤ m·π/2) or the inputs are degenerate.
    pub fn solve(&self, omega: f64, m: usize) -> Option<SlabMode> {
        if !(omega > 0.0)
            || !(self.half_width > 0.0)
            || !(self.eps_core > self.eps_clad)
            || !(self.eps_clad > 0.0)
        {
            return None;
        }
        let a = self.half_width;
        let v = self.v_number(omega);
        let lo0 = m as f64 * FRAC_PI_2;
        if v <= lo0 * (1.0 + 1e-14) + 1e-300 {
            return None; // at or below the TE_m cutoff
        }
        let hi0 = ((m + 1) as f64 * FRAC_PI_2).min(v);
        // f(u) = u·tan(u − mπ/2) − √(V² − u²): strictly increasing on the
        // bracket, f(lo⁺) < 0 (tan → 0⁺), f(hi⁻) > 0 (tan → +∞ or v-edge).
        let f = |u: f64| u * (u - lo0).tan() - (v * v - u * u).max(0.0).sqrt();
        let (mut lo, mut hi) = (lo0, hi0);
        for _ in 0..200 {
            let mid = 0.5 * (lo + hi);
            if f(mid) > 0.0 {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        let u = 0.5 * (lo + hi);
        let kappa = u / a;
        let gamma = (v * v - u * u).max(0.0).sqrt() / a;
        let beta_sq = self.eps_core * omega * omega - kappa * kappa;
        if beta_sq <= 0.0 {
            return None;
        }
        let beta = beta_sq.sqrt();
        Some(SlabMode {
            m,
            omega,
            n_eff: beta / omega,
            beta,
            kappa,
            gamma,
            half_width: a,
        })
    }
}

impl SlabMode {
    /// Transverse profile E(s) at offset `s = y − y_center` (unnormalized;
    /// the core extremum is 1). Even modes: cos(κs) in the core,
    /// cos(κa)·e^{−γ(|s|−a)} outside; odd modes: sin(κs) and
    /// sign(s)·sin(κa)·e^{−γ(|s|−a)}. Continuity of E and E' at |s| = a is
    /// built in (it is the dispersion relation).
    pub fn profile(&self, s: f64) -> f64 {
        let a = self.half_width;
        let even = self.m % 2 == 0;
        if s.abs() <= a {
            if even {
                (self.kappa * s).cos()
            } else {
                (self.kappa * s).sin()
            }
        } else {
            let edge = if even {
                (self.kappa * a).cos()
            } else {
                (self.kappa * a).sin() * s.signum()
            };
            edge * (-self.gamma * (s.abs() - a)).exp()
        }
    }

    /// Sample the profile on the Ez grid rows j = 0..ny about `y_center`
    /// (cells), zero the PEC ring rows, and normalize to unit L2 norm
    /// (Σ_j E_j² = 1) so port overlaps are direct projection coefficients.
    pub fn sample(&self, ny: usize, y_center: f64) -> Vec<f64> {
        assert!(ny >= 3, "grid too small for a mode profile");
        let mut p: Vec<f64> = (0..ny).map(|j| self.profile(j as f64 - y_center)).collect();
        p[0] = 0.0;
        p[ny - 1] = 0.0;
        let norm = p.iter().map(|x| x * x).sum::<f64>().sqrt();
        assert!(norm > 0.0, "degenerate mode profile");
        for x in &mut p {
            *x /= norm;
        }
        p
    }
}
