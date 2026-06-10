//! tei-sim-gaussian — SF-Gaussian-class (Xanadu) continuous-variable simulator.
//!
//! Functional simulator for the Gaussian quantum-optics column: N-mode
//! Gaussian states held exactly as a (mean vector, covariance matrix) pair —
//! the covariance-matrix formalism of Strawberry Fields' Gaussian backend and
//! thewalrus. There is **no Fock-space truncation anywhere**: every Gaussian
//! unitary acts as a symplectic matrix on 2N real means and a real symmetric
//! 2N×2N covariance, so states stay exact to machine precision at any
//! squeezing strength.
//!
//! Canonical reference: C. Weedbrook, S. Pirandola, R. García-Patrón,
//! N. J. Cerf, T. C. Ralph, J. H. Shapiro, S. Lloyd, *"Gaussian quantum
//! information"*, Rev. Mod. Phys. **84**, 621 (2012), arXiv:1110.3234 —
//! cited below as [Weedbrook RMP 2012].
//!
//! # Conventions (binding for the whole crate)
//!
//! * **Quadrature ordering** — interleaved x̂₁, p̂₁, x̂₂, p̂₂, …, x̂_N, p̂_N
//!   ("xpxp"): mode k owns vector indices 2k (x̂ₖ) and 2k+1 (p̂ₖ).
//! * **ħ = 2** — quadratures are x̂ = â + â†, p̂ = −i(â − â†), so
//!   [x̂, p̂] = 2i (i.e. ħ ≡ 2) and the **vacuum covariance is exactly the
//!   identity**, σ_vac = I. This matches [Weedbrook RMP 2012] §II.A and
//!   Strawberry Fields' `hbar=2` default. (The alternative x̂ = (â+â†)/√2
//!   convention gives σ_vac = I/2 and is NOT used here.)
//! * **Covariance** — σ_jk = ½⟨Δr̂_j Δr̂_k + Δr̂_k Δr̂_j⟩ with Δr̂ = r̂ − ⟨r̂⟩.
//! * **Symplectic form** — Ω = ⊕ₖ [[0, 1], [−1, 0]], so [r̂_j, r̂_k] = 2i Ω_jk.
//!   A real matrix S is symplectic iff S Ω Sᵀ = Ω; Gaussian unitaries map
//!   σ ↦ S σ Sᵀ and μ ↦ S μ ([Weedbrook RMP 2012] Eq. (30)).
//! * **Physicality** — σ is a valid quantum covariance iff σ + iΩ ⪰ 0,
//!   equivalently all symplectic eigenvalues ν_k ≥ 1 in these units
//!   ([Weedbrook RMP 2012] Eqs. (32)–(34)). See [`williamson`].
//!
//! Validation is analytic-only per the roadmap policy: squeezed-vacuum
//! variances e^{∓2r}, Sp(2N,ℝ) invariance of every operation, EPR variances
//! 2e^{−2r} of the two-mode squeezed state, Schur-complement homodyne
//! conditioning, thermal-state moments, and the σ + iΩ ⪰ 0 margin.

pub mod exec;
pub mod fidelity;
pub mod measure;
pub mod williamson;

pub use exec::{GaussianExecutor, GaussianJob, GaussianOp, HomodyneSpec};
pub use fidelity::fidelity_single_mode;
pub use measure::HomodyneOutcome;
pub use williamson::{is_physical, physicality_margin, symplectic_eigenvalues};

use tei_sim_core::linalg::Mat;

/// The symplectic form Ω = ⊕ₖ [[0, 1], [−1, 0]] on `n_modes` modes (2N×2N).
pub fn omega(n_modes: usize) -> Mat {
    let mut o = Mat::zeros(2 * n_modes, 2 * n_modes);
    for k in 0..n_modes {
        o[(2 * k, 2 * k + 1)] = 1.0;
        o[(2 * k + 1, 2 * k)] = -1.0;
    }
    o
}

/// Single-mode phase rotation R(φ): â ↦ â e^{−iφ}.
///
/// On quadratures (x̂ = â + â†, p̂ = −i(â − â†)):
/// x̂′ = x̂ cos φ + p̂ sin φ, p̂′ = −x̂ sin φ + p̂ cos φ — an SO(2) rotation,
/// hence det = 1 and (being 2×2) symplectic.
pub fn rotation(phi: f64) -> Mat {
    let (s, c) = phi.sin_cos();
    Mat::from_rows(&[&[c, s], &[-s, c]])
}

/// Single-mode squeezer S(r, φ) for Ŝ(ζ) = exp[½(ζ* â² − ζ â†²)], ζ = r e^{iφ}.
///
/// Symplectic action ([Weedbrook RMP 2012] §III.B.1, rotated through φ/2):
///
/// S = [[cosh r − sinh r cos φ,  −sinh r sin φ],
///      [−sinh r sin φ,           cosh r + sinh r cos φ]]
///
/// At φ = 0 this is diag(e^{−r}, e^{r}): x̂ squeezed, p̂ anti-squeezed; the
/// squeezing axis sits at angle φ/2 in phase space. det S = cosh²r −
/// sinh²r(cos²φ + sin²φ) = 1, so S ∈ Sp(2,ℝ).
pub fn squeezer(r: f64, phi: f64) -> Mat {
    let (ch, sh) = (r.cosh(), r.sinh());
    let (s, c) = phi.sin_cos();
    Mat::from_rows(&[&[ch - sh * c, -sh * s], &[-sh * s, ch + sh * c]])
}

/// Two-mode beamsplitter BS(θ) with transmissivity T = cos²θ.
///
/// Acts on (x̂ₐ, p̂ₐ, x̂_b, p̂_b) as [[cos θ·I₂, sin θ·I₂], [−sin θ·I₂, cos θ·I₂]]
/// ([Weedbrook RMP 2012] §III.B.2). Orthogonal with det = 1 and block-scalar
/// 2×2 structure, hence symplectic. Sign convention: the reflected port picks
/// up the minus sign (â ↦ â cos θ + b̂ sin θ, b̂ ↦ −â sin θ + b̂ cos θ).
pub fn beamsplitter(theta: f64) -> Mat {
    let (s, c) = theta.sin_cos();
    Mat::from_rows(&[
        &[c, 0.0, s, 0.0],
        &[0.0, c, 0.0, s],
        &[-s, 0.0, c, 0.0],
        &[0.0, -s, 0.0, c],
    ])
}

/// Two-mode squeezer TMS(r) for Ŝ₂(r) = exp[r(â b̂ − â† b̂†)].
///
/// S = [[cosh r·I₂, sinh r·Z], [sinh r·Z, cosh r·I₂]], Z = diag(1, −1)
/// ([Weedbrook RMP 2012] §III.B.2). On two-mode vacuum it produces the EPR
/// state with covariance [[cosh 2r·I₂, sinh 2r·Z], [sinh 2r·Z, cosh 2r·I₂]].
pub fn two_mode_squeezer(r: f64) -> Mat {
    let (ch, sh) = (r.cosh(), r.sinh());
    Mat::from_rows(&[
        &[ch, 0.0, sh, 0.0],
        &[0.0, ch, 0.0, -sh],
        &[sh, 0.0, ch, 0.0],
        &[0.0, -sh, 0.0, ch],
    ])
}

/// Embed a 2×2 symplectic acting on `mode` into the full 2N×2N identity.
pub fn embed_one_mode(n_modes: usize, mode: usize, s2: &Mat) -> Mat {
    assert!(mode < n_modes);
    assert_eq!((s2.rows, s2.cols), (2, 2));
    let mut s = Mat::identity(2 * n_modes);
    for i in 0..2 {
        for j in 0..2 {
            s[(2 * mode + i, 2 * mode + j)] = s2[(i, j)];
        }
    }
    s
}

/// Embed a 4×4 symplectic acting on (`mode_a`, `mode_b`) into 2N×2N.
pub fn embed_two_mode(n_modes: usize, mode_a: usize, mode_b: usize, s4: &Mat) -> Mat {
    assert!(mode_a < n_modes && mode_b < n_modes && mode_a != mode_b);
    assert_eq!((s4.rows, s4.cols), (4, 4));
    let mut s = Mat::identity(2 * n_modes);
    let base = [2 * mode_a, 2 * mode_a + 1, 2 * mode_b, 2 * mode_b + 1];
    for (i, &ri) in base.iter().enumerate() {
        for (j, &cj) in base.iter().enumerate() {
            s[(ri, cj)] = s4[(i, j)];
        }
    }
    s
}

fn matvec(m: &Mat, v: &[f64]) -> Vec<f64> {
    assert_eq!(m.cols, v.len());
    (0..m.rows)
        .map(|i| (0..m.cols).map(|j| m[(i, j)] * v[j]).sum())
        .collect()
}

/// An N-mode Gaussian state: mean vector μ (length 2N, xpxp order) and real
/// symmetric covariance σ (2N×2N), in ħ = 2 units (vacuum σ = I).
#[derive(Debug, Clone)]
pub struct GaussianState {
    /// ⟨r̂⟩ = (⟨x̂₁⟩, ⟨p̂₁⟩, …, ⟨x̂_N⟩, ⟨p̂_N⟩).
    pub mean: Vec<f64>,
    /// σ_jk = ½⟨{Δr̂_j, Δr̂_k}⟩.
    pub cov: Mat,
}

impl GaussianState {
    /// N-mode vacuum: μ = 0, σ = I (exact in ħ = 2 units).
    pub fn vacuum(n_modes: usize) -> Self {
        Self {
            mean: vec![0.0; 2 * n_modes],
            cov: Mat::identity(2 * n_modes),
        }
    }

    /// Coherent state ⊗ₖ |αₖ⟩, αₖ = (Re α, Im α) per mode.
    ///
    /// With x̂ = â + â†: ⟨x̂⟩ = 2 Re α, ⟨p̂⟩ = 2 Im α; the covariance is the
    /// vacuum's (displacement does not touch second moments).
    pub fn coherent(alphas: &[(f64, f64)]) -> Self {
        let mut st = Self::vacuum(alphas.len());
        for (k, &(re, im)) in alphas.iter().enumerate() {
            st.displace(k, re, im);
        }
        st
    }

    /// Squeezed vacuum ⊗ₖ Ŝ(rₖ e^{iφₖ})|0⟩, (r, φ) per mode.
    ///
    /// At φ = 0: σ = diag(e^{−2r}, e^{2r}) — Var(x̂) = e^{−2r}, Var(p̂) = e^{2r}
    /// (vacuum = 1 in ħ = 2 units).
    pub fn squeezed(params: &[(f64, f64)]) -> Self {
        let mut st = Self::vacuum(params.len());
        for (k, &(r, phi)) in params.iter().enumerate() {
            st.squeeze(k, r, phi);
        }
        st
    }

    /// Thermal state ⊗ₖ ρ_th(n̄ₖ): μ = 0, σₖ = (2n̄ₖ + 1)·I₂.
    ///
    /// Derivation: ⟨x̂²⟩ = ⟨(â + â†)²⟩ = 2⟨â†â⟩ + 1 + 2 Re⟨â²⟩ = 2n̄ + 1 for a
    /// thermal state (⟨â²⟩ = 0), and likewise for p̂ ([Weedbrook RMP 2012]
    /// §II.C, ħ = 2 units).
    pub fn thermal(nbars: &[f64]) -> Self {
        let mut st = Self::vacuum(nbars.len());
        for (k, &nb) in nbars.iter().enumerate() {
            assert!(nb >= 0.0, "thermal occupation must be ≥ 0");
            let v = 2.0 * nb + 1.0;
            st.cov[(2 * k, 2 * k)] = v;
            st.cov[(2 * k + 1, 2 * k + 1)] = v;
        }
        st
    }

    pub fn n_modes(&self) -> usize {
        self.mean.len() / 2
    }

    /// Apply a full 2N×2N symplectic: σ ← S σ Sᵀ, μ ← S μ
    /// ([Weedbrook RMP 2012] Eq. (30)).
    pub fn apply_symplectic(&mut self, s: &Mat) {
        assert_eq!((s.rows, s.cols), (self.mean.len(), self.mean.len()));
        self.cov = s.matmul(&self.cov).matmul(&s.transpose());
        self.mean = matvec(s, &self.mean);
    }

    /// Single-mode squeezer Ŝ(r e^{iφ}) on `mode`. See [`squeezer`].
    pub fn squeeze(&mut self, mode: usize, r: f64, phi: f64) {
        let s = embed_one_mode(self.n_modes(), mode, &squeezer(r, phi));
        self.apply_symplectic(&s);
    }

    /// Phase rotation R(φ) on `mode`. See [`rotation`].
    pub fn rotate(&mut self, mode: usize, phi: f64) {
        let s = embed_one_mode(self.n_modes(), mode, &rotation(phi));
        self.apply_symplectic(&s);
    }

    /// Displacement D(α), α = re + i·im: shifts the mean by (2 Re α, 2 Im α)
    /// on `mode` and leaves σ untouched (Weyl operator, [Weedbrook RMP 2012]
    /// §II.B).
    pub fn displace(&mut self, mode: usize, re: f64, im: f64) {
        assert!(mode < self.n_modes());
        self.mean[2 * mode] += 2.0 * re;
        self.mean[2 * mode + 1] += 2.0 * im;
    }

    /// Beamsplitter BS(θ) (transmissivity cos²θ) on (`mode_a`, `mode_b`).
    pub fn beamsplit(&mut self, mode_a: usize, mode_b: usize, theta: f64) {
        let s = embed_two_mode(self.n_modes(), mode_a, mode_b, &beamsplitter(theta));
        self.apply_symplectic(&s);
    }

    /// Two-mode squeezer TMS(r) on (`mode_a`, `mode_b`). See
    /// [`two_mode_squeezer`].
    pub fn two_mode_squeeze(&mut self, mode_a: usize, mode_b: usize, r: f64) {
        let s = embed_two_mode(self.n_modes(), mode_a, mode_b, &two_mode_squeezer(r));
        self.apply_symplectic(&s);
    }

    /// Mean of (x̂ₖ, p̂ₖ).
    pub fn mode_mean(&self, mode: usize) -> (f64, f64) {
        (self.mean[2 * mode], self.mean[2 * mode + 1])
    }

    /// (Var x̂ₖ, Var p̂ₖ).
    pub fn quadrature_variances(&self, mode: usize) -> (f64, f64) {
        (
            self.cov[(2 * mode, 2 * mode)],
            self.cov[(2 * mode + 1, 2 * mode + 1)],
        )
    }

    /// Reduced 2×2 covariance of `mode` (partial trace = pick the block).
    pub fn mode_cov(&self, mode: usize) -> Mat {
        let (x, p) = (2 * mode, 2 * mode + 1);
        Mat::from_rows(&[
            &[self.cov[(x, x)], self.cov[(x, p)]],
            &[self.cov[(p, x)], self.cov[(p, p)]],
        ])
    }

    /// Mean photon number ⟨n̂ₖ⟩ of `mode`.
    ///
    /// Derivation (ħ = 2): x̂² + p̂² = (â + â†)² − (â − â†)² = 2(â â† + â† â)
    /// = 2(2n̂ + 1), so ⟨n̂⟩ = (⟨x̂²⟩ + ⟨p̂²⟩ − 2)/4. With ⟨x̂²⟩ = σ_xx + μ_x²:
    ///
    /// ⟨n̂ₖ⟩ = (σ_xx + σ_pp + μ_x² + μ_p² − 2) / 4 = (Tr σₖ + |μₖ|² − 2) / 4.
    ///
    /// Checks: vacuum → 0; coherent → |α|² (μ = (2Reα, 2Imα) gives
    /// |μ|² = 4|α|²); thermal → n̄; squeezed vacuum → sinh²r
    /// ((e^{2r} + e^{−2r} − 2)/4 = (cosh 2r − 1)/2 = sinh²r).
    pub fn mean_photon(&self, mode: usize) -> f64 {
        let (mx, mp) = self.mode_mean(mode);
        let (vx, vp) = self.quadrature_variances(mode);
        (vx + vp + mx * mx + mp * mp - 2.0) / 4.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Construction sanity: coherent mean, squeezed/thermal variances.
    #[test]
    fn constructors() {
        let st = GaussianState::coherent(&[(0.7, -0.3)]);
        assert_eq!(st.mode_mean(0), (1.4, -0.6));
        let st = GaussianState::squeezed(&[(0.5, 0.0)]);
        let (vx, vp) = st.quadrature_variances(0);
        assert!((vx - (-1.0f64).exp()).abs() < 1e-12);
        assert!((vp - 1.0f64.exp()).abs() < 1e-12);
        let st = GaussianState::thermal(&[2.5]);
        assert_eq!(st.quadrature_variances(0), (6.0, 6.0));
    }

    /// embed_two_mode places blocks correctly for non-adjacent modes.
    #[test]
    fn embedding_nonadjacent() {
        let mut st = GaussianState::coherent(&[(1.0, 0.0), (0.0, 0.0), (0.0, 0.0)]);
        st.beamsplit(0, 2, std::f64::consts::FRAC_PI_4);
        let c = std::f64::consts::FRAC_1_SQRT_2;
        assert!((st.mode_mean(0).0 - 2.0 * c).abs() < 1e-12);
        assert!((st.mode_mean(2).0 + 2.0 * c).abs() < 1e-12);
        assert!(st.mode_mean(1).0.abs() < 1e-15);
    }
}
