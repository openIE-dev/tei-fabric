//! Gaussian measurements: homodyne (with conditional collapse) and heterodyne.
//!
//! # Homodyne conditioning — the Schur-complement derivation
//!
//! Homodyne at angle φ on mode k projects onto eigenstates of the rotated
//! quadrature x̂_φ = x̂ cos φ + p̂ sin φ. For a Gaussian state the joint
//! distribution of (remaining quadratures r_A, measured quadrature x_B) is a
//! classical Gaussian in the measured variable, so conditioning is the
//! textbook Gaussian-conditional / Schur-complement update. Partition
//!
//!   μ = (μ_A, μ_B),   σ = [[Σ_AA, Σ_AB], [Σ_BA, Σ_BB]],   Σ_BB scalar here,
//!
//! then given outcome x_B = m ([Weedbrook RMP 2012] §V.B, Eq. (114) with the
//! projector Π = diag(1,0) on the measured mode and the pseudo-inverse
//! (Π Σ Π)⁺ collapsing to 1/Σ_BB):
//!
//!   μ_A|B = μ_A + Σ_AB (m − μ_B) / Σ_BB
//!   Σ_A|B = Σ_AA − Σ_AB Σ_BA / Σ_BB        (Schur complement of Σ_BB).
//!
//! The measured mode's conjugate quadrature p̂_φ is left infinitely uncertain
//! by an ideal homodyne detector; we trace the measured mode out entirely
//! (drop its two rows/columns), which for Gaussian states is exactly the
//! conditional state of the remaining N−1 modes. The state therefore
//! **shrinks by one mode per homodyne measurement**.
//!
//! The outcome itself is drawn from the marginal N(μ_B, Σ_BB) via the core
//! Box-Muller normal — deterministic given the seed.

use crate::GaussianState;
use tei_sim_core::linalg::Mat;
use tei_sim_core::rng::Rng;

/// Result of a homodyne measurement.
#[derive(Debug, Clone, Copy)]
pub struct HomodyneOutcome {
    /// Marginal mean of the measured quadrature x̂_φ.
    pub mean: f64,
    /// Marginal variance of x̂_φ (vacuum = 1 in ħ = 2 units).
    pub variance: f64,
    /// The sampled outcome m ~ N(mean, variance).
    pub sample: f64,
}

impl GaussianState {
    /// Marginal (mean, variance) of x̂_φ = x̂ₖ cos φ + p̂ₖ sin φ on `mode`,
    /// without disturbing the state.
    ///
    /// mean = cᵀμₖ, variance = cᵀσₖc with c = (cos φ, sin φ) — the 1-D
    /// marginal of a Gaussian is Gaussian with the projected moments.
    pub fn homodyne_marginal(&self, mode: usize, phi: f64) -> (f64, f64) {
        assert!(mode < self.n_modes());
        let (s, c) = phi.sin_cos();
        let (x, p) = (2 * mode, 2 * mode + 1);
        let mean = c * self.mean[x] + s * self.mean[p];
        let var =
            c * c * self.cov[(x, x)] + 2.0 * c * s * self.cov[(x, p)] + s * s * self.cov[(p, p)];
        (mean, var)
    }

    /// Condition on a **given** homodyne outcome `m` for x̂_φ on `mode`:
    /// applies the Schur-complement update (module docs) to the remaining
    /// modes and removes `mode` from the state.
    pub fn homodyne_project(&mut self, mode: usize, phi: f64, m: f64) {
        assert!(mode < self.n_modes());
        // Rotate mode by φ so that x̂_φ becomes the mode's x̂ quadrature
        // (R(φ): x̂′ = x̂ cos φ + p̂ sin φ — exactly x̂_φ).
        if phi != 0.0 {
            self.rotate(mode, phi);
        }
        let q = 2 * mode; // measured index (x̂ of `mode`)
        let n2 = self.mean.len();
        let keep: Vec<usize> = (0..n2).filter(|&j| j != q && j != q + 1).collect();
        let sigma_bb = self.cov[(q, q)];
        assert!(
            sigma_bb > 0.0,
            "measured quadrature has non-positive variance"
        );
        let gain = (m - self.mean[q]) / sigma_bb;
        let mut mean = Vec::with_capacity(keep.len());
        let mut cov = Mat::zeros(keep.len(), keep.len());
        for (a, &ja) in keep.iter().enumerate() {
            mean.push(self.mean[ja] + self.cov[(ja, q)] * gain);
            for (b, &jb) in keep.iter().enumerate() {
                cov[(a, b)] = self.cov[(ja, jb)] - self.cov[(ja, q)] * self.cov[(jb, q)] / sigma_bb;
            }
        }
        self.mean = mean;
        self.cov = cov;
    }

    /// Homodyne measurement of x̂_φ on `mode`: samples an outcome from the
    /// marginal N(mean, variance) (Box-Muller via the core RNG), conditions
    /// the remaining modes, and removes the measured mode.
    pub fn homodyne(&mut self, mode: usize, phi: f64, rng: &mut Rng) -> HomodyneOutcome {
        let (mean, variance) = self.homodyne_marginal(mode, phi);
        let sample = mean + variance.sqrt() * rng.normal();
        self.homodyne_project(mode, phi, sample);
        HomodyneOutcome {
            mean,
            variance,
            sample,
        }
    }

    /// Heterodyne (double-homodyne / Husimi-Q) outcome statistics on `mode`:
    /// returns the outcome mean (μ_x, μ_p) and outcome covariance σₖ + I.
    ///
    /// Splitting the mode on a 50:50 beamsplitter against vacuum and
    /// homodyning both ports adds exactly one unit of vacuum noise to each
    /// quadrature in ħ = 2 units: the outcome distribution is the Husimi Q
    /// function, a Gaussian with covariance σₖ + I ([Weedbrook RMP 2012]
    /// §V.B.2). Returned covariance is packed as [σxx+1, σxp, σpp+1].
    pub fn heterodyne_marginal(&self, mode: usize) -> ((f64, f64), [f64; 3]) {
        assert!(mode < self.n_modes());
        let (x, p) = (2 * mode, 2 * mode + 1);
        (
            (self.mean[x], self.mean[p]),
            [
                self.cov[(x, x)] + 1.0,
                self.cov[(x, p)],
                self.cov[(p, p)] + 1.0,
            ],
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Heterodyne on vacuum: outcome covariance = 2I (one unit added).
    #[test]
    fn heterodyne_vacuum() {
        let st = GaussianState::vacuum(1);
        let ((mx, mp), c) = st.heterodyne_marginal(0);
        assert_eq!((mx, mp), (0.0, 0.0));
        assert_eq!(c, [2.0, 0.0, 2.0]);
    }

    /// Homodyne marginal at angle φ on squeezed vacuum interpolates
    /// e^{−2r}cos²φ + e^{2r}sin²φ.
    #[test]
    fn rotated_marginal_on_squeezed() {
        let r = 0.8;
        let st = GaussianState::squeezed(&[(r, 0.0)]);
        for &phi in &[0.0, 0.3, std::f64::consts::FRAC_PI_2] {
            let (_, v) = st.homodyne_marginal(0, phi);
            let want = (-2.0 * r).exp() * phi.cos().powi(2) + (2.0 * r).exp() * phi.sin().powi(2);
            assert!((v - want).abs() < 1e-12, "phi={phi}: {v} vs {want}");
        }
    }

    /// Projecting on an uncorrelated mode leaves the other mode untouched.
    #[test]
    fn projection_independent_modes() {
        let mut st = GaussianState::squeezed(&[(0.5, 0.0), (0.0, 0.0)]);
        st.homodyne_project(1, 0.0, 1.7);
        assert_eq!(st.n_modes(), 1);
        let (vx, vp) = st.quadrature_variances(0);
        assert!((vx - (-1.0f64).exp()).abs() < 1e-12);
        assert!((vp - 1.0f64.exp()).abs() < 1e-12);
        assert!(st.mean[0].abs() < 1e-15 && st.mean[1].abs() < 1e-15);
    }
}
