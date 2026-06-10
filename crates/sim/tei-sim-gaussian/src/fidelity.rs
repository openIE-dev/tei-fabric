//! Closed-form fidelity between single-mode Gaussian states.
//!
//! Uhlmann fidelity F(ρ₁, ρ₂) = (Tr √(√ρ₁ ρ₂ √ρ₁))² for one-mode Gaussians
//! has the closed form of Scutaru (J. Phys. A 31, 3659, 1998), quoted in
//! [Weedbrook RMP 2012] §IV.D. In our ħ = 2 units (vacuum σ = I, pure state
//! ⇔ det σ = 1):
//!
//!   F = 2 · exp(−½ δμᵀ (σ₁ + σ₂)⁻¹ δμ) / (√(Δ + δ) − √δ)
//!
//! with δμ = μ₂ − μ₁, Δ = det(σ₁ + σ₂), δ = (det σ₁ − 1)(det σ₂ − 1).
//!
//! Normalization cross-checked against three independent closed forms
//! (asserted in tests):
//!
//! * identical thermal states: Δ + δ = (t + 1)², √δ = t − 1 with
//!   t = (2n̄+1)², so F = 2/((t+1) − (t−1)) = 1 ✓
//! * vacuum vs coherent |α⟩: Δ = 4, δ = 0, exponent −½·4|α|²·½ = −|α|²,
//!   F = e^{−|α|²} = |⟨0|α⟩|² ✓
//! * vacuum vs squeezed vacuum: Δ = (1+e^{−2r})(1+e^{2r}) = 4cosh²r, δ = 0,
//!   F = 2/(2cosh r) = 1/cosh r = |⟨0|Ŝ(r)|0⟩|² ✓

use crate::GaussianState;

/// Fidelity F ∈ [0, 1] between two **single-mode** Gaussian states
/// (Scutaru closed form; see module docs for derivation and checks).
pub fn fidelity_single_mode(a: &GaussianState, b: &GaussianState) -> f64 {
    assert_eq!(a.n_modes(), 1, "single-mode formula");
    assert_eq!(b.n_modes(), 1, "single-mode formula");
    let det2 = |m: &tei_sim_core::linalg::Mat| m[(0, 0)] * m[(1, 1)] - m[(0, 1)] * m[(1, 0)];

    let s1 = a.mode_cov(0);
    let s2 = b.mode_cov(0);
    let sum = tei_sim_core::linalg::Mat::from_rows(&[
        &[s1[(0, 0)] + s2[(0, 0)], s1[(0, 1)] + s2[(0, 1)]],
        &[s1[(1, 0)] + s2[(1, 0)], s1[(1, 1)] + s2[(1, 1)]],
    ]);
    let big_delta = det2(&sum);
    // δ < 0 only through roundoff on pure states; clamp.
    let small_delta = ((det2(&s1) - 1.0) * (det2(&s2) - 1.0)).max(0.0);

    let du = [b.mean[0] - a.mean[0], b.mean[1] - a.mean[1]];
    // δμᵀ (σ₁+σ₂)⁻¹ δμ via the 2×2 adjugate.
    let quad = (sum[(1, 1)] * du[0] * du[0] - 2.0 * sum[(0, 1)] * du[0] * du[1]
        + sum[(0, 0)] * du[1] * du[1])
        / big_delta;

    2.0 * (-0.5 * quad).exp() / ((big_delta + small_delta).sqrt() - small_delta.sqrt())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// F(ρ, ρ) = 1 for mixed (thermal) and pure (squeezed) states.
    #[test]
    fn self_fidelity_is_one() {
        let th = GaussianState::thermal(&[1.3]);
        assert!((fidelity_single_mode(&th, &th) - 1.0).abs() < 1e-12);
        let sq = GaussianState::squeezed(&[(0.9, 0.7)]);
        assert!((fidelity_single_mode(&sq, &sq) - 1.0).abs() < 1e-12);
    }

    /// F(|0⟩, |α⟩) = e^{−|α|²}.
    #[test]
    fn vacuum_vs_coherent() {
        let vac = GaussianState::vacuum(1);
        let alpha = (0.6, -0.45);
        let coh = GaussianState::coherent(&[alpha]);
        let want = (-(alpha.0 * alpha.0 + alpha.1 * alpha.1)).exp();
        assert!((fidelity_single_mode(&vac, &coh) - want).abs() < 1e-12);
    }

    /// F(|0⟩, Ŝ(r)|0⟩) = 1/cosh r.
    #[test]
    fn vacuum_vs_squeezed() {
        let vac = GaussianState::vacuum(1);
        let r = 1.1;
        let sq = GaussianState::squeezed(&[(r, 0.0)]);
        assert!((fidelity_single_mode(&vac, &sq) - 1.0 / r.cosh()).abs() < 1e-12);
    }
}
