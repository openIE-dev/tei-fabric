//! Symplectic spectrum and the physicality check σ + iΩ ⪰ 0.
//!
//! # The bona-fide condition and how we test it without a complex eigensolver
//!
//! A real symmetric σ is a valid quantum covariance matrix iff σ + iΩ ⪰ 0
//! ([Weedbrook RMP 2012] Eq. (32); ħ = 2 units, vacuum σ = I). By Williamson's
//! theorem σ = S diag(ν₁, ν₁, …, ν_N, ν_N) Sᵀ for some symplectic S, and the
//! bona-fide condition is equivalent to all **symplectic eigenvalues**
//! ν_k ≥ 1 ([Weedbrook RMP 2012] Eq. (34)).
//!
//! The ν_k are the moduli of the (purely imaginary) eigenvalues ±iν_k of Ωσ.
//! We have no complex eigensolver, so we use the standard real reduction —
//! derived honestly here:
//!
//! 1. (Ωσ)² has eigenvalues (±iν)² = −ν².
//! 2. −(Ωσ)² = Ωσ·(−Ω)σ = Ωσ·Ωᵀσ = (ΩσΩᵀ)·σ = A·σ, with A = ΩσΩᵀ ⪰ 0
//!    symmetric whenever σ ⪰ 0 (congruence preserves semidefiniteness).
//! 3. For σ ≻ 0 the product A·σ is similar to the **symmetric** matrix
//!    M = σ^{1/2} A σ^{1/2}, via σ^{1/2}(Aσ)σ^{−1/2} = σ^{1/2} A σ^{1/2}.
//!
//! So the eigenvalues of the real symmetric PSD matrix
//! M = σ^{1/2} Ω σ Ωᵀ σ^{1/2} are exactly {ν_k², ν_k²} (each doubled), and a
//! cyclic **Jacobi rotation eigensolver** — hand-rolled below, unconditionally
//! convergent for symmetric matrices — recovers them. The matrix square root
//! σ^{1/2} comes from the same Jacobi solver (σ = VΛVᵀ ⇒ σ^{1/2} = VΛ^{1/2}Vᵀ;
//! eigenvalues below 0 are clamped, which only shrinks ν and can never make an
//! unphysical σ pass).
//!
//! Spot checks (also enforced in tests): σ = I ⇒ M = ΩΩᵀ = I ⇒ ν = 1;
//! σ = (2n̄+1)I ⇒ M = (2n̄+1)²I ⇒ ν = 2n̄+1; σ = diag(e^{−2r}, e^{2r}) ⇒
//! M = I ⇒ ν = 1 (pure squeezed vacuum saturates Heisenberg).

use crate::omega;
use tei_sim_core::linalg::Mat;

/// Cyclic Jacobi eigensolver for a real **symmetric** matrix.
///
/// Returns (eigenvalues unsorted, eigenvector matrix V with eigenvectors as
/// columns, A = V diag(λ) Vᵀ). Quadratically convergent; off-diagonal mass is
/// driven below 1e-15·‖A‖_F or 64 sweeps, whichever first (n ≤ a few dozen
/// here, so a handful of sweeps suffices).
pub fn jacobi_eigh(m: &Mat) -> (Vec<f64>, Mat) {
    assert_eq!(m.rows, m.cols, "jacobi_eigh needs a square matrix");
    let n = m.rows;
    let mut a = m.clone();
    let mut v = Mat::identity(n);
    let scale = a.frobenius_norm().max(f64::MIN_POSITIVE);
    for _sweep in 0..64 {
        let mut off = 0.0;
        for i in 0..n {
            for j in (i + 1)..n {
                off += a[(i, j)] * a[(i, j)];
            }
        }
        if off.sqrt() <= 1e-15 * scale {
            break;
        }
        for p in 0..n.saturating_sub(1) {
            for q in (p + 1)..n {
                let apq = a[(p, q)];
                if apq.abs() <= f64::MIN_POSITIVE {
                    continue;
                }
                // Rotation angle zeroing a_pq: t = tan θ from the stable root
                // of t² + 2t·τ − 1 = 0, τ = (a_qq − a_pp)/(2 a_pq).
                let tau = (a[(q, q)] - a[(p, p)]) / (2.0 * apq);
                let t = tau.signum() / (tau.abs() + (tau * tau + 1.0).sqrt());
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                // A ← Jᵀ A J with J the (p,q)-plane rotation; V ← V J.
                for k in 0..n {
                    let akp = a[(k, p)];
                    let akq = a[(k, q)];
                    a[(k, p)] = c * akp - s * akq;
                    a[(k, q)] = s * akp + c * akq;
                }
                for k in 0..n {
                    let apk = a[(p, k)];
                    let aqk = a[(q, k)];
                    a[(p, k)] = c * apk - s * aqk;
                    a[(q, k)] = s * apk + c * aqk;
                }
                for k in 0..n {
                    let vkp = v[(k, p)];
                    let vkq = v[(k, q)];
                    v[(k, p)] = c * vkp - s * vkq;
                    v[(k, q)] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[(i, i)]).collect(), v)
}

/// Symmetric PSD square root via Jacobi: σ = VΛVᵀ ⇒ σ^{1/2} = VΛ₊^{1/2}Vᵀ.
/// Negative eigenvalues (already unphysical) are clamped to zero.
pub fn sym_sqrt(m: &Mat) -> Mat {
    let n = m.rows;
    let (eig, v) = jacobi_eigh(m);
    let roots: Vec<f64> = eig.iter().map(|&l| l.max(0.0).sqrt()).collect();
    let mut out = Mat::zeros(n, n);
    for i in 0..n {
        for j in i..n {
            let mut s = 0.0;
            for k in 0..n {
                s += v[(i, k)] * roots[k] * v[(j, k)];
            }
            out[(i, j)] = s;
            out[(j, i)] = s;
        }
    }
    out
}

/// Symplectic eigenvalues ν₁ ≤ … ≤ ν_N of a 2N×2N covariance matrix.
///
/// Computed as the square roots of the eigenvalues of the real symmetric PSD
/// matrix M = σ^{1/2} Ω σ Ωᵀ σ^{1/2} (similar to −(Ωσ)²; derivation in the
/// module docs). The 2N eigenvalues come in doubled pairs {ν², ν²}; we sort
/// and average each pair, which also smooths last-bit asymmetry.
pub fn symplectic_eigenvalues(cov: &Mat) -> Vec<f64> {
    assert_eq!(cov.rows, cov.cols);
    assert_eq!(cov.rows % 2, 0, "covariance must be 2N×2N");
    let n = cov.rows / 2;
    let om = omega(n);
    let half = sym_sqrt(cov);
    let m = half
        .matmul(&om)
        .matmul(cov)
        .matmul(&om.transpose())
        .matmul(&half);
    let (mut eig, _) = jacobi_eigh(&m);
    eig.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (0..n)
        .map(|k| {
            let lo = eig[2 * k].max(0.0).sqrt();
            let hi = eig[2 * k + 1].max(0.0).sqrt();
            0.5 * (lo + hi)
        })
        .collect()
}

/// Physicality margin min_k ν_k − 1.
///
/// σ + iΩ ⪰ 0 (ħ = 2) ⟺ margin ≥ 0; pure states sit exactly at 0, thermal
/// noise pushes it positive, and any quadrature "squeezed below vacuum in
/// both directions" drives it negative.
pub fn physicality_margin(cov: &Mat) -> f64 {
    symplectic_eigenvalues(cov)
        .into_iter()
        .fold(f64::INFINITY, f64::min)
        - 1.0
}

/// Bona-fide covariance test: σ + iΩ ⪰ 0 within `tol` (i.e. ν_min ≥ 1 − tol).
pub fn is_physical(cov: &Mat, tol: f64) -> bool {
    physicality_margin(cov) >= -tol
}

#[cfg(test)]
mod tests {
    use super::*;
    use tei_sim_core::rng::Rng;

    /// Jacobi recovers a known spectrum: A = Q diag(d) Qᵀ for random
    /// orthogonal Q (from core's Householder QR).
    #[test]
    fn jacobi_known_spectrum() {
        let mut rng = Rng::new(11);
        let n = 8;
        let mut g = Mat::zeros(n, n);
        for x in g.data.iter_mut() {
            *x = rng.normal();
        }
        let (q, _) = g.qr();
        let d: Vec<f64> = (0..n).map(|i| (i as f64) - 3.5).collect();
        let mut a = Mat::zeros(n, n);
        for i in 0..n {
            for j in 0..n {
                let mut s = 0.0;
                for k in 0..n {
                    s += q[(i, k)] * d[k] * q[(j, k)];
                }
                a[(i, j)] = s;
            }
        }
        let (mut eig, _) = jacobi_eigh(&a);
        eig.sort_by(|x, y| x.partial_cmp(y).unwrap());
        for (got, want) in eig.iter().zip(&d) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }

    /// sym_sqrt squares back to the original.
    #[test]
    fn sqrt_squares_back() {
        let a = Mat::from_rows(&[&[2.0, 0.5, 0.0], &[0.5, 3.0, -0.2], &[0.0, -0.2, 1.5]]);
        let h = sym_sqrt(&a);
        let hh = h.matmul(&h);
        let mut err = 0.0f64;
        for i in 0..3 {
            for j in 0..3 {
                err = err.max((hh[(i, j)] - a[(i, j)]).abs());
            }
        }
        assert!(err < 1e-12, "‖H² − A‖∞ = {err}");
    }

    /// Closed-form symplectic spectra: thermal ν = 2n̄+1, squeezed ν = 1.
    #[test]
    fn closed_form_spectra() {
        let th = crate::GaussianState::thermal(&[0.7, 1.9]);
        let nu = symplectic_eigenvalues(&th.cov);
        assert!((nu[0] - 2.4).abs() < 1e-12 && (nu[1] - 4.8).abs() < 1e-12);
        let sq = crate::GaussianState::squeezed(&[(1.3, 0.4)]);
        let nu = symplectic_eigenvalues(&sq.cov);
        assert!((nu[0] - 1.0).abs() < 1e-12, "pure squeezed ν = {}", nu[0]);
    }
}
