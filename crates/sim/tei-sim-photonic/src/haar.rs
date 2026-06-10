//! Haar-random unitary matrices.
//!
//! F. Mezzadri, "How to generate random matrices from the classical compact
//! groups," Notices of the AMS 54, 592 (2007): if `G` is a complex Ginibre
//! matrix (i.i.d. standard complex Gaussian entries) and `G = QR` is the
//! **unique** QR factorization with `diag(R)` real and positive, then `Q` is
//! Haar-distributed on `U(N)`.
//!
//! We compute that factorization by **modified Gram-Schmidt with one
//! re-orthogonalization pass** ("twice is enough"): MGS produces
//! `R_jj = ‖v_j − proj‖ > 0` by construction, so its `Q` *is* the
//! phase-fixed factor — no separate `diag(R/|R|)` correction is needed
//! (that correction exists to repair Householder QR, whose diagonal phases
//! are arbitrary). The re-orthogonalization pass keeps `‖Q†Q − I‖` at the
//! 1e-15 level for the mesh sizes used here (N ≤ a few hundred).

use tei_sim_core::linalg::{C64, CMat};
use tei_sim_core::rng::Rng;

/// Draw an `n×n` Haar-random unitary from the given (deterministic) RNG.
pub fn haar_unitary(n: usize, rng: &mut Rng) -> CMat {
    assert!(n >= 1);
    // Complex Ginibre matrix: entries (X + iY)/√2 with X, Y ~ N(0, 1).
    let mut q = CMat::zeros(n, n);
    let inv_sqrt2 = std::f64::consts::FRAC_1_SQRT_2;
    for v in q.data.iter_mut() {
        *v = C64::new(rng.normal(), rng.normal()) * inv_sqrt2;
    }
    // Modified Gram-Schmidt over columns, two passes per column.
    for j in 0..n {
        for _pass in 0..2 {
            for i in 0..j {
                // ⟨q_i, v_j⟩ = Σ_k conj(q_i[k])·v_j[k]
                let mut dot = C64::ZERO;
                for k in 0..n {
                    dot = dot + q[(k, i)].conj() * q[(k, j)];
                }
                for k in 0..n {
                    q[(k, j)] = q[(k, j)] - dot * q[(k, i)];
                }
            }
        }
        let norm = (0..n).map(|k| q[(k, j)].norm_sq()).sum::<f64>().sqrt();
        assert!(norm > 1e-150, "Ginibre column numerically degenerate");
        let inv = 1.0 / norm;
        for k in 0..n {
            q[(k, j)] = q[(k, j)] * inv;
        }
    }
    q
}
