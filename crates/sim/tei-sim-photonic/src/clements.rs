//! Universal MZI meshes — Clements construction **and** decomposition.
//!
//! - W. R. Clements, P. C. Humphreys, B. J. Metcalf, W. S. Kolthammer,
//!   I. A. Walmsley, "Optimal design for universal multiport
//!   interferometers," Optica 3, 1460 (2016) — the rectangular mesh and the
//!   two-sided nulling algorithm implemented here.
//! - M. Reck, A. Zeilinger, H. J. Bernstein, P. Bertani, "Experimental
//!   realization of any discrete unitary operator," PRL 73, 58 (1994) — the
//!   original triangular decomposition the Clements scheme improves on
//!   (half the optical depth, balanced loss).
//!
//! ## The T block
//!
//! One programmable MZI acting on adjacent modes `(m, m+1)` is, in the
//! Clements paper's convention,
//!
//! ```text
//! T(θ, φ) = [ e^{iφ}·cosθ   −sinθ ]
//!           [ e^{iφ}·sinθ    cosθ ]      θ ∈ [0, π/2], φ ∈ [0, 2π)
//! ```
//!
//! (identity on all other modes). It is unitary for all `(θ, φ)`. The
//! physical MZI of `components::mzi_transfer` (two 50:50 couplers, internal
//! phase `θ_int`, input phase `φ`) equals
//! `i·e^{iθ_int/2}·[[e^{iφ}s, c], [e^{iφ}c, −s]]` with `s = sin(θ_int/2)`;
//! choosing `θ_int = π − 2θ` reproduces the moduli of `T`, and the leftover
//! input/output phases are absorbed by the φ shifters and the output
//! diagonal — so realizing `T` costs exactly one MZI plus phase shifters,
//! as in the paper.
//!
//! ## Decomposition (the nulling procedure, Clements §3)
//!
//! Walk the anti-diagonals `i = 0 … N−2` of `U`. On even diagonals,
//! right-multiply by `T⁻¹` (acts on **columns**) to null entries from the
//! bottom-left; on odd diagonals, left-multiply by `T` (acts on **rows**):
//!
//! - **Right null** of `U[r, m]` with `T⁻¹` on columns `(m, n=m+1)`:
//!   the new column m is `U[:,m]·e^{−iφ}cosθ − U[:,n]·sinθ`, which vanishes
//!   at row `r` for `φ = arg U[r,m] − arg U[r,n]`,
//!   `θ = atan2(|U[r,m]|, |U[r,n]|)`.
//! - **Left null** of `U[n, c]` with `T` on rows `(m, n=m+1)`:
//!   the new row n is `e^{iφ}sinθ·U[m,:] + cosθ·U[n,:]`, which vanishes at
//!   column `c` for `φ = arg(−U[n,c]) − arg U[m,c]`,
//!   `θ = atan2(|U[n,c]|, |U[m,c]|)`.
//!
//! After all `N(N−1)/2` nullings, `U` has been reduced to a diagonal `D` of
//! unit-modulus phases:
//!
//! ```text
//! T_Lk ··· T_L1 · U · T_R1⁻¹ ··· T_Rm⁻¹ = D
//!   ⇒  U = T_L1⁻¹ ··· T_Lk⁻¹ · D · T_Rm ··· T_R1
//! ```
//!
//! To reach the canonical single-sided form `U = D′·∏T`, each left inverse
//! is commuted through the diagonal with the exact identity (derived by
//! matching the four entries of the 2×2 blocks; `α, β` are the diagonal
//! phases on modes `m, n`):
//!
//! ```text
//! T⁻¹(θ, φ)·diag(e^{iα}, e^{iβ})
//!     = diag(e^{iα′}, e^{iβ′})·T(θ, φ′)
//! with  φ′ = α − β + π,   α′ = β − φ + π,   β′ = β.
//! ```
//!
//! Processing the left inverses innermost-first yields
//! `U = D′ · T′_L1 ··· T′_Lk · T_Rm ··· T_R1` — a pure ordered product of
//! `T` blocks behind one output phase screen, which is exactly what a
//! rectangular mesh of MZIs realizes.

use tei_sim_core::linalg::{C64, CMat};

/// Phase argument of a complex number (`atan2(im, re)`; `arg(0) = 0`).
#[inline]
fn arg(z: C64) -> f64 {
    z.im.atan2(z.re)
}

/// One programmed MZI: `T(θ, φ)` on modes `(mode, mode+1)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MziSetting {
    /// Upper mode index `m`; the block acts on `(m, m+1)`.
    pub mode: usize,
    pub theta: f64,
    pub phi: f64,
}

/// A programmed interferometer: `U = D · T_1 · T_2 ··· T_M` where `D` is the
/// output phase screen `diag(e^{iα_j})` and `T_1` is the **leftmost** block
/// in the matrix product (so light entering the chip passes `T_M` first).
#[derive(Debug, Clone)]
pub struct ClementsMesh {
    pub n: usize,
    /// Ordered left-to-right in the matrix product `T_1·T_2···T_M`.
    pub mzis: Vec<MziSetting>,
    /// Output phases `α_j` (the diagonal `D`).
    pub output_phases: Vec<f64>,
}

impl ClementsMesh {
    /// Construct a mesh in the canonical **rectangular** (Clements) layout
    /// from per-MZI `(θ, φ)` arrays. Layer `ℓ ∈ 0..n` holds MZIs on mode
    /// pairs starting at `ℓ mod 2` and stepping by 2; over `n` layers this
    /// places exactly `n(n−1)/2` MZIs — the universal rectangle.
    /// `settings.len()` must equal `n(n−1)/2`.
    pub fn rectangular(n: usize, settings: &[(f64, f64)], output_phases: Vec<f64>) -> Self {
        assert_eq!(settings.len(), n * (n - 1) / 2, "need n(n−1)/2 settings");
        assert_eq!(output_phases.len(), n);
        let mut mzis = Vec::with_capacity(settings.len());
        let mut k = 0;
        for layer in 0..n {
            let mut m = layer % 2;
            while m + 1 < n {
                let (theta, phi) = settings[k];
                k += 1;
                mzis.push(MziSetting {
                    mode: m,
                    theta,
                    phi,
                });
                m += 2;
            }
        }
        debug_assert_eq!(k, settings.len());
        Self {
            n,
            mzis,
            output_phases,
        }
    }

    /// Number of MZIs in the mesh.
    pub fn n_mzis(&self) -> usize {
        self.mzis.len()
    }

    /// Total number of programmable phases (2 per MZI + N output phases).
    pub fn n_phases(&self) -> usize {
        2 * self.mzis.len() + self.n
    }

    /// Assemble the full unitary `U = D·T_1···T_M` (O(N) per block).
    pub fn unitary(&self) -> CMat {
        let mut u = CMat::identity(self.n);
        for s in &self.mzis {
            right_mul_t(&mut u, s.mode, s.theta, s.phi);
        }
        for (j, &alpha) in self.output_phases.iter().enumerate() {
            let d = C64::from_polar(1.0, alpha);
            for c in 0..self.n {
                u[(j, c)] = d * u[(j, c)];
            }
        }
        u
    }

    /// Propagate an amplitude vector through the mesh: `y = U·x`, applied
    /// block-by-block (O(1) per MZI — this is the physical light path, and
    /// the O(N²)-total optical MVM).
    pub fn apply(&self, x: &[C64]) -> Vec<C64> {
        assert_eq!(x.len(), self.n);
        let mut v = x.to_vec();
        // Matrix product D·T_1···T_M·x ⇒ T_M acts first.
        for s in self.mzis.iter().rev() {
            let (m, n) = (s.mode, s.mode + 1);
            let (sn, cs) = s.theta.sin_cos();
            let eip = C64::from_polar(1.0, s.phi);
            let (a, b) = (v[m], v[n]);
            v[m] = eip * a * cs - b * sn;
            v[n] = eip * a * sn + b * cs;
        }
        for (j, &alpha) in self.output_phases.iter().enumerate() {
            v[j] = C64::from_polar(1.0, alpha) * v[j];
        }
        v
    }

    /// **Clements decomposition**: recover the mesh settings realizing an
    /// arbitrary `N×N` unitary `u`. Returns `Err` if `u` is not square or
    /// not unitary to `1e-8`. See the module docs for the full derivation.
    pub fn decompose(u: &CMat) -> Result<Self, String> {
        if u.rows != u.cols {
            return Err("decompose: matrix must be square".into());
        }
        let n = u.rows;
        let uni_err = u.dagger().matmul(u).frobenius_distance(&CMat::identity(n));
        if uni_err > 1e-8 {
            return Err(format!(
                "decompose: not unitary (‖U†U−I‖_F = {uni_err:.2e})"
            ));
        }
        if n == 1 {
            return Ok(Self {
                n: 1,
                mzis: vec![],
                output_phases: vec![arg(u[(0, 0)])],
            });
        }

        let mut w = u.clone();
        // (mode, θ, φ) in order of application.
        let mut right_ops: Vec<(usize, f64, f64)> = Vec::new();
        let mut left_ops: Vec<(usize, f64, f64)> = Vec::new();

        for i in 0..(n - 1) {
            if i % 2 == 0 {
                // Null from the bottom-left moving up the anti-diagonal,
                // multiplying by T⁻¹ on the right (column ops).
                for j in 0..=i {
                    let m = i - j;
                    let r = n - 1 - j;
                    let um = w[(r, m)];
                    let un = w[(r, m + 1)];
                    let theta = um.abs().atan2(un.abs());
                    let phi = arg(um) - arg(un);
                    right_mul_t_inv(&mut w, m, theta, phi);
                    right_ops.push((m, theta, phi));
                }
            } else {
                // Null moving down the anti-diagonal, multiplying by T on
                // the left (row ops).
                for j in 0..=i {
                    let m = n - 2 - i + j;
                    let c = j;
                    let um = w[(m, c)];
                    let un = w[(m + 1, c)];
                    let theta = un.abs().atan2(um.abs());
                    let phi = arg(-un) - arg(um);
                    left_mul_t(&mut w, m, theta, phi);
                    left_ops.push((m, theta, phi));
                }
            }
        }

        // w is now diagonal: D = T_Lk···T_L1 · U · T_R1⁻¹···T_Rm⁻¹.
        let mut alpha: Vec<f64> = (0..n).map(|j| arg(w[(j, j)])).collect();

        // Commute the left inverses through D, innermost (last applied)
        // first:  T⁻¹(θ,φ)·D = D′·T(θ, φ′)  with  φ′ = α_m − α_n + π,
        // α′_m = α_n − φ + π, α′_n = α_n.
        let mut primed: Vec<(usize, f64, f64)> = Vec::with_capacity(left_ops.len());
        for &(m, theta, phi) in left_ops.iter().rev() {
            let nn = m + 1;
            let phi_p = alpha[m] - alpha[nn] + std::f64::consts::PI;
            alpha[m] = alpha[nn] - phi + std::f64::consts::PI;
            primed.push((m, theta, phi_p));
        }
        // Processing order was Lk…L1 with each new block prepended in the
        // product; reverse so the list reads T′_L1 … T′_Lk left-to-right.
        primed.reverse();

        // U = D′ · T′_L1 ··· T′_Lk · T_Rm ··· T_R1.
        let mut mzis: Vec<MziSetting> = primed
            .into_iter()
            .map(|(mode, theta, phi)| MziSetting { mode, theta, phi })
            .collect();
        mzis.extend(
            right_ops
                .into_iter()
                .rev()
                .map(|(mode, theta, phi)| MziSetting { mode, theta, phi }),
        );

        Ok(Self {
            n,
            mzis,
            output_phases: alpha,
        })
    }
}

/// `U ← U·T(θ,φ)` on columns `(m, m+1)`: col_m ← col_m·e^{iφ}c + col_n·e^{iφ}s,
/// col_n ← −col_m·s + col_n·c (reading the columns of `T`).
fn right_mul_t(u: &mut CMat, m: usize, theta: f64, phi: f64) {
    let n = m + 1;
    let (s, c) = theta.sin_cos();
    let eip = C64::from_polar(1.0, phi);
    for r in 0..u.rows {
        let (a, b) = (u[(r, m)], u[(r, n)]);
        u[(r, m)] = (a * c + b * s) * eip;
        u[(r, n)] = -a * s + b * c;
    }
}

/// `U ← U·T⁻¹(θ,φ)` on columns `(m, m+1)` with
/// `T⁻¹ = [[e^{−iφ}c, e^{−iφ}s], [−s, c]]`.
fn right_mul_t_inv(u: &mut CMat, m: usize, theta: f64, phi: f64) {
    let n = m + 1;
    let (s, c) = theta.sin_cos();
    let eim = C64::from_polar(1.0, -phi);
    for r in 0..u.rows {
        let (a, b) = (u[(r, m)], u[(r, n)]);
        u[(r, m)] = a * eim * c - b * s;
        u[(r, n)] = a * eim * s + b * c;
    }
}

/// `U ← T(θ,φ)·U` on rows `(m, m+1)`.
fn left_mul_t(u: &mut CMat, m: usize, theta: f64, phi: f64) {
    let n = m + 1;
    let (s, c) = theta.sin_cos();
    let eip = C64::from_polar(1.0, phi);
    for col in 0..u.cols {
        let (a, b) = (u[(m, col)], u[(n, col)]);
        u[(m, col)] = eip * a * c - b * s;
        u[(n, col)] = eip * a * s + b * c;
    }
}
