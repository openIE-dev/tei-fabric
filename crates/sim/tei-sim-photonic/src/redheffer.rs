//! Redheffer star product — S-matrix composition over shared internal ports.
//!
//! R. Redheffer, "On a certain linear fractional transformation,"
//! J. Math. Phys. 39, 269 (1960). The star product is how frequency-domain
//! circuit solvers (SAX-class netlisting, RCWA layer stacking) cascade
//! multiport scattering matrices **with multiple reflections summed
//! exactly** — the matrix geometric series collapses into the
//! `(I − S₂₂S₁₁)⁻¹` resolvents below.
//!
//! ## Port partition
//!
//! A network is stored with its ports split into a *left* group (size `n_l`)
//! and a *right* group (size `n_r`):
//!
//! ```text
//! [b_left ]   [ S11  S12 ] [a_left ]
//! [b_right] = [ S21  S22 ] [a_right]
//! ```
//!
//! `A ⋆ B` connects all of A's right ports to all of B's left ports
//! (`A.n_right == B.n_left`). Writing `p` for the internal wave A→B and `q`
//! for B→A and eliminating them:
//!
//! ```text
//! p = A21·a + A22·q,   q = B11·p + B12·c
//!   ⇒ q = (I − B11·A22)⁻¹ (B11·A21·a + B12·c)
//!
//! S11 = A11 + A12·(I − B11·A22)⁻¹·B11·A21
//! S12 = A12·(I − B11·A22)⁻¹·B12
//! S21 = B21·(I − A22·B11)⁻¹·A21
//! S22 = B22 + B21·(I − A22·B11)⁻¹·A22·B12
//! ```
//!
//! The product is associative (validated against random passive networks in
//! `tests/analytic.rs`), and the perfect through-connection is its identity.

use tei_sim_core::linalg::{C64, CMat};

/// A multiport S-matrix with an explicit left/right port partition.
#[derive(Debug, Clone)]
pub struct Sparams {
    pub n_left: usize,
    pub n_right: usize,
    /// Reflection left→left, `n_l × n_l`.
    pub s11: CMat,
    /// Transmission right→left, `n_l × n_r`.
    pub s12: CMat,
    /// Transmission left→right, `n_r × n_l`.
    pub s21: CMat,
    /// Reflection right→right, `n_r × n_r`.
    pub s22: CMat,
}

impl Sparams {
    /// The star-product identity: a perfect through-connection of `n` ports
    /// (`S11 = S22 = 0`, `S12 = S21 = I`). `X ⋆ through = through ⋆ X = X`.
    pub fn through(n: usize) -> Self {
        Self {
            n_left: n,
            n_right: n,
            s11: CMat::zeros(n, n),
            s12: CMat::identity(n),
            s21: CMat::identity(n),
            s22: CMat::zeros(n, n),
        }
    }

    /// Partition a full square S-matrix: the first `n_left` ports become the
    /// left group, the rest the right group.
    pub fn from_full(s: &CMat, n_left: usize) -> Self {
        assert_eq!(s.rows, s.cols, "S must be square");
        assert!(n_left <= s.rows);
        let n_right = s.rows - n_left;
        let block = |r0: usize, c0: usize, nr: usize, nc: usize| {
            let mut m = CMat::zeros(nr, nc);
            for i in 0..nr {
                for j in 0..nc {
                    m[(i, j)] = s[(r0 + i, c0 + j)];
                }
            }
            m
        };
        Self {
            n_left,
            n_right,
            s11: block(0, 0, n_left, n_left),
            s12: block(0, n_left, n_left, n_right),
            s21: block(n_left, 0, n_right, n_left),
            s22: block(n_left, n_left, n_right, n_right),
        }
    }

    /// Lift a reflectionless transmission matrix `M` (left inputs → right
    /// outputs) into scattering form: `S21 = M`, `S12 = Mᵀ` (reciprocity),
    /// `S11 = S22 = 0`. If `M` is unitary the full S-matrix is unitary.
    pub fn from_transmission(m: &CMat) -> Self {
        let mut s12 = CMat::zeros(m.cols, m.rows);
        for i in 0..m.rows {
            for j in 0..m.cols {
                s12[(j, i)] = m[(i, j)];
            }
        }
        Self {
            n_left: m.cols,
            n_right: m.rows,
            s11: CMat::zeros(m.cols, m.cols),
            s12,
            s21: m.clone(),
            s22: CMat::zeros(m.rows, m.rows),
        }
    }

    /// Assemble the full `(n_l + n_r) × (n_l + n_r)` S-matrix.
    pub fn full(&self) -> CMat {
        let n = self.n_left + self.n_right;
        let mut s = CMat::zeros(n, n);
        for i in 0..self.n_left {
            for j in 0..self.n_left {
                s[(i, j)] = self.s11[(i, j)];
            }
            for j in 0..self.n_right {
                s[(i, self.n_left + j)] = self.s12[(i, j)];
            }
        }
        for i in 0..self.n_right {
            for j in 0..self.n_left {
                s[(self.n_left + i, j)] = self.s21[(i, j)];
            }
            for j in 0..self.n_right {
                s[(self.n_left + i, self.n_left + j)] = self.s22[(i, j)];
            }
        }
        s
    }

    /// Redheffer star product `self ⋆ other`: connect `self`'s right ports
    /// to `other`'s left ports. Panics if the shared interface is resonantly
    /// singular (`I − S₂₂S₁₁` not invertible — cannot happen for strictly
    /// passive networks, `‖S‖ < 1`).
    pub fn star(&self, other: &Sparams) -> Sparams {
        assert_eq!(
            self.n_right, other.n_left,
            "star: interface port counts must match"
        );
        let k = self.n_right;
        let eye = CMat::identity(k);
        // Y = (I − B11·A22)⁻¹ applied to right-hand sides via LU solve.
        let y_lhs = cmat_sub(&eye, &other.s11.matmul(&self.s22));
        // X = (I − A22·B11)⁻¹.
        let x_lhs = cmat_sub(&eye, &self.s22.matmul(&other.s11));
        let y_b11_a21 = csolve(&y_lhs, &other.s11.matmul(&self.s21)).expect("star: singular loop");
        let y_b12 = csolve(&y_lhs, &other.s12).expect("star: singular loop");
        let x_a21 = csolve(&x_lhs, &self.s21).expect("star: singular loop");
        let x_a22_b12 = csolve(&x_lhs, &self.s22.matmul(&other.s12)).expect("star: singular loop");
        Sparams {
            n_left: self.n_left,
            n_right: other.n_right,
            s11: cmat_add(&self.s11, &self.s12.matmul(&y_b11_a21)),
            s12: self.s12.matmul(&y_b12),
            s21: other.s21.matmul(&x_a21),
            s22: cmat_add(&other.s22, &other.s21.matmul(&x_a22_b12)),
        }
    }
}

/// Series cascade of two **2-port** S-matrices (`[[s11, s12],[s21, s22]]`,
/// port 0 = left, port 1 = right): the star product with 1/1 partition.
pub fn cascade_2port(a: &CMat, b: &CMat) -> CMat {
    assert!(a.rows == 2 && a.cols == 2 && b.rows == 2 && b.cols == 2);
    Sparams::from_full(a, 1)
        .star(&Sparams::from_full(b, 1))
        .full()
}

/// `a + b` elementwise.
fn cmat_add(a: &CMat, b: &CMat) -> CMat {
    assert!(a.rows == b.rows && a.cols == b.cols);
    let mut out = a.clone();
    for (x, y) in out.data.iter_mut().zip(&b.data) {
        *x = *x + *y;
    }
    out
}

/// `a − b` elementwise.
fn cmat_sub(a: &CMat, b: &CMat) -> CMat {
    assert!(a.rows == b.rows && a.cols == b.cols);
    let mut out = a.clone();
    for (x, y) in out.data.iter_mut().zip(&b.data) {
        *x = *x - *y;
    }
    out
}

/// Solve `A·X = B` for complex dense `A` (multiple right-hand sides) by
/// Gaussian elimination with partial (max-modulus) pivoting. Returns `None`
/// if `A` is singular to working precision. Sizes here are interface port
/// counts (≤ a few hundred), well within dense-solve territory.
fn csolve(a: &CMat, b: &CMat) -> Option<CMat> {
    assert_eq!(a.rows, a.cols);
    assert_eq!(b.rows, a.rows);
    let n = a.rows;
    let mut m = a.clone();
    let mut x = b.clone();
    for col in 0..n {
        // Pivot on the largest modulus.
        let mut p = col;
        let mut best = m[(col, col)].abs();
        for r in (col + 1)..n {
            let v = m[(r, col)].abs();
            if v > best {
                best = v;
                p = r;
            }
        }
        if best < 1e-13 {
            return None;
        }
        if p != col {
            for j in 0..n {
                let tmp = m[(col, j)];
                m[(col, j)] = m[(p, j)];
                m[(p, j)] = tmp;
            }
            for j in 0..x.cols {
                let tmp = x[(col, j)];
                x[(col, j)] = x[(p, j)];
                x[(p, j)] = tmp;
            }
        }
        let piv = m[(col, col)];
        for r in (col + 1)..n {
            let f = m[(r, col)] / piv;
            if f == C64::ZERO {
                continue;
            }
            for j in col..n {
                m[(r, j)] = m[(r, j)] - f * m[(col, j)];
            }
            for j in 0..x.cols {
                x[(r, j)] = x[(r, j)] - f * x[(col, j)];
            }
        }
    }
    // Back substitution.
    for col in (0..n).rev() {
        let piv = m[(col, col)];
        for j in 0..x.cols {
            x[(col, j)] = x[(col, j)] / piv;
        }
        for r in 0..col {
            let f = m[(r, col)];
            if f == C64::ZERO {
                continue;
            }
            for j in 0..x.cols {
                x[(r, j)] = x[(r, j)] - f * x[(col, j)];
            }
        }
    }
    Some(x)
}
