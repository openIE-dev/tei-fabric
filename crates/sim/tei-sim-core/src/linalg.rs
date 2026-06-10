//! Dense linear algebra — real and complex matrices, LU solve, QR.
//!
//! Hand-rolled per the roadmap dependency policy. Row-major storage. Scales
//! targeted: meshes ≤ 512×512, circuit cells ≤ a few hundred nodes — partial
//! pivoting LU and Householder QR are exact-enough and fast-enough here.

/// Complex number, f64 components.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub const ONE: C64 = C64 { re: 1.0, im: 0.0 };
    pub const I: C64 = C64 { re: 0.0, im: 1.0 };

    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn from_polar(r: f64, theta: f64) -> Self {
        let (s, c) = theta.sin_cos();
        Self {
            re: r * c,
            im: r * s,
        }
    }
    pub fn conj(self) -> Self {
        Self {
            re: self.re,
            im: -self.im,
        }
    }
    pub fn norm_sq(self) -> f64 {
        self.re * self.re + self.im * self.im
    }
    pub fn abs(self) -> f64 {
        self.norm_sq().sqrt()
    }
}

impl std::ops::Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}
impl std::ops::Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}
impl std::ops::Mul for C64 {
    type Output = C64;
    fn mul(self, o: C64) -> C64 {
        C64::new(
            self.re * o.re - self.im * o.im,
            self.re * o.im + self.im * o.re,
        )
    }
}
impl std::ops::Div for C64 {
    type Output = C64;
    fn div(self, o: C64) -> C64 {
        let d = o.norm_sq();
        C64::new(
            (self.re * o.re + self.im * o.im) / d,
            (self.im * o.re - self.re * o.im) / d,
        )
    }
}
impl std::ops::Mul<f64> for C64 {
    type Output = C64;
    fn mul(self, s: f64) -> C64 {
        C64::new(self.re * s, self.im * s)
    }
}
impl std::ops::Neg for C64 {
    type Output = C64;
    fn neg(self) -> C64 {
        C64::new(-self.re, -self.im)
    }
}

/// Dense row-major real matrix.
#[derive(Debug, Clone)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }
    pub fn from_rows(rows: &[&[f64]]) -> Self {
        let r = rows.len();
        let c = rows[0].len();
        let mut data = Vec::with_capacity(r * c);
        for row in rows {
            assert_eq!(row.len(), c);
            data.extend_from_slice(row);
        }
        Self {
            rows: r,
            cols: c,
            data,
        }
    }

    pub fn matmul(&self, other: &Mat) -> Mat {
        assert_eq!(self.cols, other.rows);
        let mut out = Mat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                if a == 0.0 {
                    continue;
                }
                for j in 0..other.cols {
                    out[(i, j)] += a * other[(k, j)];
                }
            }
        }
        out
    }

    pub fn transpose(&self) -> Mat {
        let mut out = Mat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)];
            }
        }
        out
    }

    pub fn frobenius_norm(&self) -> f64 {
        self.data.iter().map(|x| x * x).sum::<f64>().sqrt()
    }

    /// Solve A·x = b via LU with partial pivoting. Consumes a copy of A.
    /// Returns None if singular to working precision.
    pub fn lu_solve(&self, b: &[f64]) -> Option<Vec<f64>> {
        assert_eq!(self.rows, self.cols);
        assert_eq!(b.len(), self.rows);
        let n = self.rows;
        let mut a = self.data.clone();
        let mut x: Vec<f64> = b.to_vec();
        let idx = |i: usize, j: usize| i * n + j;

        for col in 0..n {
            // Pivot.
            let mut p = col;
            let mut max = a[idx(col, col)].abs();
            for r in (col + 1)..n {
                let v = a[idx(r, col)].abs();
                if v > max {
                    max = v;
                    p = r;
                }
            }
            if max < 1e-14 {
                return None;
            }
            if p != col {
                for j in 0..n {
                    a.swap(idx(col, j), idx(p, j));
                }
                x.swap(col, p);
            }
            // Eliminate.
            let piv = a[idx(col, col)];
            for r in (col + 1)..n {
                let f = a[idx(r, col)] / piv;
                if f == 0.0 {
                    continue;
                }
                a[idx(r, col)] = 0.0;
                for j in (col + 1)..n {
                    a[idx(r, j)] -= f * a[idx(col, j)];
                }
                x[r] -= f * x[col];
            }
        }
        // Back substitution.
        for col in (0..n).rev() {
            x[col] /= a[idx(col, col)];
            for r in 0..col {
                x[r] -= a[idx(r, col)] * x[col];
            }
        }
        Some(x)
    }

    /// Householder QR: returns (Q, R) with A = Q·R, Q orthogonal.
    pub fn qr(&self) -> (Mat, Mat) {
        let m = self.rows;
        let n = self.cols;
        let mut r = self.clone();
        let mut q = Mat::identity(m);
        for k in 0..n.min(m - 1) {
            // Householder vector for column k below the diagonal.
            let mut norm = 0.0;
            for i in k..m {
                norm += r[(i, k)] * r[(i, k)];
            }
            let norm = norm.sqrt();
            if norm < 1e-300 {
                continue;
            }
            let alpha = if r[(k, k)] > 0.0 { -norm } else { norm };
            let mut v = vec![0.0; m];
            v[k] = r[(k, k)] - alpha;
            for i in (k + 1)..m {
                v[i] = r[(i, k)];
            }
            let vtv: f64 = v.iter().map(|x| x * x).sum();
            if vtv < 1e-300 {
                continue;
            }
            // R ← (I − 2vvᵀ/vᵀv) R
            for j in 0..n {
                let mut dot = 0.0;
                for i in k..m {
                    dot += v[i] * r[(i, j)];
                }
                let f = 2.0 * dot / vtv;
                for i in k..m {
                    r[(i, j)] -= f * v[i];
                }
            }
            // Q ← Q (I − 2vvᵀ/vᵀv)
            for i in 0..m {
                let mut dot = 0.0;
                for l in k..m {
                    dot += q[(i, l)] * v[l];
                }
                let f = 2.0 * dot / vtv;
                for l in k..m {
                    q[(i, l)] -= f * v[l];
                }
            }
        }
        (q, r)
    }
}

impl std::ops::Index<(usize, usize)> for Mat {
    type Output = f64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &f64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for Mat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f64 {
        &mut self.data[i * self.cols + j]
    }
}

/// Dense row-major complex matrix (minimal surface for mesh/photonic work).
#[derive(Debug, Clone)]
pub struct CMat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<C64>,
}

impl CMat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Self {
            rows,
            cols,
            data: vec![C64::ZERO; rows * cols],
        }
    }
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n, n);
        for i in 0..n {
            m[(i, i)] = C64::ONE;
        }
        m
    }
    pub fn matmul(&self, other: &CMat) -> CMat {
        assert_eq!(self.cols, other.rows);
        let mut out = CMat::zeros(self.rows, other.cols);
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self[(i, k)];
                for j in 0..other.cols {
                    out[(i, j)] = out[(i, j)] + a * other[(k, j)];
                }
            }
        }
        out
    }
    /// Conjugate transpose.
    pub fn dagger(&self) -> CMat {
        let mut out = CMat::zeros(self.cols, self.rows);
        for i in 0..self.rows {
            for j in 0..self.cols {
                out[(j, i)] = self[(i, j)].conj();
            }
        }
        out
    }
    pub fn frobenius_distance(&self, other: &CMat) -> f64 {
        self.data
            .iter()
            .zip(&other.data)
            .map(|(a, b)| (*a - *b).norm_sq())
            .sum::<f64>()
            .sqrt()
    }
}

impl std::ops::Index<(usize, usize)> for CMat {
    type Output = C64;
    #[inline]
    fn index(&self, (i, j): (usize, usize)) -> &C64 {
        &self.data[i * self.cols + j]
    }
}
impl std::ops::IndexMut<(usize, usize)> for CMat {
    #[inline]
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut C64 {
        &mut self.data[i * self.cols + j]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rng::Rng;

    fn random_mat(rng: &mut Rng, n: usize) -> Mat {
        let mut m = Mat::zeros(n, n);
        for v in m.data.iter_mut() {
            *v = rng.normal();
        }
        m
    }

    /// LU: ‖A·x − b‖ small for random well-conditioned systems.
    #[test]
    fn lu_solves() {
        let mut rng = Rng::new(1);
        for _ in 0..20 {
            let n = 12;
            let a = random_mat(&mut rng, n);
            let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let x = a.lu_solve(&b).expect("nonsingular");
            // residual
            let mut r = 0.0;
            for i in 0..n {
                let mut ax = 0.0;
                for j in 0..n {
                    ax += a[(i, j)] * x[j];
                }
                r += (ax - b[i]).powi(2);
            }
            assert!(r.sqrt() < 1e-9, "residual {}", r.sqrt());
        }
    }

    /// QR: A = Q·R reconstruction and QᵀQ = I.
    #[test]
    fn qr_reconstructs_and_orthogonal() {
        let mut rng = Rng::new(2);
        let a = random_mat(&mut rng, 10);
        let (q, r) = a.qr();
        let qr = q.matmul(&r);
        let mut diff = a.clone();
        for i in 0..diff.data.len() {
            diff.data[i] -= qr.data[i];
        }
        assert!(
            diff.frobenius_norm() < 1e-10,
            "A−QR {}",
            diff.frobenius_norm()
        );
        let qtq = q.transpose().matmul(&q);
        let mut eye_err = 0.0;
        for i in 0..10 {
            for j in 0..10 {
                let expect = if i == j { 1.0 } else { 0.0 };
                eye_err += (qtq[(i, j)] - expect).powi(2);
            }
        }
        assert!(eye_err.sqrt() < 1e-10, "QᵀQ−I {}", eye_err.sqrt());
    }

    /// Complex arithmetic identities: i² = −1, |e^{iθ}| = 1.
    #[test]
    fn complex_identities() {
        let i2 = C64::I * C64::I;
        assert!((i2.re + 1.0).abs() < 1e-15 && i2.im.abs() < 1e-15);
        for k in 0..16 {
            let theta = k as f64 * 0.3927;
            assert!((C64::from_polar(1.0, theta).abs() - 1.0).abs() < 1e-14);
        }
    }

    /// CMat: U·U† = I for a known unitary (DFT 4×4 / 2).
    #[test]
    fn cmat_unitary_check() {
        let n = 4;
        let mut u = CMat::zeros(n, n);
        let norm = 1.0 / (n as f64).sqrt();
        for r in 0..n {
            for c in 0..n {
                u[(r, c)] =
                    C64::from_polar(norm, std::f64::consts::TAU * (r * c) as f64 / n as f64);
            }
        }
        let uud = u.matmul(&u.dagger());
        let eye = CMat::identity(n);
        assert!(uud.frobenius_distance(&eye) < 1e-12);
    }
}
