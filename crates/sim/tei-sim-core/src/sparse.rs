//! Sparse structures: CSR matrices, adjacency, greedy graph coloring, and
//! Markowitz-ordered sparse LU.
//!
//! The coloring (largest-degree-first greedy) powers chromatic Gibbs in
//! `tei-sim-stochastic`: spins within one color class share no edge, so a
//! whole class updates in parallel without violating detailed balance.
//!
//! [`SparseLu`] is the M4 rung of the circuit ladder (roadmap §3.5): a sparse
//! LU for general square matrices with Markowitz pivoting and a
//! same-pattern [`SparseLu::refactor`] fast path for transient stepping.

use crate::linalg::Mat;
use std::cmp::Reverse;
use std::collections::BinaryHeap;

/// Compressed sparse row matrix over f64.
#[derive(Debug, Clone)]
pub struct Csr {
    pub n_rows: usize,
    pub n_cols: usize,
    pub indptr: Vec<usize>,
    pub indices: Vec<u32>,
    pub data: Vec<f64>,
}

impl Csr {
    /// Build from (row, col, value) triplets. Duplicate entries are summed.
    pub fn from_triplets(n_rows: usize, n_cols: usize, triplets: &[(u32, u32, f64)]) -> Self {
        Self::from_triplets_with_map(n_rows, n_cols, triplets).0
    }

    /// [`Csr::from_triplets`] plus the slot map: `map[k]` is the index into
    /// `data` that triplet `k` was accumulated into. Re-assembling the same
    /// triplet sequence with new values can then refill `data` in O(nnz)
    /// without re-sorting — the feed for [`SparseLu::refactor`].
    pub fn from_triplets_with_map(
        n_rows: usize,
        n_cols: usize,
        triplets: &[(u32, u32, f64)],
    ) -> (Self, Vec<usize>) {
        let mut per_row: Vec<Vec<(u32, f64, u32)>> = vec![Vec::new(); n_rows];
        for (k, &(r, c, v)) in triplets.iter().enumerate() {
            per_row[r as usize].push((c, v, k as u32));
        }
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut data: Vec<f64> = Vec::new();
        let mut map = vec![0usize; triplets.len()];
        indptr.push(0);
        for row in &mut per_row {
            row.sort_by_key(|&(c, _, _)| c);
            // Merge duplicates (stable sort keeps insertion order within a
            // column, so sums are bit-identical to sequential accumulation).
            for &(c, v, k) in row.iter() {
                if indices.len() > *indptr.last().unwrap() && *indices.last().unwrap() == c {
                    *data.last_mut().unwrap() += v;
                } else {
                    indices.push(c);
                    data.push(v);
                }
                map[k as usize] = data.len() - 1;
            }
            indptr.push(indices.len());
        }
        (
            Self {
                n_rows,
                n_cols,
                indptr,
                indices,
                data,
            },
            map,
        )
    }

    /// Row slice: (column indices, values).
    #[inline]
    pub fn row(&self, i: usize) -> (&[u32], &[f64]) {
        let lo = self.indptr[i];
        let hi = self.indptr[i + 1];
        (&self.indices[lo..hi], &self.data[lo..hi])
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    /// y = A·x (dense vector).
    pub fn matvec(&self, x: &[f64], y: &mut [f64]) {
        assert_eq!(x.len(), self.n_cols);
        assert_eq!(y.len(), self.n_rows);
        for i in 0..self.n_rows {
            let (cols, vals) = self.row(i);
            let mut acc = 0.0;
            for (c, v) in cols.iter().zip(vals) {
                acc += v * x[*c as usize];
            }
            y[i] = acc;
        }
    }
}

/// Greedy graph coloring, largest-degree-first. Returns (colors, n_colors).
/// The graph is given as a symmetric CSR adjacency (self-loops ignored).
pub fn greedy_coloring(adj: &Csr) -> (Vec<u32>, u32) {
    let n = adj.n_rows;
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| std::cmp::Reverse(adj.indptr[i + 1] - adj.indptr[i]));

    let mut colors = vec![u32::MAX; n];
    let mut max_color = 0u32;
    let mut forbidden: Vec<u32> = Vec::new();
    for &i in &order {
        forbidden.clear();
        let (cols, _) = adj.row(i);
        for &c in cols {
            let cc = colors[c as usize];
            if cc != u32::MAX {
                forbidden.push(cc);
            }
        }
        let mut color = 0u32;
        while forbidden.contains(&color) {
            color += 1;
        }
        colors[i] = color;
        max_color = max_color.max(color);
    }
    (colors, max_color + 1)
}

/// No acceptable pivot at elimination step `step`: the matrix is singular to
/// working precision (e.g. a zero row/column, a floating MNA node, or values
/// that collapsed the remaining submatrix to numerical zero).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SingularError {
    pub step: usize,
}

impl std::fmt::Display for SingularError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "sparse LU: no acceptable pivot at elimination step {}",
            self.step
        )
    }
}

impl std::error::Error for SingularError {}

/// Relative pivot threshold τ for Markowitz pivoting: a candidate must
/// satisfy |a_ij| ≥ τ·max|a_*j| over its active column. 0.1 is the classic
/// SPICE compromise between fill-in and numerical stability.
const MARKOWITZ_TAU: f64 = 0.1;
/// Pivots below this fraction of max|A| are treated as numerical zero.
const PIVOT_FLOOR_REL: f64 = 1e-13;

/// Sparse LU factorization with Markowitz pivoting — `P·A·Q = L·U` with both
/// row (`P`) and column (`Q`) permutations chosen to balance fill-in against
/// numerical stability, the standard SPICE practice.
///
/// # Algorithm
///
/// Factorization runs in three phases, documented here because the roadmap
/// asks for the choice to be explicit:
///
/// 1. **Markowitz ordering** (right-looking): at each elimination step, among
///    active entries passing the threshold test |a_ij| ≥ τ·max|a_*j|
///    (τ = 0.1), pick the entry minimizing the Markowitz count
///    (r_i − 1)(c_j − 1); ties go to the largest |value|. This phase only
///    fixes the pivot order `(P, Q)`.
/// 2. **Symbolic analysis** (Gilbert–Peierls, row-wise): with the pivot order
///    fixed, compute the closed fill pattern of L and U by transitive
///    closure over the rows of U. The pattern is a (tight) superset of the
///    exact fill, so any later value set on the same sparsity fits in it.
/// 3. **Numeric factorization**: row-wise Doolittle elimination over the
///    fixed pattern with a dense scatter workspace.
///
/// Phase 3 alone is [`SparseLu::refactor`] — the transient-stepping fast
/// path: MNA reassembles the same sparsity pattern with new values every
/// step, so steps after the first cost only O(fill) flops, no pivot search,
/// no allocation beyond a scratch vector.
#[derive(Debug, Clone)]
pub struct SparseLu {
    n: usize,
    /// nnz of the factored matrix (refactor value arrays must match it).
    nnz: usize,
    /// `prow[k]` = original row eliminated at step k (row permutation P).
    prow: Vec<usize>,
    /// `pcol[k]` = original column eliminated at step k (column perm Q).
    pcol: Vec<usize>,
    // Pattern of A in pivoted coordinates, row-wise: a_src[e] indexes the
    // original csr.data slot scattered into pivoted column a_cols[e].
    a_indptr: Vec<usize>,
    a_cols: Vec<u32>,
    a_src: Vec<u32>,
    // L (unit lower triangular, diagonal implicit), row-wise, cols ascending.
    l_indptr: Vec<usize>,
    l_cols: Vec<u32>,
    l_vals: Vec<f64>,
    // U strictly upper triangular row-wise (cols ascending) + diagonal.
    u_indptr: Vec<usize>,
    u_cols: Vec<u32>,
    u_vals: Vec<f64>,
    u_diag: Vec<f64>,
}

impl SparseLu {
    /// Factor a square CSR matrix. Symbolic + numeric; the returned object
    /// can [`SparseLu::solve`] and [`SparseLu::refactor`].
    pub fn factor(a: &Csr) -> Result<SparseLu, SingularError> {
        assert_eq!(a.n_rows, a.n_cols, "sparse LU requires a square matrix");
        assert!(a.n_rows > 0, "sparse LU requires a non-empty matrix");
        let (prow, pcol) = markowitz_order(a)?;
        let mut lu = Self::symbolic(a, prow, pcol)?;
        lu.numeric(&a.data)?;
        Ok(lu)
    }

    /// Dimension of the factored system.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Entries in the L + U factors (fill diagnostic; includes the diagonal).
    pub fn fill_nnz(&self) -> usize {
        self.l_vals.len() + self.u_vals.len() + self.n
    }

    /// Re-run the numeric factorization for a matrix with the **same
    /// sparsity pattern** as the one given to [`SparseLu::factor`] but new
    /// values (`values` in that matrix's `csr.data` order). Reuses the pivot
    /// order and fill pattern — the transient/Newton fast path. Errors if a
    /// reused pivot has gone numerically zero under the new values.
    pub fn refactor(&mut self, values: &[f64]) -> Result<(), SingularError> {
        self.numeric(values)
    }

    /// Solve `A·x = b` using the current factors.
    pub fn solve(&self, b: &[f64]) -> Vec<f64> {
        assert_eq!(b.len(), self.n);
        // Forward substitution L·y = P·b (unit diagonal).
        let mut y = vec![0.0; self.n];
        for i in 0..self.n {
            let mut acc = b[self.prow[i]];
            for e in self.l_indptr[i]..self.l_indptr[i + 1] {
                acc -= self.l_vals[e] * y[self.l_cols[e] as usize];
            }
            y[i] = acc;
        }
        // Back substitution U·z = y, in place.
        for i in (0..self.n).rev() {
            let mut acc = y[i];
            for e in self.u_indptr[i]..self.u_indptr[i + 1] {
                acc -= self.u_vals[e] * y[self.u_cols[e] as usize];
            }
            y[i] = acc / self.u_diag[i];
        }
        // Undo the column permutation: x[pcol[j]] = z[j].
        let mut x = vec![0.0; self.n];
        for j in 0..self.n {
            x[self.pcol[j]] = y[j];
        }
        x
    }

    /// Dense reconstruction Â with Â[prow[i], pcol[j]] = (L·U)[i, j], i.e.
    /// Â = Pᵀ·L·U·Qᵀ mapped back to the original index space; equals A up to
    /// factorization round-off. Validation helper (O(n²) memory).
    pub fn reconstruct(&self) -> Mat {
        let mut out = Mat::zeros(self.n, self.n);
        let mut acc = vec![0.0f64; self.n];
        for i in 0..self.n {
            for a in acc.iter_mut() {
                *a = 0.0;
            }
            // Row i of L·U: Σ_k L[i,k]·U[k,:] with L[i,i] = 1 implicit.
            for e in self.l_indptr[i]..self.l_indptr[i + 1] {
                let k = self.l_cols[e] as usize;
                let l = self.l_vals[e];
                acc[k] += l * self.u_diag[k];
                for ue in self.u_indptr[k]..self.u_indptr[k + 1] {
                    acc[self.u_cols[ue] as usize] += l * self.u_vals[ue];
                }
            }
            acc[i] += self.u_diag[i];
            for ue in self.u_indptr[i]..self.u_indptr[i + 1] {
                acc[self.u_cols[ue] as usize] += self.u_vals[ue];
            }
            for (j, &v) in acc.iter().enumerate() {
                out[(self.prow[i], self.pcol[j])] = v;
            }
        }
        out
    }

    /// Phase 2: symbolic Gilbert–Peierls with the pivot order fixed.
    fn symbolic(a: &Csr, prow: Vec<usize>, pcol: Vec<usize>) -> Result<SparseLu, SingularError> {
        let n = a.n_rows;
        let mut icol = vec![0usize; n];
        for (k, &c) in pcol.iter().enumerate() {
            icol[c] = k;
        }
        // Pattern of A in pivoted coordinates, rows in elimination order.
        let mut a_indptr = Vec::with_capacity(n + 1);
        let mut a_cols = Vec::with_capacity(a.nnz());
        let mut a_src = Vec::with_capacity(a.nnz());
        a_indptr.push(0);
        let mut tmp: Vec<(u32, u32)> = Vec::new();
        for &orig in &prow {
            tmp.clear();
            for e in a.indptr[orig]..a.indptr[orig + 1] {
                tmp.push((icol[a.indices[e] as usize] as u32, e as u32));
            }
            tmp.sort_unstable_by_key(|&(c, _)| c);
            for &(c, src) in &tmp {
                a_cols.push(c);
                a_src.push(src);
            }
            a_indptr.push(a_cols.len());
        }
        // Closure pattern per row: start from the A row, and for every
        // below-diagonal column k reached (ascending), absorb U's row-k
        // pattern. Ascending traversal via a min-heap; marks deduplicate.
        let mut l_indptr = vec![0usize];
        let mut l_cols: Vec<u32> = Vec::new();
        let mut u_indptr = vec![0usize];
        let mut u_cols: Vec<u32> = Vec::new();
        let mut mark = vec![usize::MAX; n];
        let mut heap: BinaryHeap<Reverse<u32>> = BinaryHeap::new();
        let mut urow: Vec<u32> = Vec::new();
        for i in 0..n {
            urow.clear();
            debug_assert!(heap.is_empty());
            let mut visit = |c: u32, mark: &mut [usize], heap: &mut BinaryHeap<Reverse<u32>>| {
                if mark[c as usize] != i {
                    mark[c as usize] = i;
                    if (c as usize) < i {
                        heap.push(Reverse(c));
                    } else {
                        urow.push(c);
                    }
                }
            };
            for e in a_indptr[i]..a_indptr[i + 1] {
                visit(a_cols[e], &mut mark, &mut heap);
            }
            while let Some(Reverse(k)) = heap.pop() {
                l_cols.push(k);
                for e in u_indptr[k as usize]..u_indptr[k as usize + 1] {
                    visit(u_cols[e], &mut mark, &mut heap);
                }
            }
            l_indptr.push(l_cols.len());
            if mark[i] != i {
                // No path to the diagonal — structurally singular. Cannot
                // happen if phase 1 succeeded, but fail cleanly regardless.
                return Err(SingularError { step: i });
            }
            urow.sort_unstable();
            debug_assert_eq!(urow[0] as usize, i);
            u_cols.extend_from_slice(&urow[1..]);
            u_indptr.push(u_cols.len());
        }
        let (l_len, u_len) = (l_cols.len(), u_cols.len());
        Ok(SparseLu {
            n,
            nnz: a.nnz(),
            prow,
            pcol,
            a_indptr,
            a_cols,
            a_src,
            l_indptr,
            l_cols,
            l_vals: vec![0.0; l_len],
            u_indptr,
            u_cols,
            u_vals: vec![0.0; u_len],
            u_diag: vec![0.0; n],
        })
    }

    /// Phase 3: numeric row-wise elimination over the fixed pattern.
    fn numeric(&mut self, data: &[f64]) -> Result<(), SingularError> {
        assert_eq!(
            data.len(),
            self.nnz,
            "refactor values must match the factored pattern's nnz"
        );
        let scale = data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let floor = (PIVOT_FLOOR_REL * scale).max(f64::MIN_POSITIVE);
        let mut w = vec![0.0f64; self.n];
        for i in 0..self.n {
            // Zero the workspace over this row's closure pattern, then
            // scatter row i of P·A·Q into it.
            for e in self.l_indptr[i]..self.l_indptr[i + 1] {
                w[self.l_cols[e] as usize] = 0.0;
            }
            w[i] = 0.0;
            for e in self.u_indptr[i]..self.u_indptr[i + 1] {
                w[self.u_cols[e] as usize] = 0.0;
            }
            for e in self.a_indptr[i]..self.a_indptr[i + 1] {
                w[self.a_cols[e] as usize] = data[self.a_src[e] as usize];
            }
            // Eliminate with the rows above, ascending.
            for e in self.l_indptr[i]..self.l_indptr[i + 1] {
                let k = self.l_cols[e] as usize;
                let l = w[k] / self.u_diag[k];
                self.l_vals[e] = l;
                if l != 0.0 {
                    for ue in self.u_indptr[k]..self.u_indptr[k + 1] {
                        w[self.u_cols[ue] as usize] -= l * self.u_vals[ue];
                    }
                }
            }
            let d = w[i];
            if !(d.abs() >= floor) {
                // (negated ≥ also catches NaN)
                return Err(SingularError { step: i });
            }
            self.u_diag[i] = d;
            for e in self.u_indptr[i]..self.u_indptr[i + 1] {
                self.u_vals[e] = w[self.u_cols[e] as usize];
            }
        }
        Ok(())
    }
}

/// Phase 1: right-looking elimination on a working copy, recording the
/// Markowitz pivot order. See [`SparseLu`] docs for the selection rule.
fn markowitz_order(a: &Csr) -> Result<(Vec<usize>, Vec<usize>), SingularError> {
    let n = a.n_rows;
    let mut rows: Vec<Vec<(u32, f64)>> = (0..n)
        .map(|i| {
            let (cols, vals) = a.row(i);
            cols.iter().copied().zip(vals.iter().copied()).collect()
        })
        .collect();
    let mut row_alive = vec![true; n];
    let mut col_alive = vec![true; n];
    let a_scale = a.data.iter().fold(0.0f64, |m, v| m.max(v.abs()));
    let floor = (PIVOT_FLOOR_REL * a_scale).max(f64::MIN_POSITIVE);
    let mut prow = Vec::with_capacity(n);
    let mut pcol = Vec::with_capacity(n);
    // Per-step column stats, epoch-stamped to avoid O(n) clears.
    let mut col_stamp = vec![usize::MAX; n];
    let mut col_cnt = vec![0usize; n];
    let mut col_max = vec![0.0f64; n];
    let mut row_cnt = vec![0usize; n];

    for step in 0..n {
        // Pass 1: active row/column counts + per-column max |value|.
        for i in 0..n {
            if !row_alive[i] {
                continue;
            }
            let mut cnt = 0usize;
            for &(c, v) in &rows[i] {
                let cu = c as usize;
                if !col_alive[cu] {
                    continue;
                }
                cnt += 1;
                if col_stamp[cu] != step {
                    col_stamp[cu] = step;
                    col_cnt[cu] = 0;
                    col_max[cu] = 0.0;
                }
                col_cnt[cu] += 1;
                col_max[cu] = col_max[cu].max(v.abs());
            }
            row_cnt[i] = cnt;
        }
        // Pass 2: among entries with |v| ≥ max(τ·colmax, floor), take the
        // minimal Markowitz count (r−1)(c−1); ties → largest |v|.
        let mut best: Option<(usize, usize, f64, usize)> = None;
        for i in 0..n {
            if !row_alive[i] {
                continue;
            }
            for &(c, v) in &rows[i] {
                let cu = c as usize;
                if !col_alive[cu] {
                    continue;
                }
                let av = v.abs();
                if av < floor || av < MARKOWITZ_TAU * col_max[cu] {
                    continue;
                }
                let count = (row_cnt[i] - 1) * (col_cnt[cu] - 1);
                let better = match best {
                    None => true,
                    Some((_, _, bav, bcount)) => count < bcount || (count == bcount && av > bav),
                };
                if better {
                    best = Some((i, cu, av, count));
                }
            }
        }
        let Some((pi, pj, _, _)) = best else {
            return Err(SingularError { step });
        };
        prow.push(pi);
        pcol.push(pj);
        row_alive[pi] = false;
        col_alive[pj] = false;
        // Pivot row restricted to the still-active columns.
        let piv_val = rows[pi]
            .iter()
            .find(|&&(c, _)| c as usize == pj)
            .map(|&(_, v)| v)
            .expect("pivot entry present");
        let piv_row: Vec<(u32, f64)> = rows[pi]
            .iter()
            .filter(|&&(c, _)| col_alive[c as usize])
            .copied()
            .collect();
        // Update every active row holding the pivot column (right-looking).
        for r in 0..n {
            if !row_alive[r] {
                continue;
            }
            let Some(&(_, arj)) = rows[r].iter().find(|&&(c, _)| c as usize == pj) else {
                continue;
            };
            let mult = arj / piv_val;
            // rows[r] ← rows[r] − mult·piv_row over active columns
            // (dead columns dropped — they belong to the discarded L).
            let ra = &rows[r];
            let mut out: Vec<(u32, f64)> = Vec::with_capacity(ra.len() + piv_row.len());
            let (mut ia, mut ib) = (0usize, 0usize);
            while ia < ra.len() || ib < piv_row.len() {
                let ca = if ia < ra.len() {
                    let (c, _) = ra[ia];
                    if !col_alive[c as usize] {
                        ia += 1;
                        continue;
                    }
                    c
                } else {
                    u32::MAX
                };
                let cb = if ib < piv_row.len() {
                    piv_row[ib].0
                } else {
                    u32::MAX
                };
                match ca.cmp(&cb) {
                    std::cmp::Ordering::Less => {
                        out.push(ra[ia]);
                        ia += 1;
                    }
                    std::cmp::Ordering::Greater => {
                        out.push((cb, -mult * piv_row[ib].1));
                        ib += 1;
                    }
                    std::cmp::Ordering::Equal => {
                        out.push((ca, ra[ia].1 - mult * piv_row[ib].1));
                        ia += 1;
                        ib += 1;
                    }
                }
            }
            rows[r] = out;
        }
    }
    Ok((prow, pcol))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ring(n: usize) -> Csr {
        let mut t = Vec::new();
        for i in 0..n {
            let j = (i + 1) % n;
            t.push((i as u32, j as u32, 1.0));
            t.push((j as u32, i as u32, 1.0));
        }
        Csr::from_triplets(n, n, &t)
    }

    #[test]
    fn matvec_identity_like() {
        let a = Csr::from_triplets(3, 3, &[(0, 0, 2.0), (1, 1, 3.0), (2, 2, 4.0)]);
        let mut y = vec![0.0; 3];
        a.matvec(&[1.0, 1.0, 1.0], &mut y);
        assert_eq!(y, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn duplicates_sum() {
        let a = Csr::from_triplets(1, 1, &[(0, 0, 1.5), (0, 0, 2.5)]);
        assert_eq!(a.nnz(), 1);
        assert_eq!(a.data[0], 4.0);
    }

    /// Coloring validity: no edge joins two same-colored vertices.
    #[test]
    fn coloring_is_proper() {
        let adj = ring(101); // odd cycle needs 3 colors
        let (colors, n_colors) = greedy_coloring(&adj);
        for i in 0..101 {
            let (cols, _) = adj.row(i);
            for &j in cols {
                assert_ne!(colors[i], colors[j as usize], "edge ({i},{j}) same color");
            }
        }
        assert!(n_colors <= 3, "odd ring should 3-color, got {n_colors}");
    }

    /// Even cycles are 2-chromatic and greedy must achieve ≤ 3 (achieves 2
    /// with degree-first order on a cycle).
    #[test]
    fn even_ring_two_colorable() {
        let adj = ring(100);
        let (colors, _) = greedy_coloring(&adj);
        for i in 0..100 {
            let (cols, _) = adj.row(i);
            for &j in cols {
                assert_ne!(colors[i], colors[j as usize]);
            }
        }
    }

    /// Triplet→slot map: refilling values through the map reproduces `data`.
    #[test]
    fn triplet_map_refills_data() {
        let t = [
            (0u32, 1u32, 1.0),
            (1, 0, 2.0),
            (0, 1, 0.5), // duplicate of slot (0,1)
            (1, 1, 3.0),
            (0, 0, 4.0),
        ];
        let (a, map) = Csr::from_triplets_with_map(2, 2, &t);
        assert_eq!(a.nnz(), 4);
        let mut refill = vec![0.0; a.nnz()];
        for (k, &(_, _, v)) in t.iter().enumerate() {
            refill[map[k]] += v;
        }
        assert_eq!(refill, a.data);
    }

    // ---------- sparse LU (M4) ----------

    use crate::rng::Rng;

    /// Random square sparse matrix: full diagonal in [2,3) plus off-diagonal
    /// N(0,1)·½ entries with probability `density`. Deterministic per seed.
    fn random_sparse(n: usize, density: f64, rng: &mut Rng) -> Csr {
        let mut t = Vec::new();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    t.push((i as u32, j as u32, 2.0 + rng.f64()));
                } else if rng.f64() < density {
                    t.push((i as u32, j as u32, 0.5 * rng.normal()));
                }
            }
        }
        Csr::from_triplets(n, n, &t)
    }

    fn dense_of(a: &Csr) -> Mat {
        let mut m = Mat::zeros(a.n_rows, a.n_cols);
        for i in 0..a.n_rows {
            let (cols, vals) = a.row(i);
            for (&c, &v) in cols.iter().zip(vals) {
                m[(i, c as usize)] = v;
            }
        }
        m
    }

    fn frob_diff(a: &Mat, b: &Mat) -> f64 {
        a.data
            .iter()
            .zip(&b.data)
            .map(|(x, y)| (x - y) * (x - y))
            .sum::<f64>()
            .sqrt()
    }

    /// ‖A − Pᵀ·L·U·Qᵀ‖_F < 1e-10 across sizes and densities (property,
    /// deterministic RNG).
    #[test]
    fn sparse_lu_reconstruction_property() {
        let mut rng = Rng::new(42);
        for &(n, density) in &[(5usize, 0.5f64), (20, 0.3), (60, 0.1), (150, 0.04)] {
            let a = random_sparse(n, density, &mut rng);
            let lu = SparseLu::factor(&a).expect("random diag-anchored matrix nonsingular");
            let err = frob_diff(&dense_of(&a), &lu.reconstruct());
            assert!(err < 1e-10, "n={n} density={density}: ‖A−PLUQ‖={err:.3e}");
        }
    }

    /// Sparse solve matches the dense partial-pivot LU to 1e-10.
    #[test]
    fn sparse_solve_matches_dense_lu() {
        let mut rng = Rng::new(3);
        for &(n, density) in &[(8usize, 0.4f64), (40, 0.15), (120, 0.05)] {
            let a = random_sparse(n, density, &mut rng);
            let b: Vec<f64> = (0..n).map(|_| rng.normal()).collect();
            let x_sparse = SparseLu::factor(&a).unwrap().solve(&b);
            let x_dense = dense_of(&a).lu_solve(&b).unwrap();
            for (s, d) in x_sparse.iter().zip(&x_dense) {
                assert!((s - d).abs() < 1e-10, "n={n}: sparse {s} vs dense {d}");
            }
        }
    }

    /// Refactor with the *same* values is bit-identical to the original
    /// factorization; refactor with *new* values on the same pattern matches
    /// a fresh factorization of the new matrix.
    #[test]
    fn refactor_matches_fresh_factor() {
        let mut rng = Rng::new(7);
        let a = random_sparse(40, 0.15, &mut rng);
        let b: Vec<f64> = (0..40).map(|_| rng.normal()).collect();
        let mut lu = SparseLu::factor(&a).unwrap();
        let x0 = lu.solve(&b);

        // Same values → bit-identical factors and solution.
        lu.refactor(&a.data).unwrap();
        assert_eq!(lu.solve(&b), x0, "same-value refactor must be exact");
        assert!(frob_diff(&lu.reconstruct(), &dense_of(&a)) < 1e-10);

        // New values, same pattern → matches a fresh Markowitz factor.
        let new_data: Vec<f64> = a.data.iter().map(|v| v * (1.0 + 0.3 * rng.f64())).collect();
        lu.refactor(&new_data).unwrap();
        let x_re = lu.solve(&b);
        let mut a2 = a.clone();
        a2.data = new_data;
        let x_fresh = SparseLu::factor(&a2).unwrap().solve(&b);
        for (p, q) in x_re.iter().zip(&x_fresh) {
            assert!(
                (p - q).abs() < 1e-12 * p.abs().max(1.0),
                "refactor {p} vs fresh {q}"
            );
        }
    }

    /// Singular inputs error cleanly (no panic): zero row, linearly
    /// dependent rows, and a refactor that zeroes a pivot.
    #[test]
    fn singular_matrices_error_cleanly() {
        // Zero row 2.
        let a = Csr::from_triplets(3, 3, &[(0, 0, 1.0), (1, 1, 1.0), (0, 2, 1.0)]);
        assert!(SparseLu::factor(&a).is_err());
        // Duplicate rows → exactly singular after one elimination.
        let a = Csr::from_triplets(2, 2, &[(0, 0, 1.0), (0, 1, 2.0), (1, 0, 1.0), (1, 1, 2.0)]);
        assert!(SparseLu::factor(&a).is_err());
        // Refactor driving a pivot to zero.
        let a = Csr::from_triplets(2, 2, &[(0, 0, 1.0), (1, 1, 1.0)]);
        let mut lu = SparseLu::factor(&a).unwrap();
        let err = lu.refactor(&[1.0, 0.0]).unwrap_err();
        assert_eq!(err.step, 1);
    }
}
