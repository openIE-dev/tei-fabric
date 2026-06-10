//! Sparse structures: CSR matrices, adjacency, greedy graph coloring.
//!
//! The coloring (largest-degree-first greedy) powers chromatic Gibbs in
//! `tei-sim-stochastic`: spins within one color class share no edge, so a
//! whole class updates in parallel without violating detailed balance.

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
        let mut per_row: Vec<Vec<(u32, f64)>> = vec![Vec::new(); n_rows];
        for &(r, c, v) in triplets {
            per_row[r as usize].push((c, v));
        }
        let mut indptr = Vec::with_capacity(n_rows + 1);
        let mut indices = Vec::new();
        let mut data = Vec::new();
        indptr.push(0);
        for row in &mut per_row {
            row.sort_by_key(|&(c, _)| c);
            // Merge duplicates.
            let mut merged: Vec<(u32, f64)> = Vec::with_capacity(row.len());
            for &(c, v) in row.iter() {
                if let Some(last) = merged.last_mut() {
                    if last.0 == c {
                        last.1 += v;
                        continue;
                    }
                }
                merged.push((c, v));
            }
            for (c, v) in merged {
                indices.push(c);
                data.push(v);
            }
            indptr.push(indices.len());
        }
        Self {
            n_rows,
            n_cols,
            indptr,
            indices,
            data,
        }
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
}
