//! Linear-system solver selection — the M4 rung of the roadmap ladder
//! (docs/SIM-ROADMAP.md §3.5): dense LU below a node-count threshold, sparse
//! Markowitz LU (`tei_sim_core::sparse::SparseLu`) above it.
//!
//! The sparse path exploits the structure of transient stepping: every step
//! (and every Newton iteration) reassembles the **same sparsity pattern**
//! with new values, so the pivot order, fill pattern, and triplet→slot map
//! are computed once on the first solve and every later solve is a pure
//! numeric refactor + triangular solves — no pivot search, no symbolic work.
//! If a reused pivot goes numerically bad (a diode conductance can swing
//! many decades between Newton iterates), the solver re-pivots from scratch
//! once before declaring the system singular.
//!
//! [`SPARSE_NODE_THRESHOLD`] was chosen from the ignored benchmark in
//! `tests/sparse_solver.rs` (`bench_dense_vs_sparse_rc_ladders`). Measured
//! µs/step on RC ladders (release, Apple Silicon, init included):
//!
//! ```text
//! nodes    dense   sparse
//!     4     0.25     0.22
//!    16     0.93     0.62
//!    48     5.10     1.91
//!   100    21.72     4.19
//!   500   443.39    47.36   (9.4×)
//!  1000  2107.51   491.56   (4.3×)
//! ```
//!
//! The pattern-reuse refactor makes sparse at least as fast as dense at
//! *every* measured size, so the threshold is not a performance crossover:
//! it exists to keep the small cells that the analytic validation suites pin
//! down (≤ 12 nodes in tei-sim-adiabatic) on the exact, bit-identical
//! pre-M4 dense path.

use crate::mna::{self, ElemState, Mode, Topology};
use crate::netlist::Netlist;
use crate::{CircuitError, Method};
use serde::{Deserialize, Serialize};
use tei_sim_core::sparse::{Csr, SparseLu};

/// Node count above which `Auto` routes to the sparse Markowitz LU.
/// Sparse already matches dense per-step cost at 4 nodes (see module docs),
/// so this is a conservatism line, not a performance crossover: everything
/// the analytic suites validated pre-M4 stays bit-identical on dense.
pub const SPARSE_NODE_THRESHOLD: usize = 16;

/// Caller-facing solver selection for a run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverChoice {
    /// Dense at ≤ [`SPARSE_NODE_THRESHOLD`] nodes, sparse above.
    #[default]
    Auto,
    /// Force the dense LU path.
    Dense,
    /// Force the sparse Markowitz-LU path.
    Sparse,
}

/// Which path a run actually used (reported in `TransientResult`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverKind {
    Dense,
    Sparse,
}

/// Cached sparse state: built on the first solve, reused for every later
/// solve of the same (netlist, topology, mode).
pub(crate) struct SparsePattern {
    /// Matrix pattern (values hold the first assembly; refreshed on the
    /// re-pivot fallback).
    csr: Csr,
    /// Triplet index → `csr.data` slot, valid because assembly is
    /// deterministic and emits the identical stamp sequence every call.
    map: Vec<usize>,
    /// Current values in `csr.data` order, rebuilt each solve.
    values: Vec<f64>,
    lu: SparseLu,
}

/// One linear-system solver instance, owned by a transient/DC run per
/// assembly mode (Init and Step have different dimensions and patterns).
pub(crate) enum SystemSolver {
    Dense,
    Sparse {
        triplets: Vec<(u32, u32, f64)>,
        pattern: Option<SparsePattern>,
    },
}

impl SystemSolver {
    pub fn new(n_nodes: usize, choice: SolverChoice) -> Self {
        let sparse = match choice {
            SolverChoice::Auto => n_nodes > SPARSE_NODE_THRESHOLD,
            SolverChoice::Dense => false,
            SolverChoice::Sparse => true,
        };
        if sparse {
            SystemSolver::Sparse {
                triplets: Vec::new(),
                pattern: None,
            }
        } else {
            SystemSolver::Dense
        }
    }

    pub fn kind(&self) -> SolverKind {
        match self {
            SystemSolver::Dense => SolverKind::Dense,
            SystemSolver::Sparse { .. } => SolverKind::Sparse,
        }
    }

    /// Assemble and solve one MNA system at the given evaluation point.
    #[allow(clippy::too_many_arguments)]
    pub fn solve_system(
        &mut self,
        net: &Netlist,
        topo: &Topology,
        mode: Mode,
        method: Method,
        t: f64,
        h: f64,
        src_scale: f64,
        hist: &[ElemState],
        dlin: &[f64],
    ) -> Result<Vec<f64>, CircuitError> {
        match self {
            SystemSolver::Dense => {
                let (a, b) = mna::assemble(net, topo, mode, method, t, h, src_scale, hist, dlin);
                a.lu_solve(&b).ok_or(CircuitError::Singular)
            }
            SystemSolver::Sparse { triplets, pattern } => {
                triplets.clear();
                let mut b = vec![0.0; topo.dim];
                mna::assemble_into(
                    net, topo, mode, method, t, h, src_scale, hist, dlin, triplets, &mut b,
                );
                match pattern {
                    None => {
                        // First solve: build the pattern, pick pivots.
                        let (csr, map) = Csr::from_triplets_with_map(topo.dim, topo.dim, triplets);
                        let lu = SparseLu::factor(&csr).map_err(|_| CircuitError::Singular)?;
                        let x = lu.solve(&b);
                        let values = csr.data.clone();
                        *pattern = Some(SparsePattern {
                            csr,
                            map,
                            values,
                            lu,
                        });
                        Ok(x)
                    }
                    Some(p) => {
                        // Steady state: refill values through the slot map
                        // and replay the numeric factorization.
                        debug_assert_eq!(triplets.len(), p.map.len());
                        p.values.iter_mut().for_each(|v| *v = 0.0);
                        for (k, &(_, _, v)) in triplets.iter().enumerate() {
                            p.values[p.map[k]] += v;
                        }
                        if p.lu.refactor(&p.values).is_err() {
                            // Reused pivots went numerically bad — re-pivot
                            // for the current values before giving up.
                            p.csr.data.copy_from_slice(&p.values);
                            p.lu = SparseLu::factor(&p.csr).map_err(|_| CircuitError::Singular)?;
                        }
                        Ok(p.lu.solve(&b))
                    }
                }
            }
        }
    }
}
