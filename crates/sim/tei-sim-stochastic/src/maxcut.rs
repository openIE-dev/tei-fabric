//! Max-Cut ↔ Ising reduction + the problem-spec surface for /api/execute.
//!
//! maximize  cut(s) = Σ_{(i,j)∈E} w_ij (1 − sᵢsⱼ)/2
//! ⇔ minimize  Σ w_ij sᵢsⱼ  ⇔  Ising with h = 0, J_ij = −w_ij
//! (then E(s) = Σ w sᵢsⱼ and cut = (W_total − E)/2).

use crate::IsingModel;
use crate::graphs::{self, Graph};
use serde::{Deserialize, Serialize};

/// Convert a graph into the equivalent Ising minimization.
pub fn to_ising(g: &Graph) -> IsingModel {
    let edges: Vec<(u32, u32, f64)> = g.edges.iter().map(|&(i, j, w)| (i, j, -w)).collect();
    IsingModel::new(vec![0.0; g.n], &edges)
}

/// Cut value of a spin assignment.
pub fn cut_value(g: &Graph, s: &[i8]) -> f64 {
    g.edges
        .iter()
        .map(|&(i, j, w)| w * (1 - s[i as usize] * s[j as usize]) as f64 / 2.0)
        .sum()
}

/// Problem instances accepted over the API. Closed-form instances carry
/// their known optimum so the UI can show "reached optimum".
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProblemSpec {
    Complete { n: usize },
    Cycle { n: usize },
    CompleteBipartite { a: usize, b: usize },
    Petersen,
    RandomRegular { n: usize, degree: usize, seed: u64 },
}

impl ProblemSpec {
    pub fn build(&self) -> Graph {
        match *self {
            ProblemSpec::Complete { n } => graphs::complete(n),
            ProblemSpec::Cycle { n } => graphs::cycle(n),
            ProblemSpec::CompleteBipartite { a, b } => graphs::complete_bipartite(a, b),
            ProblemSpec::Petersen => graphs::petersen(),
            ProblemSpec::RandomRegular { n, degree, seed } => {
                graphs::random_regular(n, degree, seed)
            }
        }
    }

    /// Closed-form Max-Cut optimum where one exists.
    pub fn known_optimum(&self) -> Option<f64> {
        match *self {
            ProblemSpec::Complete { n } => Some(((n / 2) * n.div_ceil(2)) as f64),
            ProblemSpec::Cycle { n } => Some(if n % 2 == 0 { n as f64 } else { (n - 1) as f64 }),
            ProblemSpec::CompleteBipartite { a, b } => Some((a * b) as f64),
            ProblemSpec::Petersen => Some(12.0),
            ProblemSpec::RandomRegular { .. } => None,
        }
    }
}
