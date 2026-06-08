//! Workload IR.
//!
//! A workload is a sequence (eventually a DAG) of invocations of Periodic Stack
//! primitives. Each invocation has an `OpProfile` describing the shape and
//! dtype of the operands — the inputs a substrate cost function needs to price
//! the op.
//!
//! v0 keeps the graph flat (no explicit data dependencies between invocations).
//! That's enough for the cost-surface dispatcher: each primitive is priced
//! independently, and the dispatcher minimizes total joules. Future versions
//! add edges so the runtime can reason about staging, fusion, and recompute.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Default, Deserialize, Serialize, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum Dtype {
    I4,
    I8,
    I16,
    I32,
    Bf16,
    #[default]
    F16,
    F32,
    F64,
}

impl Dtype {
    pub fn bits(self) -> u32 {
        match self {
            Dtype::I4 => 4,
            Dtype::I8 => 8,
            Dtype::I16 => 16,
            Dtype::I32 => 32,
            Dtype::Bf16 | Dtype::F16 => 16,
            Dtype::F32 => 32,
            Dtype::F64 => 64,
        }
    }
}

/// Tensor shape for an operand. Empty `dims` means scalar / unspecified.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct TensorShape {
    pub dims: Vec<usize>,
}

impl TensorShape {
    pub fn elements(&self) -> usize {
        if self.dims.is_empty() { 0 } else { self.dims.iter().product() }
    }
}

/// What a substrate cost function needs to price one invocation.
///
/// The fields are deliberately permissive — different primitives need
/// different params (matmul wants `m,k,n`; convolution wants `c_in,c_out,k_h,k_w`;
/// sampling wants `n_spins,sweeps`). The substrate dialect inspects the fields
/// that are meaningful to it.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct OpProfile {
    /// Primary operand shape — for matmul, the result tensor; for sampling,
    /// the variable count.
    #[serde(default)]
    pub shape: TensorShape,
    /// Inner-reduction dimension for matmul-class primitives (the `k` in `m,k,n`).
    /// None when not applicable.
    #[serde(default)]
    pub reduce_dim: Option<usize>,
    /// Batch size (e.g. transformer batch, or number of independent samples).
    #[serde(default = "default_batch")]
    pub batch: usize,
    /// Operand dtype.
    #[serde(default = "default_dtype")]
    pub dtype: Dtype,
    /// Activation sparsity if known (0.0 = dense, 1.0 = fully sparse).
    #[serde(default)]
    pub sparsity: f64,
}

fn default_batch() -> usize { 1 }
fn default_dtype() -> Dtype { Dtype::F16 }

impl OpProfile {
    /// MAC count for a dense matmul `(m × k) × (k × n)`.
    /// Returns `None` if either dimension is missing.
    pub fn matmul_macs(&self) -> Option<u64> {
        let m_n: usize = if self.shape.dims.len() == 2 {
            self.shape.dims[0] * self.shape.dims[1]
        } else {
            return None;
        };
        let k = self.reduce_dim?;
        Some((m_n as u64) * (k as u64) * (self.batch as u64))
    }
}

/// One invocation of a primitive.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Invocation {
    pub primitive_id: u32,
    /// How many times per cycle this primitive runs.
    pub count: u64,
    #[serde(default)]
    pub profile: OpProfile,
    /// Free-form annotation for the viz layer.
    #[serde(default)]
    pub label: String,
}

/// Total budget the dispatcher is solving against.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Constraints {
    /// Cycles per second the workload must sustain (30 fps, 1 inference, …).
    pub cycles_per_second: f64,
    /// Power budget in watts. The dispatcher targets staying within this.
    pub power_budget_w: f64,
    /// Optional volume — used by leverage / NRE-amortization heuristics.
    #[serde(default = "default_volume")]
    pub volume: u64,
}

fn default_volume() -> u64 { 10_000 }

/// A full workload spec.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Workload {
    pub goal: String,
    pub invocations: Vec<Invocation>,
    pub constraints: Constraints,
}
