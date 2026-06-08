//! tei-import — read an ONNX model and emit a tei-ir Workload.
//!
//! Reads the .onnx protobuf with prost, walks the graph's nodes, maps each
//! ONNX op_type to a Periodic Stack primitive via `op_map`, resolves operand
//! shapes from `value_info` + `initializer` + `graph.input`, and emits one
//! `Invocation` per supported op.
//!
//! Operators we don't have a primitive mapping for (pointwise activations,
//! reshapes, slices, broadcasts, etc.) are skipped — they're not energy
//! hotspots at the resolution we care about and the dispatcher's cost
//! surface stays meaningful without them.
//!
//! Operators we can map but whose shapes don't resolve (typical for ONNX
//! models exported with dynamic axes and no shape inference baked in) are
//! reported in the result as `skipped_unresolved`. The caller can either
//! re-run ONNX shape inference upstream, or hand-fill the missing dims
//! through the existing web form.

mod op_map;
mod shape_infer;

use std::collections::HashMap;
use tei_ir::{Constraints, Dtype, Invocation, OpProfile, TensorShape, Workload};

// Generated protobuf types.
pub mod proto {
    include!(concat!(env!("OUT_DIR"), "/onnx.rs"));
}

#[derive(Debug, thiserror::Error)]
pub enum ImportError {
    #[error("could not decode .onnx protobuf: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error("graph missing")]
    NoGraph,
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Outcome of one import call.
#[derive(Debug, Clone, serde::Serialize)]
pub struct ImportReport {
    pub workload: Workload,
    pub model_name: String,
    pub producer: String,
    pub node_count: usize,
    pub mapped_count: usize,
    pub skipped_unresolved: Vec<String>,
    pub skipped_unmapped: HashMap<String, u32>,
}

/// Map an ONNX TensorProto/TypeProto::Tensor.elem_type code → tei-ir Dtype.
/// Codes per the ONNX spec (TensorProto.DataType enum).
fn map_dtype(code: i32) -> Dtype {
    match code {
        1 => Dtype::F32,    // FLOAT
        2 => Dtype::I8,     // UINT8 — collapse to i8
        3 => Dtype::I8,     // INT8
        4 => Dtype::I16,    // UINT16 — collapse to i16
        5 => Dtype::I16,    // INT16
        6 => Dtype::I32,    // INT32
        7 => Dtype::I32,    // INT64 — best we have is i32
        10 => Dtype::F16,   // FLOAT16
        11 => Dtype::F64,   // DOUBLE
        16 => Dtype::Bf16,  // BFLOAT16
        _ => Dtype::F32,
    }
}

/// Build a `name → dims` map from every shape source in the graph:
/// graph.input, graph.value_info, graph.output, initializer.
fn build_shape_map(g: &proto::GraphProto) -> HashMap<String, Vec<i64>> {
    let mut shapes: HashMap<String, Vec<i64>> = HashMap::new();

    let from_value_info = |vi: &proto::ValueInfoProto| -> Option<(String, Vec<i64>)> {
        let name = vi.name.clone();
        if name.is_empty() { return None; }
        let tensor_type = vi.r#type.as_ref()?.tensor_type.as_ref()?;
        let shape = tensor_type.shape.as_ref()?;
        let dims: Vec<i64> = shape.dim.iter()
            .map(|d| if d.dim_value > 0 { d.dim_value } else { -1 })
            .collect();
        Some((name, dims))
    };

    for vi in &g.input        { if let Some((n, d)) = from_value_info(vi) { shapes.insert(n, d); } }
    for vi in &g.output       { if let Some((n, d)) = from_value_info(vi) { shapes.insert(n, d); } }
    for vi in &g.value_info   { if let Some((n, d)) = from_value_info(vi) { shapes.insert(n, d); } }
    // Initializers (model weights) — their dims are authoritative.
    for t in &g.initializer {
        if !t.name.is_empty() {
            shapes.insert(t.name.clone(), t.dims.clone());
        }
    }
    shapes
}

/// Convert an ONNX i64 shape to usize, replacing unknown (-1, 0, or "?")
/// with 1 for batch axes; missing dims signal an unresolved op.
fn coerce_dims(dims: &[i64]) -> Option<Vec<usize>> {
    let mut out = Vec::with_capacity(dims.len());
    for (i, &d) in dims.iter().enumerate() {
        if d > 0 {
            out.push(d as usize);
        } else if i == 0 {
            // Batch axis can be dynamic — default to 1.
            out.push(1);
        } else {
            return None;
        }
    }
    Some(out)
}

/// Look up shapes for the inputs of a MatMul / Gemm node and derive (m, k, n).
fn resolve_matmul(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
) -> Option<(usize, usize, usize)> {
    if node.input.len() < 2 { return None; }
    let a = shapes.get(&node.input[0])?;
    let b = shapes.get(&node.input[1])?;
    let a = coerce_dims(a)?;
    let b = coerce_dims(b)?;
    if a.len() < 2 || b.len() < 2 { return None; }
    // For higher-rank tensors, the matmul is the last two axes; batch
    // dimensions multiply on top. Collapse the batch dims into m.
    let batch_a: usize = a[..a.len() - 2].iter().product::<usize>().max(1);
    let m = batch_a * a[a.len() - 2];
    let k_a = a[a.len() - 1];
    let k_b = b[b.len() - 2];
    let n = b[b.len() - 1];
    if k_a != k_b {
        return None; // shape mismatch — skip
    }
    Some((m, k_a, n))
}

/// Look up shapes for the inputs of a Conv node and derive a matmul-equivalent
/// `(m, k, n)`. For Conv(X[N,C,H,W], W[M,C,kH,kW]) producing Y[N,M,H',W'],
/// the canonical matmul-equivalent is `m = N*H'*W', k = C*kH*kW, n = M`.
fn resolve_conv(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
) -> Option<(usize, usize, usize)> {
    if node.input.len() < 2 { return None; }
    let x = shapes.get(&node.input[0]).and_then(|d| coerce_dims(d))?;
    let w = shapes.get(&node.input[1]).and_then(|d| coerce_dims(d))?;
    if x.len() < 3 || w.len() < 3 { return None; }
    // X = [N, C, *spatial], W = [M, C/group, *kernel]
    let n_batch = x[0];
    let c_in = x[1];
    let m_out = w[0];
    // Approximate output spatial dims with input spatial dims (ignoring
    // stride / padding for v0). Slight over-estimate of MAC count, well
    // within the cost-surface fidelity envelope.
    let spatial_x: usize = x[2..].iter().product();
    let kernel: usize = w[2..].iter().product();
    // n  = output channels
    // k  = input channels × kernel area
    // m  = batch × output spatial area  (≈ input spatial area)
    let m_dim = n_batch * spatial_x;
    let k_dim = c_in * kernel;
    let n_dim = m_out;
    Some((m_dim, k_dim, n_dim))
}

/// Pick a dtype for the workload — first input's dtype if available, else F32.
fn pick_dtype(g: &proto::GraphProto) -> Dtype {
    for vi in &g.input {
        if let Some(t) = vi.r#type.as_ref().and_then(|t| t.tensor_type.as_ref()) {
            return map_dtype(t.elem_type);
        }
    }
    Dtype::F32
}

/// Parse ONNX bytes into a tei-ir Workload.
pub fn parse_onnx(bytes: &[u8]) -> Result<ImportReport, ImportError> {
    use prost::Message;
    let model = proto::ModelProto::decode(bytes)?;
    let graph = model.graph.ok_or(ImportError::NoGraph)?;
    let mut shapes = build_shape_map(&graph);
    // Propagate shapes through identity ops + Conv + Pool so that
    // PyTorch-exported models (which only ship I/O + initializer shapes)
    // become fully resolvable. Cheap one-pass forward walk.
    shape_infer::propagate(&graph, &mut shapes);
    let dtype = pick_dtype(&graph);

    let mut invocations: Vec<Invocation> = Vec::new();
    let mut skipped_unresolved: Vec<String> = Vec::new();
    let mut skipped_unmapped: HashMap<String, u32> = HashMap::new();
    let node_count = graph.node.len();
    let mut mapped_count: usize = 0;

    for node in &graph.node {
        let mapping = op_map::map_op(&node.op_type);
        let (prim_id, kind) = match mapping {
            Some(m) => m,
            None => {
                *skipped_unmapped.entry(node.op_type.clone()).or_insert(0) += 1;
                continue;
            }
        };

        let label = if node.name.is_empty() {
            format!("{}", node.op_type)
        } else {
            format!("{} · {}", node.op_type, node.name)
        };

        let profile = match kind {
            "matmul" => {
                match resolve_matmul(node, &shapes) {
                    Some((m, k, n)) => OpProfile {
                        shape: TensorShape { dims: vec![m, n] },
                        reduce_dim: Some(k),
                        batch: 1,
                        dtype,
                        sparsity: 0.0,
                        sweeps: None,
                        variables: None,
                    },
                    None => {
                        skipped_unresolved.push(label);
                        continue;
                    }
                }
            }
            "conv" => {
                match resolve_conv(node, &shapes) {
                    Some((m, k, n)) => OpProfile {
                        shape: TensorShape { dims: vec![m, n] },
                        reduce_dim: Some(k),
                        batch: 1,
                        dtype,
                        sparsity: 0.0,
                        sweeps: None,
                        variables: None,
                    },
                    None => {
                        skipped_unresolved.push(label);
                        continue;
                    }
                }
            }
            _ => {
                // Scalar / pointwise / reductive — pass through output shape,
                // no reduce_dim.
                let out_shape = node.output.first()
                    .and_then(|n| shapes.get(n))
                    .and_then(|d| coerce_dims(d))
                    .unwrap_or_else(|| vec![1]);
                OpProfile {
                    shape: TensorShape { dims: out_shape },
                    reduce_dim: None,
                    batch: 1,
                    dtype,
                    sparsity: 0.0,
                    sweeps: None,
                    variables: None,
                }
            }
        };

        invocations.push(Invocation {
            primitive_id: prim_id,
            count: 1,
            profile,
            label,
        });
        mapped_count += 1;
    }

    let workload = Workload {
        goal: format!(
            "ONNX model · {} ({} nodes → {} mapped)",
            if graph.name.is_empty() { "unnamed".to_string() } else { graph.name.clone() },
            node_count,
            mapped_count,
        ),
        invocations,
        constraints: Constraints {
            cycles_per_second: 30.0,
            power_budget_w: 5.0,
            volume: 10_000,
        },
    };

    Ok(ImportReport {
        workload,
        model_name: graph.name,
        producer: format!("{} {}", model.producer_name, model.producer_version).trim().to_string(),
        node_count,
        mapped_count,
        skipped_unresolved,
        skipped_unmapped,
    })
}

/// Parse an ONNX file at a path.
pub fn parse_onnx_file(path: &str) -> Result<ImportReport, ImportError> {
    let bytes = std::fs::read(path)?;
    parse_onnx(&bytes)
}
