//! Minimal forward shape propagation for ONNX graphs.
//!
//! Most PyTorch-exported ONNX models only carry shape info for `graph.input`,
//! `graph.output`, and initializers (weights). The dispatcher's cost surface
//! needs intermediate-tensor shapes to compute MAC counts for Conv / MatMul /
//! Gemm operators downstream of the input.
//!
//! This module walks the graph in node order and propagates shapes through
//! the common identity / convolution / pooling ops, enough to resolve the
//! shapes for a typical CNN or transformer.
//!
//! Out of scope: data-dependent ops like Reshape (depends on a constant
//! second input), Gather/Slice (depend on indices), Resize (depends on
//! attributes we don't parse), Loop / If (control flow). Those nodes
//! pass through without shape info; the dispatcher reports them as
//! `skipped_unresolved` and the caller can re-run ONNX shape inference
//! upstream or hand-fill the dims.

use crate::proto;
use std::collections::HashMap;

/// Attribute lookup by name on a node.
fn attr<'a>(node: &'a proto::NodeProto, name: &str) -> Option<&'a proto::AttributeProto> {
    node.attribute.iter().find(|a| a.name == name)
}
fn attr_ints(node: &proto::NodeProto, name: &str) -> Vec<i64> {
    attr(node, name).map(|a| a.ints.clone()).unwrap_or_default()
}
fn attr_int(node: &proto::NodeProto, name: &str) -> Option<i64> {
    attr(node, name).map(|a| a.i)
}

/// Get a known shape for a tensor name.
fn get<'a>(shapes: &'a HashMap<String, Vec<i64>>, name: &str) -> Option<&'a Vec<i64>> {
    shapes.get(name)
}

/// Compute the output shape of a Conv node (NCHW or NCDHW).
fn conv_out(node: &proto::NodeProto, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    let x = get(shapes, &node.input[0])?;
    let w = get(shapes, &node.input[1])?;
    if x.len() < 3 || w.len() < 3 { return None; }
    let rank = x.len() - 2; // spatial dims
    let n = x[0];
    let m = w[0];
    let kernel: Vec<i64> = if !attr_ints(node, "kernel_shape").is_empty() {
        attr_ints(node, "kernel_shape")
    } else {
        w[2..].to_vec()
    };
    let strides: Vec<i64> = {
        let s = attr_ints(node, "strides");
        if s.is_empty() { vec![1; rank] } else { s }
    };
    let dilations: Vec<i64> = {
        let d = attr_ints(node, "dilations");
        if d.is_empty() { vec![1; rank] } else { d }
    };
    let pads: Vec<i64> = {
        let p = attr_ints(node, "pads");
        if p.is_empty() { vec![0; 2 * rank] } else { p }
    };
    if kernel.len() != rank || strides.len() != rank || pads.len() != 2 * rank {
        return None;
    }
    let mut out = vec![n, m];
    for i in 0..rank {
        let dim_in = x[i + 2];
        if dim_in <= 0 { return None; }
        let k = kernel[i];
        let s = strides[i];
        let d = dilations.get(i).copied().unwrap_or(1);
        let pad = pads[i] + pads[i + rank];
        // ONNX formula: floor((dim_in + pad - dilation*(k-1) - 1) / stride + 1)
        let numer = dim_in + pad - d * (k - 1) - 1;
        let dim_out = numer / s + 1;
        if dim_out <= 0 { return None; }
        out.push(dim_out);
    }
    Some(out)
}

/// Pool ops (Max/Average): NCHW → NC × pooled.
fn pool_out(node: &proto::NodeProto, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    let x = get(shapes, &node.input[0])?;
    if x.len() < 3 { return None; }
    let rank = x.len() - 2;
    let kernel = attr_ints(node, "kernel_shape");
    if kernel.is_empty() { return None; }
    let strides: Vec<i64> = {
        let s = attr_ints(node, "strides");
        if s.is_empty() { vec![1; rank] } else { s }
    };
    let pads: Vec<i64> = {
        let p = attr_ints(node, "pads");
        if p.is_empty() { vec![0; 2 * rank] } else { p }
    };
    let mut out = vec![x[0], x[1]];
    for i in 0..rank {
        let dim_in = x[i + 2];
        if dim_in <= 0 { return None; }
        let k = kernel[i];
        let s = strides[i];
        let pad = pads.get(i).copied().unwrap_or(0) + pads.get(i + rank).copied().unwrap_or(0);
        let dim_out = (dim_in + pad - k) / s + 1;
        if dim_out <= 0 { return None; }
        out.push(dim_out);
    }
    Some(out)
}

/// MatMul: contract last axis of A with second-to-last of B.
fn matmul_out(node: &proto::NodeProto, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    if node.input.len() < 2 { return None; }
    let a = get(shapes, &node.input[0])?;
    let b = get(shapes, &node.input[1])?;
    if a.len() < 2 || b.len() < 2 { return None; }
    let mut out: Vec<i64> = Vec::new();
    // Broadcast batch dims (everything except the last 2 of each).
    let a_batch = &a[..a.len() - 2];
    let b_batch = &b[..b.len() - 2];
    let n_batch = a_batch.len().max(b_batch.len());
    for i in 0..n_batch {
        let av = a_batch.get(a_batch.len().wrapping_sub(n_batch - i)).copied().unwrap_or(1);
        let bv = b_batch.get(b_batch.len().wrapping_sub(n_batch - i)).copied().unwrap_or(1);
        out.push(av.max(bv));
    }
    out.push(a[a.len() - 2]);
    out.push(b[b.len() - 1]);
    Some(out)
}

/// Gemm: Y = α A B + β C, where A and B may be transposed.
fn gemm_out(node: &proto::NodeProto, shapes: &HashMap<String, Vec<i64>>) -> Option<Vec<i64>> {
    let a = get(shapes, &node.input[0])?;
    let b = get(shapes, &node.input[1])?;
    if a.len() != 2 || b.len() != 2 { return None; }
    let trans_a = attr_int(node, "transA").unwrap_or(0) != 0;
    let trans_b = attr_int(node, "transB").unwrap_or(0) != 0;
    let m = if trans_a { a[1] } else { a[0] };
    let n = if trans_b { b[0] } else { b[1] };
    Some(vec![m, n])
}

/// One pass through the graph in node order, filling in missing output
/// shapes from the rules above. Identity ops pass the first input shape
/// through. Unknown ops are left absent (the dispatcher reports them).
pub fn propagate(g: &proto::GraphProto, shapes: &mut HashMap<String, Vec<i64>>) {
    for node in &g.node {
        if node.output.is_empty() || node.output[0].is_empty() { continue; }
        if shapes.contains_key(&node.output[0]) { continue; }

        let computed: Option<Vec<i64>> = match node.op_type.as_str() {
            "Conv" | "ConvInteger" | "QLinearConv" | "ConvTranspose" => conv_out(node, shapes),
            "MaxPool" | "AveragePool" => pool_out(node, shapes),
            "GlobalMaxPool" | "GlobalAveragePool" => {
                get(shapes, &node.input[0]).map(|x| {
                    let mut out = vec![x[0], x[1]];
                    for _ in 2..x.len() { out.push(1); }
                    out
                })
            }
            "MatMul" | "MatMulInteger" | "QLinearMatMul" => matmul_out(node, shapes),
            "Gemm" => gemm_out(node, shapes),
            // Identity-shape ops — pass first input through.
            "BatchNormalization" | "LayerNormalization" | "GroupNormalization"
            | "InstanceNormalization" | "RMSNormalization"
            | "Relu" | "Sigmoid" | "Tanh" | "Erf" | "Gelu" | "Selu" | "Elu" | "LeakyRelu"
            | "HardSigmoid" | "HardSwish" | "Softplus" | "Softsign" | "Mish"
            | "Clip" | "Exp" | "Log" | "Neg" | "Reciprocal" | "Sqrt" | "Abs" | "Floor"
            | "Ceil" | "Round" | "Cast" | "Identity" | "Dropout" | "Softmax" | "LogSoftmax"
            => get(shapes, &node.input[0]).cloned(),
            // Elementwise binary — assume broadcast to first input (handles
            // residual Add, mul-by-scale, etc.; doesn't model broadcasting).
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Where" | "Equal"
                if !node.input.is_empty() => get(shapes, &node.input[0]).cloned(),
            "Flatten" => {
                get(shapes, &node.input[0]).map(|x| {
                    let axis = attr_int(node, "axis").unwrap_or(1) as usize;
                    let pre: i64 = x[..axis.min(x.len())].iter().product();
                    let post: i64 = x[axis.min(x.len())..].iter().product();
                    vec![pre.max(1), post.max(1)]
                })
            }
            _ => None,
        };

        if let Some(shape) = computed {
            shapes.insert(node.output[0].clone(), shape);
        }
    }
}
