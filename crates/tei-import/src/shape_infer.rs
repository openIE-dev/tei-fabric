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
fn attr_tensor<'a>(node: &'a proto::NodeProto, name: &str) -> Option<&'a proto::TensorProto> {
    attr(node, name).and_then(|a| a.t.as_ref())
}

/// Decode the int64 payload of a TensorProto. Many ONNX models store small
/// integer tensors as packed `raw_data` (little-endian); newer producers
/// also populate `int64_data` directly, but our minimal proto omits that
/// field so we only read raw_data.
fn tensor_as_i64s(t: &proto::TensorProto) -> Option<Vec<i64>> {
    if !t.raw_data.is_empty() {
        // data_type 7 = INT64, 6 = INT32, 1 = FLOAT, 2 = UINT8 …
        match t.data_type {
            7 => {
                // 8 bytes per element, little-endian.
                let mut out = Vec::with_capacity(t.raw_data.len() / 8);
                for chunk in t.raw_data.chunks_exact(8) {
                    let mut b = [0u8; 8];
                    b.copy_from_slice(chunk);
                    out.push(i64::from_le_bytes(b));
                }
                Some(out)
            }
            6 => {
                let mut out = Vec::with_capacity(t.raw_data.len() / 4);
                for chunk in t.raw_data.chunks_exact(4) {
                    let mut b = [0u8; 4];
                    b.copy_from_slice(chunk);
                    out.push(i32::from_le_bytes(b) as i64);
                }
                Some(out)
            }
            _ => None,
        }
    } else {
        // No raw_data — payload may be in a typed repeated field we don't
        // declare. For our purposes (Reshape target shape, Gather indices,
        // Slice starts/ends), the producers we see in practice use raw_data.
        None
    }
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

/// Materialize a constant-tensor map keyed by tensor name.
///
/// Sources, in order of precedence:
///   1. Graph initializers (model weights and baked-in constants).
///   2. `Constant` op outputs — the value lives in the node's `value` attribute.
///
/// Constants store rank-1 integer vectors that downstream Reshape / Slice /
/// Gather / Squeeze / Unsqueeze nodes use to reshape activations. We only
/// decode int32 / int64 constants; float constants don't affect shape.
fn build_constants(g: &proto::GraphProto) -> HashMap<String, Vec<i64>> {
    let mut consts: HashMap<String, Vec<i64>> = HashMap::new();
    for t in &g.initializer {
        if t.name.is_empty() { continue; }
        if let Some(v) = tensor_as_i64s(t) {
            consts.insert(t.name.clone(), v);
        }
    }
    for node in &g.node {
        if node.op_type != "Constant" { continue; }
        if node.output.is_empty() { continue; }
        if let Some(t) = attr_tensor(node, "value") {
            if let Some(v) = tensor_as_i64s(t) {
                consts.insert(node.output[0].clone(), v);
                continue;
            }
            // Float constants still get their dims recorded.
            if !t.dims.is_empty() {
                consts.insert(node.output[0].clone(), t.dims.clone());
            }
        }
    }
    consts
}

/// Reshape: output shape is the second input, interpreted as a rank-1 i64
/// vector. ONNX allows one -1 dimension (infer) and one 0 dimension (keep);
/// resolve them against the first input's shape + element count.
fn reshape_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
    consts: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    if node.input.len() < 2 { return None; }
    let input_shape = shapes.get(&node.input[0])?;
    let target = consts.get(&node.input[1])?;
    if target.is_empty() { return None; }

    let mut out: Vec<i64> = target.iter().copied().collect();

    // Resolve `0` (keep) — replace with the same axis of the input.
    let allow_zero = attr_int(node, "allowzero").unwrap_or(0) != 0;
    if !allow_zero {
        for (i, d) in out.iter_mut().enumerate() {
            if *d == 0 {
                if let Some(&keep) = input_shape.get(i) { *d = keep; }
            }
        }
    }

    // Resolve `-1` (infer).
    let total_in: i64 = input_shape.iter().filter(|&&d| d > 0).product();
    let neg_idx = out.iter().position(|&d| d == -1);
    if let Some(i) = neg_idx {
        let rest: i64 = out.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, &d)| d.max(1))
            .product();
        if rest > 0 {
            out[i] = total_in / rest;
        }
    }
    Some(out)
}

/// Transpose: permute the input's dims according to `perm` (or reverse if
/// no `perm` attribute).
fn transpose_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let x = shapes.get(&node.input[0])?.clone();
    let perm = attr_ints(node, "perm");
    if perm.is_empty() {
        let mut r = x.clone();
        r.reverse();
        Some(r)
    } else if perm.len() == x.len() {
        Some(perm.iter().map(|&p| x[p as usize]).collect())
    } else {
        None
    }
}

/// Squeeze: drop axes where dim == 1. Axes come either from the `axes`
/// attribute (older opsets) or the second input (opset 13+, constant).
fn squeeze_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
    consts: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let x = shapes.get(&node.input[0])?.clone();
    let axes_raw: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
        consts.get(&node.input[1]).cloned().unwrap_or_default()
    } else {
        attr_ints(node, "axes")
    };
    let rank = x.len() as i64;
    let axes: Vec<usize> = axes_raw.iter().map(|&a| {
        (if a < 0 { a + rank } else { a }) as usize
    }).collect();
    let out: Vec<i64> = if axes.is_empty() {
        x.into_iter().filter(|&d| d != 1).collect()
    } else {
        x.into_iter().enumerate()
            .filter(|(i, _)| !axes.contains(i))
            .map(|(_, d)| d).collect()
    };
    Some(out)
}

/// Unsqueeze: insert 1-dims at the given axes.
fn unsqueeze_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
    consts: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let x = shapes.get(&node.input[0])?.clone();
    let axes_raw: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
        consts.get(&node.input[1]).cloned().unwrap_or_default()
    } else {
        attr_ints(node, "axes")
    };
    let target_rank = (x.len() + axes_raw.len()) as i64;
    let mut axes: Vec<usize> = axes_raw.iter().map(|&a| {
        (if a < 0 { a + target_rank } else { a }) as usize
    }).collect();
    axes.sort();
    let mut out = x;
    for a in axes {
        if a <= out.len() {
            out.insert(a, 1);
        }
    }
    Some(out)
}

/// Gather along an axis. Output shape is:
///   input.shape[..axis] + indices.shape + input.shape[axis+1..]
/// We don't know `indices.shape` unless it's a known tensor.
fn gather_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let x = shapes.get(&node.input[0])?.clone();
    let idx = shapes.get(&node.input[1])?.clone();
    let axis = attr_int(node, "axis").unwrap_or(0);
    let rank = x.len() as i64;
    let axis = (if axis < 0 { axis + rank } else { axis }) as usize;
    if axis >= x.len() { return None; }
    let mut out: Vec<i64> = Vec::with_capacity(x.len() + idx.len() - 1);
    out.extend_from_slice(&x[..axis]);
    out.extend_from_slice(&idx);
    out.extend_from_slice(&x[axis + 1..]);
    Some(out)
}

/// Slice: read starts/ends/axes/steps from inputs (opset 10+) or attrs
/// (opset 1). Compute output dims along the sliced axes.
fn slice_out(
    node: &proto::NodeProto,
    shapes: &HashMap<String, Vec<i64>>,
    consts: &HashMap<String, Vec<i64>>,
) -> Option<Vec<i64>> {
    let x = shapes.get(&node.input[0])?.clone();
    let rank = x.len();

    // Try opset-10+ input form first.
    let starts: Vec<i64>;
    let ends: Vec<i64>;
    let axes: Vec<i64>;
    let steps: Vec<i64>;
    if node.input.len() >= 3 {
        starts = consts.get(&node.input[1]).cloned()?;
        ends = consts.get(&node.input[2]).cloned()?;
        axes = if node.input.len() >= 4 {
            consts.get(&node.input[3]).cloned().unwrap_or((0..rank as i64).collect())
        } else { (0..rank as i64).collect() };
        steps = if node.input.len() >= 5 {
            consts.get(&node.input[4]).cloned().unwrap_or(vec![1; starts.len()])
        } else { vec![1; starts.len()] };
    } else {
        starts = attr_ints(node, "starts");
        ends = attr_ints(node, "ends");
        axes = if attr_ints(node, "axes").is_empty() {
            (0..starts.len() as i64).collect()
        } else { attr_ints(node, "axes") };
        steps = vec![1; starts.len()];
    }
    if starts.len() != ends.len() || starts.len() != axes.len() { return None; }

    let mut out = x.clone();
    for i in 0..starts.len() {
        let mut ax = axes[i];
        if ax < 0 { ax += rank as i64; }
        let ax = ax as usize;
        if ax >= rank { return None; }
        let dim = out[ax];
        let mut s = starts[i];
        let mut e = ends[i];
        if s < 0 { s += dim; }
        if e < 0 { e += dim; }
        s = s.clamp(0, dim);
        e = e.clamp(0, dim);
        let step = steps[i].max(1);
        out[ax] = (e - s + step - 1) / step;
    }
    Some(out)
}

/// Fixed-point shape propagation.
///
/// Repeatedly walks the graph in node order, filling in missing output
/// shapes from the rules below. Real-world ONNX graphs are usually
/// topological but converters occasionally produce out-of-order nodes
/// (e.g. tf2onnx). We iterate until no new shapes resolve in a pass,
/// up to a cap so a malformed graph can't loop forever.
pub fn propagate(g: &proto::GraphProto, shapes: &mut HashMap<String, Vec<i64>>) {
    let consts = build_constants(g);
    let max_iters = 8;
    for _ in 0..max_iters {
        let before = shapes.len();
        propagate_pass(g, shapes, &consts);
        if shapes.len() == before { break; }
    }
}

fn propagate_pass(
    g: &proto::GraphProto,
    shapes: &mut HashMap<String, Vec<i64>>,
    consts: &HashMap<String, Vec<i64>>,
) {
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
            "MatMul" | "MatMulInteger" | "QLinearMatMul" | "FusedMatMul" => matmul_out(node, shapes),
            "Gemm" => gemm_out(node, shapes),
            // Identity-shape ops — pass first input through.
            "BatchNormalization" | "LayerNormalization" | "GroupNormalization"
            | "InstanceNormalization" | "RMSNormalization"
            | "Relu" | "Sigmoid" | "Tanh" | "Erf" | "Gelu" | "Selu" | "Elu" | "LeakyRelu"
            | "HardSigmoid" | "HardSwish" | "Softplus" | "Softsign" | "Mish"
            | "Clip" | "Exp" | "Log" | "Neg" | "Reciprocal" | "Sqrt" | "Abs" | "Floor"
            | "Ceil" | "Round" | "Cast" | "Identity" | "Dropout" | "Softmax" | "LogSoftmax"
            => get(shapes, &node.input[0]).cloned(),
            // Elementwise binary — pick the higher-rank input shape so
            // broadcasts like `[1] + [B, S, H]` resolve to `[B, S, H]` instead
            // of `[1]`. Doesn't simulate full numpy broadcasting; covers the
            // common case where one operand is a scalar/vector.
            "Add" | "Sub" | "Mul" | "Div" | "Pow" | "Min" | "Max" | "Where" | "Equal"
                if !node.input.is_empty() => {
                    let a = node.input.first().and_then(|n| get(shapes, n)).cloned();
                    let b = node.input.get(1).and_then(|n| get(shapes, n)).cloned();
                    match (a, b) {
                        (Some(a), Some(b)) if b.len() > a.len() => Some(b),
                        (Some(a), _) => Some(a),
                        (None, Some(b)) => Some(b),
                        (None, None) => None,
                    }
                }
            "Flatten" => {
                get(shapes, &node.input[0]).map(|x| {
                    let axis = attr_int(node, "axis").unwrap_or(1) as usize;
                    let pre: i64 = x[..axis.min(x.len())].iter().product();
                    let post: i64 = x[axis.min(x.len())..].iter().product();
                    vec![pre.max(1), post.max(1)]
                })
            }
            "Reshape"   => reshape_out(node, shapes, &consts),
            "Transpose" => transpose_out(node, shapes),
            "Squeeze"   => squeeze_out(node, shapes, &consts),
            "Unsqueeze" => unsqueeze_out(node, shapes, &consts),
            "Gather"    => gather_out(node, shapes),
            "Slice"     => slice_out(node, shapes, &consts),
            // Reductive ops. Output = input shape with reduced axes either
            // removed or set to 1 (when keepdims=1, the ONNX default).
            "ReduceMean" | "ReduceSum" | "ReduceMax" | "ReduceMin"
            | "ReduceProd" | "ReduceL1" | "ReduceL2" | "ReduceLogSum"
            | "ReduceLogSumExp" | "ReduceSumSquare" => {
                let x = get(shapes, &node.input[0]).cloned();
                x.map(|mut s| {
                    let keepdims = attr_int(node, "keepdims").unwrap_or(1) != 0;
                    let axes_raw: Vec<i64> = if node.input.len() >= 2 && !node.input[1].is_empty() {
                        consts.get(&node.input[1]).cloned().unwrap_or_default()
                    } else {
                        attr_ints(node, "axes")
                    };
                    let rank = s.len() as i64;
                    let axes: Vec<usize> = axes_raw.iter().map(|&a| {
                        (if a < 0 { a + rank } else { a }) as usize
                    }).collect();
                    if axes.is_empty() {
                        // Reduce over all axes — output is a scalar (or [1,…,1] with keepdims).
                        if keepdims { vec![1; s.len()] } else { vec![] }
                    } else if keepdims {
                        for &a in &axes { if a < s.len() { s[a] = 1; } }
                        s
                    } else {
                        s.into_iter().enumerate()
                         .filter(|(i, _)| !axes.contains(i))
                         .map(|(_, d)| d).collect()
                    }
                })
            }
            // Quantization ops — identity shape for the main output.
            // DynamicQuantizeLinear produces three outputs (quantized,
            // scale, zero_point); the first is identity-shape.
            "DynamicQuantizeLinear" | "QuantizeLinear" | "DequantizeLinear" => {
                get(shapes, &node.input[0]).cloned()
            }
            "Expand" => {
                // Output is the second input (a shape tensor).
                if node.input.len() >= 2 {
                    consts.get(&node.input[1]).cloned()
                } else { None }
            }
            "ConstantOfShape" => {
                // Output shape is the first input (a shape tensor).
                consts.get(&node.input[0]).cloned()
            }
            "Concat" => {
                // Concat along an axis; we don't have all input shapes
                // reliably so we fall back to the first input's shape.
                get(shapes, &node.input[0]).cloned()
            }
            "Shape" => {
                // The Shape op outputs the input's shape as a 1-D tensor.
                // Record that as a constant for downstream Reshape/Gather.
                if let Some(input_shape) = get(shapes, &node.input[0]).cloned() {
                    let r = input_shape.len() as i64;
                    shapes.insert(node.output[0].clone(), vec![r]);
                    // Won't be entered into shapes again below; skip.
                }
                None
            }
            _ => None,
        };

        if let Some(shape) = computed {
            shapes.insert(node.output[0].clone(), shape);
        }
    }
}
