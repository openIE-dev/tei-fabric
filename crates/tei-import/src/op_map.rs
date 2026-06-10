//! ONNX op-type → Periodic Stack primitive ID map.
//!
//! Covers the operators that make up >95% of the compute energy in a
//! typical transformer or CNN. The remaining ONNX op_types — elementwise
//! activations, reshapes, transposes, broadcasts — are not energy hotspots
//! and are skipped in v0 (the importer emits no invocation for them).
//!
//! ONNX reference: github.com/onnx/onnx/blob/main/docs/Operators.md

/// Map an ONNX op_type to a (primitive_id, kind) pair.
///
/// `kind` is one of:
///   - `"matmul"`   — needs `(m, k, n)` shape resolution
///   - `"conv"`     — convolution-specific shape mapping
///   - `"scalar"`   — pointwise / reductive op, no MAC scaling
///   - `"sampling"` — L₁ sampling primitives (rare in ONNX)
///
/// Returns `None` for ops we deliberately ignore (Add, Reshape, Transpose, …).
pub fn map_op(op_type: &str) -> Option<(u32, &'static str)> {
    match op_type {
        // ── Linear-algebra core (matmul-class) ───────────────────────
        "MatMul" => Some((18, "matmul")),
        "Gemm" => Some((18, "matmul")),
        "MatMulInteger" => Some((18, "matmul")),
        "QLinearMatMul" => Some((18, "matmul")),
        "FusedMatMul" => Some((18, "matmul")),

        // ── Convolution ──────────────────────────────────────────────
        "Conv" => Some((24, "conv")),
        "ConvInteger" => Some((24, "conv")),
        "QLinearConv" => Some((24, "conv")),
        "ConvTranspose" => Some((24, "conv")),
        "FusedConv" => Some((24, "conv")), // ORT-contrib fused Conv + bias + activation

        // ── Linear transforms ────────────────────────────────────────
        "DFT" => Some((23, "scalar")),
        "STFT" => Some((23, "scalar")),

        // ── Reductive / non-linear blocks ────────────────────────────
        "Softmax" => Some((34, "scalar")),
        "LogSoftmax" => Some((34, "scalar")),
        "LayerNormalization" => Some((35, "scalar")),
        "BatchNormalization" => Some((35, "scalar")),
        "GroupNormalization" => Some((35, "scalar")),
        "GroupNorm" => Some((35, "scalar")), // ORT contrib alias
        "InstanceNormalization" => Some((35, "scalar")),
        "RMSNormalization" => Some((35, "scalar")),
        // ORT-contrib fused-norm ops — wrap a residual + layernorm in one op.
        "SkipLayerNormalization" => Some((35, "scalar")),
        "SkipSimplifiedLayerNormalization" => Some((35, "scalar")),
        "SimplifiedLayerNormalization" => Some((35, "scalar")),
        // ORT-contrib fused attention / GQA / MHA.
        // Fused attention takes Q/K/V as three separate inputs (not the
        // standard MatMul [A,B] pair), so it needs a different shape +
        // MAC-count resolver — kind "mha".
        "MultiHeadAttention" => Some((20, "mha")),
        "GroupQueryAttention" => Some((20, "mha")),
        "QAttention" => Some((20, "mha")),
        "FusedAttention" => Some((20, "mha")),
        "Attention" => Some((20, "mha")),

        // ── Pooling ──────────────────────────────────────────────────
        "MaxPool" => Some((26, "scalar")),
        "AveragePool" => Some((26, "scalar")),
        "GlobalMaxPool" => Some((26, "scalar")),
        "GlobalAveragePool" => Some((26, "scalar")),

        // ── Sort / hash (rare in ONNX) ───────────────────────────────
        "TopK" => Some((6, "scalar")),
        "ArgMax" => Some((6, "scalar")),
        "ArgMin" => Some((6, "scalar")),

        // Everything else: elementwise, reshape, dataflow, control —
        // not energy-relevant at our resolution. Skip.
        _ => None,
    }
}
