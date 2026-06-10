//! End-to-end importer test on a synthetic ONNX model built in code —
//! no model binaries in the repo. Exercises Conv shape inference (pads),
//! identity passthrough (Softmax), Reshape with a constant target, and
//! MatMul (m, k, n) resolution.

use prost::Message;
use tei_import::proto;

/// i64 vector → ONNX INT64 TensorProto raw_data (little-endian).
fn i64_tensor(name: &str, dims: Vec<i64>, values: &[i64]) -> proto::TensorProto {
    let mut raw = Vec::with_capacity(values.len() * 8);
    for v in values {
        raw.extend_from_slice(&v.to_le_bytes());
    }
    proto::TensorProto {
        dims,
        data_type: 7, // INT64
        name: name.to_string(),
        raw_data: raw,
    }
}

/// Weight tensor — only dims matter for shape inference.
fn weight(name: &str, dims: Vec<i64>) -> proto::TensorProto {
    proto::TensorProto {
        dims,
        data_type: 1, // FLOAT
        name: name.to_string(),
        raw_data: vec![],
    }
}

fn node(
    op: &str,
    name: &str,
    inputs: &[&str],
    outputs: &[&str],
    attrs: Vec<proto::AttributeProto>,
) -> proto::NodeProto {
    proto::NodeProto {
        input: inputs.iter().map(|s| s.to_string()).collect(),
        output: outputs.iter().map(|s| s.to_string()).collect(),
        name: name.to_string(),
        op_type: op.to_string(),
        attribute: attrs,
        doc_string: String::new(),
        domain: String::new(),
    }
}

fn ints_attr(name: &str, ints: Vec<i64>) -> proto::AttributeProto {
    proto::AttributeProto {
        name: name.to_string(),
        ints,
        r#type: 7, // INTS
        ..Default::default()
    }
}

fn tensor_input(name: &str, dims: &[i64]) -> proto::ValueInfoProto {
    proto::ValueInfoProto {
        name: name.to_string(),
        r#type: Some(proto::TypeProto {
            tensor_type: Some(proto::type_proto::Tensor {
                elem_type: 1, // FLOAT
                shape: Some(proto::TensorShapeProto {
                    dim: dims
                        .iter()
                        .map(|&d| proto::tensor_shape_proto::Dimension {
                            dim_value: d,
                            dim_param: String::new(),
                        })
                        .collect(),
                }),
            }),
        }),
        doc_string: String::new(),
    }
}

/// x[1,3,8,8] → Conv(w[4,3,3,3], pads=1) → Softmax → Reshape([1,256]) → MatMul(w2[256,10]) → y
fn build_model() -> Vec<u8> {
    let graph = proto::GraphProto {
        node: vec![
            node(
                "Conv",
                "conv0",
                &["x", "w"],
                &["c"],
                vec![
                    ints_attr("pads", vec![1, 1, 1, 1]),
                    ints_attr("strides", vec![1, 1]),
                    ints_attr("kernel_shape", vec![3, 3]),
                ],
            ),
            node("Softmax", "softmax0", &["c"], &["s"], vec![]),
            node(
                "Reshape",
                "reshape0",
                &["s", "target_shape"],
                &["r"],
                vec![],
            ),
            node("MatMul", "matmul0", &["r", "w2"], &["y"], vec![]),
        ],
        name: "tiny-test-graph".to_string(),
        initializer: vec![
            weight("w", vec![4, 3, 3, 3]),
            weight("w2", vec![256, 10]),
            i64_tensor("target_shape", vec![2], &[1, 256]),
        ],
        doc_string: String::new(),
        input: vec![tensor_input("x", &[1, 3, 8, 8])],
        output: vec![tensor_input("y", &[1, 10])],
        value_info: vec![],
    };
    let model = proto::ModelProto {
        ir_version: 8,
        producer_name: "tei-fabric-test".to_string(),
        producer_version: "0".to_string(),
        domain: String::new(),
        model_version: 1,
        doc_string: String::new(),
        graph: Some(graph),
        opset_import: vec![],
    };
    model.encode_to_vec()
}

#[test]
fn synthetic_model_imports_fully() {
    let report = tei_import::parse_onnx(&build_model()).expect("parses");
    assert_eq!(report.node_count, 4);
    // Conv + Softmax + MatMul map; Reshape is deliberately unmapped dataflow.
    assert_eq!(
        report.mapped_count, 3,
        "unresolved: {:?}",
        report.skipped_unresolved
    );
    assert!(report.skipped_unresolved.is_empty());
    assert_eq!(report.skipped_unmapped.get("Reshape"), Some(&1));

    let inv = &report.workload.invocations;
    // Conv: m = N × spatial = 1×64, k = C_in × kernel = 27, n = C_out = 4.
    assert_eq!(inv[0].primitive_id, 24);
    assert_eq!(inv[0].profile.shape.dims, vec![64, 4]);
    assert_eq!(inv[0].profile.reduce_dim, Some(27));
    // Softmax: identity shape from Conv output [1,4,8,8].
    assert_eq!(inv[1].primitive_id, 34);
    // MatMul through the Reshape: [1,256] × [256,10] → m=1, k=256, n=10.
    assert_eq!(inv[2].primitive_id, 18);
    assert_eq!(inv[2].profile.shape.dims, vec![1, 10]);
    assert_eq!(inv[2].profile.reduce_dim, Some(256));
}

#[test]
fn truncated_bytes_fail_cleanly() {
    let bytes = build_model();
    // Protobuf is resilient to suffix truncation in some cases, but a
    // mid-stream cut must either error or produce a no-graph error — never panic.
    let _ = tei_import::parse_onnx(&bytes[..bytes.len() / 3]);
}
