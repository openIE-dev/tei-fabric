//! Dispatcher routing tests — each workload class lands on the substrate
//! whose physics owns it, with the full six-substrate default set.

use std::sync::Arc;
use tei_cost_surface::{default_substrates, dispatch};
use tei_ir::{Constraints, Dtype, Invocation, OpProfile, TensorShape, Workload};
use tei_stack::Stack;

fn load_stack() -> Arc<Stack> {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/stack.json");
    Stack::load_from_path(path).expect("catalog loads")
}

fn workload(invocations: Vec<Invocation>) -> Workload {
    Workload {
        goal: "test".into(),
        invocations,
        constraints: Constraints {
            cycles_per_second: 30.0,
            power_budget_w: 5.0,
            volume: 10_000,
        },
    }
}

fn matmul_inv() -> Invocation {
    Invocation {
        primitive_id: 18,
        count: 1,
        label: "matmul".into(),
        profile: OpProfile {
            shape: TensorShape {
                dims: vec![512, 2048],
            },
            reduce_dim: Some(768),
            batch: 1,
            dtype: Dtype::F16,
            sparsity: 0.0,
            sweeps: None,
            variables: None,
        },
    }
}

fn sampling_inv(primitive_id: u32) -> Invocation {
    Invocation {
        primitive_id,
        count: 1,
        label: "sampling".into(),
        profile: OpProfile {
            shape: TensorShape { dims: vec![1024] },
            reduce_dim: None,
            batch: 1,
            dtype: Dtype::I8,
            sparsity: 0.0,
            sweeps: Some(10_000),
            variables: Some(1024),
        },
    }
}

fn snn_inv(primitive_id: u32) -> Invocation {
    Invocation {
        primitive_id,
        count: 1,
        label: "snn".into(),
        profile: OpProfile {
            shape: TensorShape { dims: vec![4096] },
            reduce_dim: Some(128),
            batch: 1,
            dtype: Dtype::I8,
            sparsity: 0.9,
            sweeps: Some(1000),
            variables: Some(4096),
        },
    }
}

#[test]
fn six_substrates_registered() {
    let stack = load_stack();
    assert_eq!(default_substrates(stack).len(), 6);
}

#[test]
fn matmul_routes_to_in_memory() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let plan = dispatch(&stack, &workload(vec![matmul_inv()]), &subs);
    assert_eq!(plan.assignments[0].chosen_substrate, "in-memory");
    // Savings factor pinned: baseline 36.48 pJ/MAC vs crossbar ≈ 4.93 fJ/MAC.
    assert!(
        plan.savings_factor > 5_000.0 && plan.savings_factor < 10_000.0,
        "savings {} outside expected band",
        plan.savings_factor
    );
}

#[test]
fn mcmc_routes_to_stochastic() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let plan = dispatch(&stack, &workload(vec![sampling_inv(39)]), &subs);
    assert_eq!(plan.assignments[0].chosen_substrate, "stochastic");
}

#[test]
fn lif_routes_to_neuromorphic() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let plan = dispatch(&stack, &workload(vec![snn_inv(50)]), &subs);
    assert_eq!(plan.assignments[0].chosen_substrate, "neuromorphic");
}

#[test]
fn dct_routes_to_reversible() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let inv = Invocation {
        primitive_id: 79,
        count: 1,
        label: "dct".into(),
        profile: OpProfile {
            shape: TensorShape {
                dims: vec![1024, 1024],
            },
            reduce_dim: Some(1024),
            batch: 1,
            dtype: Dtype::F16,
            sparsity: 0.0,
            sweeps: None,
            variables: None,
        },
    };
    let plan = dispatch(&stack, &workload(vec![inv]), &subs);
    assert_eq!(plan.assignments[0].chosen_substrate, "reversible");
}

#[test]
fn unknown_primitive_skipped_quietly() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let mut inv = matmul_inv();
    inv.primitive_id = 99_999;
    let plan = dispatch(&stack, &workload(vec![inv]), &subs);
    assert!(plan.assignments.is_empty());
}

#[test]
fn mixed_workload_fans_out() {
    let stack = load_stack();
    let subs = default_substrates(stack.clone());
    let plan = dispatch(
        &stack,
        &workload(vec![matmul_inv(), sampling_inv(258), snn_inv(50)]),
        &subs,
    );
    let chosen: Vec<&str> = plan
        .assignments
        .iter()
        .map(|a| a.chosen_substrate.as_str())
        .collect();
    assert_eq!(chosen, vec!["in-memory", "stochastic", "neuromorphic"]);
    // Total joules = sum of per-assignment joules.
    let sum: f64 = plan.assignments.iter().map(|a| a.chosen_joules_total).sum();
    assert!((plan.total_joules - sum).abs() / sum < 1e-12);
}
