//! Catalog-driven workload preset enumeration.
//!
//! Scans the Periodic Stack catalog + registered substrates and emits a set
//! of "showcase" workloads — one per non-universal substrate, each populated
//! with the substrate's native primitives at sensible default shapes. This
//! is what makes a partner's first session interesting: drop in, see the
//! fabric route a workload they didn't have to hand-author.
//!
//! The shape heuristic is intentionally simple:
//!   - L₁ primitives (sampling) get `variables=1024, sweeps=10_000`.
//!   - Matmul-class L₂ primitives get `512 × 768 × 2048` at f16.
//!   - Convolution gets a representative ResNet-style stage.
//!   - Everything else falls through to a small dense matmul default.
//!
//! Substrates with more than `MAX_OPS_PER_PRESET` native primitives are
//! truncated; the order is the order they appear in the catalog (which is
//! curated for sensibility upstream).

use crate::Substrate;
use serde::Serialize;
use std::sync::Arc;
use tei_ir::{Constraints, Dtype, Invocation, OpProfile, TensorShape, Workload};
use tei_stack::{Primitive, Stack};

const MAX_OPS_PER_PRESET: usize = 6;

#[derive(Debug, Clone, Serialize)]
pub struct Preset {
    /// Stable key for the preset selector.
    pub key: String,
    /// Human-readable label for the dropdown.
    pub label: String,
    /// Group label so the UI can `<optgroup>` correctly.
    pub category: String,
    /// One-line description.
    pub description: String,
    /// The populated workload — ready to POST to `/api/dispatch`.
    pub workload: Workload,
}

fn default_constraints() -> Constraints {
    Constraints {
        cycles_per_second: 30.0,
        power_budget_w: 5.0,
        volume: 10_000,
    }
}

/// Sensible default OpProfile for a primitive, keyed off the catalog's
/// thermodynamic class + primitive ID.
fn default_profile_for(p: &Primitive) -> OpProfile {
    if p.is_l1() {
        // Sampling-class — Gibbs / MCMC / SA / Bayesian.
        OpProfile {
            shape: TensorShape { dims: vec![1024] },
            reduce_dim: None,
            batch: 1,
            dtype: Dtype::I8,
            sparsity: 0.0,
            sweeps: Some(10_000),
            variables: Some(1024),
        }
    } else if matches!(p.id, 24) {
        // Convolution — a ResNet conv stage default.
        OpProfile {
            shape: TensorShape {
                dims: vec![56 * 56, 64],
            },
            reduce_dim: Some(64 * 3 * 3),
            batch: 1,
            dtype: Dtype::F16,
            sparsity: 0.0,
            sweeps: None,
            variables: None,
        }
    } else {
        // Everything else: a transformer-style matmul.
        OpProfile {
            shape: TensorShape {
                dims: vec![512, 2048],
            },
            reduce_dim: Some(768),
            batch: 1,
            dtype: Dtype::F16,
            sparsity: 0.0,
            sweeps: None,
            variables: None,
        }
    }
}

fn build_invocation(p: &Primitive) -> Invocation {
    Invocation {
        primitive_id: p.id,
        count: 1,
        profile: default_profile_for(p),
        label: p.name.clone(),
    }
}

/// Build all catalog-driven showcase presets for the given substrate set.
pub fn enumerate_presets(stack: &Stack, substrates: &[Arc<dyn Substrate>]) -> Vec<Preset> {
    let mut out = Vec::new();

    for s in substrates {
        // Baseline is universal — every primitive supports() returns true, so
        // a "baseline showcase" would just be the whole catalog. Not useful.
        if s.name() == "baseline" {
            continue;
        }

        let native: Vec<&Primitive> = stack
            .primitives()
            .iter()
            .filter(|p| s.supports(p))
            .take(MAX_OPS_PER_PRESET)
            .collect();

        if native.is_empty() {
            continue;
        }

        let invocations: Vec<Invocation> = native.iter().map(|p| build_invocation(p)).collect();
        let n = invocations.len();
        let display = s.display_name();

        out.push(Preset {
            key: format!("showcase-{}", s.name()),
            label: format!("{} · {} native op{}", display, n, if n == 1 { "" } else { "s" }),
            category: "Substrate showcase".to_string(),
            description: format!(
                "Auto-generated from the Periodic Stack catalog: every primitive {} natively supports, at sensible default shapes.",
                display
            ),
            workload: Workload {
                goal: format!("{} showcase", display),
                invocations,
                constraints: default_constraints(),
            },
        });
    }

    // Family scans — pull the top primitives from a few high-signal families
    // so the user can stress the dispatcher across categories at once.
    out.extend(family_preset(stack, "LA", "Linear algebra family",
        "Every native LA primitive at matmul defaults. Tests photonic vs in-memory vs reversible head-to-head."));
    out.extend(family_preset(
        stack,
        "PROB",
        "Probabilistic family",
        "Sampling-class primitives. Routes everything to the stochastic substrate.",
    ));
    out.extend(family_preset(
        stack,
        "TR",
        "Transforms family",
        "Bijective transforms (FFT, wavelet, …). Where the reversible substrate is at its best.",
    ));

    out
}

fn family_preset(
    stack: &Stack,
    family_code: &str,
    label: &str,
    description: &str,
) -> Option<Preset> {
    let idxs = stack.family(family_code);
    if idxs.is_empty() {
        return None;
    }
    let invocations: Vec<Invocation> = idxs
        .iter()
        .filter_map(|&i| stack.primitives().get(i))
        .take(MAX_OPS_PER_PRESET)
        .map(|p| build_invocation(p))
        .collect();
    if invocations.is_empty() {
        return None;
    }
    let n = invocations.len();
    Some(Preset {
        key: format!("family-{}", family_code.to_lowercase()),
        label: format!("{} · {} primitives", label, n),
        category: "Family scan".to_string(),
        description: description.to_string(),
        workload: Workload {
            goal: label.to_string(),
            invocations,
            constraints: default_constraints(),
        },
    })
}
