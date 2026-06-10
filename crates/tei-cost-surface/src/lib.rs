//! Cost-surface dispatcher.
//!
//! Given a workload (a list of primitive invocations with op profiles) and a
//! set of registered substrates, produce a `DispatchPlan` that, for each
//! invocation, picks the substrate with the lowest joule cost it supports.
//!
//! v0 picks per-invocation independently — no cross-invocation optimization,
//! no fusion, no scheduling. The dispatcher reports:
//!   - the assignment per invocation,
//!   - the assignment's total joules and seconds,
//!   - the baseline-only joules and seconds for comparison (i.e. what it
//!     would cost to run everything on the CPU/GPU baseline),
//!   - the per-substrate aggregate energy.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tei_d_in_memory::InMemoryParams;
use tei_d_neuromorphic::NeuromorphicParams;
use tei_d_photonic::PhotonicParams;
use tei_ir::{Constraints, Invocation, Workload};
use tei_stack::Stack;
use tei_substrate_traits::{Cost, Substrate};

pub mod presets;
pub use presets::{Preset, enumerate_presets};

/// Engineering parameters for substrates that expose tunable knobs.
/// Optional — missing fields fall back to dialect defaults.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct SubstrateParams {
    #[serde(default)]
    pub photonic: PhotonicParams,
    #[serde(default)]
    pub in_memory: InMemoryParams,
    #[serde(default)]
    pub neuromorphic: NeuromorphicParams,
}

/// Build the default substrate set with custom engineering parameters for
/// the dialects that expose them. The non-parameterized dialects (baseline,
/// stochastic, reversible) construct with their literature defaults.
pub fn substrates_with_params(
    stack: Arc<Stack>,
    params: &SubstrateParams,
) -> Vec<Arc<dyn Substrate>> {
    vec![
        Arc::new(tei_d_baseline::Baseline) as Arc<dyn Substrate>,
        Arc::new(tei_d_photonic::Photonic::with_params(
            params.photonic.clone(),
        )) as Arc<dyn Substrate>,
        Arc::new(tei_d_in_memory::InMemory::with_params(
            params.in_memory.clone(),
        )) as Arc<dyn Substrate>,
        Arc::new(tei_d_stochastic::Stochastic) as Arc<dyn Substrate>,
        Arc::new(tei_d_reversible::Reversible::new(stack.clone())) as Arc<dyn Substrate>,
        Arc::new(tei_d_neuromorphic::Neuromorphic::with_params(
            stack,
            params.neuromorphic.clone(),
        )) as Arc<dyn Substrate>,
    ]
}

/// One invocation's resolved assignment.
#[derive(Debug, Clone, Serialize)]
pub struct Assignment {
    pub primitive_id: u32,
    pub primitive_name: String,
    pub family: String,
    pub thermo: String,
    pub count: u64,
    pub label: String,

    pub chosen_substrate: String,
    pub chosen_display: String,
    pub chosen_cost: Cost,

    /// Joules + seconds totals for the chosen substrate at `count` invocations.
    pub chosen_joules_total: f64,
    pub chosen_seconds_total: f64,

    /// What the same op would cost on the baseline substrate (always reported
    /// for comparison even when baseline isn't the chosen one).
    pub baseline_cost: Cost,
    pub baseline_joules_total: f64,

    /// Per-substrate cost table the dispatcher considered.
    pub considered: Vec<SubstrateOption>,

    /// Short justification line.
    pub justification: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubstrateOption {
    pub substrate: String,
    pub display: String,
    pub supported: bool,
    pub cost: Option<Cost>,
    pub joules_total: Option<f64>,
}

/// The full plan.
#[derive(Debug, Clone, Serialize)]
pub struct DispatchPlan {
    pub workload_goal: String,
    pub assignments: Vec<Assignment>,

    pub total_joules: f64,
    pub baseline_total_joules: f64,
    pub savings_factor: f64,

    pub total_seconds: f64,
    pub baseline_total_seconds: f64,

    pub per_substrate_joules: Vec<SubstrateAggregate>,
    pub power_w: f64,
    pub baseline_power_w: f64,
    pub power_budget_w: f64,
    pub within_budget: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct SubstrateAggregate {
    pub substrate: String,
    pub display: String,
    pub joules: f64,
    pub op_count: u64,
}

/// Per-invocation cost-surface evaluation. Returns the chosen substrate
/// and the full considered-set, or None if the invocation references an
/// unknown primitive ID. Pure function — no aggregator state. Used by
/// both the batch `dispatch()` and the streaming SSE handler.
pub fn dispatch_invocation(
    stack: &Stack,
    inv: &Invocation,
    substrates: &[Arc<dyn Substrate>],
) -> Option<Assignment> {
    let p = stack.get(inv.primitive_id)?;
    let baseline_idx = substrates
        .iter()
        .position(|s| s.name() == "baseline")
        .expect("dispatch_invocation requires a `baseline` substrate in the set");

    let mut considered: Vec<(usize, SubstrateOption)> = Vec::with_capacity(substrates.len());
    for (idx, s) in substrates.iter().enumerate() {
        let supported = s.supports(p);
        let (cost, joules_total) = if supported {
            let c = s.cost(p, &inv.profile);
            (Some(c), Some(c.joules_total(inv.count)))
        } else {
            (None, None)
        };
        considered.push((
            idx,
            SubstrateOption {
                substrate: s.name().to_string(),
                display: s.display_name().to_string(),
                supported,
                cost,
                joules_total,
            },
        ));
    }

    let chosen_idx = considered
        .iter()
        .filter(|(_, o)| o.supported)
        .min_by(|a, b| {
            a.1.joules_total
                .unwrap()
                .total_cmp(&b.1.joules_total.unwrap())
        })
        .map(|(i, _)| *i)
        .unwrap_or(baseline_idx);

    let chosen_s = &substrates[chosen_idx];
    let chosen_cost = chosen_s.cost(p, &inv.profile);
    let chosen_joules_total = chosen_cost.joules_total(inv.count);
    let chosen_seconds_total = chosen_cost.seconds_total(inv.count);

    let baseline_cost = substrates[baseline_idx].cost(p, &inv.profile);
    let baseline_joules_total = baseline_cost.joules_total(inv.count);

    let justification = if chosen_idx == baseline_idx {
        format!(
            "No specialized substrate supports {} — stays on baseline.",
            p.name
        )
    } else {
        let speedup = if chosen_joules_total > 0.0 {
            baseline_joules_total / chosen_joules_total
        } else {
            f64::INFINITY
        };
        format!(
            "{} @ {:.2e} J/op vs baseline {:.2e} J/op — {:.0}× fewer joules.",
            chosen_s.display_name(),
            chosen_cost.joules_per_op,
            baseline_cost.joules_per_op,
            speedup
        )
    };

    Some(Assignment {
        primitive_id: p.id,
        primitive_name: p.name.clone(),
        family: p.family.clone(),
        thermo: p.thermo.clone(),
        count: inv.count,
        label: inv.label.clone(),
        chosen_substrate: chosen_s.name().to_string(),
        chosen_display: chosen_s.display_name().to_string(),
        chosen_cost,
        chosen_joules_total,
        chosen_seconds_total,
        baseline_cost,
        baseline_joules_total,
        considered: considered.into_iter().map(|(_, o)| o).collect(),
        justification,
    })
}

/// Summary of a dispatch run (everything in DispatchPlan except the per-
/// invocation assignments). Useful as the final SSE `complete` payload
/// after the invocation events have already been streamed.
#[derive(Debug, Clone, Serialize)]
pub struct DispatchSummary {
    pub workload_goal: String,
    pub total_joules: f64,
    pub baseline_total_joules: f64,
    pub savings_factor: f64,
    pub total_seconds: f64,
    pub baseline_total_seconds: f64,
    pub per_substrate_joules: Vec<SubstrateAggregate>,
    pub power_w: f64,
    pub baseline_power_w: f64,
    pub power_budget_w: f64,
    pub within_budget: bool,
    pub invocations_processed: usize,
}

/// Roll up a stream of assignments into a DispatchSummary.
pub fn summarize(
    workload_goal: &str,
    constraints: &Constraints,
    substrates: &[Arc<dyn Substrate>],
    assignments: &[Assignment],
) -> DispatchSummary {
    let mut per_substrate: Vec<SubstrateAggregate> = substrates
        .iter()
        .map(|s| SubstrateAggregate {
            substrate: s.name().to_string(),
            display: s.display_name().to_string(),
            joules: 0.0,
            op_count: 0,
        })
        .collect();
    let mut total_joules = 0.0;
    let mut baseline_total_joules = 0.0;
    let mut total_seconds = 0.0;
    let mut baseline_total_seconds = 0.0;

    for a in assignments {
        if let Some(agg) = per_substrate
            .iter_mut()
            .find(|s| s.substrate == a.chosen_substrate)
        {
            agg.joules += a.chosen_joules_total;
            agg.op_count += a.count;
        }
        total_joules += a.chosen_joules_total;
        baseline_total_joules += a.baseline_joules_total;
        total_seconds += a.chosen_seconds_total;
        baseline_total_seconds += a.baseline_cost.seconds_per_op * a.count as f64;
    }
    let cps = constraints.cycles_per_second.max(1e-9);
    let power_w = total_joules * cps;
    let baseline_power_w = baseline_total_joules * cps;
    let savings_factor = if total_joules > 0.0 {
        baseline_total_joules / total_joules
    } else {
        1.0
    };
    DispatchSummary {
        workload_goal: workload_goal.to_string(),
        total_joules,
        baseline_total_joules,
        savings_factor,
        total_seconds,
        baseline_total_seconds,
        per_substrate_joules: per_substrate,
        power_w,
        baseline_power_w,
        power_budget_w: constraints.power_budget_w,
        within_budget: power_w <= constraints.power_budget_w,
        invocations_processed: assignments.len(),
    }
}

/// Default substrate set — baseline + photonic + in-memory + stochastic +
/// reversible + neuromorphic. The reversible and neuromorphic dialects need
/// the catalog at construction time (Bennett edges / NEURO family).
pub fn default_substrates(stack: Arc<Stack>) -> Vec<Arc<dyn Substrate>> {
    vec![
        Arc::new(tei_d_baseline::Baseline) as Arc<dyn Substrate>,
        Arc::new(tei_d_photonic::Photonic::default()) as Arc<dyn Substrate>,
        Arc::new(tei_d_in_memory::InMemory::default()) as Arc<dyn Substrate>,
        Arc::new(tei_d_stochastic::Stochastic) as Arc<dyn Substrate>,
        Arc::new(tei_d_reversible::Reversible::new(stack.clone())) as Arc<dyn Substrate>,
        Arc::new(tei_d_neuromorphic::Neuromorphic::new(stack)) as Arc<dyn Substrate>,
    ]
}

/// Run the dispatcher in batch — iterate every invocation through
/// `dispatch_invocation`, then `summarize` the result.
pub fn dispatch(
    stack: &Stack,
    workload: &Workload,
    substrates: &[Arc<dyn Substrate>],
) -> DispatchPlan {
    let assignments: Vec<Assignment> = workload
        .invocations
        .iter()
        .filter_map(|inv| dispatch_invocation(stack, inv, substrates))
        .collect();
    let summary = summarize(
        &workload.goal,
        &workload.constraints,
        substrates,
        &assignments,
    );
    DispatchPlan {
        workload_goal: summary.workload_goal,
        assignments,
        total_joules: summary.total_joules,
        baseline_total_joules: summary.baseline_total_joules,
        savings_factor: summary.savings_factor,
        total_seconds: summary.total_seconds,
        baseline_total_seconds: summary.baseline_total_seconds,
        per_substrate_joules: summary.per_substrate_joules,
        power_w: summary.power_w,
        baseline_power_w: summary.baseline_power_w,
        power_budget_w: summary.power_budget_w,
        within_budget: summary.within_budget,
    }
}
