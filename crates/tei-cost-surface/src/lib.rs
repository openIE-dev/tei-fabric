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

use serde::Serialize;
use std::sync::Arc;
use tei_ir::Workload;
use tei_stack::Stack;
use tei_substrate_traits::{Cost, Substrate};

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

/// Default substrate set — baseline + photonic + in-memory + stochastic +
/// reversible. The reversible dialect needs the catalog at construction
/// time so it can consult Bennett-decomposition edges.
pub fn default_substrates(stack: Arc<Stack>) -> Vec<Arc<dyn Substrate>> {
    vec![
        Arc::new(tei_d_baseline::Baseline) as Arc<dyn Substrate>,
        Arc::new(tei_d_photonic::Photonic) as Arc<dyn Substrate>,
        Arc::new(tei_d_in_memory::InMemory) as Arc<dyn Substrate>,
        Arc::new(tei_d_stochastic::Stochastic) as Arc<dyn Substrate>,
        Arc::new(tei_d_reversible::Reversible::new(stack)) as Arc<dyn Substrate>,
    ]
}

/// Run the dispatcher.
pub fn dispatch(
    stack: &Stack,
    workload: &Workload,
    substrates: &[Arc<dyn Substrate>],
) -> DispatchPlan {
    let baseline_idx = substrates
        .iter()
        .position(|s| s.name() == "baseline")
        .expect("dispatch() requires a `baseline` substrate in the set");

    let mut assignments: Vec<Assignment> = Vec::with_capacity(workload.invocations.len());
    let mut per_substrate: Vec<SubstrateAggregate> = substrates
        .iter()
        .map(|s| SubstrateAggregate {
            substrate: s.name().to_string(),
            display: s.display_name().to_string(),
            joules: 0.0,
            op_count: 0,
        })
        .collect();

    let mut total_joules = 0.0_f64;
    let mut baseline_total_joules = 0.0_f64;
    let mut total_seconds = 0.0_f64;
    let mut baseline_total_seconds = 0.0_f64;

    for inv in &workload.invocations {
        let p = match stack.get(inv.primitive_id) {
            Some(p) => p,
            None => continue,
        };

        // Probe each substrate for cost.
        let mut considered: Vec<(usize, SubstrateOption)> = Vec::with_capacity(substrates.len());
        for (idx, s) in substrates.iter().enumerate() {
            let supported = s.supports(p);
            let (cost, joules_total) = if supported {
                let c = s.cost(p, &inv.profile);
                (Some(c), Some(c.joules_total(inv.count)))
            } else {
                (None, None)
            };
            considered.push((idx, SubstrateOption {
                substrate: s.name().to_string(),
                display: s.display_name().to_string(),
                supported,
                cost,
                joules_total,
            }));
        }

        // Choose the min-joule supported substrate.
        let chosen_idx = considered
            .iter()
            .filter(|(_, o)| o.supported)
            .min_by(|a, b| {
                a.1.joules_total.unwrap().total_cmp(&b.1.joules_total.unwrap())
            })
            .map(|(i, _)| *i)
            .unwrap_or(baseline_idx);

        let chosen_s = &substrates[chosen_idx];
        let chosen_cost = chosen_s.cost(p, &inv.profile);
        let chosen_joules_total = chosen_cost.joules_total(inv.count);
        let chosen_seconds_total = chosen_cost.seconds_total(inv.count);

        let baseline_cost = substrates[baseline_idx].cost(p, &inv.profile);
        let baseline_joules_total = baseline_cost.joules_total(inv.count);
        let baseline_seconds_total = baseline_cost.seconds_total(inv.count);

        // Accumulate per-substrate aggregate energy.
        per_substrate[chosen_idx].joules += chosen_joules_total;
        per_substrate[chosen_idx].op_count += inv.count;

        // Accumulate totals.
        total_joules += chosen_joules_total;
        baseline_total_joules += baseline_joules_total;
        total_seconds += chosen_seconds_total;
        baseline_total_seconds += baseline_seconds_total;

        // Justification.
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

        assignments.push(Assignment {
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
        });
    }

    let cps = workload.constraints.cycles_per_second.max(1e-9);
    let power_w = total_joules * cps;
    let baseline_power_w = baseline_total_joules * cps;
    let within_budget = power_w <= workload.constraints.power_budget_w;
    let savings_factor = if total_joules > 0.0 {
        baseline_total_joules / total_joules
    } else {
        1.0
    };

    DispatchPlan {
        workload_goal: workload.goal.clone(),
        assignments,
        total_joules,
        baseline_total_joules,
        savings_factor,
        total_seconds,
        baseline_total_seconds,
        per_substrate_joules: per_substrate,
        power_w,
        baseline_power_w,
        power_budget_w: workload.constraints.power_budget_w,
        within_budget,
    }
}
