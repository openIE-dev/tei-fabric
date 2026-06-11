//! Calibration loop — the roadmap's "measured, not assumed" (§4).
//!
//! After a simulator runs, its `EventLedger` is priced with the *same
//! per-event constants the cost dialect uses*, and shown beside the
//! dialect's a-priori estimate. The estimate prices assumed event counts
//! (10% spike activity, sweeps × variables proposals, 2·MACs·bits
//! modulator events); the measured figure prices the events that actually
//! happened. Divergence is a feature: it is the model being checked.
//!
//! For the adiabatic column the comparison is deeper: the reversible
//! dialect assumes a *fixed* overhead-above-Landauer
//! (`REVERSIBLE_OVERHEAD_L0 = 10³`); the simulator returns the measured
//! overhead as a *function of ramp time*, locating where on the E-vs-T
//! curve that constant actually sits.

use serde_json::{Value, json};
use std::sync::Arc;
use tei_ir::{Dtype, OpProfile, TensorShape};
use tei_sim_core::ledger::EventLedger;
use tei_stack::Stack;
use tei_substrate_traits::{K_T_LN2_300K, Substrate};

const PRIM_MATMUL: u32 = 18;
const PRIM_LIF: u32 = 50;
const PRIM_ANNEAL: u32 = 258;
/// Bits erased per L₀ atomic op (matches `tei-d-baseline`'s table).
const L0_BITS: f64 = 4.0;

/// Price `profile` on the named registered dialect. Returns
/// (joules_per_op, primitive name).
fn estimate(
    stack: &Stack,
    subs: &[Arc<dyn Substrate>],
    name: &str,
    prim_id: u32,
    profile: &OpProfile,
) -> Option<(f64, String)> {
    let prim = stack.get(prim_id)?;
    let sub = subs.iter().find(|s| s.name() == name)?;
    if !sub.supports(prim) {
        return None;
    }
    Some((sub.cost(prim, profile).joules_per_op, prim.name.clone()))
}

fn block(
    substrate: &str,
    prim_name: &str,
    estimated_j: f64,
    measured_j: f64,
    details: Value,
) -> Value {
    json!({
        "substrate": substrate,
        "primitive": prim_name,
        "estimated_j": estimated_j,
        "measured_j": measured_j,
        "measured_over_estimated": if estimated_j > 0.0 { Some(measured_j / estimated_j) } else { None },
        "details": details,
    })
}

/// Stochastic: the dialect assumes `sweeps × variables` p-bit proposals;
/// the ledger counted the proposals (and accepted flips) that happened.
pub fn stochastic(
    stack: &Stack,
    subs: &[Arc<dyn Substrate>],
    sweeps: u64,
    variables: usize,
    ledger: &EventLedger,
) -> Option<Value> {
    let profile = OpProfile {
        sweeps: Some(sweeps),
        variables: Some(variables),
        batch: 1,
        ..Default::default()
    };
    let (est, prim) = estimate(stack, subs, "stochastic", PRIM_ANNEAL, &profile)?;
    let measured = ledger.spin_updates as f64 * tei_d_stochastic::PBIT_J_PER_SAMPLE
        + ledger.sweeps as f64 * tei_d_stochastic::READOUT_J_PER_SWEEP;
    Some(block(
        "stochastic",
        &prim,
        est,
        measured,
        json!({
            "assumed_pbit_events": sweeps * variables as u64,
            "measured_spin_updates": ledger.spin_updates,
            "measured_flips": ledger.flips,
        }),
    ))
}

/// Spiking: structural fanout is taken from the built network, so the
/// assumed-vs-measured axis isolates the dialect's `DEFAULT_ACTIVITY`.
pub fn spiking(
    stack: &Stack,
    subs: &[Arc<dyn Substrate>],
    neurons: u64,
    timesteps: u64,
    n_synapses: u64,
    ledger: &EventLedger,
) -> Option<Value> {
    let fanout = if neurons > 0 { n_synapses / neurons } else { 0 };
    let profile = OpProfile {
        variables: Some(neurons as usize),
        sweeps: Some(timesteps),
        reduce_dim: Some(fanout as usize),
        batch: 1,
        ..Default::default()
    };
    let (est, prim) = estimate(stack, subs, "neuromorphic", PRIM_LIF, &profile)?;
    let p = tei_d_neuromorphic::NeuromorphicParams::default();
    let measured = ledger.sops as f64 * p.sop_j + (neurons * timesteps) as f64 * p.neuron_update_j;
    let measured_activity = if neurons * timesteps > 0 {
        ledger.spikes as f64 / (neurons * timesteps) as f64
    } else {
        0.0
    };
    Some(block(
        "neuromorphic",
        &prim,
        est,
        measured,
        json!({
            "assumed_activity": tei_d_neuromorphic::DEFAULT_ACTIVITY,
            "measured_activity": measured_activity,
            "measured_spikes": ledger.spikes,
            "measured_sops": ledger.sops,
        }),
    ))
}

/// Crossbar: the ledger's MAC and ADC counts already include tiling, so
/// the measured figure prices physical events, not the ideal matmul.
pub fn crossbar(
    stack: &Stack,
    subs: &[Arc<dyn Substrate>],
    rows: usize,
    cols: usize,
    n_queries: u64,
    ledger: &EventLedger,
) -> Option<Value> {
    let profile = OpProfile {
        shape: TensorShape {
            dims: vec![n_queries as usize, cols],
        },
        reduce_dim: Some(rows),
        dtype: Dtype::I8,
        batch: 1,
        ..Default::default()
    };
    let (est, prim) = estimate(stack, subs, "in-memory", PRIM_MATMUL, &profile)?;
    let p = tei_d_in_memory::InMemoryParams::default();
    let bits = Dtype::I8.bits() as f64;
    let measured = ledger.macs as f64 * p.device_j_per_mac
        + ledger.macs as f64 / rows.max(1) as f64 * bits * p.dac_j_per_bit
        + ledger.adc_samples as f64 * p.adc_j_per_sample;
    Some(block(
        "in-memory",
        &prim,
        est,
        measured,
        json!({
            "measured_macs": ledger.macs,
            "measured_adc_samples": ledger.adc_samples,
        }),
    ))
}

/// Photonic: the dialect assumes 2·MACs·bits modulator events (per-MAC
/// lower bound); the mesh counted one event per phase shifter per query.
pub fn photonic(
    stack: &Stack,
    subs: &[Arc<dyn Substrate>],
    n: usize,
    n_queries: usize,
    ledger: &EventLedger,
) -> Option<Value> {
    let profile = OpProfile {
        shape: TensorShape {
            dims: vec![n_queries, n],
        },
        reduce_dim: Some(n),
        batch: 1,
        ..Default::default()
    };
    let (est, prim) = estimate(stack, subs, "photonic", PRIM_MATMUL, &profile)?;
    let p = tei_d_photonic::PhotonicParams::default();
    let bits = Dtype::default().bits() as f64;
    let measured = ledger.macs as f64 * p.optical_j_per_mac / p.laser_efficiency.max(0.01)
        + ledger.modulator_events as f64 * bits * p.modulator_j_per_bit
        + ledger.adc_samples as f64 * p.adc_j_per_sample;
    Some(block(
        "photonic",
        &prim,
        est,
        measured,
        json!({
            "assumed_modulator_events": 2 * ledger.macs * bits as u64,
            "measured_modulator_events": ledger.modulator_events,
            "measured_macs": ledger.macs,
            "measured_detector_samples": ledger.adc_samples,
        }),
    ))
}

/// Adiabatic: the reversible dialect's fixed `REVERSIBLE_OVERHEAD_L0 = 10³`
/// versus the measured overhead-above-Landauer as a function of T/RC —
/// the single highest-credibility recalibration in the program. The
/// crossover ratio is where the simulated cell actually delivers the
/// dialect's assumed constant.
pub fn adiabatic(outputs: &Value) -> Option<Value> {
    let curve = outputs.get("curve")?.as_array()?;
    let floor_j = L0_BITS * K_T_LN2_300K;
    let assumed = tei_d_reversible::REVERSIBLE_OVERHEAD_L0;
    let estimated_j = assumed * floor_j;

    let pts: Vec<(f64, f64)> = curve
        .iter()
        .filter_map(|p| {
            let r = p.get("t_over_rc")?.as_f64()?;
            let e = p.get("e_diss_j")?.as_f64()?;
            Some((r, e / floor_j))
        })
        .collect();
    if pts.is_empty() {
        return None;
    }

    // Log-log interpolate the T/RC where measured overhead crosses the
    // assumed constant (overhead decreases monotonically with T/RC).
    let crossover = pts.windows(2).find_map(|w| {
        let ((r0, o0), (r1, o1)) = (w[0], w[1]);
        if (o0 - assumed) * (o1 - assumed) <= 0.0 && o0 > 0.0 && o1 > 0.0 && o0 != o1 {
            let f = (assumed.ln() - o0.ln()) / (o1.ln() - o0.ln());
            Some((r0.ln() + f * (r1.ln() - r0.ln())).exp())
        } else {
            None
        }
    });

    let measured_curve: Vec<Value> = pts
        .iter()
        .map(|(r, o)| json!({ "t_over_rc": r, "measured_overhead": o }))
        .collect();
    let measured_j_last = pts.last().map(|(_, o)| o * floor_j).unwrap_or(0.0);
    // Ready-to-use dispatch patch: the overhead this cell actually delivers
    // at the slowest ramp swept. POST it back as `substrate_params` on
    // /api/dispatch to re-price the plan with the measured constant.
    let (best_ratio, best_overhead) = *pts.last().unwrap();

    Some(json!({
        "substrate": "reversible",
        "primitive": "adiabatic L0 atomic op",
        "estimated_j": estimated_j,
        "measured_j": measured_j_last,
        "details": {
            "assumed_overhead_l0": assumed,
            "landauer_floor_j": floor_j,
            "bits_per_atomic_op": L0_BITS,
            "measured_overhead_curve": measured_curve,
            "crossover_t_over_rc": crossover,
            "patch_t_over_rc": best_ratio,
        },
        "substrate_params_patch": { "reversible": { "overhead_l0": best_overhead } },
    }))
}

/// MNIST-on-crossbar: the in-memory dialect's fixed `accuracy_loss = 0.01`
/// versus the measured loss-vs-digital of a real network under the device
/// model at the operating σ — the accuracy axis of the calibration loop.
pub fn mnist_accuracy(outputs: &Value) -> Option<Value> {
    let measured = outputs.get("measured_accuracy_loss")?.as_f64()?;
    let assumed = tei_d_in_memory::IN_MEMORY_ACCURACY_LOSS;
    let patch = tei_d_in_memory::InMemoryParams {
        accuracy_loss: measured,
        ..Default::default()
    };
    Some(json!({
        "substrate": "in-memory",
        "kind": "accuracy",
        "assumed_accuracy_loss": assumed,
        "measured_accuracy_loss": measured,
        "operating_sigma": outputs.get("operating_sigma"),
        "digital_accuracy": outputs.get("digital_accuracy"),
        "substrate_params_patch": { "in_memory": patch },
    }))
}
