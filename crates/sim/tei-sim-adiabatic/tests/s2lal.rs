//! §3.6 published-trend validation of the S2LAL MOSFET cells — closed-form
//! scaling laws and published *trends* only (DeBenedictis arXiv:2009.00448
//! charts, not its raw ngspice numbers), per the binding validation policy.
//!
//! Coverage (roadmap §3.6 validation table, MOSFET rows):
//!   i)   `s2lal_loglog_slope_is_minus_one_in_adiabatic_regime`
//!   ii)  `s2lal_recovery_ratio_improves_monotonically_with_slower_ramps`
//!   iii) `s2lal_ekv_leakage_creates_interior_energy_minimum`  (the U-curve)
//!   iv)  `s2lal_abrupt_limit_lands_at_n_cv2_scale`
//!   +)   chain linearity in N through the sparse solver path, Tellegen
//!        consistency through the MOSFET stamps, executor end-to-end.

use tei_sim_adiabatic::{
    AdiabaticExecutor, AdiabaticJob, CellSpec, fitted_slope, overhead_curve, run_cell,
};
use tei_sim_circuit::SolverKind;
use tei_sim_core::exec::Executor;

const C: f64 = 1e-12;
const V: f64 = 1.0;
const KP: f64 = 1e-4;
const VTH: f64 = 0.3;

fn s2lal_cell(n_stages: usize, leakage: bool, vth: f64) -> CellSpec {
    CellSpec::S2lalChain {
        c_f: C,
        v: V,
        n_stages,
        n_phases: 4,
        hold_rc: 8.0,
        kp: KP,
        vth,
        leakage,
        ekv_n: 1.3,
        phi_t: tei_sim_circuit::VT_300K,
    }
}

/// (i) Closed-form scaling law: with ideal (level-1, leak-free) switches the
/// S2LAL chain's E(T) follows the adiabatic −1 log-log slope over
/// T/RC ∈ [10, 1000] (within ±0.1 — the T-gate's conductance varies along
/// the swing, so the prefactor wobbles, the exponent must not), and the
/// curve is strictly monotone decreasing.
#[test]
fn s2lal_loglog_slope_is_minus_one_in_adiabatic_regime() {
    let ratios = [10.0, 31.622776601683793, 100.0, 316.22776601683796, 1000.0];
    let curve = overhead_curve(&s2lal_cell(1, false, VTH), &ratios).unwrap();
    let slope = fitted_slope(&curve);
    eprintln!("S2LAL level-1 overhead curve: {curve:?}");
    eprintln!("S2LAL fitted log-log slope = {slope:.4}");
    assert!(
        (slope + 1.0).abs() < 0.1,
        "fitted log-log slope = {slope:.4}, expected −1.0 ± 0.1"
    );
    for w in curve.windows(2) {
        assert!(w[1].1 < w[0].1, "overhead not monotone: {curve:?}");
    }
}

/// (ii) Published trend (DeBenedictis arXiv:2009.00448): the
/// energy-recovery ratio 1 − E/E_abrupt improves monotonically as the
/// power-clock ramps slow down.
#[test]
fn s2lal_recovery_ratio_improves_monotonically_with_slower_ramps() {
    let cell = s2lal_cell(1, false, VTH);
    let mut last = f64::NEG_INFINITY;
    for ratio in [2.0, 20.0, 200.0] {
        let run = run_cell(&cell, ratio).unwrap();
        let rec = run.recovery_ratio();
        eprintln!(
            "T/RC = {ratio:>5}: E = {:.4e} J, recovery ratio = {rec:.4}",
            run.e_diss_j
        );
        assert!(
            rec > last,
            "recovery ratio not improving: {rec:.4} after {last:.4} at T/RC = {ratio}"
        );
        last = rec;
    }
    // Deep adiabatic: most of the energy comes back.
    assert!(last > 0.9, "recovery ratio at T/RC = 200 only {last:.4}");
}

/// (iii) The canonical published U-curve: with EKV-lite switches the
/// subthreshold leakage through the off T-gate adds an E ∝ T floor, so the
/// energy per cycle develops an **interior minimum** in T — slowing the
/// clock stops paying once leakage dominates. Assert the minimum exists
/// strictly inside the sweep and the curve rises on both sides of it.
#[test]
fn s2lal_ekv_leakage_creates_interior_energy_minimum() {
    // vth = 0.25 raises I_off into the sweep window (the minimum sits at
    // T* ≈ sqrt(adiabatic coefficient / leak power) — see the estimate in
    // the assertion message if this moves).
    let cell = s2lal_cell(1, true, 0.25);
    let ratios = [30.0, 100.0, 300.0, 1000.0, 3000.0, 10000.0];
    let curve = overhead_curve(&cell, &ratios).unwrap();
    eprintln!("S2LAL EKV-leakage E(T) curve (T/RC, E/E_abrupt): {curve:?}");
    let (argmin, _) = curve
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.1.total_cmp(&b.1.1))
        .unwrap();
    eprintln!(
        "U-curve minimum at T/RC = {} (E/E_abrupt = {:.4e})",
        curve[argmin].0, curve[argmin].1
    );
    assert!(
        argmin > 0 && argmin < curve.len() - 1,
        "E(T) minimum not interior: argmin = {argmin}, curve = {curve:?}"
    );
    assert!(
        curve[0].1 > curve[argmin].1 && curve[curve.len() - 1].1 > curve[argmin].1,
        "no U-shape: {curve:?}"
    );
    // The level-1 (leak-free) twin keeps falling where the EKV curve has
    // already turned around — the leakage is what creates the floor.
    let ideal = overhead_curve(&s2lal_cell(1, false, 0.25), &ratios).unwrap();
    assert!(
        ideal[ratios.len() - 1].1 < ideal[argmin].1,
        "leak-free control did not keep falling: {ideal:?}"
    );
}

/// (iv) Abrupt-switching limit: at T/RC = 0.01 the per-cycle dissipation
/// lands at the N·CV² scale (full charge + discharge loss, within 10% —
/// thresholds change *where* the charge flows, not how much is lost when
/// switching is abrupt).
#[test]
fn s2lal_abrupt_limit_lands_at_n_cv2_scale() {
    let run = run_cell(&s2lal_cell(1, false, VTH), 0.01).unwrap();
    let ratio = run.e_diss_j / run.e_abrupt_j; // e_abrupt = N·CV²
    eprintln!(
        "S2LAL abrupt limit: E = {:.4e} J, E/(N·CV²) = {ratio:.4}",
        run.e_diss_j
    );
    assert!(
        (ratio - 1.0).abs() < 0.10,
        "abrupt E/(N·CV²) = {ratio:.4}, expected 1 ± 0.10"
    );
}

/// Property: stages ride their own clock phases, so chain dissipation is
/// linear in N — and a 5-stage chain (21 nodes) runs through the sparse
/// Markowitz-LU path under the default Auto choice, exercising the
/// per-Newton-iteration numeric refactor on the MOSFET cells. Tellegen
/// holds through the nonlinear stamps.
#[test]
fn s2lal_chain_linear_in_stages_through_sparse_path() {
    let ratio = 20.0;
    let r1 = run_cell(&s2lal_cell(1, false, VTH), ratio).unwrap();
    let r5 = run_cell(&s2lal_cell(5, false, VTH), ratio).unwrap();
    assert_eq!(
        r1.transient.solver,
        SolverKind::Dense,
        "5-node single stage stays dense"
    );
    assert_eq!(
        r5.transient.solver,
        SolverKind::Sparse,
        "21-node chain must route to sparse under Auto"
    );
    let rel = (r5.e_diss_j - 5.0 * r1.e_diss_j).abs() / (5.0 * r1.e_diss_j);
    assert!(
        rel < 0.03,
        "chain E(5) = {:.4e} vs 5·E(1) = {:.4e} (rel {rel:.2e})",
        r5.e_diss_j,
        5.0 * r1.e_diss_j
    );
    for t in [&r1.transient, &r5.transient] {
        assert!(
            t.tellegen_max < 1e-9,
            "per-step Tellegen residual {} W",
            t.tellegen_max
        );
        let identity = (t.source_energy - (t.dissipated_energy + t.reactive_absorbed_energy)).abs();
        assert!(
            identity < 1e-9 * t.source_energy.abs().max(1e-30),
            "energy identity residual {identity:.3e} J"
        );
        // Full recovery cycle: every load is back near 0 V.
        assert!(
            t.delta_stored_energy.abs() < 1e-2 * t.dissipated_energy,
            "Δstored = {:.3e} J after a full cycle",
            t.delta_stored_energy
        );
    }
}

/// Executor end-to-end on the new kind: a JSON `s2lal_chain` job sweeps,
/// reports the curve + fitted slope, and aggregates joules in the ledger.
#[test]
fn executor_runs_s2lal_job() {
    let job: AdiabaticJob = serde_json::from_str(
        r#"{
            "cell": {"kind":"s2lal_chain","c_f":1e-12,"v":1.0,"n_stages":2},
            "ratios": [10.0, 100.0, 1000.0]
        }"#,
    )
    .unwrap();
    let mut ticks = 0usize;
    let res = AdiabaticExecutor.execute(&job, &mut |_| ticks += 1);
    assert_eq!(ticks, 3);
    assert!(res.outputs.get("error").is_none(), "{:?}", res.outputs);
    let slope = res.outputs["fitted_loglog_slope"].as_f64().unwrap();
    assert!((slope + 1.0).abs() < 0.15, "executor slope {slope}");
    assert_eq!(res.outputs["curve"].as_array().unwrap().len(), 3);
    assert_eq!(res.outputs["cell"]["kind"].as_str().unwrap(), "s2lal_chain");
    assert!(res.ledger.joules > 0.0);
}
