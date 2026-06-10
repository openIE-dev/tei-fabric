//! Analytic validation of the adiabatic energy-analysis crate — closed-form
//! ground truth and properties only, per the roadmap's binding validation
//! policy (no foreign-tool fixtures).
//!
//! Coverage map (roadmap §3.6 validation table + spec items):
//!   1) `ramp_charge_matches_exact_closed_form_under_1pct`   (+ asymptote)
//!   2) `adiabatic_loglog_slope_is_minus_one`                (−1 ± 0.05)
//!   3) `abrupt_step_recovers_half_cv2_within_1pct`
//!   4) `charge_recovery_slow_clock_dissipates_under_10pct_of_abrupt`
//!      `charge_recovery_fast_clock_approaches_abrupt_cv2`
//!   5) `shift_register_chain_dissipation_is_linear_in_stages`
//!      `shift_register_chain_is_tellegen_consistent`
//!   6) `executor_slope_matches_direct_sweep`                (property)

use tei_sim_adiabatic::{
    AdiabaticExecutor, AdiabaticJob, CellSpec, fitted_slope, overhead_curve, ramp_charge_exact,
    run_cell,
};
use tei_sim_core::exec::Executor;

const R: f64 = 1e3;
const C: f64 = 1e-9;
const V: f64 = 1.0;
const CV2: f64 = C * V * V;

fn ramp_cell() -> CellSpec {
    CellSpec::RampCharge {
        r_ohm: R,
        c_f: C,
        v: V,
    }
}

fn recovery_cell() -> CellSpec {
    CellSpec::ChargeRecovery {
        r_ohm: R,
        c_f: C,
        v: V,
        hold_rc: 8.0,
    }
}

fn chain_cell(n_stages: usize) -> CellSpec {
    CellSpec::ShiftRegister {
        r_ohm: R,
        c_f: C,
        v: V,
        n_stages,
        n_phases: 4,
        hold_rc: 8.0,
    }
}

/// (1) Ramp charge: simulated E_R matches the exact closed form
/// E = (RC/T)·CV²·[1 − (RC/T)(1 − e^{−T/RC})] to <1% across
/// T/RC ∈ {1, 10, 100, 1000}, and the (RC/T)·CV² asymptote within 2%
/// for T/RC ≥ 100.
#[test]
fn ramp_charge_matches_exact_closed_form_under_1pct() {
    let cell = ramp_cell();
    let rc = R * C;
    for ratio in [1.0, 10.0, 100.0, 1000.0] {
        let run = run_cell(&cell, ratio).unwrap();
        let exact = ramp_charge_exact(R, C, V, ratio * rc);
        let rel = (run.e_diss_j - exact).abs() / exact;
        assert!(
            rel < 0.01,
            "T/RC={ratio}: E={:.6e} vs closed form {exact:.6e} (rel {rel:.2e})",
            run.e_diss_j
        );
        if ratio >= 100.0 {
            let asym = CV2 / ratio; // (RC/T)·CV²
            let rel = (run.e_diss_j - asym).abs() / asym;
            assert!(
                rel < 0.02,
                "T/RC={ratio}: E={:.6e} vs asymptote {asym:.6e} (rel {rel:.2e})",
                run.e_diss_j
            );
        }
    }
}

/// (2) The scaling law: log-log slope of E vs T fitted over the adiabatic
/// regime (T/RC from 10 to 1000) is −1.0 ± 0.05.
#[test]
fn adiabatic_loglog_slope_is_minus_one() {
    let ratios = [10.0, 31.622776601683793, 100.0, 316.22776601683796, 1000.0];
    let curve = overhead_curve(&ramp_cell(), &ratios).unwrap();
    let slope = fitted_slope(&curve);
    assert!(
        (slope + 1.0).abs() < 0.05,
        "fitted log-log slope = {slope:.4}, expected −1.0 ± 0.05"
    );
    // Monotone decrease across the regime.
    for w in curve.windows(2) {
        assert!(w[1].1 < w[0].1, "overhead not monotone: {curve:?}");
    }
}

/// (3) Abrupt limit: a T/RC = 0.01 step recovers ½CV² within 1%.
#[test]
fn abrupt_step_recovers_half_cv2_within_1pct() {
    let run = run_cell(&ramp_cell(), 0.01).unwrap();
    let half_cv2 = 0.5 * CV2;
    let rel = (run.e_diss_j - half_cv2).abs() / half_cv2;
    assert!(
        rel < 0.01,
        "abrupt E = {:.6e}, ½CV² = {half_cv2:.6e} (rel {rel:.2e})",
        run.e_diss_j
    );
}

/// (4a) Charge-recovery cycle, slow trapezoid (T/RC = 100): per-cycle
/// dissipation ≪ the abrupt 2·½CV² — assert < 10% (the ideal value is
/// ≈ 2·(RC/T)·CV² = 2% of abrupt, so also pin that within 20% relative).
#[test]
fn charge_recovery_slow_clock_dissipates_under_10pct_of_abrupt() {
    let run = run_cell(&recovery_cell(), 100.0).unwrap();
    let abrupt = CV2; // 2·½CV²
    assert!(
        run.e_diss_j < 0.10 * abrupt,
        "slow recovery cycle E = {:.4e} ≥ 10% of abrupt {abrupt:.4e}",
        run.e_diss_j
    );
    // Rise + fall each ≈ the ramp closed form ⇒ per-cycle ≈ 2·E_ramp(T).
    let two_ramps = 2.0 * ramp_charge_exact(R, C, V, 100.0 * R * C);
    let rel = (run.e_diss_j - two_ramps).abs() / two_ramps;
    assert!(
        rel < 0.20,
        "recovery cycle E = {:.4e} vs 2×ramp form {two_ramps:.4e} (rel {rel:.2e})",
        run.e_diss_j
    );
}

/// (4b) Charge-recovery cycle, fast ramps (T/RC = 0.01): nothing is
/// recovered — per-cycle dissipation approaches 2·½CV² = CV² (within 3%).
#[test]
fn charge_recovery_fast_clock_approaches_abrupt_cv2() {
    let run = run_cell(&recovery_cell(), 0.01).unwrap();
    let abrupt = CV2;
    let rel = (run.e_diss_j - abrupt).abs() / abrupt;
    assert!(
        rel < 0.03,
        "fast recovery cycle E = {:.4e}, abrupt CV² = {abrupt:.4e} (rel {rel:.2e})",
        run.e_diss_j
    );
}

/// (5a) Shift-register chain: total dissipation = N × the single-stage
/// value within a few % (stages ride their own clock phases; the harness
/// must not couple them).
#[test]
fn shift_register_chain_dissipation_is_linear_in_stages() {
    let ratio = 20.0;
    let e1 = run_cell(&chain_cell(1), ratio).unwrap().e_diss_j;
    let e6 = run_cell(&chain_cell(6), ratio).unwrap().e_diss_j;
    let rel = (e6 - 6.0 * e1).abs() / (6.0 * e1);
    assert!(
        rel < 0.03,
        "chain E(6) = {e6:.4e} vs 6·E(1) = {:.4e} (rel {rel:.2e})",
        6.0 * e1
    );
    // And the normalized overheads agree point-for-point.
    let r1 = run_cell(&chain_cell(1), ratio).unwrap();
    let r6 = run_cell(&chain_cell(6), ratio).unwrap();
    let (o1, o6) = (r1.e_diss_j / r1.e_abrupt_j, r6.e_diss_j / r6.e_abrupt_j);
    assert!(
        (o1 - o6).abs() / o1 < 0.03,
        "overheads {o1:.4e} vs {o6:.4e}"
    );
}

/// (5b) Chain energy ledger is Tellegen-consistent (passthrough from
/// tei-sim-circuit): per-step power residual ≈ 0 and
/// source = dissipated + reactive-absorbed to LU precision.
#[test]
fn shift_register_chain_is_tellegen_consistent() {
    let run = run_cell(&chain_cell(4), 20.0).unwrap();
    let t = &run.transient;
    // Power scale here is ~V²/R = 1 mW; residual must sit at LU epsilon.
    assert!(
        t.tellegen_max < 1e-12,
        "per-step Tellegen residual {} W",
        t.tellegen_max
    );
    let identity = (t.source_energy - (t.dissipated_energy + t.reactive_absorbed_energy)).abs();
    assert!(
        identity < 1e-10 * t.source_energy.abs().max(1e-30),
        "source − (dissipated + reactive) = {identity:.3e} J"
    );
    // Full cycle: every load returns to 0 V, so net stored energy ≈ 0 and
    // the clocks paid (almost) exactly what the resistors burned.
    assert!(
        t.delta_stored_energy.abs() < 1e-3 * t.dissipated_energy,
        "Δstored = {:.3e} J after a full recovery cycle",
        t.delta_stored_energy
    );
}

/// (6) Property: the executor's fitted_loglog_slope and curve match the
/// direct sweep exactly (same deterministic computation), the progress
/// stream ticks once per ratio with the documented metrics, and the ledger
/// aggregates the dissipated joules.
#[test]
fn executor_slope_matches_direct_sweep() {
    let ratios = vec![
        1.0,
        10.0,
        31.622776601683793,
        100.0,
        316.22776601683796,
        1000.0,
    ];
    let cell = ramp_cell();
    let job = AdiabaticJob {
        cell: cell.clone(),
        ratios: ratios.clone(),
    };

    let mut ticks = 0usize;
    let res = AdiabaticExecutor.execute(&job, &mut |p| {
        assert!((0.0..=1.0).contains(&p.fraction));
        assert!(p.metrics.get("ratio").is_some());
        assert!(p.metrics.get("e_diss_j").is_some());
        assert!(p.metrics.get("e_ratio").is_some());
        ticks += 1;
    });
    assert_eq!(ticks, ratios.len(), "one progress tick per ratio");

    // Direct sweep through the public recalibration API.
    let curve = overhead_curve(&cell, &ratios).unwrap();
    let slope_direct = fitted_slope(&curve);
    let slope_exec = res.outputs["fitted_loglog_slope"].as_f64().unwrap();
    assert!(
        (slope_exec - slope_direct).abs() < 1e-12,
        "executor slope {slope_exec} ≠ direct {slope_direct}"
    );

    // Curve points line up one-for-one, in input-ratio order.
    let out_curve = res.outputs["curve"].as_array().unwrap();
    assert_eq!(out_curve.len(), ratios.len());
    let mut total_j = 0.0;
    for (point, (ratio, overhead)) in out_curve.iter().zip(&curve) {
        assert_eq!(point["t_over_rc"].as_f64().unwrap(), *ratio);
        let o = point["e_over_half_cv2"].as_f64().unwrap();
        assert!((o - overhead).abs() < 1e-12);
        assert!(point["e_over_landauer_300k"].as_f64().unwrap() > 0.0);
        total_j += point["e_diss_j"].as_f64().unwrap();
    }

    // Ledger: aggregate joules + steps, deterministic outputs otherwise.
    assert!((res.ledger.joules - total_j).abs() < 1e-12 * total_j);
    assert!(res.ledger.sweeps > 0, "transient steps recorded in sweeps");
    assert!(res.ledger.wall_seconds.is_some());
    assert_eq!(res.outputs["abrupt_limit_j"].as_f64().unwrap(), 0.5 * CV2);
    assert_eq!(res.outputs["params"]["rc_s"].as_f64().unwrap(), R * C);
    assert_eq!(res.outputs["cell"]["kind"].as_str().unwrap(), "ramp_charge");
}
