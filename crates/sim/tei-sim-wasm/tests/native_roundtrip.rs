//! Native proof that the JSON surface is a faithful wrapper: running a job
//! through `run_*_json` (the exact code the wasm bindings call) produces
//! the same `ExecutionResult` as calling the executor directly, and the
//! forwarded progress ticks carry the SSE `progress` shape.
//!
//! `ledger.wall_seconds` is excluded from equality — it is real measured
//! wall time, distinct across the two runs by construction (and `None` on
//! wasm). Everything else must match exactly: the RNG is deterministic, so
//! same job + same seed ⇒ identical outputs.

// Native-only: wasm builds run `tests/identity.rs` under wasm-bindgen-test.
#![cfg(not(target_arch = "wasm32"))]

use tei_sim_core::exec::Executor;
use tei_sim_wasm::{run_adiabatic_json, run_stochastic_json};

const STOCHASTIC_JOB: &str = r#"{
    "problem": {"kind": "petersen"},
    "schedule": {"sweeps": 2000, "beta0": 0.1, "beta1": 5.0},
    "seed": 42
}"#;

const TEMPERING_JOB: &str = r#"{
    "problem": {"kind": "random_regular", "n": 40, "degree": 3, "seed": 7},
    "schedule": {"sweeps": 1500, "beta0": 0.1, "beta1": 5.0},
    "seed": 42,
    "tempering": {"replicas": 4, "beta_min": 0.1, "beta_max": 6.0, "swap_interval": 10}
}"#;

const ADIABATIC_JOB: &str = r#"{
    "cell": {"kind": "charge_recovery", "r_ohm": 1000.0, "c_f": 1e-9, "v": 1.0},
    "ratios": [1.0, 3.1623, 10.0, 31.623, 100.0, 316.23, 1000.0]
}"#;

/// Result JSON → Value with `ledger.wall_seconds` nulled (after asserting it
/// was measured — this is the native build).
fn normalized(result_json: &str) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(result_json).expect("result parses");
    let ws = v
        .get_mut("ledger")
        .and_then(|l| l.get_mut("wall_seconds"))
        .expect("ledger.wall_seconds present");
    assert!(ws.is_f64(), "native wall_seconds must be measured (Some)");
    *ws = serde_json::Value::Null;
    v
}

/// Each forwarded tick is the SSE `progress` payload shape.
fn assert_tick_shape(tick: &str) {
    let v: serde_json::Value = serde_json::from_str(tick).expect("tick parses");
    let fraction = v["fraction"].as_f64().expect("fraction is a number");
    assert!((0.0..=1.0).contains(&fraction), "fraction = {fraction}");
    assert!(v["metrics"].is_object(), "metrics is an object");
}

fn roundtrip<E>(exec: E, job_json: &str, run: impl Fn(&str, &mut dyn FnMut(&str)) -> String)
where
    E: Executor,
    E::Job: serde::de::DeserializeOwned,
{
    // Direct: deserialize and call the executor ourselves.
    let job: E::Job = serde_json::from_str(job_json).expect("job parses");
    let direct = exec.execute(&job, &mut |_| {});
    let direct_json = serde_json::to_string(&direct).expect("direct serializes");

    // Wrapped: through the exact function the wasm bindings call.
    let mut ticks = Vec::new();
    let wrapped_json = run(job_json, &mut |t: &str| ticks.push(t.to_string()));

    assert!(!ticks.is_empty(), "progress ticks were forwarded");
    for t in &ticks {
        assert_tick_shape(t);
    }
    assert_eq!(
        normalized(&direct_json),
        normalized(&wrapped_json),
        "JSON round-trip must equal the direct executor run"
    );
}

#[test]
fn stochastic_json_roundtrip_equals_direct() {
    roundtrip(
        tei_sim_stochastic::StochasticExecutor,
        STOCHASTIC_JOB,
        |j, cb| run_stochastic_json(j, cb),
    );
}

#[test]
fn stochastic_tempering_json_roundtrip_equals_direct() {
    roundtrip(
        tei_sim_stochastic::StochasticExecutor,
        TEMPERING_JOB,
        |j, cb| run_stochastic_json(j, cb),
    );
}

#[test]
fn adiabatic_json_roundtrip_equals_direct() {
    roundtrip(
        tei_sim_adiabatic::AdiabaticExecutor,
        ADIABATIC_JOB,
        |j, cb| run_adiabatic_json(j, cb),
    );
}

/// Petersen's Max-Cut optimum is 12 (closed form); 2000 annealing sweeps at
/// seed 42 must reach it — pins the actual numbers, not just self-consistency.
#[test]
fn stochastic_reaches_petersen_optimum() {
    let result = run_stochastic_json(STOCHASTIC_JOB, |_| {});
    let v: serde_json::Value = serde_json::from_str(&result).unwrap();
    assert_eq!(v["outputs"]["best_cut"].as_f64(), Some(12.0));
    assert_eq!(v["outputs"]["known_optimum"].as_f64(), Some(12.0));
}

/// Malformed jobs come back as `{"error": …}` JSON, never a panic.
#[test]
fn invalid_job_json_reports_error() {
    for bad in ["not json", "{}", r#"{"problem": {"kind": "klein_bottle"}}"#] {
        let v: serde_json::Value = serde_json::from_str(&run_stochastic_json(bad, |_| {})).unwrap();
        assert!(v["error"].is_string(), "{bad} -> {v}");
        let v: serde_json::Value = serde_json::from_str(&run_adiabatic_json(bad, |_| {})).unwrap();
        assert!(v["error"].is_string(), "{bad} -> {v}");
    }
}
