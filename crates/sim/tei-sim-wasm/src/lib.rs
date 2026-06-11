//! tei-sim-wasm — wasm-bindgen bindings for the tei-sim executors.
//!
//! The roadmap's §1 point 3 made concrete: every sim core compiles to
//! `wasm32-unknown-unknown`, so small instances run client-side with **the
//! exact same numerics** as the server. The RNG is hand-rolled
//! (splitmix64 + xoshiro256++) and WASM-identical, so a job with the same
//! seed produces bit-identical outputs in the browser and on the server.
//!
//! # JS API (wasm-bindgen, `wasm-pack build --target web`)
//!
//! ```js
//! import init, { run_stochastic, run_adiabatic, version }
//!   from "/path/to/pkg/tei_sim_wasm.js";
//!
//! await init();                       // fetch + instantiate the .wasm once
//!
//! const job = JSON.stringify({        // StochasticJob — the /api/execute
//!   problem:  { kind: "petersen" },   // payload *without* the "substrate" tag
//!   schedule: { sweeps: 2000, beta0: 0.1, beta1: 5.0 },
//!   seed: 42,
//! });
//! const result = JSON.parse(run_stochastic(job, (tick) => {
//!   const { fraction, metrics } = JSON.parse(tick);  // same shape as the
//!   // SSE "progress" event: { fraction, metrics: { sweep, cut, ... } }
//! }));
//! // result is the SSE "result" event payload: { ledger, outputs }
//! // (no "calibration" key, and ledger.wall_seconds is null on wasm).
//! ```
//!
//! `run_adiabatic` is identical with an `AdiabaticJob`:
//! `{"cell": {"kind": "charge_recovery", "r_ohm": 1000, "c_f": 1e-9,
//! "v": 1.0}, "ratios": [1, 10, 100]}`.
//!
//! Execution is synchronous — the progress callback fires inline. Run it in
//! a Web Worker to keep the main thread responsive on larger jobs.
//!
//! Malformed job JSON never throws: the returned JSON is
//! `{"error": "..."}`, mirroring how the executors report invalid jobs.
//!
//! # Build
//!
//! ```text
//! wasm-pack build crates/sim/tei-sim-wasm --target web --release
//! ```
//!
//! The plain-Rust `*_json` functions below are the single implementation;
//! the `#[wasm_bindgen]` exports are a thin wasm32-only shim over them, so
//! native tests exercise exactly the code the browser runs.

use serde::de::DeserializeOwned;
use tei_sim_core::exec::{Executor, Progress};

/// Crate version (also exported to JS as `version()`).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

/// Deserialize a job, run an executor, return the `ExecutionResult` as JSON
/// — `{"ledger": …, "outputs": …}`, the SSE `result` event payload.
/// Progress ticks are forwarded to `on_progress` as
/// `{"fraction": …, "metrics": …}` JSON strings, the SSE `progress` payload.
/// Errors come back as `{"error": "…"}` JSON, never a panic.
fn run_json<E>(exec: E, job_json: &str, mut on_progress: impl FnMut(&str)) -> String
where
    E: Executor,
    E::Job: DeserializeOwned,
{
    let job: E::Job = match serde_json::from_str(job_json) {
        Ok(job) => job,
        Err(e) => {
            return serde_json::json!({ "error": format!("invalid job JSON: {e}") }).to_string();
        }
    };
    let mut tick = |p: Progress| {
        let payload = serde_json::json!({ "fraction": p.fraction, "metrics": p.metrics });
        on_progress(&payload.to_string());
    };
    let result = exec.execute(&job, &mut tick);
    match serde_json::to_string(&result) {
        Ok(s) => s,
        Err(e) => serde_json::json!({ "error": format!("serialize result: {e}") }).to_string(),
    }
}

/// Run a [`tei_sim_stochastic::StochasticJob`] (JSON in, JSON out).
pub fn run_stochastic_json(job_json: &str, on_progress: impl FnMut(&str)) -> String {
    run_json(
        tei_sim_stochastic::StochasticExecutor,
        job_json,
        on_progress,
    )
}

/// Run a [`tei_sim_adiabatic::AdiabaticJob`] (JSON in, JSON out).
pub fn run_adiabatic_json(job_json: &str, on_progress: impl FnMut(&str)) -> String {
    run_json(tei_sim_adiabatic::AdiabaticExecutor, job_json, on_progress)
}

/// Fixture jobs shared by the native reference dumper
/// (`examples/reference.rs`), the wasm identity test (`tests/identity.rs`)
/// and the manual Node check (`check.mjs` keeps its own literal copy).
/// Entries are `(name, column, job_json)` with column ∈
/// {"stochastic", "adiabatic"}.
#[doc(hidden)]
pub const FIXTURE_JOBS: &[(&str, &str, &str)] = &[
    (
        "stochastic_petersen",
        "stochastic",
        r#"{"problem":{"kind":"petersen"},"schedule":{"sweeps":2000,"beta0":0.1,"beta1":5.0},"seed":42}"#,
    ),
    (
        "stochastic_tempering_rr40",
        "stochastic",
        r#"{"problem":{"kind":"random_regular","n":40,"degree":3,"seed":7},"schedule":{"sweeps":1500,"beta0":0.1,"beta1":5.0},"seed":42,"tempering":{"replicas":4,"beta_min":0.1,"beta_max":6.0,"swap_interval":10}}"#,
    ),
    (
        "adiabatic_charge_recovery",
        "adiabatic",
        r#"{"cell":{"kind":"charge_recovery","r_ohm":1000.0,"c_f":1e-9,"v":1.0},"ratios":[1.0,3.1623,10.0,31.623,100.0,316.23,1000.0]}"#,
    ),
];

/// Run one [`FIXTURE_JOBS`] entry by column name.
#[doc(hidden)]
pub fn run_fixture(column: &str, job_json: &str, on_progress: impl FnMut(&str)) -> String {
    match column {
        "stochastic" => run_stochastic_json(job_json, on_progress),
        "adiabatic" => run_adiabatic_json(job_json, on_progress),
        other => panic!("unknown fixture column {other}"),
    }
}

#[cfg(target_arch = "wasm32")]
mod bindings {
    use wasm_bindgen::prelude::*;

    /// Adapt a JS callback to the `FnMut(&str)` the runners take. Callback
    /// exceptions are swallowed — a broken progress plot must not abort the
    /// simulation.
    fn forward(cb: &js_sys::Function) -> impl FnMut(&str) + '_ {
        move |json: &str| {
            let _ = cb.call1(&JsValue::NULL, &JsValue::from_str(json));
        }
    }

    /// `run_stochastic(jobJson, (tickJson) => {}) -> resultJson`
    #[wasm_bindgen]
    pub fn run_stochastic(job_json: &str, on_progress: &js_sys::Function) -> String {
        crate::run_stochastic_json(job_json, forward(on_progress))
    }

    /// `run_adiabatic(jobJson, (tickJson) => {}) -> resultJson`
    #[wasm_bindgen]
    pub fn run_adiabatic(job_json: &str, on_progress: &js_sys::Function) -> String {
        crate::run_adiabatic_json(job_json, forward(on_progress))
    }

    /// Crate version string.
    #[wasm_bindgen]
    pub fn version() -> String {
        crate::VERSION.to_string()
    }
}
