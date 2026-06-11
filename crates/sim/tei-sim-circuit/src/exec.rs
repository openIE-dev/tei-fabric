//! `CircuitExecutor` — the `tei_sim_core::exec::Executor` for the circuit
//! column: serde job in, downsampled node traces + per-element energies +
//! event ledger out (`ledger.joules` = total dissipated, the number the
//! reversible/adiabatic cost dialect recalibrates against).

use crate::Method;
use crate::netlist::Netlist;
use crate::transient::{TransientOpts, transient_with_progress};
use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;

/// Job spec accepted by the circuit executor (mirrors /api/execute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CircuitJob {
    pub netlist: Netlist,
    /// Simulation end time [s] (snapped to the dt grid).
    pub t_stop: f64,
    /// Fixed step size [s].
    pub dt: f64,
    /// Integration method (default trapezoidal).
    #[serde(default)]
    pub method: Method,
    /// Cap on stored trace samples; traces are stride-downsampled to fit.
    /// Energy integration always uses every step.
    #[serde(default = "default_max_points")]
    pub max_points: usize,
}

fn default_max_points() -> usize {
    1024
}

/// Executor for the circuit column.
pub struct CircuitExecutor;

impl Executor for CircuitExecutor {
    type Job = CircuitJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = std::time::Instant::now();
        let n_steps = (job.t_stop / job.dt).round().max(1.0) as usize;
        let opts = TransientOpts {
            t_stop: job.t_stop,
            dt: job.dt,
            method: job.method,
            store_stride: n_steps.div_ceil(job.max_points.max(1)).max(1),
            solver: Default::default(),
        };
        match transient_with_progress(&job.netlist, &opts, on_progress) {
            Ok(res) => {
                let ledger = EventLedger {
                    joules: res.dissipated_energy,
                    wall_seconds: Some(t0.elapsed().as_secs_f64()),
                    ..EventLedger::default()
                };
                let energies: std::collections::BTreeMap<&str, f64> = res
                    .element_energy
                    .iter()
                    .map(|(n, e)| (n.as_str(), *e))
                    .collect();
                ExecutionResult {
                    ledger,
                    outputs: serde_json::json!({
                        "t": res.t,
                        "nodes": res.v,
                        "element_energy_j": energies,
                        "source_energy_j": res.source_energy,
                        "dissipated_j": res.dissipated_energy,
                        "delta_stored_j": res.delta_stored_energy,
                        "tellegen_max_w": res.tellegen_max,
                        "steps": res.steps,
                        "dt": res.dt,
                        "method": job.method,
                        "solver": res.solver,
                    }),
                }
            }
            Err(e) => ExecutionResult {
                ledger: EventLedger {
                    wall_seconds: Some(t0.elapsed().as_secs_f64()),
                    ..EventLedger::default()
                },
                outputs: serde_json::json!({ "error": e.to_string() }),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A job deserializes from the documented JSON shape with defaults.
    #[test]
    fn job_deserializes_with_defaults() {
        let job: CircuitJob = serde_json::from_str(
            r#"{
                "netlist": { "elements": [
                    {"kind":"voltage_source","name":"vs","p":1,"n":0,"wave":{"shape":"dc","v":1.0}},
                    {"kind":"resistor","name":"r1","p":1,"n":2,"r":1000.0},
                    {"kind":"capacitor","name":"c1","p":2,"n":0,"c":1e-9}
                ]},
                "t_stop": 1e-5,
                "dt": 1e-8
            }"#,
        )
        .unwrap();
        assert_eq!(job.method, Method::Trapezoidal);
        assert_eq!(job.max_points, 1024);
        assert_eq!(job.netlist.elements.len(), 3);
    }

    /// An invalid netlist surfaces as an "error" output, not a panic.
    #[test]
    fn invalid_job_reports_error_output() {
        let job: CircuitJob =
            serde_json::from_str(r#"{"netlist": {"elements": []}, "t_stop": 1.0, "dt": 0.1}"#)
                .unwrap();
        let res = CircuitExecutor.execute(&job, &mut |_| {});
        assert!(res.outputs.get("error").is_some());
    }
}
