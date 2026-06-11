//! `AdiabaticExecutor` — the `tei_sim_core::exec::Executor` for the
//! adiabatic column: a `CellSpec` with its T/RC ratio list in, the overhead
//! curve plus fitted log-log slope out, with one progress tick per completed
//! ratio and the aggregate dissipation in `ledger.joules` (transient steps
//! land in `ledger.sweeps`, the generic iteration counter).

use crate::cells::CellSpec;
use crate::sweep::{E_LANDAUER_300K, SweepPoint, fitted_slope, sweep_with_progress};
use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;

/// Job spec accepted by the adiabatic executor (mirrors /api/execute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdiabaticJob {
    /// Which cell template to sweep.
    pub cell: CellSpec,
    /// T/RC ratios to run (ramp time in units of the switch time constant).
    pub ratios: Vec<f64>,
}

/// Executor for the adiabatic column.
pub struct AdiabaticExecutor;

impl Executor for AdiabaticExecutor {
    type Job = AdiabaticJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = tei_sim_core::exec::WallTimer::start();
        let n = job.ratios.len().max(1);
        let mut tick = |done: usize, p: &SweepPoint| {
            on_progress(Progress {
                fraction: done as f64 / n as f64,
                metrics: serde_json::json!({
                    "ratio": p.t_over_rc,
                    "e_diss_j": p.e_diss_j,
                    "e_ratio": p.e_over_abrupt,
                }),
            });
        };
        match sweep_with_progress(&job.cell, &job.ratios, &mut tick) {
            Ok(points) => {
                let curve_pairs: Vec<(f64, f64)> = points
                    .iter()
                    .map(|p| (p.t_over_rc, p.e_over_abrupt))
                    .collect();
                let slope = fitted_slope(&curve_pairs);
                let total_j: f64 = points.iter().map(|p| p.e_diss_j).sum();
                let total_steps: u64 = points.iter().map(|p| p.steps).sum();
                let ledger = EventLedger {
                    joules: total_j,
                    sweeps: total_steps,
                    wall_seconds: t0.elapsed_seconds(),
                    ..EventLedger::default()
                };
                ExecutionResult {
                    ledger,
                    outputs: serde_json::json!({
                        "curve": points.iter().map(|p| serde_json::json!({
                            "t_over_rc": p.t_over_rc,
                            "e_diss_j": p.e_diss_j,
                            "e_over_half_cv2": p.e_over_abrupt,
                            "e_over_landauer_300k": p.e_diss_j / E_LANDAUER_300K,
                        })).collect::<Vec<_>>(),
                        "fitted_loglog_slope": slope,
                        "abrupt_limit_j": job.cell.abrupt_limit_j(),
                        "cell": job.cell,
                        "params": {
                            "r_ohm": job.cell.r_ohm(),
                            "c_f": job.cell.c_f(),
                            "v_v": job.cell.v(),
                            "rc_s": job.cell.rc(),
                            "n_ratios": job.ratios.len(),
                        },
                    }),
                }
            }
            Err(e) => ExecutionResult {
                ledger: EventLedger {
                    wall_seconds: t0.elapsed_seconds(),
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

    /// A job deserializes from the documented JSON shape with cell defaults.
    #[test]
    fn job_deserializes_with_defaults() {
        let job: AdiabaticJob = serde_json::from_str(
            r#"{
                "cell": {"kind":"charge_recovery","r_ohm":1000.0,"c_f":1e-9,"v":1.0},
                "ratios": [1.0, 10.0, 100.0]
            }"#,
        )
        .unwrap();
        assert_eq!(job.ratios.len(), 3);
        let CellSpec::ChargeRecovery { hold_rc, .. } = job.cell else {
            panic!()
        };
        assert_eq!(hold_rc, 8.0);
    }

    /// An invalid job surfaces as an "error" output, not a panic.
    #[test]
    fn invalid_job_reports_error_output() {
        let job: AdiabaticJob = serde_json::from_str(
            r#"{"cell": {"kind":"ramp_charge","r_ohm":1000.0,"c_f":1e-9,"v":1.0}, "ratios": []}"#,
        )
        .unwrap();
        let res = AdiabaticExecutor.execute(&job, &mut |_| {});
        assert!(res.outputs.get("error").is_some());
        assert!(res.ledger.wall_seconds.is_some());
    }
}
