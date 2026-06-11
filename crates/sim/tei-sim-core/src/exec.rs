//! Executor contract for the sim layer.
//!
//! Each sim crate defines its own typed job and implements `Executor`. The
//! progress callback streams intermediate metrics (the server forwards them
//! over SSE; the browser draws them live).

use crate::ledger::EventLedger;
use serde::Serialize;

/// A progress tick emitted mid-run.
#[derive(Debug, Clone, Serialize)]
pub struct Progress {
    /// Fraction complete in [0, 1].
    pub fraction: f64,
    /// Free-form metrics for live plots (e.g. {"sweep": 1200, "best_cut": 87}).
    pub metrics: serde_json::Value,
}

/// Final result of an execution.
#[derive(Debug, Clone, Serialize)]
pub struct ExecutionResult {
    /// What physically happened — feeds cost-dialect recalibration.
    pub ledger: EventLedger,
    /// Column-specific outputs (best state, cut value, samples, …).
    pub outputs: serde_json::Value,
}

/// The contract every simulator implements.
pub trait Executor {
    type Job;
    /// Run the job; call `on_progress` at a sensible cadence (the
    /// implementation decides — typically every few hundred sweeps/events).
    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult;
}

/// Wall-clock timer for `EventLedger::wall_seconds` that degrades cleanly on
/// `wasm32-unknown-unknown`, where `std::time::Instant::now()` panics
/// (no monotonic clock). Natively `elapsed_seconds()` measures real wall
/// time; on wasm32 it returns `None` — exactly the ledger's documented
/// "None on wasm" contract.
#[derive(Debug, Clone, Copy)]
pub struct WallTimer {
    #[cfg(not(target_arch = "wasm32"))]
    start: std::time::Instant,
}

impl WallTimer {
    /// Start the timer (a no-op on wasm32).
    #[must_use]
    pub fn start() -> Self {
        Self {
            #[cfg(not(target_arch = "wasm32"))]
            start: std::time::Instant::now(),
        }
    }

    /// Seconds since `start()` — `Some` natively, `None` on wasm32.
    #[must_use]
    pub fn elapsed_seconds(&self) -> Option<f64> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            Some(self.start.elapsed().as_secs_f64())
        }
        #[cfg(target_arch = "wasm32")]
        {
            None
        }
    }
}
