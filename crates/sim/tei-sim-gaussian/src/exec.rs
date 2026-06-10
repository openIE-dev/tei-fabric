//! GaussianExecutor — the `/api/execute` entry point for the Gaussian column.
//!
//! A [`GaussianJob`] is a circuit of symplectic ops applied to the N-mode
//! vacuum, optionally followed by repeated homodyne sampling of one
//! quadrature. Outputs: per-mode means/variances/photon numbers, the minimum
//! symplectic eigenvalue (physicality margin), homodyne sample statistics +
//! histogram, and an [`EventLedger`] whose `macs` field carries the
//! matrix-op flop estimate (each op costs two dense 2N×2N matmuls for
//! σ ← SσSᵀ plus one mat-vec; `spin_updates` is unused in this column).

use crate::{GaussianState, physicality_margin};
use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

/// One circuit element. Angles in radians; `theta` gives transmissivity
/// cos²θ on the beamsplitter; `(re, im)` is the displacement amplitude α.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "op", rename_all = "snake_case")]
pub enum GaussianOp {
    Squeeze {
        mode: usize,
        r: f64,
        #[serde(default)]
        phi: f64,
    },
    Rotate {
        mode: usize,
        phi: f64,
    },
    Displace {
        mode: usize,
        re: f64,
        #[serde(default)]
        im: f64,
    },
    Beamsplitter {
        mode_a: usize,
        mode_b: usize,
        theta: f64,
    },
    TwoModeSqueeze {
        mode_a: usize,
        mode_b: usize,
        r: f64,
    },
}

/// Repeated homodyne of x̂_φ on `mode`: each shot is an independent
/// preparation of the circuit output (the state is not consumed between
/// shots), so the sample stream draws from the marginal N(μ_φ, σ_φ).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct HomodyneSpec {
    pub mode: usize,
    #[serde(default)]
    pub phi: f64,
    pub shots: u64,
}

/// Job spec accepted by the Gaussian executor (mirrors /api/execute).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct GaussianJob {
    pub n_modes: usize,
    pub circuit: Vec<GaussianOp>,
    #[serde(default)]
    pub homodyne: Option<HomodyneSpec>,
    #[serde(default)]
    pub seed: u64,
}

pub struct GaussianExecutor;

const SHOT_BATCH: u64 = 4096;
const HIST_BINS: usize = 41;

fn apply_op(state: &mut GaussianState, op: &GaussianOp) {
    match *op {
        GaussianOp::Squeeze { mode, r, phi } => state.squeeze(mode, r, phi),
        GaussianOp::Rotate { mode, phi } => state.rotate(mode, phi),
        GaussianOp::Displace { mode, re, im } => state.displace(mode, re, im),
        GaussianOp::Beamsplitter {
            mode_a,
            mode_b,
            theta,
        } => state.beamsplit(mode_a, mode_b, theta),
        GaussianOp::TwoModeSqueeze { mode_a, mode_b, r } => {
            state.two_mode_squeeze(mode_a, mode_b, r)
        }
    }
}

/// Flop estimate for one symplectic op: two (2N)³ matmuls (σ ← SσSᵀ) plus a
/// (2N)² mat-vec. Displacement is 2 adds; counted as 2.
fn op_macs(op: &GaussianOp, n_modes: usize) -> u64 {
    let d = 2 * n_modes as u64;
    match op {
        GaussianOp::Displace { .. } => 2,
        _ => 2 * d * d * d + d * d,
    }
}

impl Executor for GaussianExecutor {
    type Job = GaussianJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = std::time::Instant::now();
        let mut ledger = EventLedger::default();
        let mut state = GaussianState::vacuum(job.n_modes);

        let shots = job.homodyne.as_ref().map(|h| h.shots).unwrap_or(0);
        let shot_batches = shots.div_ceil(SHOT_BATCH);
        let total_steps = (job.circuit.len() as u64 + shot_batches).max(1);

        // ── Circuit ──
        for (i, op) in job.circuit.iter().enumerate() {
            apply_op(&mut state, op);
            ledger.macs += op_macs(op, job.n_modes);
            on_progress(Progress {
                fraction: (i + 1) as f64 / total_steps as f64,
                metrics: serde_json::json!({ "stage": "circuit", "ops_applied": i + 1 }),
            });
        }

        // ── Physicality margin (ν_min − 1; ≥ 0 within roundoff for any
        //    state produced by symplectic ops on vacuum) ──
        let nu = crate::symplectic_eigenvalues(&state.cov);
        let nu_min = nu.iter().copied().fold(f64::INFINITY, f64::min);
        let margin = physicality_margin(&state.cov);

        // ── Per-mode moments ──
        let modes: Vec<serde_json::Value> = (0..state.n_modes())
            .map(|k| {
                let (mx, mp) = state.mode_mean(k);
                let (vx, vp) = state.quadrature_variances(k);
                serde_json::json!({
                    "mean_x": mx, "mean_p": mp,
                    "var_x": vx, "var_p": vp,
                    "mean_photons": state.mean_photon(k),
                })
            })
            .collect();

        // ── Homodyne sampling (independent preparations per shot) ──
        let homodyne_out = job.homodyne.as_ref().map(|spec| {
            let mut rng = Rng::new(job.seed);
            let (marg_mean, marg_var) = state.homodyne_marginal(spec.mode, spec.phi);
            let sd = marg_var.sqrt();
            let (lo, hi) = (marg_mean - 5.0 * sd, marg_mean + 5.0 * sd);
            let bin_w = (hi - lo) / HIST_BINS as f64;
            let mut hist = vec![0u64; HIST_BINS];
            let (mut sum, mut sumsq) = (0.0, 0.0);
            let (mut min, mut max) = (f64::INFINITY, f64::NEG_INFINITY);
            let mut done = 0u64;
            for batch in 0..shot_batches {
                let n = SHOT_BATCH.min(spec.shots - done);
                for _ in 0..n {
                    let m = marg_mean + sd * rng.normal();
                    sum += m;
                    sumsq += m * m;
                    min = min.min(m);
                    max = max.max(m);
                    let b = (((m - lo) / bin_w) as i64).clamp(0, HIST_BINS as i64 - 1);
                    hist[b as usize] += 1;
                }
                done += n;
                ledger.adc_samples += n;
                on_progress(Progress {
                    fraction: (job.circuit.len() as u64 + batch + 1) as f64 / total_steps as f64,
                    metrics: serde_json::json!({ "stage": "homodyne", "shots_done": done }),
                });
            }
            let n = spec.shots.max(1) as f64;
            let mean = sum / n;
            let var = (sumsq / n - mean * mean).max(0.0);
            serde_json::json!({
                "mode": spec.mode, "phi": spec.phi, "shots": spec.shots,
                "marginal_mean": marg_mean, "marginal_variance": marg_var,
                "sample_mean": mean, "sample_variance": var,
                "sample_min": min, "sample_max": max,
                "histogram": { "lo": lo, "hi": hi, "counts": hist },
            })
        });

        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());
        ExecutionResult {
            ledger,
            outputs: serde_json::json!({
                "n_modes": job.n_modes,
                "modes": modes,
                "symplectic_eigenvalues": nu,
                "symplectic_eig_min": nu_min,
                "physicality_margin": margin,
                "homodyne": homodyne_out,
            }),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn epr_job(shots: u64, seed: u64) -> GaussianJob {
        serde_json::from_value(serde_json::json!({
            "n_modes": 2,
            "circuit": [
                { "op": "two_mode_squeeze", "mode_a": 0, "mode_b": 1, "r": 0.5 },
                { "op": "displace", "mode": 0, "re": 0.25 },
            ],
            "homodyne": { "mode": 0, "phi": 0.0, "shots": shots },
            "seed": seed,
        }))
        .expect("job deserializes")
    }

    /// End-to-end: serde round-trip, progress monotone, physical output,
    /// homodyne marginal = cosh 2r, ledger counts ops and shots.
    #[test]
    fn executor_end_to_end() {
        let job = epr_job(10_000, 7);
        let mut fractions = Vec::new();
        let res = GaussianExecutor.execute(&job, &mut |p| fractions.push(p.fraction));
        assert!(fractions.windows(2).all(|w| w[0] <= w[1]));
        assert!((fractions.last().unwrap() - 1.0).abs() < 1e-12);
        let out = &res.outputs;
        assert!(out["physicality_margin"].as_f64().unwrap() >= -1e-12);
        let want_var = 1.0f64.cosh(); // cosh 2r, r = 0.5
        let got = out["homodyne"]["marginal_variance"].as_f64().unwrap();
        assert!((got - want_var).abs() < 1e-12);
        assert_eq!(res.ledger.adc_samples, 10_000);
        assert!(res.ledger.macs > 0);
    }
}
