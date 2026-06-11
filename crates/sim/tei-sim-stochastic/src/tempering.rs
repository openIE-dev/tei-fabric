//! Parallel tempering (replica exchange) on top of the chromatic Gibbs sampler.
//!
//! K replicas of the same Ising model run at fixed inverse temperatures
//! β₀ < β₁ < … < β_{K−1} (the ladder). Every `swap_interval` sweeps an
//! exchange round attempts adjacent-pair state swaps with the Metropolis rule
//!
//!   P(accept) = min(1, exp(Δβ·ΔE)),  Δβ = β_{k+1} − β_k,  ΔE = E_{k+1} − E_k,
//!
//! which preserves the product distribution Π_k π_{β_k}(s_k) — detailed
//! balance on the joint chain, so every rung keeps its exact Boltzmann
//! marginal. Hot replicas cross energy barriers; cold replicas exploit;
//! accepted exchanges hand good configurations down the ladder. Even pairs
//! (0,1)(2,3)… swap on even rounds, odd pairs (1,2)(3,4)… on odd rounds.
//!
//! Determinism: each replica owns a sub-seeded RNG (master stream order:
//! replica 0..K−1, then the dedicated swap RNG last), replicas advance
//! independently between exchange rounds (rayon-parallel, no shared mutable
//! state), and the exchange phase is sequential — results are bit-identical
//! across runs and across thread counts.

use crate::{GibbsSampler, IsingModel, TracePoint};
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use tei_sim_core::exec::Progress;
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

/// β-ladder interpolation between `beta_min` and `beta_max`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Ladder {
    #[default]
    Geometric,
    Linear,
}

/// Replica-exchange configuration (mirrors /api/execute JSON).
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct TemperingSpec {
    /// Number of replicas K (ladder rungs).
    #[serde(default = "default_replicas")]
    pub replicas: usize,
    /// Hottest inverse temperature (rung 0).
    #[serde(default = "default_beta_min")]
    pub beta_min: f64,
    /// Coldest inverse temperature (rung K−1).
    #[serde(default = "default_beta_max")]
    pub beta_max: f64,
    /// Sweeps between exchange rounds.
    #[serde(default = "default_swap_interval")]
    pub swap_interval: u64,
    /// Ladder interpolation: geometric (default) or linear in β.
    #[serde(default)]
    pub ladder: Ladder,
}

fn default_replicas() -> usize {
    8
}
fn default_beta_min() -> f64 {
    0.1
}
fn default_beta_max() -> f64 {
    6.0
}
fn default_swap_interval() -> u64 {
    10
}

impl Default for TemperingSpec {
    fn default() -> Self {
        Self {
            replicas: default_replicas(),
            beta_min: default_beta_min(),
            beta_max: default_beta_max(),
            swap_interval: default_swap_interval(),
            ladder: Ladder::default(),
        }
    }
}

impl TemperingSpec {
    /// Ascending β ladder with K rungs spanning [beta_min, beta_max].
    pub fn betas(&self) -> Vec<f64> {
        assert!(self.replicas >= 1, "tempering needs at least one replica");
        assert!(
            self.beta_min > 0.0 && self.beta_max >= self.beta_min,
            "ladder requires 0 < beta_min ≤ beta_max"
        );
        let k = self.replicas;
        if k == 1 {
            return vec![self.beta_max];
        }
        (0..k)
            .map(|i| {
                let t = i as f64 / (k - 1) as f64;
                match self.ladder {
                    Ladder::Linear => self.beta_min + (self.beta_max - self.beta_min) * t,
                    Ladder::Geometric => self.beta_min * (self.beta_max / self.beta_min).powf(t),
                }
            })
            .collect()
    }
}

/// One recorded exchange attempt (for diagnostics and validation).
#[derive(Debug, Clone, Copy, Serialize)]
pub struct SwapDecision {
    /// Exchange round index.
    pub round: u64,
    /// Lower rung of the attempted pair (k, k+1).
    pub pair: usize,
    /// Log acceptance ratio x = Δβ·ΔE — accept iff u < eˣ, u ∈ [0,1).
    pub log_ratio: f64,
    pub accepted: bool,
}

/// Aggregate exchange statistics.
#[derive(Debug, Clone, Default, Serialize)]
pub struct SwapStats {
    pub attempts: u64,
    pub accepts: u64,
    pub per_pair_attempts: Vec<u64>,
    pub per_pair_accepts: Vec<u64>,
    /// Full decision log (one entry per attempted pair swap).
    pub decisions: Vec<SwapDecision>,
}

impl SwapStats {
    pub fn acceptance_overall(&self) -> f64 {
        if self.attempts == 0 {
            0.0
        } else {
            self.accepts as f64 / self.attempts as f64
        }
    }

    /// Acceptance rate per adjacent pair, length K−1 (0.0 where unattempted).
    pub fn per_pair_acceptance(&self) -> Vec<f64> {
        self.per_pair_attempts
            .iter()
            .zip(&self.per_pair_accepts)
            .map(|(&a, &acc)| if a == 0 { 0.0 } else { acc as f64 / a as f64 })
            .collect()
    }
}

struct Replica {
    state: Vec<i8>,
    energy: f64,
    rng: Rng,
    ledger: EventLedger,
    best_energy: f64,
    best_state: Vec<i8>,
}

/// Replica-exchange engine: K fixed-β Gibbs chains plus the exchange kernel.
pub struct Tempering {
    pub betas: Vec<f64>,
    pub stats: SwapStats,
    replicas: Vec<Replica>,
    sampler: GibbsSampler,
    swap_rng: Rng,
    round: u64,
}

impl Tempering {
    /// Spin updates per chunk below which `advance` skips the rayon pool.
    #[cfg(feature = "parallel")]
    const PAR_THRESHOLD_UPDATES: u64 = 4096;

    pub fn new(model: &IsingModel, spec: &TemperingSpec, seed: u64) -> Self {
        let betas = spec.betas();
        let k = betas.len();
        // Sub-seed one stream per replica from the master, swap RNG last.
        let mut master = Rng::new(seed);
        let replica_seeds: Vec<u64> = (0..k).map(|_| master.next_u64()).collect();
        let swap_rng = Rng::new(master.next_u64());

        let replicas: Vec<Replica> = replica_seeds
            .iter()
            .map(|&rs| {
                let mut rng = Rng::new(rs);
                let state: Vec<i8> = (0..model.n)
                    .map(|_| if rng.bernoulli(0.5) { 1 } else { -1 })
                    .collect();
                let energy = model.energy(&state);
                Replica {
                    best_energy: energy,
                    best_state: state.clone(),
                    state,
                    energy,
                    rng,
                    ledger: EventLedger::default(),
                }
            })
            .collect();

        Self {
            stats: SwapStats {
                per_pair_attempts: vec![0; k.saturating_sub(1)],
                per_pair_accepts: vec![0; k.saturating_sub(1)],
                ..SwapStats::default()
            },
            betas,
            replicas,
            sampler: GibbsSampler::new(model),
            swap_rng,
            round: 0,
        }
    }

    pub fn replica_count(&self) -> usize {
        self.replicas.len()
    }

    /// Current state at rung `k`.
    pub fn state(&self, k: usize) -> &[i8] {
        &self.replicas[k].state
    }

    /// Current energy at rung `k`.
    pub fn energy(&self, k: usize) -> f64 {
        self.replicas[k].energy
    }

    /// Best (lowest-energy) configuration seen across all rungs.
    pub fn best(&self) -> (f64, &[i8]) {
        let mut bi = 0;
        for k in 1..self.replicas.len() {
            if self.replicas[k].best_energy < self.replicas[bi].best_energy {
                bi = k;
            }
        }
        (self.replicas[bi].best_energy, &self.replicas[bi].best_state)
    }

    /// Per-replica event ledgers merged into one (sweeps = K × per-replica).
    pub fn merged_ledger(&self) -> EventLedger {
        let mut total = EventLedger::default();
        for r in &self.replicas {
            total.merge(&r.ledger);
        }
        total
    }

    /// Advance every replica `sweeps` sweeps at its fixed β. Replicas touch
    /// only their own state/RNG/ledger, so the execution schedule cannot
    /// change the result — bit-identical at any thread count, and identical
    /// with the `parallel` feature disabled (the wasm32 build). Under
    /// `parallel`, chunks below `PAR_THRESHOLD_UPDATES` spin updates run
    /// serially (identical output, no rayon dispatch overhead on tiny
    /// models).
    pub fn advance(&mut self, model: &IsingModel, sweeps: u64) {
        let sampler = &self.sampler;
        let betas = &self.betas;
        let advance_one = |k: usize, r: &mut Replica| {
            let beta = betas[k];
            for _ in 0..sweeps {
                sampler.sweep(model, &mut r.state, beta, &mut r.rng, &mut r.ledger);
                r.energy = model.energy(&r.state);
                if r.energy < r.best_energy {
                    r.best_energy = r.energy;
                    r.best_state.copy_from_slice(&r.state);
                }
            }
        };
        #[cfg(feature = "parallel")]
        {
            let work = model.n as u64 * sweeps * self.replicas.len() as u64;
            if work >= Self::PAR_THRESHOLD_UPDATES {
                self.replicas
                    .par_iter_mut()
                    .enumerate()
                    .for_each(|(k, r)| advance_one(k, r));
                return;
            }
        }
        for (k, r) in self.replicas.iter_mut().enumerate() {
            advance_one(k, r);
        }
    }

    /// One exchange round (sequential, dedicated RNG): even pairs on even
    /// rounds, odd pairs on odd rounds. Accepted swaps exchange states and
    /// energies; the βs stay attached to their rungs.
    pub fn swap_round(&mut self) {
        let mut k = (self.round % 2) as usize;
        while k + 1 < self.replicas.len() {
            let dbeta = self.betas[k + 1] - self.betas[k];
            let de = self.replicas[k + 1].energy - self.replicas[k].energy;
            let x = dbeta * de;
            // u ∈ [0,1) ⇒ u < eˣ is exactly min(1, eˣ): x ≥ 0 always accepts.
            let u = self.swap_rng.f64();
            let accepted = u < x.exp();
            self.stats.attempts += 1;
            self.stats.per_pair_attempts[k] += 1;
            if accepted {
                self.stats.accepts += 1;
                self.stats.per_pair_accepts[k] += 1;
                let (lo, hi) = self.replicas.split_at_mut(k + 1);
                std::mem::swap(&mut lo[k].state, &mut hi[0].state);
                std::mem::swap(&mut lo[k].energy, &mut hi[0].energy);
            }
            self.stats.decisions.push(SwapDecision {
                round: self.round,
                pair: k,
                log_ratio: x,
                accepted,
            });
            k += 2;
        }
        self.round += 1;
    }
}

/// Outcome of a parallel-tempering run.
#[derive(Debug, Clone, Serialize)]
pub struct TemperOutcome {
    pub best_energy: f64,
    pub best_state: Vec<i8>,
    pub trace: Vec<TracePoint>,
    pub ledger: EventLedger,
    pub stats: SwapStats,
    pub betas: Vec<f64>,
}

/// Run parallel tempering: `sweeps_per_replica` sweeps on each of the K
/// rungs, exchange rounds every `spec.swap_interval` sweeps. Trace points
/// (and progress ticks) land every `trace_every` sweeps, chunk-aligned.
pub fn parallel_temper(
    model: &IsingModel,
    spec: &TemperingSpec,
    sweeps_per_replica: u64,
    seed: u64,
    trace_every: u64,
    mut on_progress: Option<&mut dyn FnMut(Progress)>,
) -> TemperOutcome {
    let mut pt = Tempering::new(model, spec, seed);
    let interval = spec.swap_interval.max(1);
    let trace_every = trace_every.max(1);
    let total = sweeps_per_replica;

    let mut trace = Vec::new();
    let mut next_trace = trace_every;
    let mut done = 0u64;
    while done < total {
        let chunk = interval.min(total - done);
        pt.advance(model, chunk);
        done += chunk;
        if done % interval == 0 {
            pt.swap_round();
        }
        if done >= next_trace || done == total {
            while next_trace <= done {
                next_trace += trace_every;
            }
            let energy = (0..pt.replica_count())
                .map(|k| pt.energy(k))
                .fold(f64::INFINITY, f64::min);
            let (best_energy, _) = pt.best();
            trace.push(TracePoint {
                sweep: done,
                energy,
                best_energy,
            });
            if let Some(cb) = on_progress.as_deref_mut() {
                cb(Progress {
                    fraction: done as f64 / total as f64,
                    metrics: serde_json::json!({
                        "sweep": done,
                        "energy": energy,
                        "best_energy": best_energy,
                        "swap_acceptance_rate": pt.stats.acceptance_overall(),
                        "replica_betas": pt.betas,
                    }),
                });
            }
        }
    }

    let (best_energy, best_state) = pt.best();
    let best_state = best_state.to_vec();
    TemperOutcome {
        best_energy,
        best_state,
        trace,
        ledger: pt.merged_ledger(),
        stats: pt.stats,
        betas: pt.betas,
    }
}
