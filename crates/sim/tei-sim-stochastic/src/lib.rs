//! tei-sim-stochastic — thrml-class (Extropic) Ising / EBM sampler.
//!
//! Functional simulator for the stochastic substrate column: sparse Ising
//! energy-based models sampled by **chromatic block Gibbs** — the graph is
//! colored once, then every spin in a color class updates simultaneously
//! (no two share an edge, so the conditional distributions stay exact).
//! Simulated-annealing schedules layer on top for optimization workloads.
//!
//! Energy convention:  E(s) = −Σᵢ hᵢ sᵢ − Σ_{i<j} J_{ij} sᵢ sⱼ ,  s ∈ {−1,+1}ⁿ
//! Gibbs conditional:  p(sᵢ=+1 | rest) = σ(2β ℓᵢ),  ℓᵢ = hᵢ + Σⱼ J_{ij} sⱼ
//!
//! Validation: analytic + published only — exact enumeration ≤ 20 spins,
//! Boltzmann χ², chromatic ≡ sequential, Onsager's T_c on the 2D lattice,
//! closed-form Max-Cut optima (Kₙ, cycles, Petersen, K_{a,b}).

pub mod graphs;
pub mod maxcut;
pub mod tempering;

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;
use tei_sim_core::sparse::{Csr, greedy_coloring};

/// A sparse Ising model: fields `h` and symmetric couplings `J`.
#[derive(Debug, Clone)]
pub struct IsingModel {
    pub n: usize,
    pub h: Vec<f64>,
    /// Symmetric adjacency with coupling values — both (i,j) and (j,i) stored.
    pub j: Csr,
}

impl IsingModel {
    /// Build from fields and an undirected edge list (i, j, J_ij).
    pub fn new(h: Vec<f64>, edges: &[(u32, u32, f64)]) -> Self {
        let n = h.len();
        let mut triplets = Vec::with_capacity(edges.len() * 2);
        for &(i, j, w) in edges {
            assert_ne!(i, j, "self-coupling not allowed");
            triplets.push((i, j, w));
            triplets.push((j, i, w));
        }
        Self {
            n,
            h,
            j: Csr::from_triplets(n, n, &triplets),
        }
    }

    /// E(s) = −Σ h s − Σ_{i<j} J s s.
    pub fn energy(&self, s: &[i8]) -> f64 {
        let mut e = 0.0;
        for i in 0..self.n {
            e -= self.h[i] * s[i] as f64;
            let (cols, vals) = self.j.row(i);
            for (c, w) in cols.iter().zip(vals) {
                let jj = *c as usize;
                if jj > i {
                    e -= w * (s[i] as f64) * (s[jj] as f64);
                }
            }
        }
        e
    }

    /// Local field ℓᵢ = hᵢ + Σⱼ J_{ij} sⱼ.
    #[inline]
    pub fn local_field(&self, i: usize, s: &[i8]) -> f64 {
        let (cols, vals) = self.j.row(i);
        let mut l = self.h[i];
        for (c, w) in cols.iter().zip(vals) {
            l += w * s[*c as usize] as f64;
        }
        l
    }
}

/// Chromatic Gibbs sampler over an Ising model.
pub struct GibbsSampler {
    /// Spin order grouped by color class (color-class boundaries in `starts`).
    class_members: Vec<u32>,
    class_starts: Vec<usize>,
}

impl GibbsSampler {
    pub fn new(model: &IsingModel) -> Self {
        let (colors, n_colors) = greedy_coloring(&model.j);
        let mut class_members = Vec::with_capacity(model.n);
        let mut class_starts = Vec::with_capacity(n_colors as usize + 1);
        class_starts.push(0);
        for color in 0..n_colors {
            for (i, &c) in colors.iter().enumerate() {
                if c == color {
                    class_members.push(i as u32);
                }
            }
            class_starts.push(class_members.len());
        }
        Self {
            class_members,
            class_starts,
        }
    }

    /// One full sweep: every color class in sequence; within a class, every
    /// spin's conditional depends only on other classes, so the class update
    /// is exact regardless of order (and parallelizable).
    pub fn sweep(
        &self,
        model: &IsingModel,
        s: &mut [i8],
        beta: f64,
        rng: &mut Rng,
        ledger: &mut EventLedger,
    ) {
        for w in self.class_starts.windows(2) {
            for &iu in &self.class_members[w[0]..w[1]] {
                let i = iu as usize;
                let l = model.local_field(i, s);
                // p(s_i = +1) = σ(2βℓ)
                let p_up = 1.0 / (1.0 + (-2.0 * beta * l).exp());
                let new = if rng.f64() < p_up { 1i8 } else { -1i8 };
                ledger.spin_updates += 1;
                if new != s[i] {
                    s[i] = new;
                    ledger.flips += 1;
                }
            }
        }
        ledger.sweeps += 1;
    }
}

/// Annealing schedule: inverse temperature β swept from `beta0` to `beta1`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Schedule {
    pub sweeps: u64,
    pub beta0: f64,
    pub beta1: f64,
    /// "linear" or "geometric" interpolation in β.
    #[serde(default = "default_kind")]
    pub kind: String,
}

fn default_kind() -> String {
    "geometric".to_string()
}

impl Schedule {
    pub fn beta_at(&self, sweep: u64) -> f64 {
        if self.sweeps <= 1 {
            return self.beta1;
        }
        let t = sweep as f64 / (self.sweeps - 1) as f64;
        match self.kind.as_str() {
            "linear" => self.beta0 + (self.beta1 - self.beta0) * t,
            _ => self.beta0 * (self.beta1 / self.beta0).powf(t),
        }
    }
}

/// A point on the annealing trace.
#[derive(Debug, Clone, Serialize)]
pub struct TracePoint {
    pub sweep: u64,
    pub energy: f64,
    pub best_energy: f64,
}

/// Outcome of an annealing run.
#[derive(Debug, Clone, Serialize)]
pub struct AnnealOutcome {
    pub best_energy: f64,
    pub best_state: Vec<i8>,
    pub trace: Vec<TracePoint>,
    pub ledger: EventLedger,
}

/// Run simulated annealing on a model. `trace_every` controls trace density.
pub fn anneal(
    model: &IsingModel,
    schedule: &Schedule,
    seed: u64,
    trace_every: u64,
    mut on_progress: Option<&mut dyn FnMut(Progress)>,
) -> AnnealOutcome {
    let mut rng = Rng::new(seed);
    let sampler = GibbsSampler::new(model);
    let mut ledger = EventLedger::default();

    // Random initial state.
    let mut s: Vec<i8> = (0..model.n)
        .map(|_| if rng.bernoulli(0.5) { 1 } else { -1 })
        .collect();
    let mut energy = model.energy(&s);
    let mut best_energy = energy;
    let mut best_state = s.clone();
    let mut trace = Vec::new();

    for sweep in 0..schedule.sweeps {
        let beta = schedule.beta_at(sweep);
        sampler.sweep(model, &mut s, beta, &mut rng, &mut ledger);
        energy = model.energy(&s);
        if energy < best_energy {
            best_energy = energy;
            best_state.copy_from_slice(&s);
        }
        if sweep % trace_every == 0 || sweep + 1 == schedule.sweeps {
            trace.push(TracePoint {
                sweep,
                energy,
                best_energy,
            });
            if let Some(cb) = on_progress.as_deref_mut() {
                cb(Progress {
                    fraction: (sweep + 1) as f64 / schedule.sweeps as f64,
                    metrics: serde_json::json!({
                        "sweep": sweep,
                        "energy": energy,
                        "best_energy": best_energy,
                        "beta": beta,
                    }),
                });
            }
        }
    }

    AnnealOutcome {
        best_energy,
        best_state,
        trace,
        ledger,
    }
}

/// Fixed-temperature sampling: burn-in then measure ⟨E⟩ and ⟨|m|⟩.
pub fn sample_observables(
    model: &IsingModel,
    beta: f64,
    burn_in: u64,
    measure: u64,
    seed: u64,
) -> (f64, f64) {
    let mut rng = Rng::new(seed);
    let sampler = GibbsSampler::new(model);
    let mut ledger = EventLedger::default();
    let mut s: Vec<i8> = (0..model.n)
        .map(|_| if rng.bernoulli(0.5) { 1 } else { -1 })
        .collect();
    for _ in 0..burn_in {
        sampler.sweep(model, &mut s, beta, &mut rng, &mut ledger);
    }
    let (mut e_sum, mut m_sum) = (0.0, 0.0);
    for _ in 0..measure {
        sampler.sweep(model, &mut s, beta, &mut rng, &mut ledger);
        e_sum += model.energy(&s);
        let m: i64 = s.iter().map(|&x| x as i64).sum();
        m_sum += (m as f64 / model.n as f64).abs();
    }
    (e_sum / measure as f64, m_sum / measure as f64)
}

// ───────────────────────────── Executor ─────────────────────────────

/// Job spec accepted by the stochastic executor (mirrors /api/execute).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StochasticJob {
    pub problem: maxcut::ProblemSpec,
    pub schedule: Schedule,
    #[serde(default)]
    pub seed: u64,
    /// Optional replica exchange. When present, `schedule.sweeps` is the
    /// per-replica budget and the ladder overrides `beta0`/`beta1`.
    #[serde(default)]
    pub tempering: Option<tempering::TemperingSpec>,
}

pub struct StochasticExecutor;

impl Executor for StochasticExecutor {
    type Job = StochasticJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let graph = job.problem.build();
        let model = maxcut::to_ising(&graph);
        let total_weight: f64 = graph.edges.iter().map(|e| e.2).sum();

        let t0 = std::time::Instant::now();
        // Wrap progress so the browser sees cut values, not raw energies.
        let mut wrapped = |p: Progress| {
            let mut metrics = p.metrics.clone();
            if let (Some(e), Some(be)) = (
                metrics.get("energy").and_then(|v| v.as_f64()),
                metrics.get("best_energy").and_then(|v| v.as_f64()),
            ) {
                metrics["cut"] = serde_json::json!((total_weight - e) / 2.0);
                metrics["best_cut"] = serde_json::json!((total_weight - be) / 2.0);
            }
            on_progress(Progress {
                fraction: p.fraction,
                metrics,
            });
        };
        let trace_every = (job.schedule.sweeps / 200).max(1);

        if let Some(spec) = &job.tempering {
            let mut outcome = tempering::parallel_temper(
                &model,
                spec,
                job.schedule.sweeps,
                job.seed,
                trace_every,
                Some(&mut wrapped),
            );
            outcome.ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());
            let best_cut = (total_weight - outcome.best_energy) / 2.0;
            return ExecutionResult {
                ledger: outcome.ledger.clone(),
                outputs: serde_json::json!({
                    "best_cut": best_cut,
                    "total_weight": total_weight,
                    "best_energy": outcome.best_energy,
                    "n_nodes": model.n,
                    "n_edges": graph.edges.len(),
                    "known_optimum": job.problem.known_optimum(),
                    "replica_count": outcome.betas.len(),
                    "replica_betas": outcome.betas,
                    "swap_attempts": outcome.stats.attempts,
                    "swap_accepts": outcome.stats.accepts,
                    "swap_acceptance_overall": outcome.stats.acceptance_overall(),
                    "per_pair_acceptance": outcome.stats.per_pair_acceptance(),
                    "trace": outcome.trace.iter().map(|t| serde_json::json!({
                        "sweep": t.sweep,
                        "cut": (total_weight - t.energy) / 2.0,
                        "best_cut": (total_weight - t.best_energy) / 2.0,
                    })).collect::<Vec<_>>(),
                }),
            };
        }

        let mut outcome = anneal(
            &model,
            &job.schedule,
            job.seed,
            trace_every,
            Some(&mut wrapped),
        );
        outcome.ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

        let best_cut = (total_weight - outcome.best_energy) / 2.0;
        ExecutionResult {
            ledger: outcome.ledger.clone(),
            outputs: serde_json::json!({
                "best_cut": best_cut,
                "total_weight": total_weight,
                "best_energy": outcome.best_energy,
                "n_nodes": model.n,
                "n_edges": graph.edges.len(),
                "known_optimum": job.problem.known_optimum(),
                "trace": outcome.trace.iter().map(|t| serde_json::json!({
                    "sweep": t.sweep,
                    "cut": (total_weight - t.energy) / 2.0,
                    "best_cut": (total_weight - t.best_energy) / 2.0,
                })).collect::<Vec<_>>(),
            }),
        }
    }
}
