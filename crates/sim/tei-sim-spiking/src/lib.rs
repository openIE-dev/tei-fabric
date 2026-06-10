//! tei-sim-spiking — Lava-class (Intel / NIR community) spiking-neural-network
//! simulator.
//!
//! Functional simulator for the spiking substrate column: populations of
//! leaky integrate-and-fire (LIF) neurons connected by sparse, delayed,
//! current-based synapses, with constant-current and Poisson external drive and
//! optional pair-based STDP. Clock-driven with an **exact exponential
//! integrator** for the subthreshold membrane (so simulated trajectories match
//! the closed-form ODE solution to floating-point precision) and a per-timestep
//! delay ring buffer for spike propagation.
//!
//! The job schema mirrors **NIR semantics** (LIF nodes, linear/connection
//! nodes, integer delays) as plain JSON — no HDF5 dependency; an offline
//! NIR-file converter can come later.
//!
//! # Modules
//! * [`lif`] — the single-neuron LIF model and its closed-form solutions.
//! * [`network`] — populations, sparse delayed synapses, the clock-driven core.
//! * [`stdp`] — pair-based spike-timing-dependent plasticity.
//!
//! # Validation (see `tests/analytic.rs` and docs/SIM-ROADMAP.md §3.2)
//! Analytic / published only — no foreign-tool fixtures:
//!  * membrane trajectory vs the closed-form charging curve,
//!  * f–I curve vs `f = 1/(τ ln[RI/(RI + v_rest − v_th)] + t_ref)`,
//!  * refractory rate cap `≤ 1/t_ref`,
//!  * Brunel 2000 balanced-network regimes (low/irregular vs high rate),
//!  * STDP window asymmetry (Bi & Poo 1998),
//!  * ledger consistency `sops = Σ spikes × fan-out`, and determinism.

pub mod lif;
pub mod network;
pub mod stdp;

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::rng::Rng;

pub use lif::NeuronParams;
pub use network::{Network, NetworkBuilder, PopulationSpec, RasterPoint, RunResult};
pub use stdp::{StdpConfig, StdpState};

fn one() -> f64 {
    1.0
}

/// A layer (population) in a [`SpikingJob`]: a set of identical LIF neurons
/// plus its external drive. Times are seconds, rates Hz.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct LayerSpec {
    pub name: String,
    pub n: usize,
    pub tau: f64,
    #[serde(default)]
    pub v_rest: f64,
    #[serde(default)]
    pub v_reset: f64,
    pub v_th: f64,
    #[serde(default)]
    pub t_ref: f64,
    #[serde(default = "one")]
    pub r: f64,
    /// Constant injected current.
    #[serde(default)]
    pub i_ext: f64,
    /// Independent Poisson input rate per neuron, Hz.
    #[serde(default)]
    pub poisson_rate: f64,
    /// Voltage jump per external Poisson spike.
    #[serde(default)]
    pub poisson_weight: f64,
}

impl LayerSpec {
    fn to_population(&self) -> PopulationSpec {
        PopulationSpec {
            name: self.name.clone(),
            n: self.n,
            params: NeuronParams {
                tau: self.tau,
                v_rest: self.v_rest,
                v_reset: self.v_reset,
                v_th: self.v_th,
                t_ref: self.t_ref,
                r: self.r,
            },
            i_ext: self.i_ext,
            poisson_rate: self.poisson_rate,
            poisson_weight: self.poisson_weight,
        }
    }
}

/// A connection rule between two layers: random Bernoulli connectivity with the
/// given per-pair `probability`, uniform `weight` (signed: negative ⇒
/// inhibitory) and integer `delay` in timesteps.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ConnSpec {
    /// Presynaptic layer index.
    pub pre: usize,
    /// Postsynaptic layer index.
    pub post: usize,
    pub probability: f64,
    pub weight: f64,
    /// Delay in timesteps (≥ 1).
    pub delay: u32,
}

/// A complete spiking job: layers, connectivity, drive, timing and seed.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct SpikingJob {
    pub layers: Vec<LayerSpec>,
    #[serde(default)]
    pub connections: Vec<ConnSpec>,
    /// Simulated duration, seconds.
    pub duration: f64,
    /// Timestep, seconds.
    pub dt: f64,
    #[serde(default)]
    pub seed: u64,
}

impl SpikingJob {
    /// Build the network (connectivity drawn from `rng`) and the step count.
    pub fn build(&self, rng: &mut Rng) -> (Network, usize) {
        let mut b = NetworkBuilder::new(self.dt);
        for layer in &self.layers {
            b.add_population(layer.to_population());
        }
        for c in &self.connections {
            b.connect_random(c.pre, c.post, c.probability, c.weight, c.delay, rng);
        }
        let n_steps = (self.duration / self.dt).round() as usize;
        (b.build(), n_steps)
    }
}

/// Per-population summary statistics in the executor output.
#[derive(Clone, Debug, Serialize)]
pub struct PopulationStats {
    pub name: String,
    pub n: usize,
    /// Mean firing rate, Hz.
    pub mean_rate_hz: f64,
    /// Mean CV of inter-spike intervals.
    pub cv_isi: f64,
}

/// The spiking-column executor.
pub struct SpikingExecutor;

impl Executor for SpikingExecutor {
    type Job = SpikingJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        // One deterministic RNG stream seeds both connectivity and Poisson
        // input, so the whole run is reproducible from `job.seed`.
        let mut rng = Rng::new(job.seed);
        let (net, n_steps) = job.build(&mut rng);
        let result = net.run(n_steps, &mut rng, on_progress);

        let pops: Vec<PopulationStats> = (0..net.n_populations())
            .map(|p| {
                let (name, start, len) = net.population(p);
                PopulationStats {
                    name: name.to_string(),
                    n: len,
                    mean_rate_hz: result.pop_rate_hz(start, len),
                    cv_isi: result.pop_cv_isi(start, len),
                }
            })
            .collect();

        let raster = result.raster_sample(2000);

        ExecutionResult {
            ledger: result.ledger.clone(),
            outputs: serde_json::json!({
                "n_neurons": net.n,
                "n_synapses": net.n_synapses(),
                "duration": result.duration,
                "dt": result.dt,
                "mean_rate_hz": result.mean_rate_hz(),
                "cv_isi": result.mean_cv_isi(),
                "populations": pops,
                "raster_sample": raster,
            }),
        }
    }
}
