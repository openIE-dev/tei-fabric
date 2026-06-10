//! Networks of LIF populations with sparse, delayed, current-based synapses.
//!
//! # Synapse model
//!
//! Synapses are **current-based delta** synapses: an arriving spike instantly
//! increments the postsynaptic membrane by the synaptic weight `w` (a voltage
//! jump, in the same units as `v`). This is the Brunel 2000 convention and lets
//! the balanced-network analytics apply directly. Excitatory synapses carry
//! `w > 0`, inhibitory `w < 0`.
//!
//! # Delays and propagation
//!
//! Each synapse has an integer delay `d ≥ 1` (in timesteps). Propagation uses a
//! per-timestep **delay ring buffer** of length `max_delay + 1`: when neuron
//! `i` fires at step `t`, every outgoing synapse `(i → j, w, d)` deposits `w`
//! into ring slot `(t + d) mod L` at position `j`; that slot is read and
//! cleared `d` steps later. Because `1 ≤ d ≤ max_delay < L`, deposits never
//! land in the slot being consumed this step, so there is no within-step
//! feedback (clock-driven, the simplest correct scheme).
//!
//! # External drive
//!
//! Two independent sources, both optional per population:
//!  * a constant injected current `i_ext` (folded into `v_∞`), and
//!  * independent homogeneous **Poisson** input spikes at `poisson_rate` Hz per
//!    neuron, each adding `poisson_weight` to `v` (delta excitation). The number
//!    of external arrivals per step is drawn `~ Poisson(rate·dt)` from the
//!    deterministic core RNG, so runs are reproducible from the seed.
//!
//! # Update order (per timestep)
//!
//! For each non-refractory neuron: exact exponential decay toward `v_∞`, add
//! Poisson input, add delivered synaptic input, then test threshold. All
//! threshold crossings in a step are collected, *then* resets, refractory
//! arming and outgoing deposits are applied — so a spike influences its targets
//! only after its delay.

use crate::lif::NeuronParams;
use serde::Serialize;
use tei_sim_core::exec::Progress;
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

/// One population of identical LIF neurons plus its external drive.
#[derive(Clone, Debug)]
pub struct PopulationSpec {
    pub name: String,
    pub n: usize,
    pub params: NeuronParams,
    /// Constant injected current (folded into v_∞).
    pub i_ext: f64,
    /// Independent Poisson input rate per neuron, Hz (0 disables).
    pub poisson_rate: f64,
    /// Voltage jump per external Poisson spike.
    pub poisson_weight: f64,
}

/// Builder that accumulates populations and synapses, then freezes a `Network`.
pub struct NetworkBuilder {
    dt: f64,
    pops: Vec<PopulationSpec>,
    offsets: Vec<usize>,
    n_total: usize,
    // (pre_global, post_global, weight, delay_steps)
    syn: Vec<(u32, u32, f64, u32)>,
}

impl NetworkBuilder {
    pub fn new(dt: f64) -> Self {
        Self {
            dt,
            pops: Vec::new(),
            offsets: Vec::new(),
            n_total: 0,
            syn: Vec::new(),
        }
    }

    /// Add a population; returns its population index. Global neuron indices for
    /// population `p` are `offset(p) .. offset(p) + n`.
    pub fn add_population(&mut self, spec: PopulationSpec) -> usize {
        let idx = self.pops.len();
        self.offsets.push(self.n_total);
        self.n_total += spec.n;
        self.pops.push(spec);
        idx
    }

    /// Global index of the first neuron in population `p`.
    pub fn offset(&self, p: usize) -> usize {
        self.offsets[p]
    }

    /// Random Bernoulli connectivity from `pre_pop` to `post_pop`: each ordered
    /// pair connects independently with probability `prob`. Self-connections
    /// (same global neuron) are excluded. Draws from `rng` so connectivity is
    /// reproducible from the seed.
    pub fn connect_random(
        &mut self,
        pre_pop: usize,
        post_pop: usize,
        prob: f64,
        weight: f64,
        delay_steps: u32,
        rng: &mut Rng,
    ) {
        assert!(delay_steps >= 1, "synaptic delay must be ≥ 1 timestep");
        let pre0 = self.offsets[pre_pop];
        let post0 = self.offsets[post_pop];
        let npre = self.pops[pre_pop].n;
        let npost = self.pops[post_pop].n;
        for a in 0..npre {
            let pre = (pre0 + a) as u32;
            for b in 0..npost {
                let post = (post0 + b) as u32;
                if pre == post {
                    continue;
                }
                if rng.bernoulli(prob) {
                    self.syn.push((pre, post, weight, delay_steps));
                }
            }
        }
    }

    /// Add a single explicit synapse by global neuron index (handy for small
    /// deterministic test networks).
    pub fn connect_explicit(&mut self, pre: u32, post: u32, weight: f64, delay_steps: u32) {
        assert!(delay_steps >= 1, "synaptic delay must be ≥ 1 timestep");
        self.syn.push((pre, post, weight, delay_steps));
    }

    /// Freeze into a runnable network (builds the flattened outgoing adjacency
    /// and precomputes per-neuron integrator constants).
    pub fn build(self) -> Network {
        let n = self.n_total;
        let dt = self.dt;

        let mut decay = vec![0.0; n];
        let mut v_inf = vec![0.0; n];
        let mut v_reset = vec![0.0; n];
        let mut v_th = vec![0.0; n];
        let mut ref_steps = vec![0u32; n];
        let mut lambda = vec![0.0; n];
        let mut pweight = vec![0.0; n];
        let mut pop_of = vec![0u32; n];

        let mut pop_names = Vec::with_capacity(self.pops.len());
        let mut pop_start = Vec::with_capacity(self.pops.len());
        let mut pop_len = Vec::with_capacity(self.pops.len());

        for (p, spec) in self.pops.iter().enumerate() {
            let start = self.offsets[p];
            pop_names.push(spec.name.clone());
            pop_start.push(start);
            pop_len.push(spec.n);
            let d = spec.params.decay(dt);
            let vi = spec.params.v_inf(spec.i_ext);
            let rs = spec.params.ref_steps(dt);
            for k in 0..spec.n {
                let i = start + k;
                decay[i] = d;
                v_inf[i] = vi;
                v_reset[i] = spec.params.v_reset;
                v_th[i] = spec.params.v_th;
                ref_steps[i] = rs;
                lambda[i] = spec.poisson_rate * dt;
                pweight[i] = spec.poisson_weight;
                pop_of[i] = p as u32;
            }
        }

        // Build flattened outgoing adjacency sorted by presynaptic neuron.
        let mut syn = self.syn;
        syn.sort_by_key(|&(pre, post, _, _)| (pre, post));
        let mut out_indptr = vec![0usize; n + 1];
        for &(pre, _, _, _) in &syn {
            out_indptr[pre as usize + 1] += 1;
        }
        for i in 0..n {
            out_indptr[i + 1] += out_indptr[i];
        }
        let m = syn.len();
        let mut out_target = vec![0u32; m];
        let mut out_weight = vec![0.0; m];
        let mut out_delay = vec![0u32; m];
        let mut cursor = out_indptr.clone();
        let mut max_delay = 1u32;
        for &(pre, post, w, d) in &syn {
            let e = cursor[pre as usize];
            cursor[pre as usize] += 1;
            out_target[e] = post;
            out_weight[e] = w;
            out_delay[e] = d;
            max_delay = max_delay.max(d);
        }

        Network {
            dt,
            n,
            decay,
            v_inf,
            v_reset,
            v_th,
            ref_steps,
            lambda,
            pweight,
            pop_names,
            pop_start,
            pop_len,
            pop_of,
            out_indptr,
            out_target,
            out_weight,
            out_delay,
            max_delay,
        }
    }
}

/// A frozen, runnable spiking network.
pub struct Network {
    pub dt: f64,
    pub n: usize,
    decay: Vec<f64>,
    v_inf: Vec<f64>,
    v_reset: Vec<f64>,
    v_th: Vec<f64>,
    ref_steps: Vec<u32>,
    lambda: Vec<f64>,
    pweight: Vec<f64>,
    pop_names: Vec<String>,
    pop_start: Vec<usize>,
    pop_len: Vec<usize>,
    #[allow(dead_code)]
    pop_of: Vec<u32>,
    out_indptr: Vec<usize>,
    out_target: Vec<u32>,
    out_weight: Vec<f64>,
    out_delay: Vec<u32>,
    max_delay: u32,
}

impl Network {
    /// Number of populations.
    pub fn n_populations(&self) -> usize {
        self.pop_names.len()
    }

    /// `(name, start, len)` of population `p`.
    pub fn population(&self, p: usize) -> (&str, usize, usize) {
        (&self.pop_names[p], self.pop_start[p], self.pop_len[p])
    }

    /// Out-degree (fan-out) of global neuron `i`.
    pub fn out_degree(&self, i: usize) -> usize {
        self.out_indptr[i + 1] - self.out_indptr[i]
    }

    /// Total number of synapses.
    pub fn n_synapses(&self) -> usize {
        self.out_target.len()
    }

    /// Run the clock-driven simulation for `n_steps` timesteps.
    ///
    /// `rng` supplies the Poisson input stream; passing the same RNG state and
    /// network yields identical spike trains. `on_progress` is called at ~1%
    /// cadence with `{t, total_spikes, rate_hz}`.
    pub fn run(
        &self,
        n_steps: usize,
        rng: &mut Rng,
        on_progress: &mut dyn FnMut(Progress),
    ) -> RunResult {
        let n = self.n;
        let l = self.max_delay as usize + 1;
        let mut v = self.v_reset.clone();
        let mut refr = vec![0u32; n];
        let mut ring = vec![0.0f64; l * n];
        let mut spike_times: Vec<Vec<f64>> = vec![Vec::new(); n];
        let mut ledger = EventLedger::default();
        let report_every = (n_steps / 100).max(1);
        let mut fired: Vec<u32> = Vec::new();

        let t0 = std::time::Instant::now();
        for step in 0..n_steps {
            let slot = step % l;
            let base = slot * n;
            fired.clear();
            for i in 0..n {
                if refr[i] > 0 {
                    refr[i] -= 1;
                    v[i] = self.v_reset[i];
                    continue;
                }
                let vi = self.v_inf[i];
                v[i] = vi + (v[i] - vi) * self.decay[i];
                if self.lambda[i] > 0.0 {
                    let k = poisson(rng, self.lambda[i]);
                    if k > 0 {
                        v[i] += k as f64 * self.pweight[i];
                    }
                }
                v[i] += ring[base + i];
                if v[i] >= self.v_th[i] {
                    fired.push(i as u32);
                }
            }
            // Consume (clear) this step's delivery slot.
            for x in &mut ring[base..base + n] {
                *x = 0.0;
            }
            // Apply spikes: reset, arm refractory, deposit outgoing.
            for &iu in &fired {
                let i = iu as usize;
                v[i] = self.v_reset[i];
                refr[i] = self.ref_steps[i];
                spike_times[i].push(step as f64 * self.dt);
                ledger.spikes += 1;
                let lo = self.out_indptr[i];
                let hi = self.out_indptr[i + 1];
                ledger.sops += (hi - lo) as u64;
                for e in lo..hi {
                    let j = self.out_target[e] as usize;
                    let d = self.out_delay[e] as usize;
                    let s = (step + d) % l;
                    ring[s * n + j] += self.out_weight[e];
                }
            }

            if step % report_every == 0 || step + 1 == n_steps {
                let t = step as f64 * self.dt;
                let rate = if t > 0.0 {
                    ledger.spikes as f64 / (t * n as f64)
                } else {
                    0.0
                };
                on_progress(Progress {
                    fraction: (step + 1) as f64 / n_steps as f64,
                    metrics: serde_json::json!({
                        "t": t,
                        "total_spikes": ledger.spikes,
                        "rate_hz": rate,
                    }),
                });
            }
        }
        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

        RunResult {
            spike_times,
            ledger,
            dt: self.dt,
            duration: n_steps as f64 * self.dt,
            n,
        }
    }
}

/// Per-step Poisson count via Knuth's algorithm (exact; intended for small
/// `λ = rate·dt`, the regime of per-timestep input rates).
fn poisson(rng: &mut Rng, lambda: f64) -> u64 {
    let l = (-lambda).exp();
    let mut k = 0u64;
    let mut p = 1.0;
    loop {
        p *= rng.f64();
        if p <= l {
            break;
        }
        k += 1;
    }
    k
}

/// Outcome of a network run: per-neuron spike trains plus the event ledger.
#[derive(Clone, Debug)]
pub struct RunResult {
    /// Spike times (seconds) per global neuron index.
    pub spike_times: Vec<Vec<f64>>,
    pub ledger: EventLedger,
    pub dt: f64,
    pub duration: f64,
    pub n: usize,
}

impl RunResult {
    /// Mean firing rate (Hz) over all neurons.
    pub fn mean_rate_hz(&self) -> f64 {
        if self.duration <= 0.0 || self.n == 0 {
            return 0.0;
        }
        self.ledger.spikes as f64 / (self.duration * self.n as f64)
    }

    /// Mean firing rate (Hz) over neurons `start .. start+len`.
    pub fn pop_rate_hz(&self, start: usize, len: usize) -> f64 {
        if self.duration <= 0.0 || len == 0 {
            return 0.0;
        }
        let count: usize = self.spike_times[start..start + len]
            .iter()
            .map(|s| s.len())
            .sum();
        count as f64 / (self.duration * len as f64)
    }

    /// Coefficient of variation of inter-spike intervals for one neuron, or
    /// `None` if it has fewer than `min_isi + 1` spikes. `CV = σ_ISI / μ_ISI`;
    /// Poisson-like (irregular) spiking has `CV ≈ 1`, clock-like (regular)
    /// spiking `CV ≈ 0`.
    pub fn neuron_cv(&self, i: usize, min_isi: usize) -> Option<f64> {
        let s = &self.spike_times[i];
        if s.len() < min_isi + 1 {
            return None;
        }
        let isis: Vec<f64> = s.windows(2).map(|w| w[1] - w[0]).collect();
        let mean = isis.iter().sum::<f64>() / isis.len() as f64;
        if mean <= 0.0 {
            return None;
        }
        let var = isis.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / isis.len() as f64;
        Some(var.sqrt() / mean)
    }

    /// Mean CV of ISIs over all neurons that fired enough (≥ 3 spikes ⇒ ≥ 2
    /// ISIs). Returns 0 if no neuron qualifies.
    pub fn mean_cv_isi(&self) -> f64 {
        let mut acc = 0.0;
        let mut count = 0usize;
        for i in 0..self.n {
            if let Some(cv) = self.neuron_cv(i, 2) {
                acc += cv;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { acc / count as f64 }
    }

    /// Mean CV of ISIs over neurons `start .. start+len`.
    pub fn pop_cv_isi(&self, start: usize, len: usize) -> f64 {
        let mut acc = 0.0;
        let mut count = 0usize;
        for i in start..start + len {
            if let Some(cv) = self.neuron_cv(i, 2) {
                acc += cv;
                count += 1;
            }
        }
        if count == 0 { 0.0 } else { acc / count as f64 }
    }

    /// First `cap` spikes in time order as `[t_seconds, neuron_id]` pairs
    /// (raster sample for visualization).
    pub fn raster_sample(&self, cap: usize) -> Vec<RasterPoint> {
        let mut all: Vec<RasterPoint> = Vec::new();
        for (i, s) in self.spike_times.iter().enumerate() {
            for &t in s {
                all.push(RasterPoint {
                    t,
                    neuron: i as u32,
                });
            }
        }
        all.sort_by(|a, b| a.t.total_cmp(&b.t).then_with(|| a.neuron.cmp(&b.neuron)));
        all.truncate(cap);
        all
    }
}

/// One point of a spike raster.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct RasterPoint {
    /// Spike time, seconds.
    pub t: f64,
    /// Global neuron index.
    pub neuron: u32,
}
