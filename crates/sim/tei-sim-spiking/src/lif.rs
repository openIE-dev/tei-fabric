//! Leaky integrate-and-fire (LIF) neuron — the single-compartment core.
//!
//! # Subthreshold dynamics
//!
//! The membrane potential `v` obeys the first-order linear ODE
//!
//! ```text
//!     τ_m dv/dt = −(v − v_rest) + R·I
//! ```
//!
//! where `τ_m` is the membrane time constant, `v_rest` the resting potential,
//! `R` the membrane resistance and `I` an injected current. For a *constant*
//! input over a timestep `dt` the equation integrates **exactly** (it is linear
//! with constant coefficients), giving the exponential-Euler / exact-integrator
//! update
//!
//! ```text
//!     v(t+dt) = v_∞ + (v(t) − v_∞)·exp(−dt/τ_m),     v_∞ = v_rest + R·I.
//! ```
//!
//! We use this closed form rather than forward Euler so the simulated
//! trajectory matches the analytic solution to floating-point precision:
//! starting from `v_rest`,
//!
//! ```text
//!     v(t) = v_rest + R·I·(1 − exp(−t/τ_m)),
//! ```
//! which is validation check (a). See Gerstner & Kistler, *Spiking Neuron
//! Models* (2002), §4.1; Dayan & Abbott, *Theoretical Neuroscience* (2001), §5.4.
//!
//! # Threshold, reset, refractoriness
//!
//! When `v` reaches `v_th` the neuron emits a spike and `v` is clamped to
//! `v_reset` for an absolute refractory period `t_ref` (no integration, inputs
//! ignored — the Brunel 2000 convention).
//!
//! # f–I curve (constant suprathreshold current)
//!
//! Integrating from `v_reset` toward `v_∞ = v_rest + R·I` and solving for the
//! first passage to `v_th`,
//!
//! ```text
//!     v_th = v_∞ + (v_reset − v_∞)·exp(−T/τ_m)
//!  ⇒  T   = τ_m · ln[ (v_reset − v_∞) / (v_th − v_∞) ].
//! ```
//!
//! Both numerator and denominator are negative (`v_∞ > v_th > v_reset` in the
//! suprathreshold regime), so the ratio is `> 1` and `T > 0`. With `v_reset =
//! v_rest` this reduces to the familiar textbook form
//!
//! ```text
//!     T = τ_m · ln[ R·I / (R·I + v_rest − v_th) ],
//!     f = 1 / (T + t_ref).
//! ```
//! (validation check (b); validation table in docs/SIM-ROADMAP.md §3.2 prints
//! the `v_rest = 0` special case `f = [τ ln(RI/(RI − V_th))]⁻¹`).
//!
//! As `R·I → ∞` the ratio → 1, `T → 0`, and the rate saturates at `1/t_ref`
//! (validation check (c)).

use serde::{Deserialize, Serialize};

fn one() -> f64 {
    1.0
}

/// Parameters of a homogeneous LIF neuron (SI units: seconds, volts, ohms —
/// any self-consistent unit system works since only ratios appear).
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct NeuronParams {
    /// Membrane time constant τ_m.
    pub tau: f64,
    /// Resting potential v_rest.
    pub v_rest: f64,
    /// Reset potential after a spike.
    pub v_reset: f64,
    /// Firing threshold.
    pub v_th: f64,
    /// Absolute refractory period.
    pub t_ref: f64,
    /// Membrane resistance R (defaults to 1.0 — current measured in volts).
    #[serde(default = "one")]
    pub r: f64,
}

impl NeuronParams {
    /// Per-step decay factor exp(−dt/τ_m) of the exact integrator.
    #[inline]
    pub fn decay(&self, dt: f64) -> f64 {
        (-dt / self.tau).exp()
    }

    /// Steady-state potential v_∞ = v_rest + R·I for a constant current `i_ext`.
    #[inline]
    pub fn v_inf(&self, i_ext: f64) -> f64 {
        self.v_rest + self.r * i_ext
    }

    /// Refractory period expressed in whole timesteps (rounded to nearest).
    #[inline]
    pub fn ref_steps(&self, dt: f64) -> u32 {
        (self.t_ref / dt).round() as u32
    }

    /// Continuous-time inter-spike interval `T` (excluding refractoriness) for a
    /// constant current, or `None` if the input is subthreshold (`v_∞ ≤ v_th`).
    pub fn analytic_period(&self, i_ext: f64) -> Option<f64> {
        let v_inf = self.v_inf(i_ext);
        if v_inf <= self.v_th {
            return None;
        }
        Some(self.tau * ((self.v_reset - v_inf) / (self.v_th - v_inf)).ln())
    }

    /// Closed-form firing rate `f = 1/(T + t_ref)` for a constant current, or
    /// `None` if subthreshold.
    pub fn analytic_rate(&self, i_ext: f64) -> Option<f64> {
        self.analytic_period(i_ext).map(|t| 1.0 / (t + self.t_ref))
    }
}

/// Free (no-threshold) membrane trajectory under constant current, sampled
/// after each of `n_steps` timesteps. Starting from `v0` this reproduces the
/// analytic charging curve exactly (validation check (a)).
pub fn membrane_trace(p: &NeuronParams, i_ext: f64, dt: f64, n_steps: usize, v0: f64) -> Vec<f64> {
    let decay = p.decay(dt);
    let v_inf = p.v_inf(i_ext);
    let mut v = v0;
    let mut out = Vec::with_capacity(n_steps);
    for _ in 0..n_steps {
        v = v_inf + (v - v_inf) * decay;
        out.push(v);
    }
    out
}

/// Clock-driven single-neuron LIF simulation under constant current. Returns
/// the timestep indices at which spikes occurred. Includes threshold, reset
/// and absolute refractoriness — the same arithmetic the network core uses.
pub fn simulate_single(p: &NeuronParams, i_ext: f64, dt: f64, n_steps: usize) -> Vec<usize> {
    let decay = p.decay(dt);
    let v_inf = p.v_inf(i_ext);
    let ref_steps = p.ref_steps(dt);
    let mut v = p.v_reset;
    let mut refr = 0u32;
    let mut spikes = Vec::new();
    for t in 0..n_steps {
        if refr > 0 {
            refr -= 1;
            v = p.v_reset;
            continue;
        }
        v = v_inf + (v - v_inf) * decay;
        if v >= p.v_th {
            spikes.push(t);
            v = p.v_reset;
            refr = ref_steps;
        }
    }
    spikes
}

/// Measured steady-state firing rate (Hz) of a single neuron under constant
/// current, taken from the last inter-spike interval (the transient has
/// settled). Returns 0 if fewer than two spikes occur.
pub fn measured_rate(p: &NeuronParams, i_ext: f64, dt: f64, n_steps: usize) -> f64 {
    let s = simulate_single(p, i_ext, dt, n_steps);
    if s.len() < 2 {
        return 0.0;
    }
    let isi = (s[s.len() - 1] - s[s.len() - 2]) as f64 * dt;
    1.0 / isi
}
