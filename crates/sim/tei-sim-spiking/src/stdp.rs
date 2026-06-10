//! Pair-based spike-timing-dependent plasticity (STDP).
//!
//! # Window
//!
//! The canonical pair-based rule (Bi & Poo 1998; Song, Miller & Abbott 2000)
//! makes the weight change depend on the relative timing `Δt = t_post − t_pre`
//! of a pre/post spike pair:
//!
//! ```text
//!     Δw(Δt) =  A₊ · exp(−Δt/τ₊)    if Δt > 0   (pre before post → potentiation)
//!     Δw(Δt) = −A₋ · exp( Δt/τ₋)    if Δt < 0   (post before pre → depression)
//! ```
//!
//! The window is **asymmetric** (potentiation and depression have independent
//! amplitudes and time constants) and decays exponentially away from
//! coincidence — the qualitative shape measured by Bi & Poo, *J. Neurosci.*
//! 18(24):10464 (1998), Fig. 7.
//!
//! # Online traces
//!
//! Computing `Δt` for every pair is `O(N²)`; the standard implementation keeps
//! one exponentially decaying *trace* per synapse end instead:
//!
//! ```text
//!     dx/dt = −x/τ₊   (presynaptic trace, jumps +1 on a pre spike)
//!     dy/dt = −y/τ₋   (postsynaptic trace, jumps +1 on a post spike)
//! ```
//!
//! On a **post** spike the weight is potentiated by `A₊·x` (x = residual of the
//! most recent pre spikes); on a **pre** spike it is depressed by `A₋·y`. For an
//! isolated pre/post pair this reproduces the window above exactly — the
//! property verified in the tests (`stdp_trace_matches_window`).

use serde::{Deserialize, Serialize};

/// STDP rule configuration. Weights are clamped to `[w_min, w_max]`.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct StdpConfig {
    /// Potentiation amplitude A₊.
    pub a_plus: f64,
    /// Depression amplitude A₋ (positive number; applied as a decrement).
    pub a_minus: f64,
    /// Potentiation time constant τ₊.
    pub tau_plus: f64,
    /// Depression time constant τ₋.
    pub tau_minus: f64,
    /// Lower weight bound.
    pub w_min: f64,
    /// Upper weight bound.
    pub w_max: f64,
}

impl StdpConfig {
    /// Closed-form weight change for an isolated pre/post pair separated by
    /// `dt_post_minus_pre = t_post − t_pre`.
    pub fn window(&self, dt_post_minus_pre: f64) -> f64 {
        let dt = dt_post_minus_pre;
        if dt > 0.0 {
            self.a_plus * (-dt / self.tau_plus).exp()
        } else if dt < 0.0 {
            -self.a_minus * (dt / self.tau_minus).exp()
        } else {
            // Exactly coincident: convention is the (positive) potentiation limit.
            self.a_plus
        }
    }

    #[inline]
    fn clamp(&self, w: f64) -> f64 {
        w.clamp(self.w_min, self.w_max)
    }
}

/// Online STDP state for a single synapse (one presynaptic + one postsynaptic
/// trace). Drive it with `on_pre`/`on_post` in spike-time order.
#[derive(Clone, Debug)]
pub struct StdpState {
    cfg: StdpConfig,
    /// Presynaptic trace x.
    x: f64,
    /// Postsynaptic trace y.
    y: f64,
    /// Last time either trace was updated.
    t_last: f64,
    /// Current weight.
    pub weight: f64,
}

impl StdpState {
    pub fn new(cfg: StdpConfig, weight: f64) -> Self {
        Self {
            cfg,
            x: 0.0,
            y: 0.0,
            t_last: 0.0,
            weight,
        }
    }

    /// Decay both traces forward to time `t`.
    fn advance(&mut self, t: f64) {
        let dt = t - self.t_last;
        if dt > 0.0 {
            self.x *= (-dt / self.cfg.tau_plus).exp();
            self.y *= (-dt / self.cfg.tau_minus).exp();
            self.t_last = t;
        }
    }

    /// Presynaptic spike at time `t`: depress by `A₋·y`, then bump the pre trace.
    pub fn on_pre(&mut self, t: f64) {
        self.advance(t);
        self.weight = self.cfg.clamp(self.weight - self.cfg.a_minus * self.y);
        self.x += 1.0;
    }

    /// Postsynaptic spike at time `t`: potentiate by `A₊·x`, then bump the post
    /// trace.
    pub fn on_post(&mut self, t: f64) {
        self.advance(t);
        self.weight = self.cfg.clamp(self.weight + self.cfg.a_plus * self.x);
        self.y += 1.0;
    }
}
