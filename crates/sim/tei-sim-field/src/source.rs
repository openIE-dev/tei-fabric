//! Soft sources — additive Ez excitations.
//!
//! All sources here are **soft** (transparent): each step they *add*
//! `amplitude·s(t)` onto Ez instead of overwriting it, so scattered waves
//! pass through the source cells unimpeded (Taflove & Hagness 2005, §5.1).

use crate::grid::Grid2d;
use serde::{Deserialize, Serialize};

/// Source time signature s(t).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum TimeProfile {
    /// Baseband Gaussian pulse s(t) = exp(−((t − t0)/τ)²). Its spectrum is
    /// a Gaussian centred at ω = 0 with σ_ω = √2/τ; pick τ large enough
    /// that the significant band stays well-resolved on the grid.
    Gaussian { t0: f64, tau: f64 },
    /// Modulated Gaussian pulse s(t) = exp(−((t − t0)/τ)²)·sin(ω(t − t0)):
    /// spectrum is a Gaussian centred at ±ω with the same σ_ω = √2/τ as the
    /// baseband pulse, and the sin carrier gives an exact null at ω = 0 (no
    /// DC component left on the grid). This is the natural driver for
    /// waveguide mode sources, whose band must sit on the guided dispersion
    /// branch.
    ModulatedGaussian { omega: f64, t0: f64, tau: f64 },
    /// Continuous wave s(t) = w(t)·sin(ωt), where w is a raised-cosine
    /// turn-on envelope over the first `ramp` time units (suppresses the
    /// broadband switch-on transient). `ramp = 0` means hard turn-on.
    Cw {
        omega: f64,
        #[serde(default)]
        ramp: f64,
    },
}

impl TimeProfile {
    /// Evaluate s(t).
    pub fn eval(&self, t: f64) -> f64 {
        match *self {
            TimeProfile::Gaussian { t0, tau } => {
                let u = (t - t0) / tau;
                (-u * u).exp()
            }
            TimeProfile::ModulatedGaussian { omega, t0, tau } => {
                let u = (t - t0) / tau;
                (-u * u).exp() * (omega * (t - t0)).sin()
            }
            TimeProfile::Cw { omega, ramp } => {
                let w = if ramp <= 0.0 || t >= ramp {
                    1.0
                } else {
                    0.5 * (1.0 - (std::f64::consts::PI * t / ramp).cos())
                };
                w * (omega * t).sin()
            }
        }
    }
}

/// Spatial footprint of the source (which Ez cells are driven).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum SourceShape {
    /// A single Ez cell.
    Point { i: usize, j: usize },
    /// A vertical line at column `i` spanning rows `j0..=j1`. Defaults span
    /// the full interior (1..=ny−2; the outermost ring is PEC).
    Line {
        i: usize,
        #[serde(default)]
        j0: Option<usize>,
        #[serde(default)]
        j1: Option<usize>,
    },
}

fn default_amplitude() -> f64 {
    1.0
}

/// A soft Ez source: spatial footprint × time signature × amplitude.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Source {
    pub shape: SourceShape,
    pub time: TimeProfile,
    #[serde(default = "default_amplitude")]
    pub amplitude: f64,
}

impl Source {
    /// Add `amplitude·s(t)` onto the driven Z cells (call once per step,
    /// after the field update, with t the post-update time).
    pub fn inject(&self, g: &mut Grid2d, t: f64) {
        let s = self.amplitude * self.time.eval(t);
        match self.shape {
            SourceShape::Point { i, j } => g.add_ez(i, j, s),
            SourceShape::Line { i, j0, j1 } => {
                let lo = j0.unwrap_or(1);
                let hi = j1.unwrap_or(g.ny - 2);
                for j in lo..=hi {
                    g.add_ez(i, j, s);
                }
            }
        }
    }
}
