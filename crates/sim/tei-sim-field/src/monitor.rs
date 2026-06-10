//! Monitors — time-series probes and on-the-fly DFT accumulation.
//!
//! The DFT monitor implements the running discrete Fourier transform
//!
//! ```text
//!   F(ω) ≈ Σₙ Ez(tₙ)·e^{−iωtₙ}·Δt
//! ```
//!
//! accumulated one term per step, so no time series has to be stored — the
//! standard MEEP-style frequency monitor. When the accumulation window in
//! steady CW state spans an integer number of periods that is also an
//! integer number of steps, the conjugate (−ω) spectral line cancels
//! exactly and the accumulated value is the clean single-sided amplitude.

use crate::grid::Grid2d;
use tei_sim_core::linalg::C64;

/// Time-series point probe: records Ez(i, j) every step it is given.
#[derive(Debug, Clone)]
pub struct Probe {
    pub i: usize,
    pub j: usize,
    /// One sample per `record` call, in call order.
    pub trace: Vec<f64>,
}

impl Probe {
    pub fn new(i: usize, j: usize) -> Self {
        Self {
            i,
            j,
            trace: Vec::new(),
        }
    }

    /// Sample Ez(i, j) from the grid's current state.
    pub fn record(&mut self, g: &Grid2d) {
        self.trace.push(g.ez_at(self.i, self.j));
    }

    /// Most recent sample (0.0 if none yet).
    pub fn last(&self) -> f64 {
        self.trace.last().copied().unwrap_or(0.0)
    }
}

/// On-the-fly DFT monitor at a single Ez cell for a set of angular
/// frequencies ω (rad per unit time; normalized units c = Δ = 1).
#[derive(Debug, Clone)]
pub struct DftMonitor {
    pub i: usize,
    pub j: usize,
    omegas: Vec<f64>,
    accum: Vec<C64>,
    samples: u64,
}

impl DftMonitor {
    pub fn new(i: usize, j: usize, omegas: Vec<f64>) -> Self {
        let n = omegas.len();
        Self {
            i,
            j,
            omegas,
            accum: vec![C64::ZERO; n],
            samples: 0,
        }
    }

    /// Accumulate Ez(i, j)·e^{−iωt} for every registered ω. `t` is the
    /// physical time of the grid's current Ez (i.e. (n+1)·Δt after step n).
    pub fn record(&mut self, g: &Grid2d, t: f64) {
        let ez = g.ez_at(self.i, self.j);
        for (a, &w) in self.accum.iter_mut().zip(&self.omegas) {
            *a = *a + C64::from_polar(ez, -w * t);
        }
        self.samples += 1;
    }

    pub fn omegas(&self) -> &[f64] {
        &self.omegas
    }

    /// Raw accumulated sums Σ Ez·e^{−iωt} (one per ω).
    pub fn accum(&self) -> &[C64] {
        &self.accum
    }

    /// Riemann-sum DFT amplitudes: the raw sums scaled by Δt.
    pub fn spectra(&self, dt: f64) -> Vec<C64> {
        self.accum.iter().map(|&a| a * dt).collect()
    }

    /// Number of `record` calls so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }
}
