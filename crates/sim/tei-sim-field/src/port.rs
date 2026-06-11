//! Port monitors — mode-overlap amplitudes with running DFTs (F2).
//!
//! A **port** is a transverse line (one grid column `i`, all interior rows)
//! crossing a waveguide, with an associated guided-mode profile per
//! requested frequency. Every timestep the monitor records the modal
//! projection
//!
//! ```text
//!   o(t) = Σ_j Ez(i, j, t) · E_mode(j)        (E_mode unit-L2 normalized)
//! ```
//!
//! and accumulates the running DFT `A(ω) += o(t)·e^{+iωt}` — the same
//! on-the-fly style as [`crate::monitor::DftMonitor`], one complex
//! accumulator per frequency. Because the mode profile is itself
//! frequency-dependent (n_eff(ω), γ(ω)), each frequency carries its own
//! sampled profile.
//!
//! **Kernel sign.** For a real field written in the photonics convention
//! Ez = Re[A(x)·e^{−iωt}], A(x) = e^{+iβx} forward, the coefficient A is
//! recovered by averaging against `e^{+iωt}` (the `e^{−iωt}` kernel of the
//! F1 [`crate::monitor::DftMonitor`] yields the conjugate A*). Port
//! monitors therefore use the **+iωt** kernel so extracted S-parameters
//! land directly in `tei-sim-photonic`'s `e^{+iβL}` forward-propagation
//! convention: a straight guide of length L gives arg S₂₁ = +βL.
//!
//! The projection does not by itself distinguish forward from backward
//! travel; directional separation happens at extraction time by
//! reference-run subtraction (see [`crate::sparams`]).

use crate::grid::Grid2d;
use serde::{Deserialize, Serialize};
use tei_sim_core::linalg::C64;

/// Port description in a [`crate::FieldJob`] (serde: `{"i": …, "mode": …}`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PortSpec {
    /// Grid column (x index) of the port plane.
    pub i: usize,
    /// TE mode order m to project onto (0 = fundamental).
    #[serde(default)]
    pub mode: usize,
}

/// A live port monitor: per-frequency mode profiles plus DFT accumulators.
#[derive(Debug, Clone)]
pub struct PortMonitor {
    /// Grid column of the port plane.
    pub i: usize,
    omegas: Vec<f64>,
    /// `profiles[f][j]` — unit-L2 mode profile for frequency f on Ez rows.
    profiles: Vec<Vec<f64>>,
    accum: Vec<C64>,
    samples: u64,
}

impl PortMonitor {
    /// `profiles` must hold one length-ny, unit-L2 profile per ω.
    pub fn new(i: usize, omegas: Vec<f64>, profiles: Vec<Vec<f64>>) -> Self {
        assert_eq!(omegas.len(), profiles.len(), "one profile per frequency");
        let n = omegas.len();
        Self {
            i,
            omegas,
            profiles,
            accum: vec![C64::ZERO; n],
            samples: 0,
        }
    }

    /// Project the current Ez column onto each frequency's mode profile and
    /// accumulate the DFT terms. `t` is the physical time of the grid's
    /// current Ez (i.e. (n+1)·Δt after step n).
    pub fn record(&mut self, g: &Grid2d, t: f64) {
        let col = &g.ez[self.i * g.ny..(self.i + 1) * g.ny];
        for ((acc, &w), prof) in self.accum.iter_mut().zip(&self.omegas).zip(&self.profiles) {
            let o: f64 = col.iter().zip(prof).map(|(&e, &p)| e * p).sum();
            *acc = *acc + C64::from_polar(o, w * t);
        }
        self.samples += 1;
    }

    /// Riemann-sum DFT mode amplitudes a(ω) (raw accumulators × Δt),
    /// one per registered frequency.
    pub fn amplitudes(&self, dt: f64) -> Vec<C64> {
        self.accum.iter().map(|&a| a * dt).collect()
    }

    /// Number of `record` calls so far.
    pub fn samples(&self) -> u64 {
        self.samples
    }
}
