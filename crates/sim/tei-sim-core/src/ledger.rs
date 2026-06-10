//! EventLedger — typed counters every simulator emits.
//!
//! The ledger is the calibration loop: measured event counts feed back into
//! the cost dialects, replacing assumed constants with observed quantities.

use serde::Serialize;

#[derive(Debug, Clone, Default, Serialize)]
pub struct EventLedger {
    /// Gibbs / annealing sweeps actually run.
    pub sweeps: u64,
    /// Individual spin-update proposals.
    pub spin_updates: u64,
    /// Accepted state flips.
    pub flips: u64,
    /// Spikes fired (spiking column).
    pub spikes: u64,
    /// Synaptic operations = spikes × fanout.
    pub sops: u64,
    /// Analog samples digitized (ADC events).
    pub adc_samples: u64,
    /// Modulator events (photonic column).
    pub modulator_events: u64,
    /// Multiply-accumulates executed.
    pub macs: u64,
    /// Integrated dissipation, joules (circuit column ∫i·v dt).
    pub joules: f64,
    /// Wall-clock seconds of the simulation run (None on wasm).
    pub wall_seconds: Option<f64>,
}

impl EventLedger {
    pub fn merge(&mut self, other: &EventLedger) {
        self.sweeps += other.sweeps;
        self.spin_updates += other.spin_updates;
        self.flips += other.flips;
        self.spikes += other.spikes;
        self.sops += other.sops;
        self.adc_samples += other.adc_samples;
        self.modulator_events += other.modulator_events;
        self.macs += other.macs;
        self.joules += other.joules;
    }
}
