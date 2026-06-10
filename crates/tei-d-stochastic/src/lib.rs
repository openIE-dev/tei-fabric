//! Stochastic substrate — sMTJ / p-bit / thermodynamic-sampling physics.
//!
//! A stochastic substrate samples from a probability distribution defined by
//! a network of probabilistic bits (p-bits). Each p-bit is a stochastic MTJ
//! whose magnetization thermally fluctuates between two states, gated by an
//! externally biased input. The compute *is* the noise — the thermal bath
//! does the sampling. This is the architectural premise of Extropic's
//! Thermodynamic Sampling Unit (Z1) and the broader Camsari-group p-bit
//! lineage.
//!
//! Energy decomposition per Gibbs sweep:
//!
//!   1. **Per-p-bit sample energy.** sMTJ devices dissipate roughly thermal
//!      noise's worth of energy per flip — well below conventional logic
//!      transition energies. Camsari lab measurements and Extropic device
//!      modeling cluster around 1 fJ per p-bit sample at room temperature.
//!   2. **Coupling matrix update.** For each sweep, the bias of each p-bit
//!      depends on the weighted sum of its neighbors' states. On Extropic-
//!      class hardware this is integrated analog; the energy term scales
//!      with edge count, not vertex count. We use a per-sweep readout
//!      overhead modeling the integration + the digital control path.
//!   3. **Sample readout / ADC.** Periodic snapshots of the p-bit array are
//!      digitized for the host. We charge one ADC sample per snapshot
//!      (every N sweeps); negligible per-sample under typical chain lengths.
//!
//! Net per-sample energy on shipping / pre-production stochastic accelerators
//! lands in the **1-10 fJ per p-bit-update range**. This crate uses 1 fJ
//! per p-bit-update + a 1 pJ per-sweep readout-and-coupling overhead.
//!
//! Fidelity: accuracy_loss = 0. Stochastic substrates are intentionally
//! random; their statistical output matches a software PRNG-based sampler
//! given enough samples. The cost difference is in *energy per equivalent
//! sample*, not output quality.
//!
//! Citations:
//!   - Camsari et al. 2017, *Phys. Rev. X* 7: 031014 — p-bit foundational.
//!   - Borders et al. 2019, *Nature* 573: 390 — integer factorization with sMTJ p-bits.
//!   - Aadit et al. 2024, *Nature Comms* 15: 8977 — sparse + higher-order Ising machines.
//!   - Extropic Z1 architecture announcement (Oct 2025) + thrml v0.1.3 docs.
//!   - thrml (github.com/extropic-ai/thrml) — JAX reference sampler.

use tei_ir::OpProfile;
use tei_stack::Primitive;
use tei_substrate_traits::{Cost, Substrate};

/// sMTJ per-sample energy, joules. Source: Camsari group + Extropic device modeling.
pub const PBIT_J_PER_SAMPLE: f64 = 1.0e-15;

/// Per-sweep coupling + readout overhead, joules. Dominated by the analog
/// integration network and any ADC snapshots. Roughly one small ADC sample
/// per sweep at the system level.
pub const READOUT_J_PER_SWEEP: f64 = 1.0e-12;

/// Sustained sweep rate at the system level (sweeps per second).
/// Stochastic substrates run sweep cycles at GHz rates per Camsari & Extropic
/// publications; the system rate is throttled by ADC + DMA. Modeling 1 GHz
/// sustained sweep rate as a defensible system-level number.
const STOCHASTIC_SWEEPS_PER_SEC: f64 = 1.0e9;

/// Native primitives — anything that naturally maps to thermodynamic sampling.
fn primitive_is_stochastic_native(p: &Primitive) -> bool {
    matches!(
        p.id,
        8   |  // Stochastic rounding
        38  |  // Monte Carlo accept/reject
        39  |  // MCMC / Langevin step
        99  |  // Bayesian posterior
        245 |  // Discrete Gaussian sampling
        251 |  // Bootstrap resampling
        258 |  // Simulated annealing
        274 // Lattice Boltzmann step
    )
}

/// Stochastic substrate (p-bit array / TSU).
pub struct Stochastic;

impl Stochastic {
    fn sampling_energy(&self, profile: &OpProfile) -> Cost {
        let (sweeps, variables) = match (profile.sweeps, profile.variables) {
            (Some(s), Some(v)) => (s, v as u64),
            // Sensible defaults if the workload didn't say.
            (Some(s), None) => (s, 64),
            (None, Some(v)) => (1000, v as u64),
            (None, None) => (1000, 64),
        };
        let batch = profile.batch as u64;

        let pbit_events = sweeps * variables * batch;
        let pbit_j = pbit_events as f64 * PBIT_J_PER_SAMPLE;
        let readout_j = sweeps as f64 * batch as f64 * READOUT_J_PER_SWEEP;

        let joules_per_op = pbit_j + readout_j;
        let seconds_per_op = (sweeps * batch) as f64 / STOCHASTIC_SWEEPS_PER_SEC;

        Cost {
            joules_per_op,
            seconds_per_op,
            accuracy_loss: 0.0,
        }
    }
}

impl Substrate for Stochastic {
    fn name(&self) -> &str {
        "stochastic"
    }
    fn display_name(&self) -> &str {
        "Stochastic (sMTJ p-bits)"
    }

    fn supports(&self, primitive: &Primitive) -> bool {
        primitive_is_stochastic_native(primitive)
    }

    fn cost(&self, _primitive: &Primitive, profile: &OpProfile) -> Cost {
        self.sampling_energy(profile)
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "Camsari et al. 2017, Phys. Rev. X 7: 031014 — p-bit foundational.",
            "Borders et al. 2019, Nature 573: 390 — sMTJ p-bit integer factorization.",
            "Aadit et al. 2024, Nature Comms 15: 8977 — sparse + higher-order Ising.",
            "Extropic Z1 architecture announcement (Oct 2025) + thrml v0.1.3.",
            "thrml (github.com/extropic-ai/thrml) — JAX reference sampler.",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tei_ir::{Dtype, TensorShape};

    fn mcmc() -> tei_stack::Primitive {
        serde_json::from_str(
            r#"{
            "id": 39, "name": "MCMC / Langevin step", "family": "PROB",
            "B": "kernel", "C": "L2+L3", "D": "sequential",
            "existing": "EM", "silicon_target": null, "wave": null
        }"#,
        )
        .unwrap()
    }

    /// Anchor: 1024 p-bits × 10k sweeps = 1.024e7 p-bit events × 1 fJ
    /// + 1e4 sweeps × 1 pJ readout = 20.24 nJ exactly.
    #[test]
    fn ising_anchor_20_24_nj() {
        let s = Stochastic;
        let prof = OpProfile {
            shape: TensorShape { dims: vec![1024] },
            reduce_dim: None,
            batch: 1,
            dtype: Dtype::I8,
            sparsity: 0.0,
            sweeps: Some(10_000),
            variables: Some(1024),
        };
        let cost = s.cost(&mcmc(), &prof);
        let expected = 1024.0 * 10_000.0 * 1.0e-15 + 10_000.0 * 1.0e-12;
        assert!(
            (cost.joules_per_op - expected).abs() / expected < 1e-9,
            "{:.4e} != {expected:.4e}",
            cost.joules_per_op
        );
    }
}
