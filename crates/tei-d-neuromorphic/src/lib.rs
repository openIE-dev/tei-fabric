//! Neuromorphic substrate — LIF spike-event physics.
//!
//! A neuromorphic substrate executes spiking-neural-network primitives as
//! physical events: a neuron's membrane potential integrates weighted input
//! spikes and fires when it crosses threshold. Energy is spent **per event**,
//! not per clock — a silent neuron costs (almost) nothing. That event-driven
//! sparsity is the substrate's structural advantage over dense simulation:
//! a GPU pays for every synapse every timestep; neuromorphic silicon pays
//! only for the spikes that actually happen.
//!
//! Cost decomposition for one SNN run (`neurons × timesteps`, activity `a`,
//! fan-out `f` synapses per neuron):
//!
//!   1. **Synaptic operations.** Each spike event propagates to `f`
//!      downstream synapses. Published per-SOP energies on shipping
//!      neuromorphic silicon cluster at 20-30 pJ:
//!      Loihi 23.6 pJ/SOP (Davies 2018), TrueNorth 26 pJ/SOP (Merolla
//!      2014); Loihi 2 improves on this. We use 20 pJ/SOP.
//!   2. **Neuron updates.** Every neuron's membrane state advances each
//!      timestep whether or not it fires (leak + integrate). Loihi-class
//!      figures: ~60-81 pJ per active neuron update. We use 60 pJ.
//!   3. **Plasticity (STDP only).** On-line learning pays a weight-update
//!      per synaptic event on top of the SOP itself. Loihi's programmable
//!      learning engine lands at roughly 2-3× the inference SOP energy.
//!      We add 40 pJ per plastic synaptic event.
//!
//! The `OpProfile` mapping reuses the sampling-shape fields:
//!   - `variables` → neuron count
//!   - `sweeps`    → timesteps
//!   - `reduce_dim`→ fan-out (synapses per neuron); default 128
//!   - `sparsity`  → inactivity: activity factor = `1 - sparsity`. When the
//!     workload leaves sparsity at 0 (the "dense" default that makes sense
//!     for matmuls), we substitute the typical SNN activity of 0.1 — a
//!     fully-dense spike train is not a meaningful operating point.
//!
//! Citations:
//!   - Davies et al. 2018, *IEEE Micro* 38(1): 82 — Loihi, 23.6 pJ/SOP.
//!   - Merolla et al. 2014, *Science* 345: 668 — TrueNorth, 26 pJ/SOP.
//!   - Orchard et al. 2021, *IEEE SiPS* — Loihi 2 architecture.
//!   - Höppner et al. 2021, arXiv:2103.08392 — SpiNNaker 2 energy figures.
//!   - NIR (Pedersen et al. 2024, *Neuromorphic Comput. Eng.*) — the
//!     cross-platform SNN IR this dialect would lower through in a full
//!     compiler flow.

use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tei_ir::OpProfile;
use tei_stack::{Primitive, Stack};
use tei_substrate_traits::{Cost, Substrate};

/// Tunable engineering parameters for the neuromorphic substrate.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NeuromorphicParams {
    /// Energy per synaptic operation, joules. Loihi 23.6 pJ, TrueNorth 26 pJ.
    pub sop_j: f64,
    /// Energy per neuron membrane update per timestep, joules.
    pub neuron_update_j: f64,
    /// Extra energy per plastic (learning) synaptic event, joules.
    pub plasticity_j: f64,
    /// Default fan-out when the workload doesn't specify one.
    pub default_fanout: u32,
}

impl Default for NeuromorphicParams {
    fn default() -> Self {
        Self {
            sop_j: 20.0e-12,
            neuron_update_j: 60.0e-12,
            plasticity_j: 40.0e-12,
            default_fanout: 128,
        }
    }
}

/// Sustained timestep rate at the system level. Loihi-class chips run small
/// networks at ~10 kHz algorithmic timesteps.
const TIMESTEPS_PER_SEC: f64 = 1.0e4;

/// Typical SNN activity factor substituted when the workload leaves
/// `sparsity` at the dense default.
pub const DEFAULT_ACTIVITY: f64 = 0.1;

/// Catalog primitive ID for STDP (NEURO family) — the only primitive that
/// pays the plasticity term. LIF (id 50) and future NEURO additions route
/// through the same SOP + membrane-update model.
const PRIM_STDP: u32 = 51;

pub struct Neuromorphic {
    stack: Arc<Stack>,
    pub params: NeuromorphicParams,
}

impl Neuromorphic {
    pub fn new(stack: Arc<Stack>) -> Self {
        Self {
            stack,
            params: NeuromorphicParams::default(),
        }
    }

    pub fn with_params(stack: Arc<Stack>, params: NeuromorphicParams) -> Self {
        Self { stack, params }
    }

    fn snn_energy(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        let p = &self.params;
        let neurons = profile.variables.unwrap_or(1024) as f64;
        let timesteps = profile.sweeps.unwrap_or(1000) as f64;
        let fanout = profile.reduce_dim.unwrap_or(p.default_fanout as usize) as f64;
        let activity = if profile.sparsity > 0.0 {
            (1.0 - profile.sparsity).clamp(0.0, 1.0)
        } else {
            DEFAULT_ACTIVITY
        };
        let batch = profile.batch as f64;

        // Spike events across the run.
        let events = neurons * timesteps * activity * batch;
        // Every event fans out to `fanout` synapses.
        let sop_j = events * fanout * p.sop_j;
        // Membrane update for every neuron every timestep, firing or not.
        let update_j = neurons * timesteps * batch * p.neuron_update_j;
        // STDP pays the learning-engine update per plastic synaptic event.
        let plastic_j = if primitive.id == PRIM_STDP {
            events * fanout * p.plasticity_j
        } else {
            0.0
        };

        Cost {
            joules_per_op: sop_j + update_j + plastic_j,
            seconds_per_op: timesteps * batch / TIMESTEPS_PER_SEC,
            // Rate-coded SNN inference trades timesteps for precision; at
            // typical operating points published accuracy loss vs the ANN
            // baseline is a few percent.
            accuracy_loss: 0.02,
        }
    }
}

impl Substrate for Neuromorphic {
    fn name(&self) -> &str {
        "neuromorphic"
    }
    fn display_name(&self) -> &str {
        "Neuromorphic (LIF spiking)"
    }

    /// Claims the NEURO family from the catalog — auto-expands as the
    /// Periodic Stack gains spiking primitives.
    fn supports(&self, primitive: &Primitive) -> bool {
        self.stack
            .family("NEURO")
            .iter()
            .filter_map(|&i| self.stack.primitives().get(i))
            .any(|p| p.id == primitive.id)
    }

    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        self.snn_energy(primitive, profile)
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "Davies et al. 2018, IEEE Micro 38(1): 82 — Loihi, 23.6 pJ/SOP.",
            "Merolla et al. 2014, Science 345: 668 — TrueNorth, 26 pJ/SOP.",
            "Orchard et al. 2021, IEEE SiPS — Loihi 2 architecture.",
            "Höppner et al. 2021, arXiv:2103.08392 — SpiNNaker 2 energy figures.",
            "Pedersen et al. 2024, Neuromorphic Comput. Eng. — NIR cross-platform SNN IR.",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tei_ir::{Dtype, TensorShape};

    fn load_stack() -> Arc<Stack> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/stack.json");
        Stack::load_from_path(path).expect("catalog loads")
    }

    fn snn_profile() -> OpProfile {
        OpProfile {
            shape: TensorShape { dims: vec![4096] },
            reduce_dim: Some(128),
            batch: 1,
            dtype: Dtype::I8,
            sparsity: 0.9, // 10% activity
            sweeps: Some(1000),
            variables: Some(4096),
        }
    }

    /// Anchor: 4096 neurons × 1000 steps × 0.1 activity × 128 fan-out
    /// × 20 pJ/SOP + 4096 × 1000 × 60 pJ membrane updates = 1.294 mJ.
    #[test]
    fn lif_anchor_1_294_mj() {
        let stack = load_stack();
        let s = Neuromorphic::new(stack.clone());
        let lif = stack.get(50).expect("LIF in catalog");
        let cost = s.cost(lif, &snn_profile());
        let events = 4096.0 * 1000.0 * 0.1;
        let expected = events * 128.0 * 20.0e-12 + 4096.0 * 1000.0 * 60.0e-12;
        assert!(
            (cost.joules_per_op - expected).abs() / expected < 1e-9,
            "{:.4e} != {expected:.4e}",
            cost.joules_per_op
        );
    }

    /// STDP pays the plasticity term on top of LIF inference.
    #[test]
    fn stdp_costs_more_than_lif() {
        let stack = load_stack();
        let s = Neuromorphic::new(stack.clone());
        let lif = stack.get(50).unwrap();
        let stdp = stack.get(51).unwrap();
        let prof = snn_profile();
        assert!(s.cost(stdp, &prof).joules_per_op > s.cost(lif, &prof).joules_per_op);
    }

    /// supports() is catalog-driven: claims exactly the NEURO family.
    #[test]
    fn claims_neuro_family_only() {
        let stack = load_stack();
        let s = Neuromorphic::new(stack.clone());
        assert!(s.supports(stack.get(50).unwrap())); // LIF
        assert!(s.supports(stack.get(51).unwrap())); // STDP
        assert!(!s.supports(stack.get(18).unwrap())); // Dense MatMul
    }
}
