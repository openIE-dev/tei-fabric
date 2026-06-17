//! Price a [`Placement`] in joules per inference — step 3 of the mapper plan,
//! the hook to the "measured not assumed" ledger.
//!
//! The whole point of the deterministic, connectivity-aware mapper is that
//! cross-core spike routing costs energy. This module makes that concrete: a
//! placement that leaves more synapses crossing cores prices **higher**, so
//! the mapper's objective (`cross_core_synapses`) shows up as joules. That is
//! the lever — a better mapping lets the chip realize more of its theoretical
//! efficiency, measured in J.
//!
//! ## Structural estimate, honest provenance
//!
//! A placement gives *structure* (neurons, synapses, the cross-core split),
//! not *activity* (how many spikes actually fire). So the estimate multiplies
//! structure by an **assumed** average spike rate and a per-substrate energy
//! cost table — exactly the `JoulesSource::Table` tier of the embedded ledger.
//! When a dev kit (Akida Cloud, a Speck/Xylo board) measures real J/inference,
//! the same shape becomes `Measured` and the costs are recalibrated. The
//! per-substrate constants here are **illustrative defaults, not vendor
//! specs** — calibration replaces them.

use crate::{Placement, SpikingGraph};

/// Provenance of an energy figure — mirrors the embedded ledger's tiers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Provenance {
    /// Assumed activity × a shipped cost table (honest default).
    Table,
    /// Calibrated against a real measurement on the target.
    Measured,
}

/// Per-substrate energy constants + the assumed activity that turns structure
/// into joules.
#[derive(Debug, Clone, Copy)]
pub struct NeuroCost {
    /// Joules per neuron state update.
    pub e_neuron_j: f64,
    /// Joules per on-core synaptic operation.
    pub e_synop_j: f64,
    /// Cross-core synaptic-op energy multiplier (≥ 1) — the routing premium
    /// the mapper exists to minimize.
    pub route_factor: f64,
    /// Assumed average spikes per neuron per inference (the activity the
    /// structure is multiplied by). `Table`-tier until measured.
    pub spikes_per_neuron: f64,
    /// Where the constants came from.
    pub source: Provenance,
}

impl NeuroCost {
    /// **Illustrative** order-of-magnitude defaults — NOT a vendor spec. Real
    /// numbers come from calibrating on the target (then `source = Measured`).
    /// Picked only so the routing premium is visible in tests/demos: a
    /// cross-core synop costs 8× an on-core one.
    pub fn illustrative() -> Self {
        Self {
            e_neuron_j: 1.0e-12,  // 1 pJ / neuron update
            e_synop_j: 1.0e-13,   // 0.1 pJ / on-core synop
            route_factor: 8.0,    // cross-core spikes are dramatically pricier
            spikes_per_neuron: 1.0,
            source: Provenance::Table,
        }
    }
}

/// The priced result of a placement.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnergyEstimate {
    /// Estimated joules for one inference.
    pub joules: f64,
    pub neuron_updates: u64,
    pub synops: u64,
    pub cross_core_synops: u64,
    pub source: Provenance,
}

/// Price a placement: structure (from `graph` + `placement`) × assumed
/// activity × `cost`, with the routing premium applied to cross-core synops.
pub fn price(graph: &SpikingGraph, placement: &Placement, cost: &NeuroCost) -> EnergyEstimate {
    let total_neurons: u64 = placement.per_core_neurons.iter().map(|&n| n as u64).sum();
    let total_synapses: u64 = graph.connections.iter().map(|c| c.synapses).sum();
    let cross = placement.cross_core_synapses.min(total_synapses);
    let on_core = total_synapses - cross;

    // synops = activity × synapses (each presynaptic spike traverses its fan-out)
    let synops = (cost.spikes_per_neuron * total_synapses as f64).round() as u64;
    let cross_core_synops = (cost.spikes_per_neuron * cross as f64).round() as u64;
    let on_core_synops = (cost.spikes_per_neuron * on_core as f64).round() as u64;

    let joules = total_neurons as f64 * cost.e_neuron_j
        + on_core_synops as f64 * cost.e_synop_j
        + cross_core_synops as f64 * cost.e_synop_j * cost.route_factor;

    EnergyEstimate {
        joules,
        neuron_updates: total_neurons,
        synops,
        cross_core_synops,
        source: cost.source,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{place, NirGraph, Target};

    const MLP: &str = r#"{
        "nodes": {
            "in":{"type":"Input","shape":[2]},
            "fc1":{"type":"Affine","weight_shape":[4,2]},
            "lif1":{"type":"LIF","shape":[4]},
            "fc2":{"type":"Affine","weight_shape":[3,4]},
            "lif2":{"type":"LIF","shape":[3]},
            "out":{"type":"Output","shape":[3]}
        },
        "edges":[["in","fc1"],["fc1","lif1"],["lif1","fc2"],["fc2","lif2"],["lif2","out"]]
    }"#;

    fn graph() -> SpikingGraph {
        crate::lower(&NirGraph::from_json(MLP).unwrap())
    }

    #[test]
    fn pricing_is_deterministic_and_table_tier() {
        let g = graph();
        let p = place(&g, &Target { cores: 2, neurons_per_core: 8 });
        let c = NeuroCost::illustrative();
        let a = price(&g, &p, &c);
        let b = price(&g, &p, &c);
        assert_eq!(a, b);
        assert_eq!(a.source, Provenance::Table);
        assert_eq!(a.neuron_updates, 7); // 4 + 3 LIF neurons
        assert_eq!(a.synops, 20); // 8 + 12 synapses at 1 spike/neuron
        assert!(a.joules > 0.0);
    }

    #[test]
    fn a_worse_placement_prices_higher() {
        // Same graph; fit-on-fewer-cores vs forced-to-split. The split pays the
        // routing premium → strictly more joules. This is the mapper's value,
        // in joules.
        let g = graph();
        let c = NeuroCost::illustrative();
        let roomy = price(&g, &place(&g, &Target { cores: 1, neurons_per_core: 100 }), &c);
        let tight = price(&g, &place(&g, &Target { cores: 4, neurons_per_core: 4 }), &c);
        assert_eq!(roomy.cross_core_synops, 0); // everything on one core
        assert!(tight.cross_core_synops > 0); // forced to route
        assert!(tight.joules > roomy.joules, "cross-core routing costs more joules");
    }

    #[test]
    fn measured_costs_carry_through_provenance() {
        let g = graph();
        let p = place(&g, &Target { cores: 2, neurons_per_core: 8 });
        let mut c = NeuroCost::illustrative();
        c.source = Provenance::Measured; // as if calibrated on a dev kit
        assert_eq!(price(&g, &p, &c).source, Provenance::Measured);
    }
}
