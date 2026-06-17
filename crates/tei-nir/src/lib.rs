//! Deterministic place-and-route for spiking-neural graphs.
//!
//! The neuromorphic toolchain's highest-leverage gap (see
//! `docs/NEUROMORPHIC.md`): mapping a spiking graph onto a chip's cores is an
//! NP-hard partition-under-routing-constraints problem, today solved by slow,
//! **non-deterministic** Python heuristics that leave the silicon's advantage
//! on the table. This is the OpenIE-shaped answer: a fast, **reproducible,
//! attested** mapper — the same graph + target produces a byte-identical
//! placement and a stable [`Placement::checksum`], so a mapping can be cached,
//! diffed, and audited.
//!
//! This crate is the **place-and-route core** (step 2 of the plan). It is
//! host-side compiler tooling (std + alloc), like `tei-cost-surface` — not
//! firmware. What it deliberately does NOT do yet, and why each is a clean
//! follow-on:
//! - **NIR ingest** (step 1): parse the real NIR serialization into
//!   [`SpikingGraph`]. Here the graph is constructed directly; the mapper
//!   doesn't care where it came from.
//! - **Joule pricing** (step 3): feed a [`Placement`] through the cost surface
//!   (a substrate per registered target). The placement already exposes the
//!   routing cost that pricing keys on.
//! - **Per-chip constraint models**: real Loihi-2 / SpiNNaker2 / Speck core
//!   limits. [`Target`] carries the two that dominate (neuron capacity + core
//!   count); richer fan-in/out/memory limits slot into the same fit check.
//!
//! ## Why it's not "just bin-packing"
//!
//! Pure capacity bin-packing ignores connectivity and pays for it in
//! cross-core spike routing (the real energy + latency cost on these chips).
//! [`place`] is **connectivity-aware**: it places the heaviest populations
//! first and, among the cores a population fits on, picks the one that adds the
//! fewest cross-core synapses to already-placed neighbors — deterministically
//! (fixed order, lowest-core-index tie-break). It reports the resulting
//! `cross_core_synapses` as the routing-cost objective a real solver (or the
//! cost surface) optimizes against.

use std::collections::BTreeMap;

mod nir;
pub use nir::{classify, lower, NirGraph, NirKind, NirNode};

#[cfg(feature = "hdf5")]
mod hdf5_reader;
#[cfg(feature = "hdf5")]
pub use hdf5_reader::from_hdf5;

/// A node: one neuron population (or layer), with its neuron count.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Population {
    pub id: u32,
    pub neurons: u32,
}

/// An edge: a synaptic projection between two populations, sized by the number
/// of synapses (the thing that costs to route across cores).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Connection {
    pub from: u32,
    pub to: u32,
    pub synapses: u64,
}

/// A spiking graph — NIR-shaped, but constructed directly here.
#[derive(Debug, Clone, Default)]
pub struct SpikingGraph {
    pub populations: Vec<Population>,
    pub connections: Vec<Connection>,
}

impl SpikingGraph {
    pub fn new() -> Self {
        Self::default()
    }
    pub fn population(mut self, id: u32, neurons: u32) -> Self {
        self.populations.push(Population { id, neurons });
        self
    }
    pub fn connect(mut self, from: u32, to: u32, synapses: u64) -> Self {
        self.connections.push(Connection { from, to, synapses });
        self
    }
}

/// The mapping target: how many cores, and each core's neuron capacity. The
/// two constraints that dominate placement; richer limits extend [`fits`].
///
/// [`fits`]: Target::fits
#[derive(Debug, Clone, Copy)]
pub struct Target {
    pub cores: usize,
    pub neurons_per_core: u32,
}

impl Target {
    fn fits(&self, used: u32, add: u32) -> bool {
        used.saturating_add(add) <= self.neurons_per_core
    }
}

/// The result of a placement: which core each population landed on, per-core
/// neuron utilization, the cross-core routing cost, whether anything had to
/// overflow its core, and a stable checksum of the assignment.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Placement {
    /// population id → core index.
    pub core_of: BTreeMap<u32, usize>,
    /// neurons assigned to each core (length == `target.cores`).
    pub per_core_neurons: Vec<u32>,
    /// synapses whose endpoints landed on different cores — the routing cost.
    pub cross_core_synapses: u64,
    /// true if some population didn't fit any core's capacity (placed on the
    /// least-full core anyway, so the mapping is reported, not silently lost).
    pub overflowed: bool,
    /// stable hash of the (id, core) assignment — same placement ⇒ same value.
    pub checksum: u64,
}

/// Deterministically place `graph` onto `target`, connectivity-aware.
///
/// Order: populations by neuron count descending, then id ascending (so the
/// result never depends on input order or hash iteration). Each population
/// goes on the fitting core that adds the fewest cross-core synapses to its
/// already-placed neighbors; ties break to the lowest core index. If no core
/// fits, it lands on the least-full core and `overflowed` is set.
pub fn place(graph: &SpikingGraph, target: &Target) -> Placement {
    let cores = target.cores.max(1);
    let mut per_core_neurons = vec![0u32; cores];
    let mut core_of: BTreeMap<u32, usize> = BTreeMap::new();

    // Deterministic placement order.
    let mut order: Vec<&Population> = graph.populations.iter().collect();
    order.sort_by(|a, b| b.neurons.cmp(&a.neurons).then(a.id.cmp(&b.id)));

    let mut overflowed = false;
    for pop in order {
        // Cost of placing `pop` on core `c`: synapses to already-placed
        // neighbors that would then cross a core boundary.
        let added_cross = |c: usize| -> u64 {
            let mut x = 0u64;
            for conn in &graph.connections {
                let other = if conn.from == pop.id {
                    Some(conn.to)
                } else if conn.to == pop.id {
                    Some(conn.from)
                } else {
                    None
                };
                if let Some(o) = other {
                    if let Some(&oc) = core_of.get(&o) {
                        if oc != c {
                            x = x.saturating_add(conn.synapses);
                        }
                    }
                }
            }
            x
        };

        // Prefer a fitting core minimizing added cross-core synapses (lowest
        // index breaks ties). Fall back to the least-full core on overflow.
        let mut best_fit: Option<(usize, u64)> = None;
        for c in 0..cores {
            if target.fits(per_core_neurons[c], pop.neurons) {
                let cost = added_cross(c);
                if best_fit.map(|(_, bc)| cost < bc).unwrap_or(true) {
                    best_fit = Some((c, cost));
                }
            }
        }
        let chosen = match best_fit {
            Some((c, _)) => c,
            None => {
                overflowed = true;
                // least-full core (lowest index tie-break)
                (0..cores).min_by_key(|&c| per_core_neurons[c]).unwrap()
            }
        };
        per_core_neurons[chosen] = per_core_neurons[chosen].saturating_add(pop.neurons);
        core_of.insert(pop.id, chosen);
    }

    // Routing cost over the whole graph.
    let mut cross_core_synapses = 0u64;
    for conn in &graph.connections {
        if let (Some(&a), Some(&b)) = (core_of.get(&conn.from), core_of.get(&conn.to)) {
            if a != b {
                cross_core_synapses = cross_core_synapses.saturating_add(conn.synapses);
            }
        }
    }

    Placement {
        checksum: checksum(&core_of),
        core_of,
        per_core_neurons,
        cross_core_synapses,
        overflowed,
    }
}

/// FNV-1a over the sorted `(id, core)` assignment — the attestation hash.
/// `BTreeMap` iterates in key order, so this is order-stable by construction.
fn checksum(core_of: &BTreeMap<u32, usize>) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut feed = |b: u8| {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    };
    for (&id, &core) in core_of {
        for b in id.to_le_bytes() {
            feed(b);
        }
        for b in (core as u64).to_le_bytes() {
            feed(b);
        }
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_same_inputs_same_placement() {
        let g = SpikingGraph::new()
            .population(1, 100)
            .population(2, 100)
            .population(3, 100)
            .connect(1, 2, 500)
            .connect(2, 3, 500);
        let t = Target { cores: 2, neurons_per_core: 200 };
        let a = place(&g, &t);
        let b = place(&g, &t);
        assert_eq!(a, b);
        assert_eq!(a.checksum, b.checksum);
        // ...and independent of input order (the heart of "attested").
        let g2 = SpikingGraph::new()
            .population(3, 100)
            .population(1, 100)
            .population(2, 100)
            .connect(2, 3, 500)
            .connect(1, 2, 500);
        assert_eq!(place(&g2, &t).checksum, a.checksum);
    }

    #[test]
    fn fits_on_one_core_means_zero_routing() {
        let g = SpikingGraph::new()
            .population(1, 50)
            .population(2, 50)
            .connect(1, 2, 999);
        let t = Target { cores: 4, neurons_per_core: 1000 };
        let p = place(&g, &t);
        assert_eq!(p.cross_core_synapses, 0); // both on one core
        assert!(!p.overflowed);
        assert_eq!(p.core_of[&1], p.core_of[&2]);
    }

    #[test]
    fn connectivity_aware_keeps_a_clique_together() {
        // Two tightly-coupled pops (heavy synapse count) + a third that forces
        // a split. A capacity-only packer might separate 1&2; the
        // connectivity-aware one keeps the heavy edge on one core.
        let g = SpikingGraph::new()
            .population(1, 100)
            .population(2, 100)
            .population(3, 100)
            .connect(1, 2, 10_000) // heavy — must not cross
            .connect(2, 3, 1); // light — fine to cross
        let t = Target { cores: 2, neurons_per_core: 200 }; // 3 pops, 2 cores → a split is forced
        let p = place(&g, &t);
        assert!(!p.overflowed);
        assert_eq!(p.core_of[&1], p.core_of[&2], "heavy edge stays on-core");
        assert_eq!(p.cross_core_synapses, 1, "only the light edge crosses");
    }

    #[test]
    fn overflow_is_flagged_not_dropped() {
        let g = SpikingGraph::new().population(1, 500).population(2, 500);
        let t = Target { cores: 1, neurons_per_core: 600 }; // can't hold both
        let p = place(&g, &t);
        assert!(p.overflowed);
        assert_eq!(p.core_of.len(), 2); // both still placed (reported, not lost)
    }
}
