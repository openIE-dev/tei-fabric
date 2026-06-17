//! NIR ingest — lower a Neuromorphic Intermediate Representation graph into a
//! [`SpikingGraph`] the mapper can place.
//!
//! NIR (the neuromorphs group's interop IR) is a static graph of typed nodes +
//! edges: ~11 computational primitives — neuron dynamics (`LIF`, `CubaLIF`,
//! `LI`, `IF`), weight maps (`Affine`, `Linear`, `Conv1d/2d`), `Input`/`Output`
//! boundaries, and structural ops (`Flatten`, `SumPool2d`, `Threshold`, …).
//! Crucially NIR **interleaves** populations and weights: a network reads
//! `Input → Affine(W) → LIF → Affine(W) → LIF → Output`, where the `Affine`
//! nodes ARE the synapses and the `LIF` nodes are the neuron populations.
//!
//! Lowering therefore **contracts** the weight nodes onto edges:
//! - neuron nodes → [`crate::Population`] (neurons = its parameter shape size);
//! - `Input`/`Output` → 0-neuron boundary populations (anchor connections, take
//!   no core capacity);
//! - each weight node → a [`crate::Connection`] between the nearest upstream and
//!   downstream populations, `synapses` = the weight's element count (walking
//!   through structural nodes like `Flatten`/`Pool`).
//!
//! ## Transport
//!
//! NIR's reference serialization is **HDF5**. This module ingests the NIR graph
//! *model* from a JSON projection (`{nodes:{name:{type, shape|weight_shape}},
//! edges:[[from,to]]}`) — dependency-light and the part that actually lowers.
//! An HDF5 reader (the `hdf5` crate) that produces the same [`NirGraph`] is a
//! thin, clean follow-on; everything below is transport-agnostic.

use crate::{Connection, Population, SpikingGraph};
use serde::Deserialize;
use std::collections::BTreeMap;

/// How a NIR node maps into the spiking graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NirKind {
    /// Neuron dynamics → a population (carries neurons).
    Neuron,
    /// Weight map → synapses on an edge (carries a synapse count).
    Weight,
    /// Graph input boundary.
    Input,
    /// Graph output boundary.
    Output,
    /// Reshape/pool/threshold/etc. — passthrough for connectivity.
    Structural,
}

/// Classify a NIR node `type` string.
pub fn classify(ty: &str) -> NirKind {
    match ty {
        "LIF" | "CubaLIF" | "LI" | "IF" | "I" => NirKind::Neuron,
        "Affine" | "Linear" => NirKind::Weight,
        _ if ty.starts_with("Conv") => NirKind::Weight,
        "Input" => NirKind::Input,
        "Output" => NirKind::Output,
        // Flatten, SumPool2d, AvgPool2d, Threshold, Scale, Delay, Sequence…
        _ => NirKind::Structural,
    }
}

/// One ingested NIR node, with its size resolved: neuron count for
/// neuron/boundary nodes, synapse count for weight nodes, 0 for structural.
#[derive(Debug, Clone)]
pub struct NirNode {
    pub name: String,
    pub ty: String,
    pub kind: NirKind,
    pub size: u64,
}

/// A parsed NIR graph (nodes sorted by name for determinism).
#[derive(Debug, Clone, Default)]
pub struct NirGraph {
    pub nodes: Vec<NirNode>,
    pub edges: Vec<(String, String)>,
}

#[derive(Deserialize)]
struct RawNode {
    #[serde(rename = "type")]
    ty: String,
    #[serde(default)]
    shape: Vec<u64>,
    #[serde(default)]
    weight_shape: Vec<u64>,
}

#[derive(Deserialize)]
struct RawGraph {
    nodes: BTreeMap<String, RawNode>,
    #[serde(default)]
    edges: Vec<(String, String)>,
}

fn product(dims: &[u64]) -> u64 {
    if dims.is_empty() {
        0
    } else {
        dims.iter().product()
    }
}

impl NirGraph {
    /// Parse the JSON projection of a NIR graph.
    pub fn from_json(s: &str) -> Result<NirGraph, serde_json::Error> {
        let raw: RawGraph = serde_json::from_str(s)?;
        let mut nodes: Vec<NirNode> = raw
            .nodes
            .into_iter()
            .map(|(name, n)| {
                let kind = classify(&n.ty);
                // Weights size by their weight matrix; everything else by its
                // (neuron) shape. Fall back to the other field if one's absent.
                let size = match kind {
                    NirKind::Weight => {
                        let w = product(&n.weight_shape);
                        if w > 0 {
                            w
                        } else {
                            product(&n.shape)
                        }
                    }
                    _ => {
                        let s = product(&n.shape);
                        if s > 0 {
                            s
                        } else {
                            product(&n.weight_shape)
                        }
                    }
                };
                NirNode { name, ty: n.ty, kind, size }
            })
            .collect();
        nodes.sort_by(|a, b| a.name.cmp(&b.name)); // determinism
        Ok(NirGraph { nodes, edges: raw.edges })
    }

    fn node(&self, name: &str) -> Option<&NirNode> {
        self.nodes.iter().find(|n| n.name == name)
    }

    fn is_pop(&self, name: &str) -> bool {
        matches!(
            self.node(name).map(|n| n.kind),
            Some(NirKind::Neuron) | Some(NirKind::Input) | Some(NirKind::Output)
        )
    }
}

/// Lower a NIR graph to a [`SpikingGraph`]: populations from neuron/boundary
/// nodes, connections from contracted weight nodes (+ any direct
/// population→population edges, as 0-synapse links).
pub fn lower(g: &NirGraph) -> SpikingGraph {
    // Stable population ids, assigned in (sorted) node order.
    let mut id_of: BTreeMap<&str, u32> = BTreeMap::new();
    let mut populations = Vec::new();
    let mut next = 1u32;
    for n in &g.nodes {
        if matches!(n.kind, NirKind::Neuron | NirKind::Input | NirKind::Output) {
            id_of.insert(n.name.as_str(), next);
            populations.push(Population {
                id: next,
                // boundaries take no core capacity; only neuron nodes do.
                neurons: if n.kind == NirKind::Neuron { n.size as u32 } else { 0 },
            });
            next += 1;
        }
    }

    // Walk structural nodes to the nearest population up/downstream.
    let succ = |name: &str| -> Vec<&str> {
        g.edges
            .iter()
            .filter(|(a, _)| a == name)
            .map(|(_, b)| b.as_str())
            .collect()
    };
    let pred = |name: &str| -> Vec<&str> {
        g.edges
            .iter()
            .filter(|(_, b)| b == name)
            .map(|(a, _)| a.as_str())
            .collect()
    };
    fn resolve<'a>(
        g: &'a NirGraph,
        start: &'a str,
        step: &dyn Fn(&str) -> Vec<&'a str>,
        depth: u32,
    ) -> Option<&'a str> {
        if depth > 64 {
            return None; // cycle / pathological — bail deterministically
        }
        if g.is_pop(start) {
            return Some(start);
        }
        match g.node(start).map(|n| n.kind) {
            // hop through structural nodes; weights terminate (handled apart)
            Some(NirKind::Structural) => {
                step(start).into_iter().find_map(|n| resolve(g, n, step, depth + 1))
            }
            _ => None,
        }
    }

    let mut connections = Vec::new();
    for w in g.nodes.iter().filter(|n| n.kind == NirKind::Weight) {
        for p in pred(&w.name) {
            let Some(src) = (if g.is_pop(p) { Some(p) } else { resolve(g, p, &pred, 0) }) else {
                continue;
            };
            for s in succ(&w.name) {
                let Some(dst) = (if g.is_pop(s) { Some(s) } else { resolve(g, s, &succ, 0) })
                else {
                    continue;
                };
                if let (Some(&from), Some(&to)) = (id_of.get(src), id_of.get(dst)) {
                    connections.push(Connection { from, to, synapses: w.size });
                }
            }
        }
    }
    // Direct population→population edges (no weight between) — 0-synapse links
    // that still express connectivity to the placer.
    for (a, b) in &g.edges {
        if g.is_pop(a) && g.is_pop(b) {
            if let (Some(&from), Some(&to)) = (id_of.get(a.as_str()), id_of.get(b.as_str())) {
                connections.push(Connection { from, to, synapses: 0 });
            }
        }
    }

    SpikingGraph { populations, connections }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{place, Target};

    // Input(2) → fc1 Affine[4,2] → lif1 LIF(4) → fc2 Affine[3,4] → lif2 LIF(3)
    //          → out Output(3)
    const MLP: &str = r#"{
        "nodes": {
            "in":   {"type":"Input",  "shape":[2]},
            "fc1":  {"type":"Affine", "weight_shape":[4,2]},
            "lif1": {"type":"LIF",    "shape":[4]},
            "fc2":  {"type":"Affine", "weight_shape":[3,4]},
            "lif2": {"type":"LIF",    "shape":[3]},
            "out":  {"type":"Output", "shape":[3]}
        },
        "edges": [["in","fc1"],["fc1","lif1"],["lif1","fc2"],["fc2","lif2"],["lif2","out"]]
    }"#;

    #[test]
    fn lowers_an_mlp_snn() {
        let g = NirGraph::from_json(MLP).unwrap();
        let sg = lower(&g);
        // populations: in, lif1, lif2, out (4 pop-nodes); fc1/fc2 are weights.
        assert_eq!(sg.populations.len(), 4);
        let neurons = |id: u32| sg.populations.iter().find(|p| p.id == id).unwrap().neurons;
        // ids assigned in sorted name order: in,lif1,lif2,out
        // (fc* are not populations). Find the two LIFs by neuron count.
        let mut counts: Vec<u32> = sg.populations.iter().map(|p| p.neurons).collect();
        counts.sort_unstable();
        assert_eq!(counts, vec![0, 0, 3, 4]); // 2 boundaries (0), lif2=3, lif1=4
        let _ = neurons;
        // connections: fc1 (in→lif1, 8 syn), fc2 (lif1→lif2, 12 syn).
        let syns: u64 = sg.connections.iter().map(|c| c.synapses).sum();
        assert_eq!(syns, 4 * 2 + 3 * 4); // 8 + 12 = 20
        assert_eq!(sg.connections.iter().filter(|c| c.synapses > 0).count(), 2);
    }

    #[test]
    fn structural_node_is_traversed() {
        // Input → flat Flatten → fc Affine[2,4] → lif LIF(2)
        let j = r#"{
            "nodes": {
                "in":  {"type":"Input","shape":[2,2]},
                "flat":{"type":"Flatten"},
                "fc":  {"type":"Affine","weight_shape":[2,4]},
                "lif": {"type":"LIF","shape":[2]}
            },
            "edges": [["in","flat"],["flat","fc"],["fc","lif"]]
        }"#;
        let sg = lower(&NirGraph::from_json(j).unwrap());
        // fc's upstream is `in` via the structural `flat`; downstream is `lif`.
        let c = sg.connections.iter().find(|c| c.synapses == 8).expect("in→lif edge");
        let in_id = 1; // 'in' sorts first
        assert_eq!(c.from, in_id);
    }

    #[test]
    fn ingest_then_place_is_deterministic_and_attested() {
        let sg = lower(&NirGraph::from_json(MLP).unwrap());
        let t = Target { cores: 2, neurons_per_core: 8 };
        let a = place(&sg, &t);
        let b = place(&sg, &t);
        assert_eq!(a.checksum, b.checksum);
        assert!(!a.overflowed); // 4+3 neurons across 2 cores of 8
    }
}
