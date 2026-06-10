//! Periodic Stack of Computation — loader and index.
//!
//! The Periodic Stack catalogs 258 compute primitives across 33 families
//! along with three edge types:
//!   - **composition**: primitive X is built from primitives Y, Z, …
//!   - **bennett_decomposition**: primitive X has a reversible (L₀) kernel
//!     and an irreversible projection (L₂) phase
//!   - **thermodynamic_inheritance**: primitive X inherits the thermo class
//!     of parent Y under composition
//!
//! The canonical source lives at compute.openie.dev. The mirror at
//! `data/stack.json` (workspace root) is what this crate loads at runtime.
//!
//! Thermodynamic classes used throughout:
//!   - **L0**       bijective / reversible-in-principle
//!   - **L1**       requires entropy (measurement, sampling)
//!   - **L2**       irreversible (information-erasing) at a fixed rate
//!   - **L2max**   maximally reductive (wide reduction, hash, sort)
//!   - **L0/L2**   bijective at the core, lossy on output projection

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Top-level catalog as deserialized from `stack.json`.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StackData {
    pub version: String,
    pub generated: String,
    pub totals: Totals,
    pub families: Vec<Family>,
    pub primitives: Vec<Primitive>,
    pub edges: Edges,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Totals {
    pub primitives: u32,
    pub families: u32,
    pub l0_free: u32,
    pub has_hardware: u32,
    pub hardware_gap: u32,
    pub composition_edges: u32,
    pub bennett_decompositions: u32,
    pub thermodynamic_inheritance_edges: u32,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Family {
    pub code: String,
    pub name: String,
    pub description: String,
    pub count: u32,
}

/// A single primitive in the Periodic Stack.
///
/// The lettered fields are catalog axes — kept short in the JSON for size:
///   - `B` (abstraction): the abstraction level the primitive lives at
///   - `C` (thermo):      thermodynamic class — L0, L1, L2, L2max, L0/L2
///   - `D` (concurrency): whether the primitive is naturally parallel
///   - `E` (state):       state-bit count
///   - `F` (info):        information-theoretic notes
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Primitive {
    pub id: u32,
    pub name: String,
    pub family: String,
    #[serde(rename = "B")]
    pub abstraction: String,
    #[serde(rename = "C")]
    pub thermo: String,
    #[serde(rename = "D")]
    pub concurrency: String,
    #[serde(rename = "E", default)]
    pub state: Option<u32>,
    #[serde(rename = "F", default)]
    pub info: Option<String>,
    /// Existing implementation status: "HW" / "ISA" / "EM" / "TH" / unset (software).
    pub existing: Option<String>,
    /// Targeted silicon node (e.g. "7nm", "SKY130").
    pub silicon_target: Option<String>,
    /// Wave / sub-family routing data; opaque to this crate.
    pub wave: serde_json::Value,
}

impl Primitive {
    /// Whether the primitive has dedicated silicon today.
    pub fn has_hardware(&self) -> bool {
        matches!(self.existing.as_deref(), Some("HW") | Some("ISA"))
    }

    /// L₀ bijective / reversible in principle.
    pub fn is_l0(&self) -> bool {
        self.thermo.starts_with("L0")
    }

    /// L₂max — maximally reductive.
    pub fn is_l2max(&self) -> bool {
        self.thermo == "L2max"
    }

    /// L₁ — requires entropy.
    pub fn is_l1(&self) -> bool {
        self.thermo == "L1"
    }
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Edges {
    pub composition: Vec<CompositionEdge>,
    pub bennett_decomposition: Vec<BennettEdge>,
    pub thermodynamic_inheritance: Vec<InheritanceEdge>,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CompositionEdge {
    pub source: u32,
    pub target: u32,
    pub label: String,
    #[serde(default)]
    pub wave: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BennettEdge {
    pub target: u32,
    pub l0_phase: String,
    pub l2_phase: String,
    #[serde(default)]
    pub wave: serde_json::Value,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InheritanceEdge {
    pub parent: u32,
    pub child: u32,
    pub note: String,
    #[serde(default)]
    pub wave: serde_json::Value,
}

/// The loaded catalog with O(1) lookup indices built at load time.
#[derive(Debug)]
pub struct Stack {
    pub data: StackData,
    by_id: HashMap<u32, usize>,
    by_family: HashMap<String, Vec<usize>>,
    dependencies: HashMap<u32, Vec<u32>>,
    dependents: HashMap<u32, Vec<u32>>,
    bennett_by_id: HashMap<u32, BennettEdge>,
}

#[derive(Debug, thiserror::Error)]
pub enum LoadError {
    #[error("could not read catalog file {0}: {1}")]
    Read(String, #[source] std::io::Error),
    #[error("could not parse catalog JSON: {0}")]
    Parse(#[from] serde_json::Error),
}

impl Stack {
    /// Load and index the catalog from a JSON file.
    pub fn load_from_path(path: &str) -> Result<Arc<Self>, LoadError> {
        let text =
            std::fs::read_to_string(path).map_err(|e| LoadError::Read(path.to_string(), e))?;
        Self::load_from_str(&text)
    }

    /// Load and index the catalog from a JSON string.
    pub fn load_from_str(json: &str) -> Result<Arc<Self>, LoadError> {
        let data: StackData = serde_json::from_str(json)?;

        let mut by_id = HashMap::with_capacity(data.primitives.len());
        let mut by_family: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, p) in data.primitives.iter().enumerate() {
            by_id.insert(p.id, i);
            by_family.entry(p.family.clone()).or_default().push(i);
        }

        let mut dependencies: HashMap<u32, Vec<u32>> = HashMap::new();
        let mut dependents: HashMap<u32, Vec<u32>> = HashMap::new();
        for e in &data.edges.composition {
            dependencies.entry(e.target).or_default().push(e.source);
            dependents.entry(e.source).or_default().push(e.target);
        }

        let mut bennett_by_id: HashMap<u32, BennettEdge> = HashMap::new();
        for b in &data.edges.bennett_decomposition {
            bennett_by_id.insert(b.target, b.clone());
        }

        Ok(Arc::new(Self {
            data,
            by_id,
            by_family,
            dependencies,
            dependents,
            bennett_by_id,
        }))
    }

    pub fn get(&self, id: u32) -> Option<&Primitive> {
        self.by_id
            .get(&id)
            .and_then(|i| self.data.primitives.get(*i))
    }

    pub fn count(&self) -> usize {
        self.data.primitives.len()
    }

    pub fn family(&self, code: &str) -> &[usize] {
        self.by_family
            .get(code)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    pub fn primitives(&self) -> &[Primitive] {
        &self.data.primitives
    }

    /// Composition sources for a primitive — what it depends on.
    pub fn deps_of(&self, id: u32) -> &[u32] {
        self.dependencies
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Composition targets for a primitive — what depends on it (the leverage list).
    pub fn dependents_of(&self, id: u32) -> &[u32] {
        self.dependents
            .get(&id)
            .map(|v| v.as_slice())
            .unwrap_or(&[])
    }

    /// Bennett decomposition for a primitive, if one exists.
    pub fn bennett_for(&self, id: u32) -> Option<&BennettEdge> {
        self.bennett_by_id.get(&id)
    }

    /// True if the primitive has a known reversible (L₀) kernel.
    pub fn has_bennett(&self, id: u32) -> bool {
        self.bennett_by_id.contains_key(&id)
    }

    /// Transitive closure over composition edges from `target`, inclusive.
    pub fn dependency_closure(&self, target: u32) -> Vec<u32> {
        let mut visited: HashMap<u32, ()> = HashMap::new();
        let mut stack: Vec<u32> = vec![target];
        while let Some(id) = stack.pop() {
            if visited.insert(id, ()).is_none() {
                for d in self.deps_of(id) {
                    stack.push(*d);
                }
            }
        }
        let mut result: Vec<u32> = visited.into_keys().collect();
        result.sort();
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn load() -> Arc<Stack> {
        let path = concat!(env!("CARGO_MANIFEST_DIR"), "/../../data/stack.json");
        Stack::load_from_path(path).expect("catalog loads")
    }

    /// Catalog invariants the whole workspace depends on.
    #[test]
    fn catalog_shape() {
        let s = load();
        assert_eq!(s.count(), 258, "primitive count");
        assert_eq!(s.data.families.len(), 33, "family count");
        assert_eq!(
            s.data.edges.bennett_decomposition.len(),
            26,
            "bennett edges"
        );
    }

    /// IDs the dialects hard-reference must exist and keep their identities.
    #[test]
    fn load_bearing_primitives() {
        let s = load();
        for (id, name) in [
            (18, "Dense MatMul"),
            (20, "Attention"),
            (24, "Convolution"),
            (34, "Softmax"),
            (35, "Normalization"),
            (39, "MCMC / Langevin step"),
            (50, "Spike / LIF neuron"),
            (51, "STDP"),
            (79, "DCT"),
            (258, "Simulated annealing"),
        ] {
            let p = s
                .get(id)
                .unwrap_or_else(|| panic!("primitive {id} missing"));
            assert_eq!(p.name, name, "primitive {id} renamed");
        }
    }

    /// Dependency closure walks composition edges transitively.
    #[test]
    fn dependency_closure_includes_target() {
        let s = load();
        let closure = s.dependency_closure(18);
        assert!(closure.contains(&18));
    }
}
