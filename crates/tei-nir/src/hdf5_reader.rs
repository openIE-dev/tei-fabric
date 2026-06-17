//! Read a real `.nir` file (NIR's reference HDF5 format) into a [`NirGraph`].
//!
//! Layout (from `nir/serialization.py`): root `version` dataset + a `node`
//! group (the root `NIRGraph`) holding a `nodes` subgroup — one group per node,
//! each with a `type` string dataset + parameter datasets (`weight`, `tau`, …)
//! — and an `edges` dataset of UTF-8 name pairs. We size weight nodes by their
//! `weight` dataset's dims and neuron nodes by a per-neuron param dataset's
//! dims (no array data is read — just shapes), then hand off to [`crate::lower`].
//!
//! Feature-gated (`hdf5`): binds libhdf5 via `hdf5-metno`, so the core crate
//! stays dependency-light. Verified against a vlen-UTF-8 fixture
//! (`tests/fixtures/mlp.nir`) matching what `nir.write` emits.

use crate::nir::{classify, NirGraph, NirKind, NirNode};
use hdf5_metno as hdf5;
use hdf5::types::VarLenUnicode;

/// Read a NIR HDF5 file into a [`NirGraph`] (same shape as `from_json`).
pub fn from_hdf5(path: &str) -> Result<NirGraph, hdf5::Error> {
    let file = hdf5::File::open(path)?;
    let root = file.group("node")?;
    let nodes_grp = root.group("nodes")?;

    let mut nodes = Vec::new();
    for name in nodes_grp.member_names()? {
        let g = nodes_grp.group(&name)?;
        let ty = read_scalar_string(&g, "type")?;
        let kind = classify(&ty);
        let size = match kind {
            // synapses = weight matrix element count (dims only, no data read)
            NirKind::Weight => dataset_elems(&g, "weight"),
            // neurons = a per-neuron param's element count
            NirKind::Neuron => ["tau", "v_threshold", "r", "v_leak"]
                .iter()
                .map(|p| dataset_elems(&g, p))
                .find(|&n| n > 0)
                .unwrap_or(0),
            // boundaries/structural take no core capacity
            _ => 0,
        };
        nodes.push(NirNode { name, ty, kind, size });
    }
    nodes.sort_by(|a, b| a.name.cmp(&b.name)); // determinism

    // edges: an (M, 2) vlen-string dataset, row-major → name pairs.
    let mut edges = Vec::new();
    if let Ok(ds) = root.dataset("edges") {
        let flat = ds.read_raw::<VarLenUnicode>()?;
        let mut it = flat.into_iter();
        while let (Some(a), Some(b)) = (it.next(), it.next()) {
            edges.push((a.as_str().to_owned(), b.as_str().to_owned()));
        }
    }

    Ok(NirGraph { nodes, edges })
}

/// Element count of a node's dataset (product of dims), or 0 if absent.
fn dataset_elems(g: &hdf5::Group, name: &str) -> u64 {
    g.dataset(name)
        .map(|d| d.shape().iter().product::<usize>() as u64)
        .unwrap_or(0)
}

/// Read a scalar vlen-UTF-8 string dataset (e.g. a node's `type`).
fn read_scalar_string(g: &hdf5::Group, name: &str) -> Result<String, hdf5::Error> {
    let s = g.dataset(name)?.read_scalar::<VarLenUnicode>()?;
    Ok(s.as_str().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{lower, place, Target};

    #[test]
    fn reads_a_real_nir_hdf5_file() {
        let g = from_hdf5(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/mlp.nir"))
            .expect("read mlp.nir");
        // Same shape the JSON ingest produces for this MLP-SNN.
        assert_eq!(g.nodes.len(), 6); // in, fc1, lif1, fc2, lif2, out
        let sg = lower(&g);
        assert_eq!(sg.populations.len(), 4); // 2 LIF + 2 boundaries
        let syns: u64 = sg.connections.iter().map(|c| c.synapses).sum();
        assert_eq!(syns, 4 * 2 + 3 * 4); // fc1 (8) + fc2 (12) = 20
        // and it places deterministically, same as the JSON path.
        let p = place(&sg, &Target { cores: 2, neurons_per_core: 8 });
        assert!(!p.overflowed);
    }
}
