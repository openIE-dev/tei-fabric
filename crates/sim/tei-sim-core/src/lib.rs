//! tei-sim-core — shared numerics for the native simulation layer.
//!
//! Policy (see docs/SIM-ROADMAP.md): hand-rolled, dependency-light
//! (`serde` only here; `rayon` lives in the sim crates that parallelize),
//! deterministic across platforms including wasm32. Validation is
//! analytic-only — every module carries property tests against closed-form
//! ground truth.

pub mod events;
pub mod exec;
pub mod ledger;
pub mod linalg;
pub mod ode;
pub mod rng;
pub mod sparse;
