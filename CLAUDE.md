# tei-fabric

Rust orchestration layer for the TEI compute fabric. The catalog is in `data/stack.json` (mirror of `compute-openie-web/src/data/stack.json`).

## Architecture

Each substrate dialect is its own crate (`crates/tei-d-*`), implementing the `Substrate` trait from `tei-substrate-traits`. The dispatcher in `tei-cost-surface` consumes a workload from `tei-ir` and returns a `DispatchPlan` choosing the lowest-joule substrate per primitive. The Axum binary `bin/tei-serve` exposes this over HTTP.

## Substrate physics — first principles, not multipliers

Per-substrate cost functions model the physics of the compute paradigm. Each dialect documents its constants with citations to published literature (gdsfactory/SAX, Sandia CrossSim, IBM AIHWKIT, Extropic thrml, etc.). The Landauer floor `kT·ln(2)` is the absolute minimum; the substrate model defines its overhead above that floor as a function of the workload.

The earlier maturity-tier model (`SW / EM / TH / HW / reversible` multipliers from `compute-sim-server/src/inversion.rs`) is **superseded** by per-physics models here. The Bennett-decomposition data stays as a property of the primitive (it informs the reversible substrate but isn't a substrate on its own).

## Coupling with compute-sim-server

`compute-sim-server` continues to run the legacy JLandauer engine on port 9650. `tei-fabric` runs the new dialect-based dispatcher on port 9651 (or env-configured). Both consume the same `stack.json`. Eventually `compute-sim-server` collapses into a thin binary inside this workspace.

## Conventions

- Apache-2.0 license, OSS-first.
- Edition 2024, rust 1.96+.
- `cargo fmt` + `cargo clippy -- -D warnings` clean.
- All published cost constants must have a comment citing the source.
- No unsafe outside FFI bridges.
- Browser viz follows the orbit.thermoedge.ai pattern: Astro + WebGPU, no heavy bundler.

## Build / run

```bash
cargo check
cargo run -p tei-serve              # port 9651 by default
```
