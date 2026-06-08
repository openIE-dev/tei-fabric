# tei-fabric

**The Rust runtime, simulation, and visualization for the Thermodynamic Edge Intelligence compute fabric.**

`tei-fabric` is the orchestration layer that sits above the open-source ecosystem of substrate-specific simulators (MEEP, SAX, thrml, CrossSim, Lava, ngspice+aa, Strawberry Fields, …). It takes a workload, maps it onto the Periodic Stack of Computation, and dispatches each primitive to the substrate whose physics performs it at the lowest joule cost.

The 258-primitive catalog and the Bennett-decomposition metadata come from the [Periodic Stack of Computation](https://compute.openie.dev). Each substrate dialect models the physics of one compute paradigm — photonic, stochastic, reversible, neuromorphic, in-memory crossbar, quantum-photonic — from first principles, with citations to the published literature.

## What's in v0

- **Three substrate dialects:** baseline CPU/GPU, photonic (MZI mesh), in-memory (RRAM crossbar).
- **One workload:** dense matrix multiply, parametric in shape and dtype.
- **A cost-surface dispatcher** that picks the lowest-joule substrate per primitive.
- **An Axum HTTP service** that exposes the dispatcher as a JSON API.
- **A browser-side viz** (Astro + WebGPU) showing the workload graph, the per-primitive substrate assignment, and the energy comparison vs the baseline.

Subsequent versions add the remaining substrate dialects (stochastic / reversible / neuromorphic / quantum-photonic), the bridges to external OSS reference simulators, and the full Periodic Stack workload import.

## Workspace layout

```
tei-fabric/
├── crates/
│   ├── tei-stack/             load + index the Periodic Stack catalog
│   ├── tei-ir/                workload graph + op-profile types
│   ├── tei-substrate-traits/  the Substrate trait
│   ├── tei-d-baseline/        CPU/GPU physics
│   ├── tei-d-photonic/        MZI mesh physics
│   ├── tei-d-in-memory/       RRAM crossbar physics
│   └── tei-cost-surface/      dispatcher
├── bin/
│   └── tei-serve/             Axum API
├── data/
│   └── stack.json             Periodic Stack catalog (mirror of compute.openie.dev)
└── web/                       browser viz (Astro + WebGPU)
```

## Build

```bash
cargo check          # whole workspace
cargo run -p tei-serve
```

## License

Apache-2.0. See `LICENSE-APACHE`.

The Periodic Stack catalog (`data/stack.json`) is content from compute.openie.dev and tracks that source.
