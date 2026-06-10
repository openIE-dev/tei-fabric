# tei-fabric

**The Rust runtime, simulation, and visualization for the Thermodynamic Edge Intelligence compute fabric.**

Every operation has a physics that performs it best. A matrix multiply is cheapest as light through an interferometer mesh. A random sample is cheapest as thermal noise in a magnetic junction. A reversible transform is cheapest on logic that recovers its charge. `tei-fabric` reads a workload, decomposes it into primitives from the [Periodic Stack of Computation](https://compute.openie.dev) (258 primitives, 33 families), and dispatches each primitive to the substrate whose physics performs it at the lowest joule cost.

Live demo: **[fabric.thermoedge.ai](https://fabric.thermoedge.ai)**

## The substrate dialects

| Dialect | Physics | Anchor | Key citations |
|---|---|---|---|
| `tei-d-baseline` | Modern CPU/GPU (the comparator) | ~36 pJ/MAC | Landauer 1961 · Bérut 2012 |
| `tei-d-photonic` | MZI mesh, WDM, modulators + ADC | ~30 fJ/MAC | Shen 2017 · Wang 2018 · Hamerly 2019 |
| `tei-d-in-memory` | RRAM crossbar MVM, tiled | ~2-5 fJ/MAC | Wan 2022 · Khaddam-Aljameh 2022 · CrossSim |
| `tei-d-stochastic` | sMTJ p-bit thermodynamic sampling | ~1 fJ/sample | Camsari 2017 · Borders 2019 · thrml |
| `tei-d-reversible` | Adiabatic CMOS + Bennett decomposition | ~10³ × Landauer (L₀ phase) | DeBenedictis 2020 · Bennett 1989 |
| `tei-d-neuromorphic` | LIF spike events + STDP | ~20 pJ/SOP | Davies 2018 · Merolla 2014 · NIR |

Every constant carries a citation. Every constant is also an engineering knob — the API accepts per-request `substrate_params`, and the web UI exposes them as live sliders with physical diagrams.

## What it does

- **ONNX import** (`tei-import`) — pure-Rust protobuf decode + fixed-point shape inference with constant folding. Handles CNNs (MobileNetV2), quantized transformers (BERT), and diffusion UNets (Stable Diffusion v1, 1.6 GB) including ORT contrib fused ops (`FusedMatMul`, `MultiHeadAttention`, `GroupNorm`, …).
- **Cost-surface dispatch** (`tei-cost-surface`) — per-invocation lowest-joule substrate selection with the full considered set reported.
- **HTTP service** (`tei-serve`) — Axum: batch dispatch, SSE streaming dispatch, chunked uploads (to 2.5 GB), catalog-driven workload presets.
- **Browser viz** (`web/`) — Astro: Sankey flow of primitives → substrates, live engineering-parameter sweeps, per-invocation cards.

## Workspace layout

```
tei-fabric/
├── crates/
│   ├── tei-stack/             Periodic Stack catalog loader + index
│   ├── tei-ir/                workload IR (matmul / sampling / spiking profiles)
│   ├── tei-substrate-traits/  the Substrate trait + Cost
│   ├── tei-d-*/               six substrate dialects (see table)
│   ├── tei-cost-surface/      dispatcher + presets + summaries
│   └── tei-import/            ONNX → Workload (protobuf + shape inference)
├── bin/tei-serve/             Axum API
├── data/stack.json            Periodic Stack catalog (mirror of compute.openie.dev)
└── web/                       browser viz (Astro 5 + Tailwind 4)
```

## Build & run

```bash
cargo test --workspace        # physics anchors + routing + ONNX fixtures
cargo run -p tei-serve        # API on :9651

cd web && pnpm install && pnpm dev   # viz on :4321
```

Requires `protoc` (protobuf compiler) for the vendored ONNX schema: `brew install protobuf` / `apt install protobuf-compiler`.

## API sketch

```
GET  /health                   liveness + catalog size
GET  /api/stack                full Periodic Stack catalog
GET  /api/substrates           registered dialects + citations
GET  /api/presets              catalog-driven workload presets
POST /api/dispatch             Workload [+ substrate_params] → DispatchPlan
POST /api/dispatch/stream      same, as SSE (started/invocation/complete)
POST /api/import/onnx          single-shot ONNX import (small models)
POST /api/import/onnx/chunk    chunked import (X-Upload-Id / X-Chunk-Index / X-Chunk-Total)
```

## License

Apache-2.0. See `LICENSE-APACHE`.

The Periodic Stack catalog (`data/stack.json`) is content from compute.openie.dev and tracks that source. tei-fabric composes the open substrate-simulation ecosystem (gdsfactory, SAX, thrml, CrossSim, Lava, NIR, the DeBenedictis adiabatic-analysis flow) rather than reinventing it — the fabric is the joule-aware orchestration above tools those communities validate.

Built by [Thermodynamic Edge Intelligence Corp.](https://thermoedge.ai) — a subsidiary of Open Interface Engineering.
