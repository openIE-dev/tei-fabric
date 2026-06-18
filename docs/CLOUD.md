# cloud.thermoedge.ai — teiOS, hardware-free

**Status**: internal architecture · never deployed (lives in `docs/` with
LANDSCAPE.md / NEUROMORPHIC.md). Pins the product boundary before code.

## The three domains

teiOS, Studio, and the fabric engine split cleanly across three surfaces. Each
owns a verb; none duplicates another.

| domain | runs on | owns | verb |
|---|---|---|---|
| **cloud**.thermoedge.ai | **simulated** substrates (server / GPU) | the complete teiOS experience with **zero hardware** — every target emulated, running the real algorithms through the real runtime | **try · learn · run teiOS** |
| **studio**.thermoedge.ai | **real field devices** | author algorithms + code, build/flash images, edge **fleet management** of deployed hardware | **build · ship · operate** |
| **fabric**.thermoedge.ai | the physics engine (`tei-serve`) | the cost-surface dispatcher + the 9 native simulators + the Periodic Stack — the layer the other two consume | **the engine** |

The Akida-Cloud idea, done right and bigger: not "rent one chip emulated on an
FPGA," but **the entire teiOS runtime over every substrate, simulated** — write
an app, watch it dispatch across simulated cores / meshes / crossbars / spiking
fabric, and see the *same* ledger / dispatch / calibration stream you'd get on
metal. The only thing missing versus a real board is the board.

## What cloud IS (and is NOT)

- **IS**: the full teiOS loop — app → lowest-joule dispatch across simulated
  substrates → event ledger → calibration — presented as "instant, own
  nothing, install nothing." A learning + evaluation + development surface.
- **IS NOT** a neuromorphic (or any) *datacenter compute* offering. teiOS's
  value is energy-optimized edge compute; cloud doesn't sell production cycles,
  it sells the **experience and the dev loop** without procurement. The
  compute still belongs at the edge — that's Studio's job.

## The hardware-free contract (why the three are continuous)

The point that makes this one product, not three: a teiOS app is **substrate-
agnostic** (it calls `tei-rt`: `run`/`dispatch`/`calibrate`, never a board
API). So:

> An app you run on a **simulated** target in cloud runs **identically** when
> Studio builds it for a **real** device — same source, same ledger shape,
> same dispatch logic. cloud → studio is a deploy, not a rewrite.

cloud swaps the *substrate implementations* for simulators; the runtime, the
ledger, the cost table, the dispatch rule are the same code paths.

## What's reused vs new

**Reused (cloud is ~80% already built):**
- `tei-rt` — the runtime kernel (dispatch + ledger + calibration), host-runnable.
- `tei-sim-*` (9 simulators) + `tei-d-*` substrate physics — already execute
  real workloads and report joules/time/accuracy.
- `tei-serve` endpoints: **`/api/execute`** already runs a workload on a native
  simulator and streams it; **`/api/dispatch(/stream)`**, **`/api/nir/price`**.
- Studio's CONSOLE **SIMULATE** mode — already emits a protocol-faithful ledger
  with zero hardware (the seed of the cloud stream).

**New (the cloud surface):**
- A cloud runtime entry — run a teiOS app over a chosen *simulated* target and
  stream the full loop (dispatch + ledger + calibration), e.g.
  `POST /api/cloud/run { app, target }` over the existing sim engine.
- The `cloud.thermoedge.ai` web app: pick a virtual target → run → watch the
  ledger/dispatch, the same console Studio shows for a real board.
- Infra: a Caddy entry (via `site-ops/Caddyfile` on .10.10.10.2 — never edit
  prod directly) + a `deploy/manifest.toml` entry.

## Fidelity boundary (the honest line, same as everywhere in teiOS)

Simulated joules are **model-tier**, not measured-on-silicon — the cost-surface
physics models, the analog of `JoulesSource::Table` / "compile-verified, not
hardware-verified." cloud gets you, faithfully: the **full loop**, the
**dispatch logic**, and **relative** substrate comparisons (which substrate
wins, and roughly by how much). It does **not** give absolute measured energy —
that's a real board (Studio + the bench) or a calibrated dev kit. cloud must
label its joules as simulated, never imply Measured. The value is the
experience, the dev velocity, and correct *relative* dispatch — not a power
meter.

## Build order (after this doc)

1. **Runtime endpoint** — `/api/cloud/run`: execute a teiOS app across the
   chosen simulated substrate set, stream dispatch + ledger + calibration.
   Verify on the fabric server.
2. **cloud web app** — the hosted "virtual board" experience over (1).
3. **Infra** — Caddy + deploy manifest for `cloud.thermoedge.ai`.

Each is a bounded, separately-verifiable step; (1) is pure software over the
existing engine and needs no new hardware or site.
