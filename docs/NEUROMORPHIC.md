# Neuromorphic — landscape, and where tei-fabric fits

**Status**: internal positioning · never deployed (lives in `docs/` with
LANDSCAPE.md). Competitor/vendor names are fine here, not on any public page
(`feedback-no-comparative-marketing`). Specs below are web-verified
(2026-06) — no fabricated numbers; correct before relying on any.

## TL;DR — the wedge

The neuromorphic field's highest-leverage *missing* piece is **not another
chip**. It is a **deterministic, optimizing, attested place-and-route
compiler** that maps a spiking graph onto a chip's cores. Today that NP-hard
mapping is done by slow, non-deterministic Python heuristics that **forfeit
much of the silicon's theoretical advantage** (the "hardware lottery" — the
under-engineered toolchain eats the win). That is exactly OpenIE's house
style: a fast, reproducible, **energy-priced** Rust solver. It plugs straight
into what tei-fabric already has — the cost surface, the Periodic Stack
primitive ids, the "measured not assumed" ledger.

## The hardware, by what you can actually get

**Commercially accessible — now in `ofpga-chipdb` (BoardCategory::Neuromorphic,
energy_source = accelerator):**

| chip | vendor | access | note |
|---|---|---|---|
| Xylo (Audio 3) | SynSense | dev kit (contact-vendor) | ~1K neurons, µW always-on |
| Speck | SynSense | dev kit since 2022 | DynapCNN + DVS event vision, ~0.32M neurons, ~3µs/spike |
| DynapCNN | SynSense | dev kit | conv SNN for event vision |
| Akida (AKD1000) | BrainChip | **Akida Cloud / FPGA Cloud (browser), Edge AI Box** | event SNN; gen-2 IP ~1.2M neurons / 10B synapses; MetaTF/TF flow |
| Pulsar | Innatera | shipping into consumer devices 2026 | first commercial spiking-neural MCU; 12 digital + 4 analog cores; ~300µW audio; PyTorch (Talamo) flow |

**Research-access only — landscape, NOT bench targets** (deliberately kept out
of the registry per "buy real hardware"): Intel **Loihi 2 / Hala Point** (INRC
program, Lava SDK, up to 1.15B neurons at Sandia); **SpiNNaker2 / SpiNNcloud**
(TU Dresden spinout, ARM-core, crossing into commercial sale); IBM
**NorthPole / TrueNorth** (not for sale); **BrainScaleS-2** (Heidelberg, analog
~1000× bio-time, free remote via EBRAINS). Adjacent: Syntiant NDP (ultra-low-
power DNN, already in earbuds — David's Nicla Voice carries one), Prophesee /
iniVation event cameras (the sensor front-end).

## The interop layer: NIR

**NIR** (Neuromorphic Intermediate Representation, the neuromorphs group) is
the hardware-agnostic graph IR for SNNs — a serialized set of primitives, not
an optimizing compiler. It's what registries like Synfire (synfire.dev — a
"Hugging Face for SNNs" layer on NIR; new, org/domain mismatch noted) index,
and what compiles one model down to Loihi 2 / SpiNNaker2 / Xylo / etc. NIR is
the natural front-door for tei-fabric: **NIR primitives map onto Periodic
Stack ids**, so a spiking model becomes priceable in the same joules table as
every other primitive.

## Why the simulators seem to "prove" the incumbents win (and the honest moat)

GPUs have beaten neuromorphic on fair-fight cortical-sim benchmarks (Knight &
Nowotny). Three reasons, in order: (1) most "SNN benchmarks" are the **wrong
workload** — dense rate-coded ANN conversions that hand the GPU its best case
and the SNN chip its worst (per-spike routing with no sparsity to amortize);
(2) the **GPU stack is mature** (60–80% of peak) while Lava/MetaTF/place-and-
route extract far less — the toolchain forfeits the lead; (3) the genuine win
is **narrow**: microwatt always-on sensing (keyword spotting, event vision,
vibration/anomaly) plus a few optimization/sampling problems. The general-
purpose claims are oversold; the µW edge niche survives fair benchmarking.
Crucially, (1) and (2) are **software-fixable** — which is again the wedge.

## How it lands in tei-fabric (the build, if we take it)

A `tei-nir` / mapper crate, in the house style:

1. **Ingest NIR** → an internal spiking graph; map its primitives to Periodic
   Stack ids (reuse `tei-stack`).
2. **Deterministic place-and-route**: partition the graph across a target's N
   cores under fan-out / routing / memory constraints — a real solver
   (ILP/heuristic-with-bound), reproducible, **attested** (the same graph +
   target → byte-identical placement + a checksum), unlike today's
   non-deterministic Python.
3. **Price the mapping in joules**: feed the placement through the existing
   cost surface; a substrate per registry target (Akida/Speck/Xylo/Loihi…),
   "measured not assumed" once a dev kit calibrates.
4. **Honest scope**: this does **not** speed a single inference (the ASIC runs
   async) — it lets the chip *realize more of its theoretical advantage*, and
   makes the mapping reproducible + auditable. That is the leverage.

This is the first thing in the whole neuromorphic ecosystem that is missing
*and* squarely ours. The chips exist; the registry now knows them; the
deterministic energy-priced mapper is the piece nobody built.
