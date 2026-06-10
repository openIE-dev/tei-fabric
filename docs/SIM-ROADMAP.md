# tei-sim — Native Simulation Layer Roadmap

**Status**: approved plan · **Owner**: tei-fabric core · **Scope**: everything, including the mountains

---

## 1. Vision

tei-fabric today **prices** primitives — six substrate dialects with citation-grounded
analytical cost models. This roadmap adds the layer that **executes** them: a functional
simulator per substrate column, in pure Rust, with no Python, no FFI, no foreign
frameworks anywhere in the build, runtime, or CI.

The open ecosystem this fabric composes (thrml, CrossSim, Lava, SAX, neuroptica, the
DeBenedictis adiabatic-analysis flow, MEEP, Strawberry Fields) consists of small,
well-defined numerical cores wrapped in Python/JAX/PyTorch ergonomics. We port the
**capabilities** — the cores — not the tools. Each port credits its lineage
("thrml-class", "CrossSim-class") and is validated against analytic ground truth and
published results, never against the foreign implementation at runtime.

What execution buys over pricing:

1. **Measured, not assumed.** Cost dialects today assume constants (10% spike
   activity, sweeps-to-convergence, phase-error accuracy loss). Simulators count
   actual events and feed an **event ledger** back into the cost surface.
2. **A real runtime.** Dispatch routes a Max-Cut to the stochastic column — and the
   stochastic simulator anneals it and returns the cut, with a live convergence
   curve in the browser.
3. **WASM-native.** Every sim core compiles to `wasm32-unknown-unknown`. Small
   instances run client-side. None of the Python-hosted reference tools can do this.

---

## 2. Architecture

```
crates/sim/
├── tei-sim-core         shared numerics: RNG, dense+sparse linalg, event queue,
│                        ODE integrators, units, EventLedger, Executor trait
├── tei-sim-stochastic   block Gibbs / Ising / annealing          (thrml-class)
├── tei-sim-spiking      LIF/AdEx event-driven SNN, NIR semantics (Lava-class)
├── tei-sim-crossbar     MVM + device non-idealities              (CrossSim-class)
├── tei-sim-photonic     S-matrix circuits + universal MZI meshes (SAX/neuroptica-class)
├── tei-sim-circuit      MNA transient solver — MOUNTAIN 1        (SPICE-class, scoped)
├── tei-sim-adiabatic    S2LAL cell library + energy analysis     (aa-class, on -circuit)
├── tei-sim-field        Yee-grid FDTD — MOUNTAIN 2               (MEEP-class, scoped)
└── tei-sim-gaussian     CV Gaussian-state quantum optics         (SF-Gaussian-class)
```

Cross-crate flows (the integration story):

- `tei-sim-field` extracts **S-parameters from port monitors** → component models
  consumed by `tei-sim-photonic` (device sim feeds circuit sim — our own
  closed loop, no external EDA).
- `tei-sim-circuit` provides the **exact IR-drop mode** for `tei-sim-crossbar`
  (parasitic wire-resistance mesh solve) and is the foundation `tei-sim-adiabatic`
  builds its cell templates on.
- Every simulator emits an `EventLedger` (spikes, sweeps, flips, modulator events,
  detector samples, ∫i·v per device) that **recalibrates the corresponding cost
  dialect** — measured constants replacing assumed ones.

### Dependency policy

`rayon` (parallelism), `serde`/`serde_json` (I/O), nothing else in the sim layer.
PRNG is hand-rolled (splitmix64 seeding + xoshiro256++), deterministic and
WASM-identical for reproducible runs. Dense linear algebra (complex + real LU, QR)
is hand-rolled in `tei-sim-core::linalg`; sims at our scales (meshes ≤ 512×512,
circuit cells ≤ a few hundred nodes) do not justify a BLAS dependency. Sparse LU
(CSR, Markowitz ordering) lands in core when `tei-sim-circuit` M4 needs it.
If profiling ever demands more, the fallback is `faer` (pure Rust) — never a
C/Fortran FFI.

### Validation policy (binding)

**Analytic ground truth and published numbers only.** No golden fixtures generated
by the foreign tools, no runtime cross-checks against Python. Three admissible
sources of truth:

1. **Closed-form math** — exact partition functions, ODE solutions, unitarity,
   symplectic invariants, quantization SNR, adiabatic scaling laws.
2. **Published results** — numbers printed in the cited literature (Onsager 1944,
   Brunel 2000, Bi & Poo 1998, Clements 2016, best-known G-set cuts, …).
3. **Property tests** — invariants that must hold for all inputs (energy
   conservation, probability normalization, CFL stability, Sp(2n) preservation).

Every sim crate ships its validation table (below) as `cargo test` suites. CI runs all.

---

## 3. Per-crate specifications

### 3.0 `tei-sim-core` — shared numerics

| Module | Contents |
|---|---|
| `rng` | splitmix64 + xoshiro256++, `Rng` trait, normal/exponential/categorical draws |
| `linalg` | `Mat<f64>` / `Mat<Complex>`: matmul, LU solve (partial pivot), QR (Householder), determinant |
| `sparse` | CSR matrix, adjacency lists, graph coloring (greedy DSATUR — chromatic Gibbs needs it) |
| `events` | binary-heap event queue with stable tie-breaking; calendar queue if profiling demands |
| `ode` | RK4, trapezoidal, backward Euler; adaptive step with LTE control |
| `ledger` | `EventLedger` — typed counters (sweeps, flips, spikes, sops, samples, joules-by-element) |
| `exec` | `Executor` trait: `fn execute(&self, inv: &Invocation, opts: &ExecOpts) -> ExecutionResult` |

**Validation**: RNG statistical batteries (mean/var/autocorrelation bounds), LU/QR
reconstruction ‖A − LU‖ < 1e-12, integrator convergence order measured empirically
(RK4 must show 4th-order slope), event-queue ordering property tests.

### 3.1 `tei-sim-stochastic` — thrml-class

**Model**: Ising/EBM factor graph (sparse `h`, `J`), **chromatic block Gibbs**
(graph-colored parallel sweeps — the thrml core trick, rayon-parallel per color
class), simulated-annealing schedules (linear/geometric/adaptive), parallel
tempering as a stretch goal.

**Executes**: primitives 38 (MC accept/reject), 39 (MCMC), 99 (Bayesian posterior),
258 (simulated annealing).

**Validation table**:
| Check | Source of truth |
|---|---|
| Magnetization + energy vs exact enumeration, N ≤ 20 spins | closed form (2^N sum) |
| 2D Ising critical temperature T_c = 2/ln(1+√2) | Onsager 1944 |
| Detailed balance: empirical transition ratios vs Boltzmann | closed form |
| Max-Cut on G-set instances ≥ 95% of best-known cut | published G-set records |
| Chromatic = sequential Gibbs in distribution (KS test) | property |

**Ledger → dialect**: measured sweeps-to-convergence and flip counts replace the
user-guessed `sweeps` in `tei-d-stochastic` pricing.

### 3.2 `tei-sim-spiking` — Lava-class

**Model**: populations of LIF neurons (AdEx, Izhikevich later), sparse delayed
synapses, hybrid clock/event-driven stepping, pair-based STDP. Graph input is a
JSON schema mirroring **NIR semantics** (LIF, Linear, Delay node types) — no HDF5
dependency; a NIR-file converter can come later as a separate offline tool.

**Executes**: primitives 50 (Spike/LIF), 51 (STDP).

**Validation table**:
| Check | Source of truth |
|---|---|
| Membrane trajectory under constant current | closed-form LIF ODE solution |
| f–I curve: f = [τ ln(RI/(RI−V_th))]⁻¹ | closed form |
| Balanced-network regimes (AI/SR/SI states, rates) | Brunel 2000 analytic phase diagram |
| STDP window shape (potentiation/depression asymmetry) | Bi & Poo 1998 |
| Event-driven ≡ small-step clock-driven (spike trains match) | property |

**Ledger → dialect**: measured spike counts replace `DEFAULT_ACTIVITY = 0.1` in
`tei-d-neuromorphic`; measured SOPs price the run exactly.

### 3.3 `tei-sim-crossbar` — CrossSim-class

**Model**: `y = G·v` with the non-ideality stack: lognormal programming
variability, per-read shot/thermal noise, PCM power-law drift, quantized DAC
inputs, ADC transfer (offset + INL + clipping), IR drop in three fidelity modes —
ideal / closed-form first-order / **exact resistive-mesh solve via
`tei-sim-circuit`**. Includes a minimal forward-only tensor executor (MLP + Conv,
f32) so full-network accuracy-vs-noise curves can be produced natively; weights
arrive via the existing ONNX importer. Stretch: train a small MLP on raw MNIST in
pure Rust so the accuracy demo is end-to-end self-contained.

**Executes**: primitives 17/18/19/24-class MVMs in noisy-hardware mode.

**Validation table**:
| Check | Source of truth |
|---|---|
| Output variance under independent device noise σ_y² = Σ vᵢ²σᵢ² | closed form |
| ADC quantization SNR = 6.02·b + 1.76 dB | closed form |
| Drift exponent recovery from synthetic PCM traces | closed form (power law) |
| IR-drop first-order mode vs exact mesh mode convergence | internal consistency |
| Accuracy-degradation trend vs noise matches published direction/magnitude | NeuRRAM (Wan 2022) |

**Ledger → dialect**: measured `accuracy_loss` (from real inference under noise)
replaces the fixed 1% in `tei-d-in-memory`.

### 3.4 `tei-sim-photonic` — SAX/neuroptica-class

**Model**: S-parameter netlists composed via the **Redheffer star product** over
frequency points. Component library: waveguide (β, loss), directional coupler,
phase shifter (thermo-optic), MZI, photodetector (responsivity + shot noise),
grating coupler. **Universal meshes**: Reck and Clements topologies; weight
loading = target unitary → Clements decomposition → phase settings; forward pass
with injected phase quantization + thermal crosstalk.

**Executes**: primitives 18/20/24/53 in photonic-hardware mode.

**Validation table**:
| Check | Source of truth |
|---|---|
| Clements round-trip ‖U_target − U_mesh‖ < 1e-12 | Clements et al. 2016 (constructive) |
| MZI transfer = sin²/cos² of phase | closed form |
| Lossless network ⇒ S unitary | property |
| Star-product associativity | property |
| Mesh accuracy vs phase-error σ trend | Fang/Hughes-style published curves |

**Ledger → dialect**: measured modulator events + detector samples + realized
accuracy loss recalibrate `tei-d-photonic`.

### 3.5 `tei-sim-circuit` — MOUNTAIN 1 (SPICE-class MNA)

Planned from day one with a four-stage ladder; "scoped" means staged, not skipped.

| Stage | Contents | Unlocks |
|---|---|---|
| **M1** | MNA assembly; DC operating point (Newton-Raphson + source stepping); linear R/C/L/V/I; transient (trap + BE, adaptive LTE) | RC/RLC analytics, crossbar exact IR-drop |
| **M2** | Nonlinear devices: Shockley diode; MOSFET level-1; EKV-lite for near/sub-threshold | transistor circuits |
| **M3** | Parametric power-clock sources (trapezoid/sinusoid ramps); **per-element energy instrumentation ∫i·v dt** | the entire point: dissipation per cell vs ramp time |
| **M4** | Sparse CSR + Markowitz-ordered LU; performance pass | cells > ~500 nodes |

**Deliberately out**, documented: BSIM model cards, RF/harmonic balance, noise
analysis, PDK parsing. SKY130-class behavior enters by manually mapping the 2-3
transistor flavors adiabatic cells need onto EKV-lite parameters.

**Validation table**:
| Check | Source of truth |
|---|---|
| RC step response v(t) = V(1 − e^(−t/RC)) | closed form |
| RLC underdamped ringing frequency + envelope | closed form |
| Diode half-wave rectifier averages | closed form |
| **Adiabatic ramp: E → (RC/T)·CV² for T ≫ RC; E → ½CV² for T ≪ RC** | closed form — the canonical result |
| Integrator order + LTE controller behavior | property |

### 3.6 `tei-sim-adiabatic` — aa-class (on top of -circuit M3)

**Model**: S2LAL / Q2LAL cells as parametric subcircuit templates; 2N-phase
power-clock generators; the DeBenedictis shift-register testbench; ramp-time
sweeps producing **energy-per-op vs frequency curves** and overhead-above-Landauer.

**Validation table**:
| Check | Source of truth |
|---|---|
| log-log slope of E vs T → −1 in the adiabatic regime | closed form scaling law |
| Abrupt-switching limit recovers ½CV² | closed form |
| Energy-recovery ratio trends vs published S2LAL data | DeBenedictis arXiv:2009.00448 |

**Ledger → dialect**: measured overhead curve replaces the fixed
`REVERSIBLE_OVERHEAD_L0 = 10³` with a frequency-dependent function. This is the
single highest-credibility recalibration in the program.

### 3.7 `tei-sim-field` — MOUNTAIN 2 (Yee FDTD)

| Stage | Contents | Unlocks |
|---|---|---|
| **F1** | 2D TE/TM Yee grid; CPML boundaries; point/line sources; on-the-fly DFT monitors; dielectric media | device physics |
| **F2** | Waveguide mode sources; port monitors; **S-parameter extraction → `tei-sim-photonic` component models** | the device→circuit closed loop |
| **F3** | 3D Yee grid; rayon domain decomposition; Lorentz/Drude dispersive media | real components |
| **F4** | wgpu compute-shader kernel — FDTD in the browser on GPU | the demo nobody else has |

**Validation table**:
| Check | Source of truth |
|---|---|
| Numerical dispersion relation vs analytic ω(k) on the Yee grid | closed form |
| Slab-waveguide effective index | analytic transcendental solution |
| Ring-resonator FSR = c/(n_g·L) | closed form |
| CPML reflection < −60 dB | property/threshold |
| CFL stability boundary behavior | property |

### 3.8 `tei-sim-gaussian` — SF-Gaussian-class

**Model**: Gaussian states as (mean, covariance) in xp ordering; symplectic
operations (rotation, squeezing, displacement, beamsplitter); homodyne/heterodyne
measurement; photon statistics of Gaussian states. Small-n boson-sampling
permanents (Ryser, ≤ 20 modes) as stretch. Build order is last; the column ships
regardless of how the public TEQuantum branding question resolves — the math is
the math.

**Validation table**:
| Check | Source of truth |
|---|---|
| Squeezed-vacuum quadrature variances e^(∓2r)/2 | closed form |
| Heisenberg bound preserved by all ops | property (symplectic invariant) |
| Hong-Ou-Mandel dip at 50:50 beamsplitter | closed form |
| Covariance stays valid (σ + iΩ/2 ⪰ 0) | property |

---

## 4. Integration layer

- **`Executor` trait** (`tei-sim-core::exec`) — implemented by each sim crate;
  registered alongside the cost dialect for its column.
- **`POST /api/execute`** — dispatch + execute: route the invocation, run the
  simulator, stream progress (SSE: `routed` → `progress`×N → `result` with ledger
  + outputs + recalibrated cost).
- **Calibration loop** — `ExecutionResult.ledger` → `MeasuredCost` shown side by
  side with the analytical estimate in the UI ("estimated 20.2 nJ · measured
  18.7 nJ"). Divergence is a feature: it shows the model being checked.
- **Web** — per-column live visualizations: energy-trace convergence (stochastic),
  spike raster (spiking), accuracy-vs-noise curve (crossbar), mesh phase map
  (photonic), E-vs-T log-log (adiabatic), field animation (FDTD).
- **WASM** — sim cores feature-gated `no_std`-adjacent (no filesystem, no threads
  unless `rayon` feature) so the browser can run small instances locally with zero
  server round-trip.

---

## 5. Phasing

| Phase | Deliverable | Estimate |
|---|---|---|
| **0** | `tei-sim-core` + `Executor` + `/api/execute` scaffold | 1 session |
| **1** | `tei-sim-stochastic` + live browser Max-Cut annealing | 1-2 |
| **2** | `tei-sim-spiking` + Brunel validation + live raster | 2 |
| **3** | `tei-sim-crossbar` (ideal + first-order IR) + accuracy curves | 2 |
| **4** | `tei-sim-photonic` + Clements + phase-noise sweeps | 2 |
| **5** | `tei-sim-circuit` M1→M3 + `tei-sim-adiabatic` + E-vs-T curves | 3-5 |
| **6** | `tei-sim-field` F1→F2 + S-param handoff to photonic | 3 |
| **7** | `tei-sim-gaussian` | 1-2 |
| **8** | Mountains' summits: circuit M4 sparse, field F3 3D + F4 wgpu, parallel tempering, MNIST-in-Rust | ongoing |

Phases 1-4 are independent after Phase 0 and can interleave; 5 unblocks the exact
IR-drop mode of 3 and all of 6's consumers. Each phase lands with its validation
table green in CI before the next begins.

## 6. Performance budgets (honesty checks, not promises)

- Stochastic: ≥ 10⁸ spin-flips/s/core (chromatic Gibbs, rayon across colors)
- Spiking: ≥ 10⁷ synaptic events/s/core event-driven
- Crossbar: 512×512 noisy MVM < 1 ms ideal mode
- Photonic: 64×64 Clements decomposition + forward < 10 ms
- Circuit: 100-node adiabatic cell, 10⁶ timesteps < 10 s
- FDTD 2D: ≥ 50 MCells/s/core

Budgets are CI-tracked (non-blocking benches) so regressions are visible.

## 7. Risk register

| Risk | Mitigation |
|---|---|
| MNA Newton non-convergence on stiff adiabatic cells | source stepping + gmin stepping from day one; BE fallback |
| FDTD memory at 3D | F3 is staged; domain decomposition designed in F1 |
| Hand-rolled linalg correctness | reconstruction property tests; `faer` as documented fallback |
| Hafnian/permanent blowup | hard cap n ≤ 20; Gaussian backend is the product, sampling is the stretch |
| Scope creep toward "general SPICE/MEEP" | the **deliberately-out** lists in 3.5/3.7 are part of this contract |

## 8. Credit & naming

Public docs and code headers name lineage with respect: "thrml-class (Extropic)",
"CrossSim-class (Sandia)", "Lava-class (Intel/NIR community)", "SAX-class
(Ghent/gdsfactory ecosystem)", "aa-class (DeBenedictis)", "MEEP-class (MIT)",
"SF-Gaussian-class (Xanadu)". We port capabilities into a different language and
runtime model; the ideas carry their authors' names.
