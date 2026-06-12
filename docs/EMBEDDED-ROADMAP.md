# TEI Embedded — the fabric model on current-generation hardware

**Status**: exploration draft · **Owner**: tei-fabric core · **Thesis date**: 2026-06-11

---

## 1. Why

tei-fabric prices computational primitives in joules, executes them, counts
what physically happened in an event ledger, and feeds measured energy back
to replace assumed constants. Purpose-built substrates (p-bit arrays,
photonic meshes, adiabatic logic) are coming — but the *programming model*
should not wait for them.

Current microcontrollers already are heterogeneous fabrics in miniature:
a main core plus PIO state machines, low-power coprocessors, filter/CORDIC
accelerators, NPUs, DMA engines, and hardware event routing that does work
while the CPU sleeps. Each block has a different joules-per-op. Nobody
dispatches across them on measured energy; everybody dispatches on
folklore.

**The goal**: ship the toolchain and ecosystem support that lets today's
chips *exhibit the fabric's features* — priced primitives, event ledgers,
measured-not-assumed calibration, lowest-joule dispatch — so that when the
purpose-built hardware arrives, the developer community already speaks the
model. The chips change; the contract doesn't.

## 2. The governing principle: turnkey wins

Raspberry Pi and Arduino did not win on hardware merit. They won because
the first-run experience is minutes: flash an image, `begin()`, blink.
Every deliverable below is judged by **minutes-to-first-ledger** — the
time from "heard about it" to seeing your own board print measured joules
for work it just did. Depth (calibration, dispatch policy, fabric upload)
unlocks progressively behind that first moment, never in front of it.

Concretely, the turnkey artifact per ecosystem is defined *first*, and the
architecture serves it:

| Ecosystem | Turnkey artifact | First-run target |
|---|---|---|
| Arduino | Library Manager install → `TEI.begin(); TEI.run(FFT, buf)` → ledger on Serial | < 5 min |
| MicroPython / CircuitPython | `mip install tei` / bundle → `import tei` → ledger repr | < 5 min |
| Rust / Embassy | `cargo add tei-ledger` + one wrapper type around the executor | < 15 min |
| Raspberry Pi / Yocto | flashable demo image; `teictl run fft` prints ledger + dispatch choice | < 10 min |
| ESP-IDF | component registry `idf.py add-dependency "openie/tei"` | < 10 min |
| Zephyr | west module + `CONFIG_TEI=y`, energy tables in devicetree | < 30 min |

## 3. The device contract (what is identical everywhere)

Three shapes, all already canon in the fabric, all serializable small:

1. **Primitive identity** — the Periodic Stack id space (`stack.json`).
   Embedded profile = a small subset (matmul/MAC, FFT/DCT, filter, sort,
   hash, sample/threshold, CRC/crypto, …) with the same ids the fabric
   uses, so a ledger from an RP2350 and one from the photonic column are
   rows in the same table.
2. **The embedded EventLedger** — counters that current hardware can
   count cheaply, replacing the sim columns' counters:
   `{cycles, instructions?, dma_transfers, adc_samples, accel_invocations,
   sleep_us, active_us, joules?: Option, joules_source: measured|cycles_proxy|table}`.
   `joules` is honest about provenance — a board with a coulomb counter
   reports `measured`; a bare board reports `cycles_proxy` against its
   calibrated per-state power table; an uncalibrated board reports `table`
   (shipped defaults) and says so.
3. **The calibration report** — `{board_id, substrate ("cpu@…MHz",
   "pio", "ulp", "npu", …), primitive_id, n_ops, ledger, j_per_op}` —
   POSTable to the fabric's existing `/api/calibration` family. The
   fabric-side store we already built becomes the aggregation point for
   community-measured J/op tables per board, the same way it already
   holds measured constants for the sim columns.

The dispatch rule is also identical: given a primitive and the board's
calibrated cost table, run it on the lowest-joule substrate available.
On an RP2350 that choice set is {M33 core, Hazard3 core, PIO, DMA+sniffer};
on purpose-built hardware it becomes {CPU, p-bit array, mesh}. Same API.

## 3.5 The product trio

- **teiOS** — the runtime. Not Linux: a `no_std` TEI executor (instrumented
  tasks, the embedded ledger, lowest-joule dispatch, calibration agent)
  over Embassy-class HALs on MCUs and a flat bare-metal image on Pi-class
  boards. teiOS is what every flashable artifact in EMBEDDED-TARGETS.md
  contains.
- **The forge** — the recipe system ("our Yocto, not Linux"): MACHINE
  recipe in → reproducible teiOS image out (`.uf2`/`.bin`/`.bit`/`.img`),
  with the board's energy tables baked in.
- **TEI Studio** — the turnkey face (the Arduino-IDE/Thonny lesson, applied).
  **A web page — install nothing, ever** (doctrine; see the web-only
  verdict in §8): connect a board → flash teiOS from the browser →
  **live ledger console** streaming joules per primitive → cost-table
  browser → run a calibration → publish the measurement to the fabric.
  The /run live-view components already on fabric.thermoedge.ai are
  reused verbatim. Code editing comes later; flashing + *seeing your
  board's joules* is the wedge. Studio invokes the forge's published
  image catalog; users never meet the forge directly.

Minutes-to-first-ledger through Studio: plug in a Pico 2 → Studio offers
"Flash teiOS" → 10 seconds later the ledger view is live. That is the
whole pitch, demonstrated.

## 3.6 The full ecosystem map

The trio is the seed of something larger: every category that made
embedded development *easy* gets a thermodynamic-compute counterpart.
Ease of use is not a feature of this list — it is the founding doctrine.
Each row ships only when it is SUPER easy: one click, one line, one
drag-drop. If a row needs a manual, it is not done.

| Category (the thing that won) | TEI counterpart | What changes under thermodynamic paradigms |
|---|---|---|
| Yocto (image builder) | **the forge** | recipes emit teiOS images with energy tables baked in; reproducible joules, not just reproducible bits |
| Raspberry Pi OS (the OS) | **teiOS** | the scheduler's first-class resource is joules; every task has a ledger; sleep is a substrate |
| Arduino (IDE + library + `begin()`) | **TEI Studio + tei-arduino** | `TEI.run(FFT, buf)` returns a ledger; the Serial monitor is a joule monitor |
| MicroPython / CircuitPython | **tei modules** | `import tei` — the REPL prints measured joules; education-first on-ramp |
| PlatformIO (multi-platform studio) | **TEI Studio** | one app, every board in EMBEDDED-TARGETS.md, one flash button |
| ROS (robotics middleware) | **tei-ros / joule-aware pub-sub** | nodes and topics carry joule budgets; a robot's compute graph dispatches like the fabric dispatches — lowest-joule substrate wins; energy is a first-class QoS field |
| Edge Impulse (end-to-end edge MLOps) | **fabric pipelines** | collect → train → deploy, but every deploy candidate is priced in measured J/inference per board before you flash |
| Qualcomm AI Hub (model zoo + per-device profiling) | **the fabric hub** (fabric.thermoedge.ai grows a deploy arm) | the community calibration store IS the per-device profile database; pick a primitive/workload, see measured J/op on every board, click deploy |
| OpenMV (domain board + IDE) | **domain bundles** (vision first — Joulo camera lineage) | a camera pipeline where each stage reports joules and the dispatcher moves stages between core/NPU/PIO |
| Edge AI runtimes (TFLM et al.) | **teiOS primitive runtime** | kernels are Periodic Stack primitives with ledgers; delegates are substrates; selection is by measured joules |

The sequencing discipline stays the same: each counterpart enters the
world as a turnkey artifact (a flashable image, a one-line install, a
web page with a Connect button) and earns depth afterward. The fabric
hub deserves emphasis: it already exists as the calibration store +
cost surface — the embedded program turns it into the place where a
developer asks "what does THIS workload cost on THIS board" and gets a
measured answer with a flash button next to it.

## 4. Architecture: one core, thin bindings

One `no_std` Rust core crate owns the contract types + dispatch logic +
cycles-proxy energy model. Everything else is a thin binding:

```
tei-ledger (no_std core: types, cost tables, proxy model, serde)
├── tei-embassy        Rust: executor instrumentation, per-task ledgers
├── tei-arduino        C++ wrapper (core compiled as staticlib, C ABI)
├── tei-micropython    native module (same C ABI)
├── tei-circuitpython  shared-bindings module
├── tei-zephyr         west module; energy tables via devicetree bindings
└── tei-linux (teid)   Yocto/RPi daemon: hwmon/INA sensors, RAPL-class,
                       systemd service, talks to fabric.thermoedge.ai
```

The C-ABI staticlib trick (one Rust core, every ecosystem links it) is the
maintenance-minimizing shape; precedent: TinyUSB/lvgl ship one C core into
every ecosystem, TFLM ships one C++ core. We invert the language but keep
the shape.

**Energy measurement tiers** (a board is in exactly one tier per substrate):

- **T0 measured** — on-board coulomb counter / power monitor readable by
  firmware (INA228/PAC1934-class, EFM32 AEM, fuel gauges). Real joules.
- **T1 calibrated proxy** — DWT CYCCNT / mcycle × a per-power-state table
  calibrated once on a bench (PPK2/Joulescope/Otii with GPIO markers) or
  crowd-sourced from T0 boards of the same family via the fabric store.
- **T2 shipped table** — defaults from datasheets; honest `joules_source:
  table`. Still useful: dispatch *ratios* between substrates are far more
  stable than absolute watts.

## 5. Reference targets (proposed; verify in research pass)

- **RP2350 (Pico 2)** — the heterogeneity demo: M33 vs Hazard3 vs PIO for
  the same primitive, huge community, cheap. Likely first Rust+Arduino+
  MicroPython target.
- **ESP32-C6 or -P4** — LP core vs HP core dispatch + Wi-Fi upload of
  calibration reports to the fabric; ESP-IDF component registry reach.
- **Raspberry Pi (4/5/Zero 2)** — the Linux/Yocto/teid flagship and the
  turnkey image; hwmon/INA HATs for T0 measurement; the board everyone
  already owns.
- Stretch: one Ethos-U board (STM32N6 / MCX N / Alif) for the NPU column,
  one nRF54 for the VPR coprocessor + PPI story, MSP430 EnergyTrace as the
  T0-measurement reference.

## 6. Phasing (each phase ends with a turnkey artifact)

| Phase | Deliverable | Turnkey artifact |
|---|---|---|
| **E0** | `tei-ledger` no_std core: types, embedded ledger, cost tables, proxy model, C ABI | crates.io publish; `cargo add tei-ledger` works on stable |
| **E1** | RP2350: Embassy instrumentation + PIO-vs-core dispatch demo; bench calibration kit (PPK2 scripts) | Pico 2 UF2 demo: hold BOOTSEL, drag, serial prints a live ledger + "FFT ran on PIO, 41 µJ vs 96 µJ on core" |
| **E2** | Arduino + MicroPython bindings over the same core | Library Manager + mip packages; copy-paste sketch prints a ledger |
| **E3** | `teid` for Linux (RPi/Yocto): hwmon/INA, systemd, fabric upload; meta-tei layer | flashable RPi image; `teictl run` + web ledger view |
| **E4** | Fabric integration: `/api/calibration` accepts board reports; fabric.thermoedge.ai page showing community J/op tables per board | your board's measurement appears on the public cost surface |
| **E5** | Zephyr module + ESP-IDF component; devicetree energy bindings | `CONFIG_TEI=y`; `idf.py add-dependency` |

## 7. What this is not

- Not an RTOS, not a scheduler replacement — it instruments and advises
  the executor you already use (Embassy task, FreeRTOS task, loop()).
- Not a power-management framework — Zephyr PM et al. manage states;
  TEI *prices work* and chooses where work runs. It composes with PM.
- Not vendor benchmarking — tables are measured, sourced, and reproducible;
  no comparative marketing, ever (the numbers speak or stay out).

## 8. Verified engineering decisions (research pass 1, 2026-06-11)

Web-verified findings that freeze previously-open choices. Sources in the
research transcripts; key URLs inline.

**Repo + release shape**
- **LVGL's in-repo-manifests pattern wins**: one `tei-embedded` repo carrying
  `library.properties` (Arduino), `idf_component.yml` (ESP registry),
  `zephyr/module.yml`, `library.json` (PlatformIO) — every channel releases
  atomically with the core. (TinyUSB's separate-wrapper-repo lags by
  construction; TFLM's no-releases model is the cautionary tale — its
  abandoned Arduino wrapper got delisted and the namespace ceded.)
- **Memfault's SDK is the structural precedent**: portable core + `ports/`,
  a `tei_platform_*.h` contract the integrator implements, out-of-tree
  Zephyr module + listing in Zephyr's external-modules docs index (the
  Memfault/Golioth route; default-manifest inclusion is not the goal).
- One config header (`tei_config.h`, tusb_config/lv_conf-style) + optional
  Kconfig shims per RTOS.

**Rust core, per-ecosystem**
- **Embassy needs no fork or wrapper**: `embassy-executor`'s `trace` feature
  exposes seven `_embassy_trace_*` extern fns (task new/exec begin/end/
  ready, executor idle) resolved at link time — tei-embassy implements
  them and per-task ledgers come free; optionally also the `rtos-trace`
  backend (SystemView users get TEI data) and an embassy-time driver for
  active/sleep attribution. Pin embassy-executor 0.8–0.10 (pre-1.0 churn).
- **Ledger telemetry channel**: defmt 1.0 (wire format now stable) for
  human logs; **postcard-rpc topics** (no_std serde, embassy-usb transport
  out of the box) for the structured ledger stream Studio consumes.
- **Arduino**: `precompiled=true` + per-ABI `.a` under `src/{build.mcu}/`
  (cortex-m0plus / cortex-m33 / esp32* / riscv32*) + thin camelCase C++
  singleton (`TEI.begin(); TEI.run(FFT, buf)`; ledger implements
  `Printable` so `Serial.println(TEI.ledger())` just works). Caveat
  verified: per-core `compiler.libraries.ldflags` support is uneven —
  CI must link-test each core. A Rust-core Arduino library appears to be
  genuinely novel; tei would be the early mover. ESP32 PlatformIO docs
  must point at the pioarduino fork (official platform froze at Arduino 2.x).
- **MicroPython**: the emlearn dual route — per-arch native `.mpy`
  (Rust staticlib via `MPY_LD_FLAGS`, `LINK_RUNTIME=1`, armv7emsp/
  armv7emdp/xtensawin/rv32imc × ABI 6.3) on a self-hosted mip index, plus
  USER_C_MODULES for firmware builds needing DMA/counter access natmods
  can't reach. **CircuitPython cannot load native .mpy** (confirmed open
  issue) — ship a pure-Python shim in the Community Bundle now; upstream
  a shared-bindings module as the long game (the ulab path).
- **Zephyr**: west module; **`tei,energy-table` devicetree binding** for
  per-board J/op data — novel for Zephyr (zephyr,power-state carries no
  power numbers) but validated by Linux DT precedent (`opp-microwatt`,
  `dynamic-power-coefficient`). Rust core ships as `west blobs` prebuilt
  `.a` per arch (zephyr-lang-rust is official but too narrow to depend on).
- **ESP-IDF**: registry component with per-target prebuilt `.a` —
  **Slint already ships Rust-as-staticlib on the ESP registry**, the
  direct precedent. `targets:` gating + examples/ + upload-components-ci.

**TEI Studio verdict — WEB-ONLY install, for every device (doctrine)**

Decision (David, 2026-06-11, overriding the research's Tauri-first
recommendation): there is no desktop app to install. TEI Studio is a web
page; every supported device gets teiOS from the browser. Where the
browser "can't," we innovate around it. Three install tiers, all web:

- **Tier W1 — direct browser flash** (Chromium WebSerial/WebUSB, all
  production-proven): esptool-js for the ESP family (the ESP Launchpad /
  Web Tools stack); **WebUSB PICOBOOT** for RP2040/RP2350 (picoflash.org
  and Arm's picotool.js prove it, Pico 2 included); **WebUSB DFU**
  (webdfu) for STM32 DfuSe ROM bootloaders — and for FPGAs we
  preferentially target the DFU-bootloader boards (**Fomu, OrangeCrab**),
  which makes "flash a bitstream from a web page" real; DAPjs CMSIS-DAP
  for boards with DAPLink interface chips (micro:bit-class — flashing
  over WebUSB-HID since 2018). probe-rs-in-WASM (upstream work in
  flight, demo at inspect.probe.rs) is the future unification of this tier.
- **Tier W2 — file-write install (works in EVERY browser incl. Safari)**:
  UF2 targets are a file download + drag to the BOOTSEL drive — no API
  needed at all. **And the Pi-class innovation: because teiOS is not
  Linux, a Pi install is just FILES on a FAT partition** (the GPU
  bootloader loads our flat kernel img + config.txt from FAT — no ext4,
  no partition surgery). `showDirectoryPicker()` → user picks the SD
  card → Studio writes the teiOS boot set. The "not Linux at all"
  decision is precisely what makes browser Pi-install possible where
  Raspberry Pi OS needs a native imager. (Cards >32 GB ship exFAT and
  need a one-time FAT32 format via the OS dialog — documented, still
  zero install.)
- **Tier W3 — the teiProbe bridge ($4 universal web programmer)**: for
  SWD-only parts (nRF, STM32 without DFU, SAMD…) and FPGA SPI flash,
  Studio offers a one-time drag-drop: a **teiProbe UF2 onto any Pico**
  (debugprobe/CMSIS-DAP lineage, PIO-driven SWD/JTAG/SPI). The browser
  then drives the bridge over WebUSB to flash the real target. One
  drag-drop mints the programmer for everything the browser can't reach
  directly. This also gives Safari/Firefox users a path: the bridge
  setup is Tier W2 (a file drag), and only the final hop needs Chromium.

Honest envelope: the full matrix is Chromium-first (Safari opposes
WebSerial/WebUSB; Firefox is Nightly-experimental). Tier W2 covers every
browser; Studio detects capability and routes. A Tauri wrapper can come
later for offline/fleet use — it is a wrapper around the same web app,
never the primary path.

UX patterns to copy verbatim: Thonny's "bootloader volume detected →
offer firmware" dialog; Raspberry Pi Imager's three-choice flow and
runtime JSON image catalog (the forge publishes exactly that shape, à la
ESP Launchpad's vendor-hostable TOML); Nordic PPK2's live scrolling
current trace as the live-ledger view's gold standard. The live ledger
console is WebSerial (or WebUSB vendor CDC) streaming postcard frames.

**CI norms adopted**: compile-sketches matrix + arduino-lint (Arduino),
wokwi-ci with wait-serial ledger assertions (functional only — simulated
cycles ≠ energy), QEMU/unix-port natmod tests (MicroPython), twister +
native_sim (Zephyr), embedded-test + self-hosted RP2350 runner (HIL),
yocto-check-layer + layer index (meta-tei).

## 9. Open questions for research pass 2 (in flight)

(Three research sweeps were scoped — per-family energy measurement reality,
on-die substrate inventory + energy-aware dispatch prior art, ecosystem
packaging norms. They were interrupted; re-run before E0 freezes the
ledger counter set.)

- Which counters are cheaply countable per family (decides ledger fields).
- Whether any on-die T0 measurement exists beyond EFM32 AEM / EnergyTrace.
- Ethos-U delegate model details for the NPU substrate column.
- MicroPython native-module vs frozen-py tradeoff for the binding.
- Whether devicetree is the right home for Zephyr energy tables.
