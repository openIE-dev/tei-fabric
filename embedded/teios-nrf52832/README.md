# teios-nrf52832 — teiOS E1c on the Nicla Voice / Nicla Sense ME

The EMBEDDED-ROADMAP E1c artifact — the **Nordic nRF52832** port
(Cortex-M4F @ 64 MHz), the host MCU on Arduino's Nicla Voice and Nicla
Sense ME. Every second the firmware prices the **Hash primitive
(Periodic Stack id 36)** — CRC32 over a 64 KiB workload — on the M4
software path, streams the ledger, and issues a dispatch verdict:

| substrate | how | counted |
|---|---|---|
| `cpu-m4@64mhz` | software table-driven CRC32 on the Cortex-M4F | **true DWT CYCCNT cycles** + active µs |
| `ndp120` / `bhi260` | the always-on AI accelerator on the module — a **priced cost-table menu entry**, not yet a runnable CRC substrate | (offload kernel = bench stretch) |

The board-independent half (CRC32, JSON writers, cost table, ledger) is
the shared [`teios-core`](../teios-core) crate, used verbatim by the
RP2040/H7 images. Same JSON-lines protocol, same `tei-ledger` shape.

> The shipped J/op values are **ILLUSTRATIVE Table-tier defaults**
> (`joules_source: "table"`), pending the E1 bench measurement.

## The TEI story: "sleep is a substrate"

The nRF52832 is a *single* general-compute core, so for the Hash
primitive there is exactly **one** runnable substrate — the M4 software
path. What makes these boards interesting to the fabric is the *other*
die on the module:

- **Nicla Voice** — Syntiant **NDP120** always-on audio NPU, listening
  at sub-milliwatt while the host M4 sleeps.
- **Nicla Sense ME** — Bosch **BHI260AP** self-learning sensor-fusion
  hub, fusing while the host sleeps.

Those are *fixed-function* accelerators — they don't run CRC32 — so they
appear in the shipped cost table as a **priced menu entry**
(`SUBSTRATE_ACCEL`). Actually offloading a primitive to them is the
documented bring-up stretch (mirroring the H7's second M4 core). When an
accelerator kernel lands, it becomes a real second row and the dispatch
verdict can flip from "compute on the M4" to "let the host sleep".

## ⚠ Hardware-verification status — read this

Compile-verified for `thumbv7em-none-eabihf`; links to a valid image
(~88 KB); 3/3 host lib tests green. **Not yet hardware-verified.** Two
board-specific items are the on-bench step (silent at compile time):

1. **UART pins + transport.** The nRF52832 has **no USB peripheral**, so
   the ledger stream goes out a UARTE at 115200 8N1 — *not* USB CDC like
   the RP2040/H7. The default TX/RX (P0.06 / P0.08, the nRF52 reset
   defaults) must be mapped to whatever the Nicla carrier exposes (ESLOV
   header / debug pads / an external USB-UART bridge). That pin map is
   the bench step.
2. **Flash base.** `memory.x` is the bare-metal layout (app at 0x0). A
   Nicla running its Arduino (mbed) bootloader places the app **above**
   the bootloader; set `FLASH ORIGIN` to that offset when flashing
   through it.

The parts that **are** verified now: the M4 software substrate, the
DWT-counted cycles (`cyccnt:true`), the cost table + dispatch, the
ledger wire protocol, and the **RAM-constrained design** (below). The
UART pin map and flash base are the documented hardware-pending
boundary, delivered the same way E1a/E1b were.

## RAM-constrained: the workload lives in flash

The nRF52832 has only **64 KB of RAM** — exactly `BUF_LEN`. So unlike
the RP2040 (264 KB) and H7 (512 KB), the 64 KiB workload **cannot** sit
in RAM. It is a `const`-built xorshift32 array in flash (`.rodata`),
byte-for-byte identical to `teios_core::fill_pattern`, so the CRC
matches every other board. The M4 reads it straight from flash (XIP).
(EasyDMA can't read flash, but the UART TX buffers are small RAM
strings, so that constraint doesn't bite here.)

## Build + flash

```sh
cargo build --release --target thumbv7em-none-eabihf
"$(find "$(rustc --print sysroot)" -name llvm-objcopy | head -1)" \
  -O binary target/thumbv7em-none-eabihf/release/teios-nrf52832 teios-nrf52832.bin
```

Flash with a debug probe (the most reliable bench path):

```sh
probe-rs download --chip nRF52832_xxAA --binary-format bin \
  --base-address 0x0 teios-nrf52832.bin
probe-rs reset --chip nRF52832_xxAA
```

(Through the Nicla Arduino bootloader, use that bootloader's DFU path
and the app-offset base instead — bench step.) Then read the UART at
115200 (a USB-UART adapter on the carrier's TX, or TEI Studio's web
console once a bridge is attached).

### Board variants

The board feature sets `board_id` and the accelerator name. Exactly one
must be on:

```sh
# Nicla Voice (default): board_id "nicla-voice", accelerator "ndp120"
cargo build --release --target thumbv7em-none-eabihf
# Nicla Sense ME: board_id "nicla-sense", accelerator "bhi260"
cargo build --release --target thumbv7em-none-eabihf \
  --no-default-features --features board-nicla-sense
```

## Host tests

```sh
cargo test --lib --target aarch64-apple-darwin   # or your host triple
(cd ../teios-core && cargo test)                 # the shared logic
```

## What differs from the RP2040/H7 images (the port deltas)

- **Target**: `thumbv7em-none-eabihf` (Cortex-M4F), `embassy-nrf`
  `nrf52832` + `time-driver-rtc1`. `critical-section-single-core` from
  cortex-m.
- **Transport**: **UART** (UARTE0), because the nRF52832 has no USB —
  the first teiOS image not on USB CDC.
- **Workload in flash**: forced by the 64 KB RAM (see above).
- **Single general-compute substrate**: one M4 path + an accelerator
  menu entry, vs the RP2040's CPU-vs-DMA-sniffer and the H7's M7-vs-CRC
  races. The dispatch story here is "sleep is a substrate", not a
  two-engine race — honestly so until an NDP120/BHI260 offload kernel
  lands.
