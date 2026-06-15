# teios-ra6m5 — teiOS E1d on the Arduino Portenta C33

The EMBEDDED-ROADMAP E1d artifact — the **Renesas RA6M5** port
(R7FA6M5, Cortex-M33 @ 200 MHz), the third MCU vendor family in the
matrix after Raspberry Pi (RP2040) and ST/Nordic (STM32H7 / nRF52832).
Every second the firmware prices the **Hash primitive (Periodic Stack
id 36)** — CRC32 over a 64 KiB workload — on two on-die substrates,
cross-checks them, and dispatches the cheaper:

| substrate | how | counted |
|---|---|---|
| `cpu-m33@200mhz` | software table-driven CRC32 on the Cortex-M33 | **true DWT CYCCNT cycles** |
| `crca` | the RA6M5 **CRCA hardware-CRC peripheral** computes CRC-32 while the core idles | accel invocations |

A genuine hardware-vs-software race with a real `check` — unlike the
single-core Nicla nRF52832. The board-independent half (CRC32, JSON
writers, cost table, ledger) is the shared [`teios-core`](../teios-core)
crate, used verbatim. Same JSON-lines protocol, same `tei-ledger` shape.

> The shipped J/op values are **ILLUSTRATIVE Table-tier defaults**
> (`joules_source: "table"`), pending the E1 bench measurement.

## ⚠ Hardware-verification status — read this

Compile-verified for `thumbv8m.main-none-eabihf`; links to a valid
image (~85 KB); 3/3 host lib tests green. **Not yet hardware-verified.**
The bench items (silent at compile time):

1. **Transport = semihosting.** The RA6M5 has no embassy HAL and its
   SCI-UART bring-up (clock tree + baud + PFS pin-mux) is involved, so
   the ledger streams over **semihosting** (`hstdout`) to whatever debug
   probe is attached — which is exactly how you bring a board up before
   its UART is wired, and needs zero clock setup. Production moves to an
   SCI UART (that pin/clock work is the bench step).
2. **CRCA correctness + clock.** The CRCA register sequence targets
   zlib/IEEE CRC-32 (GPS=CRC-32, LSB-first/reflected, seed 0xFFFF_FFFF,
   software final XOR); on-die agreement with the M33 path is the
   `check` line every pass. The 1 Hz cadence is a fixed cycle-delay
   against the *reset* clock (no PLL bring-up), so it is approximate
   until the clock tree is configured.

Verified now: the two-substrate dispatch logic, the CRCA register
sequence (against the `ra6m5-pac` SVD), the DWT-counted M33 cycles
(`cyccnt:true`), the cost table, and the ledger wire protocol.

## No mature async HAL → bare-metal

There is no embassy time-driver for the Renesas RA family, so this is
plain `cortex-m-rt` + the `ra6m5-pac` register crate — a **synchronous**
`app` (no async runtime), unlike the embassy-based RP2040/H7/nRF images.
The `app(tei)` signature here takes `&mut Tei` and returns
`Result<(), TeiError>` *without* `async`.

## RAM note

The 64 KiB workload is a `const`-built xorshift32 array in flash
(`.rodata`), byte-for-byte identical to `teios_core::fill_pattern`, so
the CRC matches every other board (and the M33 reads it via XIP).

## Build + flash

```sh
cargo build --release --target thumbv8m.main-none-eabihf
"$(find "$(rustc --print sysroot)" -name llvm-objcopy | head -1)" \
  -O binary target/thumbv8m.main-none-eabihf/release/teios-ra6m5 teios-ra6m5.bin
```

Flash with a debug probe and watch the semihosting stream:

```sh
probe-rs download --chip R7FA6M5BH --binary-format bin \
  --base-address 0x0 teios-ra6m5.bin
probe-rs run --chip R7FA6M5BH \
  target/thumbv8m.main-none-eabihf/release/teios-ra6m5   # semihosting → console
```

(Through the Portenta C33 Arduino bootloader, use that bootloader's DFU
path and the app-offset base instead — bench step.)

## Host tests

```sh
cargo test --lib --target aarch64-apple-darwin   # or your host triple
(cd ../teios-core && cargo test)                 # the shared logic
```

## What differs from the other images (the port deltas)

- **Target**: `thumbv8m.main-none-eabihf` (Cortex-M33), `ra6m5-pac`
  (svd2pac register crate) — **no HAL, no embassy**. `cortex-m-rt`
  entry + cycle-delay cadence; `critical-section-single-core`.
- **Transport**: **semihosting** (probe-attached), not USB/UART — the
  most verifiable bench transport with no clock bring-up.
- **HW-CRC substrate**: the RA6M5 **CRCA** peripheral, vs the RP2040 DMA
  sniffer and the H7 STM32-CRC — same primitive, third silicon family,
  same dispatch story.
- **Synchronous app**: no async runtime exists for RA, so `app` is a
  plain `fn`, not `async fn`.

## Measured joules — pending a bare-metal IIC driver

The other skeletons gain `JoulesSource::Measured` via the `tei-ina228`
EnergyMeter over `embedded-hal` 1.0 I²C (`--features measured-ina228`). The
RA6M5 is bare-metal `cortex-m-rt` + `ra6m5-pac` with **no `embedded-hal` I²C
implementation**, so the INA228 can't be driven here yet — it would need a
small RA6M5 **IIC** (`R_IIC`) driver that implements `embedded_hal::i2c::I2c`.
Until then this board stays Table-tier (or semihosting-reported cycles); the
measured path lands with that IIC driver. (Everything else — the energy math
in `tei-ina228`, the EnergyMeter contract — is board-agnostic and ready.)
