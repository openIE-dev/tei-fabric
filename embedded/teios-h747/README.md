# teios-h747 — teiOS E1b on the Arduino Portenta H7 (STM32H747XI)

The EMBEDDED-ROADMAP E1b artifact — the first **Cortex-M7** teiOS
target and the first with a board that has a *real* DWT cycle counter.
Every second the firmware prices the same **Hash primitive (Periodic
Stack id 36)** — CRC32 over a 64 KiB buffer — on two on-die substrates,
cross-checks them, and dispatches to the cheaper one:

| substrate | how | counted |
|---|---|---|
| `cpu-m7@480mhz` | software table-driven CRC32 on the Cortex-M7 | **true DWT CYCCNT cycles** + active µs |
| `crc-hw` | the STM32 hardware CRC peripheral computes the CRC while the core idles | accel invocations + active µs |

Both results are cross-checked, both runs become `tei_ledger::EventLedger`s,
both substrates are priced from the shipped cost table, and
`CostTable::cheapest` issues the dispatch verdict — all streamed as JSON
lines over USB CDC serial, the identical protocol the RP2040/RP2350
images speak. The board-independent half (CRC32, workload, JSON
writers) is the shared [`teios-core`](../teios-core) crate, used verbatim.

A third substrate, `cpu-m4@240mhz` (the second core), is present in the
shipped cost table as a priced menu entry; actually *booting the M4 and
offloading a run to it* is the inter-core bring-up stretch (see below),
so the default app does not call it yet.

> The shipped J/op values are **ILLUSTRATIVE Table-tier defaults**
> (`joules_source: "table"`), pending the E1 bench measurement. The
> ledger never overclaims: the provenance field says exactly what the
> numbers are.

## ⚠ Hardware-verification status — read this

This firmware is **compile-verified for `thumbv7em-none-eabihf` and
links to a valid image at the correct flash base (0x08040000), but it
is NOT yet hardware-verified.** Two things need on-bench bring-up with a
debugger before the serial stream will appear, and a wrong guess at
either is *silent* (the host build is clean; only a scope/debugger
shows the fault):

1. **USB high-speed over the external ULPI PHY.** The Portenta's USB-C
   is wired to a USB3300 ULPI PHY, not the STM32's internal FS PHY. The
   firmware uses `Driver::new_hs_ulpi` with the standard STM32H7
   `OTG_HS_ULPI` alternate-function pin map (PA5 CK, PC2 DIR, PC3 NXT,
   PC0 STP, PA3/PB0/PB1/PB10/PB11/PB12/PB13/PB5 D0–D7). PHY reset/power
   sequencing and the 60 MHz ULPI clock are the bench step.
2. **The 480 MHz clock tree** from the 25 MHz HSE crystal. The RCC PLL
   dividers in `fw.rs` are a conservative skeleton; the exact `pll1` /
   voltage-scale / flash-latency set for a stable 480 MHz + 48 MHz USB
   clock is the bench step.

The parts that **are** verified now: the substrate/dispatch logic, the
ledger wire protocol, the CRC-peripheral configuration (poly 0x04C11DB7,
init 0xFFFF_FFFF, byte-reflected in/out, software final XOR =
zlib/IEEE CRC32, matching `teios-core::crc32_software`), and the link
layout (app above Arduino's MCUboot bootloader). These are exercised by
the host tests below; the USB-HS/RCC items are the documented
hardware-pending boundary, delivered the same way E1a was
(compile-verified, bench-pending) — honest about what a debugger still
has to confirm.

## Cycles on the M7 (a real counter, finally)

Unlike the RP2040's Cortex-M0+ (no DWT) the M7 has an **architectural
DWT CYCCNT**. The firmware enables `DCB.TRCENA` + `DWT.CYCCNTENA` and
reads `DWT::cycle_count()` directly:

- The boot line reports `cyccnt:true` — the truth: a real counter.
- `cpu-m7@480mhz` ledgers carry true per-cycle counts, not a timer
  proxy. (The counter is 32-bit; the CRC32 region is ~10⁵–10⁶ cycles,
  well under the ~9 s wrap at 480 MHz, and `delta` is computed in u32
  space so a single wrap is still correct.)

Energy provenance is a separate axis: it lives in `joules_source`, and
this image ships at the Table tier.

## Build + flash

```sh
cargo build --release --target thumbv7em-none-eabihf
# objcopy to a raw image (llvm-objcopy ships with the rust toolchain):
"$(find "$(rustc --print sysroot)" -name llvm-objcopy | head -1)" \
  -O binary target/thumbv7em-none-eabihf/release/teios-h747 teios-h747.bin
```

Put the board in DFU (double-tap the reset button — the green LED
pulses), then flash to the application slot **above** Arduino's
bootloader:

```sh
dfu-util -d 2341:035b -a 0 -s 0x08040000:leave -D teios-h747.bin
```

(`2341:035b` is the Portenta H7 DFU VID:PID; `-a 0` is the internal
flash alt-setting; `:leave` resets into the app.) Then open the serial
port (`screen /dev/tty.usbmodem* 115200`, or TEI Studio's web console)
— the ledger stream starts on USB connect.

### Board variants

The H7 and H7 Lite share silicon and firmware; only the reported
`board_id` differs. Exactly one board feature must be on:

```sh
# Arduino Portenta H7 (default): board_id "portenta-h7"
cargo build --release --target thumbv7em-none-eabihf
# Arduino Portenta H7 Lite: board_id "portenta-h7-lite"
cargo build --release --target thumbv7em-none-eabihf \
  --no-default-features --features board-portenta-h7-lite
```

### `.cargo/config.toml`

Ships activated: [`.cargo/config.toml`](.cargo/config.toml) sets the
default target (`thumbv7em-none-eabihf`) and the `--nmagic` link flag,
so the build is just `cargo build --release`. The `-Tlink.x` link arg
itself comes from `build.rs`, emitted only for the bare-metal target so
host `cargo test` links normally.

## Host tests

The wire format, the CRC32 implementation, and the cost table are
host-testable; the generic logic's tests live in `teios-core`, and this
crate's tests pin the board identity, the three-substrate cost table
(crc-hw cheapest), and `cyccnt:true`:

```sh
cargo test --lib --target aarch64-apple-darwin   # or your host triple
(cd ../teios-core && cargo test)                 # the shared logic
```

The `build.rs` is target-gated: it only emits `memory.x` and the
`-Tlink.x` link arg for the bare-metal target, so the host `cargo test`
of the lib links against the normal system toolchain.

## What differs from the RP2040/RP2350 images (the port deltas)

- **Target**: `thumbv7em-none-eabihf` (Cortex-M7, hard-float),
  `embassy-stm32` `stm32h747xi-cm7`. The M7 has native atomics, so
  `critical-section` uses cortex-m's `critical-section-single-core`
  (the M7 runs the firmware alone; the M4 is not booted).
- **Dual-core init**: the STM32H747 is two CPUs on one die, so embassy's
  `init` is replaced by `init_primary(config, &SHARED_DATA)` — the M7
  (primary) brings up the clock tree and publishes it to the M4 via a
  `SharedData` handshake region.
- **Flash base 0x08040000**: Arduino ships an MCUboot-style bootloader
  in the first sectors; the teiOS app links above it (`memory.x` FLASH
  ORIGIN = 0x08040000, LEN 1792K), which is why DFU writes to that
  offset rather than 0x08000000.
- **Hardware CRC substrate**: instead of the RP2040 DMA sniffer, the
  cheap path is the STM32 CRC peripheral — same primitive, different
  silicon, same dispatch story.
- **Real cycle counter**: `cyccnt:true` (DWT CYCCNT), vs the RP2040's
  timer proxy and `cyccnt:false`.
