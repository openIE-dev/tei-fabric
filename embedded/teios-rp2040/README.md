# teios-rp2040 — teiOS E1a on the Adafruit Feather RP2040 (and Pico 1)

The EMBEDDED-ROADMAP E1a artifact — the RP2040 port of teios-rp2350,
targeting the boards on the bench: hold BOOTSEL, drag a UF2, open the
serial port, and watch your board price the same primitive on two
on-die substrates and dispatch to the cheaper one — the fabric's
programming model (priced primitives, event ledgers, lowest-joule
dispatch) on the original $4 silicon.

Every second the firmware runs the **Hash primitive (Periodic Stack
id 36)** — CRC32 over a 64 KiB buffer — on:

| substrate | how | counted |
|---|---|---|
| `cpu@125mhz` | software table-driven CRC32 on the Cortex-M0+ | timer-proxy cycles + active µs |
| `dma-sniffer` | RP2040 DMA sniffer computes the CRC in hardware while the DMA channel streams the buffer | DMA transfers + active µs |

Both results are cross-checked, both runs become
`tei_ledger::EventLedger`s, both substrates are priced from the
shipped cost table, and `CostTable::cheapest` issues the dispatch
verdict — all streamed as JSON lines over USB CDC serial. The
board-independent half (CRC32, workload, JSON writers) is the shared
[`teios-core`](../teios-core) crate, used verbatim by teios-rp2350.

> The shipped J/op values are **ILLUSTRATIVE Table-tier defaults**
> (`joules_source: "table"`), pending the E1 bench measurement — which
> will be a novel public result (no published DMA-vs-core energy
> figure exists for this part). The ledger never overclaims: the
> provenance field says exactly what the numbers are.

## Cycles on the M0+ (no DWT CYCCNT exists)

The Cortex-M0+ architecturally has **no cycle counter** — DWT CYCCNT
does not exist on this core (unlike the RP2350's M33, where it is
optional and present). Per the roadmap's per-substrate `CycleSource`
doctrine, the firmware's `TimerCycleSource` derives cycles from the
RP2040 TIMER (1 MHz, 64-bit, monotonic from reset):

```
cycles = elapsed_us × clk_sys_MHz        (125 at the default clock)
```

- The boot line reports `cyccnt:false` — the truth: no counter.
- cpu ledgers carry the timer-derived proxy in `cycles`; quantization
  is ±1 µs per span = **±125 cycles** at 125 MHz (under 0.02% on the
  ~10⁶-cycle CRC32 region this firmware measures).
- `clk_sys` is read from the live clock tree at init, so the scale
  factor stays honest if the clock is reconfigured.
- The rejected alternative — SysTick (24-bit) — gives true cycle
  granularity but wraps every ~134 ms at 125 MHz and needs
  wrap-counting interrupts; the timer proxy is the simpler honest
  choice at E1's span lengths.

Energy provenance is a separate axis and unchanged: it lives in
`joules_source`, and this image ships at the Table tier.

## Build + flash (two commands)

```sh
cargo build --release --target thumbv6m-none-eabi
python3 scripts/elf2uf2.py target/thumbv6m-none-eabi/release/teios-rp2040 teios-rp2040.uf2
```

Hold the **BOOTSEL button while plugging the Feather in** (or while
tapping Reset), then drag `teios-rp2040.uf2` onto the `RPI-RP2` drive.
The board reboots into the firmware immediately. (With picotool
installed — `brew install picotool` — the conversion is equivalently
`picotool uf2 convert <elf> -t elf teios-rp2040.uf2 --family rp2040`,
and `cargo run --release --target thumbv6m-none-eabi` flashes a
BOOTSEL-mode board directly via the runner in `.cargo/config.toml`.)

Then open the serial port (e.g. `screen /dev/tty.usbmodem* 115200`,
or TEI Studio's web console) — the ledger stream starts on connect.

### Board variants

The board feature sets `board_id` on every JSON line **and** the boot2
flash bootloader (see below). Exactly one must be on:

```sh
# Adafruit Feather RP2040 (default): board_id "feather-rp2040", GD25Q64C boot2
cargo build --release --target thumbv6m-none-eabi
# Raspberry Pi Pico 1: board_id "pico", W25Q080-class boot2
cargo build --release --target thumbv6m-none-eabi --no-default-features --features board-pico
```

### Recommended `.cargo/config.toml`

A staged copy ships as [`cargo-config-staged.toml`](cargo-config-staged.toml);
activate it with:

```sh
mkdir -p .cargo && mv cargo-config-staged.toml .cargo/config.toml
```

With it in place the build is just `cargo build --release` and
`cargo run --release` flashes a BOOTSEL-mode board.

## Host tests

The wire format, the CRC32 implementation, and the cost table are
host-testable; the generic logic's tests live in `teios-core`, and
this crate's tests pin the board identity and wire shape:

```sh
cargo test --lib --target aarch64-apple-darwin   # or your host triple
(cd ../teios-core && cargo test)                 # the shared logic
```

## What differs from teios-rp2350 (the port deltas)

- **Target**: `thumbv6m-none-eabi` (Cortex-M0+), embassy-rp's `rp2040`
  feature. No atomic CAS on this core, so `portable-atomic` runs
  through `critical-section` (provided by embassy-rp).
- **boot2**: the RP2040 boot ROM loads a 256-byte second-stage
  bootloader from the start of flash to configure the QSPI chip for
  XIP — `memory.x` carves the `BOOT2` region and embassy-rp's
  `link-rp.x` places the blob (selected per board feature: GD25Q64C on
  the Feather, W25Q080-class on the Pico). The RP2350 replaced this
  whole mechanism with `IMAGE_DEF`, so this section exists only here.
- **Cycles**: `TimerCycleSource` proxy (above) instead of DWT CYCCNT.
- **Binary info**: RP2040 picotool layout (`.boot_info` header after
  the vector table + `.bi_entries`) instead of the RP2350's
  `.start_block`/`.end_block`.
- **pac**: the DMA sniffer block (SNIFF_DATA/SNIFF_CTRL, CRC32R,
  OUT_REV/OUT_INV) is register-identical; the one difference is
  `TRANS_COUNT`, a plain u32 on RP2040 (the RP2350 added a MODE field).
- **UF2 family**: `0xe48bff56` (rp2040) instead of rp2350-arm-s;
  `scripts/elf2uf2.py` here takes the family as an optional third
  argument and defaults to rp2040.

## The JSON-lines protocol (TEI Studio web console)

One JSON object per `\n`-terminated line, discriminated by `"type"`:
`boot` (once per connect), then per second `ledger` ×2, `check`,
`dispatch`. snake_case fields; `None` fields **omitted**, never
`null`; numbers may use exponent notation. The `ledger` object is
exactly `tei_ledger::EventLedger` in serde form. Full field-by-field
spec in the crate docs at the top of [`src/lib.rs`](src/lib.rs).

```json
{"type":"boot","board_id":"feather-rp2040","firmware":"teios-rp2040/0.1.0","primitive_id":36,"buf_len":65536,"cyccnt":false}
{"type":"ledger","board_id":"feather-rp2040","substrate":"cpu@125mhz","primitive_id":36,"n_ops":1,"ledger":{"cycles":1250000,"dma_transfers":0,"adc_samples":0,"accel_invocations":0,"sleep_us":0,"active_us":10000,"joules_source":"table"}}
{"type":"ledger","board_id":"feather-rp2040","substrate":"dma-sniffer","primitive_id":36,"n_ops":1,"ledger":{"cycles":0,"dma_transfers":16384,"adc_samples":0,"accel_invocations":1,"sleep_us":0,"active_us":280,"joules_source":"table"}}
{"type":"check","ok":true,"crc_cpu":2378668581,"crc_dma":2378668581}
{"type":"dispatch","primitive_id":36,"chosen":"dma-sniffer","j_per_op":4e-6,"joules_source":"table","alternatives":[{"substrate":"cpu@125mhz","j_per_op":1.8e-4,"joules_source":"table"}]}
```

Parser notes for Studio: split on `\n`, `JSON.parse` each line, switch
on `type`; tolerate unknown line types and unknown fields (the
contract grows additively). On this board `cyccnt` is always `false`
and cpu `cycles` is the timer proxy — nonzero, honest, ±125.

## How the sniffer is driven

Same recipe as the RP2350 (the sniffer survived the generation jump
unchanged): embassy-rp does not expose it, so `src/fw.rs` programs the
rp-pac registers on channel 0 (reserved by holding the `DMA_CH0`
singleton): `SNIFF_DATA` seeded `0xFFFF_FFFF`, `SNIFF_CTRL.CALC =
CRC32R` with `OUT_REV | OUT_INV` — which makes the hardware result
identical to the zlib/IEEE reflected CRC32 the software substrate
computes. The channel streams the buffer word-by-word
(`TREQ = PERMANENT`, non-incrementing dummy write target, `SNIFF_EN`
set) and the result is read back through the output transforms. E1
busy-waits on `BUSY` for simple timing attribution; IRQ + WFE (CPU
truly asleep while the sniffer works) is the next step and the
configuration the bench measurement wants.

## Next

- **The first REAL ledger**: this board exists on the bench — PPK2
  calibration replaces the illustrative Table entries with measured
  `CyclesProxy` values, POSTed as a `CalibrationReport` to the
  fabric's `/api/calibration`.
- IRQ+WFE during the DMA run; `sleep_us` attribution.
- Third substrate: PIO CRC (the roadmap's heterogeneity demo), then
  core1 via the executor.
