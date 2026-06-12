# teios-rp2350 — teiOS E1 on the Raspberry Pi Pico 2

The EMBEDDED-ROADMAP E1 artifact: hold BOOTSEL, drag a UF2, open the
serial port, and watch your board price the same primitive on two
on-die substrates and dispatch to the cheaper one — the fabric's
programming model (priced primitives, event ledgers, lowest-joule
dispatch) running on a $5 board.

Every second the firmware runs the **Hash primitive (Periodic Stack
id 36)** — CRC32 over a 64 KiB buffer — on:

| substrate | how | counted |
|---|---|---|
| `cpu@150mhz` | software table-driven CRC32 on the Cortex-M33 | DWT CYCCNT cycles + active µs |
| `dma-sniffer` | RP2350 DMA sniffer computes the CRC in hardware while the DMA channel streams the buffer | DMA transfers + active µs |

Both results are cross-checked, both runs become
`tei_ledger::EventLedger`s, both substrates are priced from the
shipped cost table, and `CostTable::cheapest` issues the dispatch
verdict — all streamed as JSON lines over USB CDC serial.

> The shipped J/op values are **ILLUSTRATIVE Table-tier defaults**
> (`joules_source: "table"`), pending the E1 bench measurement — which
> will be a novel public result (no published PIO/DMA-vs-core energy
> figure exists for this part). The ledger never overclaims: the
> provenance field says exactly what the numbers are.

## Build + flash (two commands)

```sh
cargo build --release --target thumbv8m.main-none-eabihf
picotool uf2 convert target/thumbv8m.main-none-eabihf/release/teios-rp2350 -t elf teios-rp2350.uf2
```

Hold BOOTSEL while plugging the Pico 2 in, then drag `teios-rp2350.uf2`
onto the `RP2350` drive. (`brew install picotool` on macOS;
`scripts/elf2uf2.py` is a stdlib-only Python fallback that does the
same conversion: `python3 scripts/elf2uf2.py <elf> <uf2>`.)

With a board already in BOOTSEL mode and picotool installed,
`cargo run --release --target thumbv8m.main-none-eabihf` flashes
directly (runner configured in `.cargo/config.toml` — see below).

Then open the serial port (e.g. `screen /dev/tty.usbmodem* 115200`,
or TEI Studio's web console) — the ledger stream starts on connect.

### Recommended `.cargo/config.toml`

A staged copy ships as [`cargo-config-staged.toml`](cargo-config-staged.toml);
activate it with:

```sh
mkdir -p .cargo && mv cargo-config-staged.toml .cargo/config.toml
```

With it in place the build is just `cargo build --release` and
`cargo run --release` flashes a BOOTSEL-mode board. (The `--target`
flag form above works identically without it.)

## Host tests

The wire format, the CRC32 implementation, and the cost table are
host-testable. The board-independent half (CRC32, workload, JSON
writers) lives in the shared [`teios-core`](../teios-core) crate (also
used by the RP2040 port, [`teios-rp2040`](../teios-rp2040)); this
crate's `src/lib.rs` pins the pico2 identity and locks the emitted
lines to `tei-ledger`'s serde shape by comparing against `serde_json`
output:

```sh
cargo test --lib --target aarch64-apple-darwin   # or your host triple
(cd ../teios-core && cargo test)                 # the shared logic
```

## The JSON-lines protocol (TEI Studio web console)

One JSON object per `\n`-terminated line, discriminated by `"type"`:
`boot` (once per connect), then per second `ledger` ×2, `check`,
`dispatch`. snake_case fields; `None` fields **omitted**, never
`null`; numbers may use exponent notation. The `ledger` object is
exactly `tei_ledger::EventLedger` in serde form. Full field-by-field
spec in the crate docs at the top of [`src/lib.rs`](src/lib.rs).

```json
{"type":"boot","board_id":"pico2","firmware":"teios-rp2350/0.1.0","primitive_id":36,"buf_len":65536,"cyccnt":true}
{"type":"ledger","board_id":"pico2","substrate":"cpu@150mhz","primitive_id":36,"n_ops":1,"ledger":{"cycles":524288,"dma_transfers":0,"adc_samples":0,"accel_invocations":0,"sleep_us":0,"active_us":3495,"joules_source":"table"}}
{"type":"ledger","board_id":"pico2","substrate":"dma-sniffer","primitive_id":36,"n_ops":1,"ledger":{"cycles":0,"dma_transfers":16384,"adc_samples":0,"accel_invocations":1,"sleep_us":0,"active_us":230,"joules_source":"table"}}
{"type":"check","ok":true,"crc_cpu":2378668581,"crc_dma":2378668581}
{"type":"dispatch","primitive_id":36,"chosen":"dma-sniffer","j_per_op":4e-6,"joules_source":"table","alternatives":[{"substrate":"cpu@150mhz","j_per_op":9e-5,"joules_source":"table"}]}
```

Parser notes for Studio: split on `\n`, `JSON.parse` each line, switch
on `type`; tolerate unknown line types and unknown fields (the
contract grows additively); `cyccnt:false` in `boot` means cpu ledgers
carry `cycles:0` (M33 cycle counter is architecturally optional —
`DWT_CTRL.NOCYCCNT` is honored, not assumed).

## How the sniffer is driven

embassy-rp does not expose the DMA sniffer, so `src/fw.rs` programs it
through the rp-pac registers (`unstable-pac` feature) on channel 0
(reserved by holding the `DMA_CH0` singleton): `SNIFF_DATA` seeded
`0xFFFF_FFFF`, `SNIFF_CTRL.CALC = CRC32R` with `OUT_REV | OUT_INV` —
which makes the hardware result identical to the zlib/IEEE reflected
CRC32 the software substrate computes. The channel streams the buffer
word-by-word (`TREQ = PERMANENT`, non-incrementing dummy write target,
`SNIFF_EN` set) and the result is read back through the output
transforms. E1 busy-waits on `BUSY` for simple timing attribution;
IRQ + WFE (CPU truly asleep while the sniffer works) is the next step
and the configuration the bench measurement wants.

## Next

- Bench calibration kit (PPK2 scripts) → replace the illustrative
  Table entries with measured `CyclesProxy` values, POST a
  `CalibrationReport` to the fabric's `/api/calibration`.
- IRQ+WFE during the DMA run; `sleep_us` attribution.
- Third substrate: PIO CRC (the roadmap's heterogeneity demo), then
  the Hazard3 cores via the architecture switch.
