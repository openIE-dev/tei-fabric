# tei-ina228 — Measured-tier joules for teiOS

A board-agnostic driver for the **TI INA228** high-precision I²C power/energy
monitor, implementing `tei_ledger::EnergyMeter`. The INA228 integrates power ×
time on-chip into a 40-bit `ENERGY` register, so a teiOS board reads accumulated
**joules directly** — turning the ledger from `JoulesSource::Table` (datasheet
estimate) into `JoulesSource::Measured` (the proof the whole fabric rests on).

Generic over `embedded-hal` 1.0 `I2c`, so the same code runs on the RP2040 /
STM32 / nRF / RA forge skeletons.

## Energy math (TI INA228 datasheet, SBOS725) — host-tested

- `CURRENT_LSB = max_expected_current / 2^19`
- `SHUNT_CAL  = 13107.2e6 × CURRENT_LSB × R_shunt`  (×4 when `ADCRANGE = 1`)
- `Energy[J]  = 51.2 × CURRENT_LSB × ENERGY_register`  (= 16 × 3.2 × CURRENT_LSB)

The three pure functions (`current_lsb`, `shunt_cal`, `energy_joules`) are unit-
tested; the I²C plumbing is generic + compile-checked.

## Bench wiring (the hardware-pending step)

1. Put the INA228's **shunt in-line on the rail whose energy you want** (the
   board's main supply, not the I²C-comms rail). Adafruit's INA228 STEMMA QT
   breakout has a 0.015 Ω shunt; pass that + your max current to `Ina228::new`.
2. Wire the INA228's **I²C** to the board (Feather RP2040: STEMMA QT = GP2 SDA /
   GP3 SCL = `I2C1`).
3. Build the firmware with the meter enabled, e.g. for the RP2040 skeleton:
   `cargo build --release --target thumbv6m-none-eabi --features measured-ina228`.

Each `run_on` then zeroes the accumulator before the primitive and reads joules
after, so the ledger line carries the measured energy for that exact window.

## Use

```rust,ignore
use tei_ina228::{Ina228, DEFAULT_ADDR};
use tei_ledger::EnergyMeter;

let mut meter = Ina228::new(i2c, DEFAULT_ADDR, 0.015, 5.0, true)?; // 15 mΩ, 5 A, low range
meter.reset();
// … run the primitive …
let joules = meter.joules(); // Some(measured J) or None if the bus is down
```
