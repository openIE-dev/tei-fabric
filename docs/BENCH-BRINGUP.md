# Bench bring-up — the first hardware-verified Measured joule

**Status**: runbook · the physical step that converts teiOS from
compile/link-verified to **hardware-verified**.

Everything in teiOS is verified in software today: the kernel (`tei-rt`) is
host-tested, all four MCU images cross-compile through the forge, the
calibration loop and the Embassy trace binding link. The one thing software
can't do is read a real joule. This is the checklist that does.

**Goal**: flash an RP2040, wire an INA228 in-line on its supply, and watch the
first `joules_source: measured` ledger line relay into the HUB cost surface +
FLEET roster — a real measured J/op for the Hash primitive, not a Table-tier
default.

## Bill of materials

| item | notes |
|---|---|
| Adafruit Feather RP2040 | on the bench. Has a STEMMA QT port = I²C1 (SDA=GP2, SCL=GP3) — exactly what the firmware's measured build expects. (A Pico works too; wire I²C to GP2/GP3 by hand.) |
| Adafruit INA228 breakout (#5832) | 20-bit power/energy monitor, **15 mΩ shunt** (= the firmware's 0.015 Ω default), STEMMA QT, I²C addr **0x40**. The 40-bit hardware ENERGY accumulator is what makes this Measured-tier, not proxy. |
| STEMMA QT ↔ STEMMA QT cable | I²C + 3V3 to the breakout |
| 2 wires | to put the INA228 shunt in series with the rail you measure |
| Chromium browser | Studio's flash + live ledger console are WebSerial/WebUSB |

## Step 1 — build + flash (Studio, turnkey)

1. Open **studio.thermoedge.ai**.
2. **CODE** — the default app already does the full loop:
   `run_on(cpu)` → `run_on(dma)` → `check` → `dispatch` → `run` (dispatched)
   → `publish` (emits a calibration report per substrate).
3. **BUILD** — pick `Adafruit Feather RP2040`, tick **Measured joules**
   (wires in the INA228 EnergyMeter), BUILD. ~2 s warm → a `.uf2`.
4. **FLASH** — download the UF2, hold **BOOTSEL** while tapping reset so the
   `RPI-RP2` drive mounts, drag the UF2 on. The board reboots into teiOS.

## Step 2 — wire the INA228

Two independent connections — don't conflate them:

- **Data + breakout power** (STEMMA QT): Feather ↔ INA228. This is just I²C
  (SDA=GP2, SCL=GP3) + 3V3/GND for the chip itself. Matches the firmware's
  `I2c::new_blocking(I2C1, GP3=SCL, GP2=SDA)`.
- **The shunt, in series with the rail you measure** (IN+ / IN−): the INA228
  measures current through its shunt, so the shunt must sit **in the supply
  path**, not on the STEMMA bus. To measure the Feather's own draw, route its
  input power through the shunt:

  ```
  bench/USB 5V ──▶ INA228 IN+ ──[15 mΩ shunt]──▶ INA228 IN− ──▶ Feather VBUS
  ```

  The Feather then reads its own consumption over I²C. (Honest note: this is
  upstream of the on-board regulator, so it's whole-board input power — the
  rail to state in any report. Measuring a downstream rail means moving the
  shunt there.)

Firmware constants (the measured build's defaults, `teios-app-rp2040/src/fw.rs`):
addr `0x40`, shunt `0.015 Ω`, max-current `5 A`, **low-range**. The Adafruit
ADA5832 matches addr + shunt out of the box, so the defaults *work as-is* —
the energy reading is accurate (the SHUNT_CAL calibration is exact at any
`max_a`).

**Resolution tune (optional but worth it for a small board).** Low-range is
ADCRANGE=1 = ±40.96 mV full-scale; across the 15 mΩ shunt that ceils
measurable current at ≈ **2.73 A** (40.96 mV / 0.015 Ω) — far above any MCU's
draw, so low-range is the right pick. But `max_a = 5 A` sets a coarse
`CURRENT_LSB = 5 / 2¹⁹ ≈ 9.5 µA`. A Feather pulls ~50–200 mA, so dropping
`max_a` to ~`1.0` in the `Ina228::new(..)` call gives ~5× finer resolution
(15 mΩ × 1 A = 15 mV, still inside ±40.96 mV) — accuracy unchanged, just more
bits where the signal lives. (Verified against the TI INA228 datasheet
SBOS725 + the Adafruit ADA5832 spec, 2026-06.)

If your breakout's shunt differs from 15 mΩ, the J/op scales linearly — set
the real shunt value in the same one-line constant and re-build.

## Step 3 — observe (CONSOLE)

In Studio **CONSOLE**, **Connect a board** (WebSerial, 115200). Per pass you
should see:

- `ledger cpu@125mhz … <J> (measured)` and `ledger dma-sniffer … <J> (measured)`
  — **`measured`, green**, not `table`. That is the INA228 reading real energy.
- `dispatch → <lowest-joule substrate>` from the re-priced cost table.
- `report …` lines (from `publish`) — the device's calibrated prices.

If joules still say `table`, the meter isn't being read: check the STEMMA
connection, the 0x40 address, and that the BUILD had **Measured** ticked.

## Step 4 — verify the loop closed (HUB / FLEET)

Studio relays each `report` line to the fabric. Confirm it landed:

- **HUB** — `feather-rp2040` now shows a **Measured** J/op for the Hash
  primitive (id 36), sourced from your bench. Coverage `boards_measured`
  ticks up.
- **FLEET** — `feather-rp2040` appears in the roster with a recent
  `last_seen` and its measured J/op.
- Or check the API directly:
  ```sh
  curl -s http://<host>:9651/api/hub | python3 -m json.tool | grep -A3 feather
  ```

**Done** = the HUB cost surface holds a J/op for the Hash primitive on the
Feather that came from a meter on your bench, not from a datasheet. From that
point the dispatch verdict on this board is priced on reality.

## What this bench session also confirms (beyond the joule)

The RP2040 image's other firmware-unverified pieces get exercised at the same
time: USB-CDC transport, the DMA-sniffer hardware CRC vs the M0+ software CRC
agreement (`check`), and the runtime's dispatch + re-pricing on real silicon.
The INA228 energy math is exact (40-bit hardware accumulator); the shunt value
and placement are the only calibration variables.
