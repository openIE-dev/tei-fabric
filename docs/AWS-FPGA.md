# AWS-FPGA tier — live F2 path (scoping)

**Status**: internal scoping · the software seam is built + deployed; the live
instance + bitstream below is a **deliberate, cost-incurring step that needs an
explicit go**. Lives in `docs/` with CLOUD.md / NEUROMORPHIC.md / LANDSCAPE.md.

## Why FPGA, and only for digital columns

An FPGA is the *one* rentable fabric where we control the **datapath** and read
**real card power** — so it produces `Measured × StandIn`, then de-rates to a
target ASIC estimate with bounds (the T2 tier). It is honest **only for
digital-datapath substrates** — stochastic/p-bit, reversible, baseline, and the
digital control of in-memory/neuromorphic. For the analog-physics columns
(photonic, analog RRAM crossbar, analog neuromorphic) an FPGA measures a
*digital emulation* whose energy has no fixed relation to the target; those stay
`Modeled`. This boundary is enforced in code: `fpga_to_asic_default` returns
`None` for analog substrates and `/api/cloud/fpga-estimate` answers `422`.

## What's already built (no spend)

- `tei-meter::correction` — `Correction { factor, factor_lo, factor_hi, basis,
  source, method }` + `TargetEstimate`. `apply()` turns measured FPGA joules →
  target estimate with bounds. Default factors = Kuon & Rose 2007 dynamic-power
  gap (7.1× hard-block .. 14× logic-only, ~4.6× favorable), node-agnostic.
- `POST /api/cloud/fpga-estimate` — the calculator: measured FPGA joules +
  substrate → target estimate + provenance, or honest `422` for analog. **Live
  on cloud.thermoedge.ai.** It is the seam the live path feeds.

## The live path — steps (each separately verifiable)

1. **First datapath: the stochastic p-bit annealer.** Cleanest digital
   datapath and the flagship "thermodynamic computing" story. Implement the
   `tei-d-stochastic` anneal kernel (LFSR p-bits + coupling array + sweep
   schedule) in RTL/HLS so the FPGA runs the *same* computation the simulator
   does, byte-comparable on the Max-Cut result.
2. **Instance + toolchain.** AWS **F2** (AMD/Xilinx Virtex UltraScale+).
   Build the bitstream with Vitis/Vivado (off-instance where possible to keep
   instance hours down). Confirm **live F2 on-demand pricing** before
   committing — do not assume a figure.
3. **Card power telemetry → `FpgaMeter`.** Read the F2 board power (AMD
   sysmon / the shell's power monitor) and integrate over the run, exactly like
   `MacmonMeter` does for the Mac. New `tei-meter` meter: `Measured`,
   `Identity::StandIn { "AWS F2 (Virtex UltraScale+)" }`. Feeds `measure_run`.
4. **Wire to the correction.** `measure_run` on the FPGA → `fpga_joules` →
   `Correction::apply` → `TargetEstimate`. Same `/api/cloud/fpga-estimate`
   response shape, now fed by a real card reading instead of a posted number.
5. **Calibrate the correction (T3).** Replace the `LiteratureDefault` factor
   with `MeasuredCalibrated` against an analogous hardened block (or our own
   tapeout) to tighten the bounds. Closes the loop into the cost surface.

## Cost / effort reality

F2 is metered hourly (non-trivial — verify current pricing) and a bitstream is
real RTL/Vitis engineering. This is a lighthouse investment for one digital
column, not a switch. The payoff: the first true *measured-energy → target
estimate with a published correction* artifact — the thing no competitor
bothers to produce.

## Go-decision gate

Do **not** spin up an F2 instance or start the bitstream without an explicit
go. Until then the calculator endpoint + correction model stand on their own and
are exercised by posting measured numbers.
