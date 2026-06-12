//! tei-sim-adiabatic — **aa-class (DeBenedictis)** adiabatic-logic energy
//! analysis, built on the `tei-sim-circuit` MNA transient solver.
//!
//! Lineage: Erik DeBenedictis's adiabatic-analysis (aa) flow — ngspice
//! sweeps over S2LAL shift-register testbenches, arXiv:2009.00448 — ported
//! as a pure-Rust capability per the roadmap (§3.6): parametric cell
//! templates emit `tei_sim_circuit::Netlist`s, a rayon-parallel ramp-time
//! sweep harness runs them across T/RC ratios with the circuit crate's
//! per-element ∫i·v energy instrumentation, and the result is the
//! **energy-per-op vs ramp-time overhead curve** that recalibrates the
//! reversible cost dialect — replacing its fixed
//! `REVERSIBLE_OVERHEAD_L0 = 10³` constant with a measured,
//! frequency-dependent function ([`overhead_curve`] + [`fitted_slope`]).
//!
//! ## What's modeled
//!
//! Two switch abstractions, both emitting plain `tei_sim_circuit::Netlist`s:
//!
//! - **R–C templates** — switches as linear on-resistances, the abstraction
//!   the adiabatic scaling law E_R(T) = (RC/T)·CV²·[1 − (RC/T)(1 − e^{−T/RC})]
//!   (Athas et al. 1994) is derived for; every validation check is
//!   closed-form. These follow the ideal −1 slope all the way down.
//! - **S2LAL MOSFET templates** (`tei-sim-circuit` M2) — CMOS transmission
//!   gates between trapezoid power-clocks and the loads, complementary
//!   control trapezoids, and the opposing off T-gate per stage
//!   ([`s2lal_chain`]). Level-1 switches add threshold/triode realism;
//!   EKV-lite switches add the honest subthreshold leakage that floors the
//!   curve, so E(T) develops the published interior minimum. Clocks and
//!   controls stay ideal sources — measuring transistor + wire dissipation
//!   is the point (see [`cells`] for the full list of kept simplifications).
//!
//! ## Validation (tests/analytic.rs + tests/s2lal.rs — analytic ground
//! truth and published trends only)
//!
//! | Check | Source of truth |
//! |---|---|
//! | Ramp charge matches the exact closed form to <1% over T/RC ∈ {1,10,100,1000} | closed form (derived in [`sweep`]) |
//! | log-log slope of E vs T → −1.0 ± 0.05 for T/RC ∈ [10, 1000] | closed-form scaling law |
//! | Abrupt limit T/RC = 0.01 recovers ½CV² within 1% | closed form |
//! | Charge-recovery cycle ≪ abrupt 2·½CV² when slow; → 2·½CV² when fast | closed form / property |
//! | N-stage chain dissipation = N × single stage; Tellegen residual ≈ 0 | property (passthrough from -circuit) |
//! | Executor's fitted slope ≡ direct-sweep slope | property |
//! | S2LAL chain: slope → −1 in the adiabatic regime | closed-form scaling law |
//! | S2LAL energy-recovery ratio improves monotonically with slower ramps | published trend (DeBenedictis arXiv:2009.00448) |
//! | S2LAL + EKV leakage: E(T) has an interior minimum (the U-curve) | published trend (leakage-floored adiabatic energy) |
//! | S2LAL abrupt limit lands at the N·CV² scale | closed form |
//!
//! Citations:
//!   - DeBenedictis 2020, *arXiv:2009.00448* — S2LAL static 2-level adiabatic
//!     (the aa shift-register testbench this crate's chain template mirrors).
//!   - DeBenedictis adiabatic-analysis (aa) ngspice scripts
//!     (github.com/erikdebenedictis) — the flow being ported.
//!   - Athas, Svensson, Koller, Tzartzanis, Chou, "Low-power digital systems
//!     based on adiabatic-switching principles", IEEE Trans. VLSI Syst.
//!     2(4):398–407, 1994 — the (RC/T)·CV² ramp-dissipation law.
//!   - Frank 2017, *Computing Communities Consortium* — fundamental
//!     adiabatic limits.

pub mod cells;
pub mod exec;
pub mod sweep;

pub use cells::{
    CellSpec, SwitchModel, charge_recovery_cell, ramp_charge_cell, s2lal_buffer_stage, s2lal_chain,
    shift_register_chain, transmission_gate,
};
pub use exec::{AdiabaticExecutor, AdiabaticJob};
pub use sweep::{
    ADIABATIC_REGIME_MIN_RATIO, CellRun, E_LANDAUER_300K, SweepPoint, fitted_slope, overhead_curve,
    ramp_charge_exact, run_cell, sweep_with_progress,
};
