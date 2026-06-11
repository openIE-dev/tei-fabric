//! tei-sim-circuit — SPICE-class MNA transient solver (**MOUNTAIN 1**, scoped).
//!
//! Stages shipped (per docs/SIM-ROADMAP.md §3.5):
//!
//! - **M1** — modified nodal analysis over linear R / C / L / V / I; DC
//!   operating point; fixed-step transient with trapezoidal (default) and
//!   backward-Euler integration. The system matrix is rebuilt and dense
//!   LU-factored every step (and every Newton iteration): at our scales —
//!   adiabatic cells of at most a few hundred nodes — the O(n³) factor is
//!   microseconds, and factor reuse / sparsity is the M4 ladder, not this one.
//! - **M3** — parametric power-clock waveforms (DC, linear ramp, trapezoid,
//!   sine, PWL) and **per-element energy instrumentation ∫ i·v dt** — the
//!   entire point: dissipation per element versus ramp time is what feeds the
//!   reversible/adiabatic cost dialect.
//! - **M2 (partial)** — Shockley diode with a Newton-Raphson inner loop and
//!   SPICE-style junction-voltage limiting (`pnjlim`). MOSFET level-1 and
//!   EKV-lite remain on the M2 ladder.
//! - **M4** — sparse CSR + Markowitz-ordered LU (`tei_sim_core::sparse`)
//!   behind a solver abstraction: cells at ≤ [`SPARSE_NODE_THRESHOLD`]
//!   (16) nodes keep the dense rebuild-and-factor path bit-for-bit; larger cells
//!   assemble triplets once, factor once, and every later step/Newton
//!   iteration reuses the pivot order + fill pattern via a numeric-only
//!   refactor. Unlocks cells beyond ~500 nodes within the roadmap §6 budget
//!   (100-node adiabatic cell, 10⁶ timesteps < 10 s).
//!
//! Deliberately out (contract per the roadmap §3.5): BSIM model cards,
//! RF / harmonic balance, noise analysis, PDK parsing. Adaptive-LTE stepping
//! is deferred — fixed-step trapezoidal is second-order and exact
//! enough for cell-scale energy analytics, as the convergence tests measure.
//!
//! ## Energy accounting (the M3 contract)
//!
//! Every element accumulates absorbed energy by the trapezoid rule on its
//! instantaneous power p = i·v (current oriented n⁺→n⁻ through the element).
//! Because the MNA solve enforces KCL exactly at every accepted time point —
//! with the *companion-model* currents standing in for the reactive branches —
//! Tellegen's theorem holds **discretely**: Σ_elements i·v = 0 at every step,
//! so `source_energy = dissipated + Δstored` to LU precision. The validation
//! suite asserts both that identity and the physical closed forms
//! (½CV² abrupt-charge loss, the (RC/T)·CV² adiabatic-ramp law).
//!
//! Validation: analytic ground truth only (closed-form RC/RLC responses,
//! the canonical adiabatic scaling law, integrator-order measurements) —
//! never foreign-tool fixtures.

use serde::{Deserialize, Serialize};

pub mod exec;
mod mna;
pub mod netlist;
mod solver;
pub mod transient;

pub use exec::{CircuitExecutor, CircuitJob};
pub use netlist::{Element, Netlist, Node, VT_300K, Waveform};
pub use solver::{SPARSE_NODE_THRESHOLD, SolverChoice, SolverKind};
pub use transient::{
    DcSolution, TransientOpts, TransientResult, solve_dc, transient, transient_with_progress,
};

/// Time-integration method for the transient loop.
///
/// Trapezoidal is second-order accurate and A-stable (the SPICE default);
/// backward Euler is first-order and L-stable — useful as a damped fallback
/// for stiff start-ups, exposed for the integrator-order tests.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Method {
    #[default]
    Trapezoidal,
    BackwardEuler,
}

/// Errors from netlist validation and the solvers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CircuitError {
    /// Malformed netlist or run options (message says what).
    Invalid(String),
    /// MNA matrix singular to working precision (floating node, source loop,
    /// or inconsistent initial conditions across a hard voltage loop).
    Singular,
    /// Newton-Raphson failed to converge on the nonlinear devices.
    NoConvergence,
}

impl std::fmt::Display for CircuitError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            CircuitError::Invalid(msg) => write!(f, "invalid circuit: {msg}"),
            CircuitError::Singular => write!(f, "singular MNA system"),
            CircuitError::NoConvergence => write!(f, "Newton-Raphson did not converge"),
        }
    }
}

impl std::error::Error for CircuitError {}
