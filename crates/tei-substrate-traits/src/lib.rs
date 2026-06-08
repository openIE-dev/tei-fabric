//! The Substrate trait.
//!
//! A substrate is a physical compute paradigm — photonic mesh, RRAM crossbar,
//! adiabatic CMOS, spiking silicon, p-bit fabric, GPU, … — that can execute
//! some subset of the 258 Periodic Stack primitives. Each substrate models
//! its physics from first principles and reports the joule + time + accuracy
//! cost of running a primitive on it.
//!
//! The dispatcher in `tei-cost-surface` consumes this trait and picks the
//! lowest-joule substrate per primitive (subject to fidelity constraints).

use serde::Serialize;
use tei_ir::OpProfile;
use tei_stack::Primitive;

/// Landauer's limit at 300 K: `kT · ln(2) ≈ 2.85 × 10⁻²¹ J` per erased bit.
///
/// Source: Landauer 1961 *IBM J. Res. Dev.* and the Bérut-Lutz experimental
/// verification (Bérut et al., *Nature* 483, 187 (2012)).
pub const K_T_LN2_300K: f64 = 2.85e-21;

/// Cost of one invocation of a primitive on a particular substrate.
#[derive(Debug, Clone, Copy, Serialize)]
pub struct Cost {
    /// Energy per invocation, in joules.
    pub joules_per_op: f64,
    /// Wall-clock per invocation, in seconds.
    pub seconds_per_op: f64,
    /// Fidelity loss vs an exact computation. 0.0 means lossless;
    /// 1.0 means useless output. Substrate dialects with intrinsic noise
    /// (photonic, in-memory, stochastic) report a small positive number.
    pub accuracy_loss: f64,
}

impl Cost {
    /// Total joules for `count` invocations.
    #[inline]
    pub fn joules_total(self, count: u64) -> f64 {
        self.joules_per_op * count as f64
    }

    /// Total wall-clock for `count` invocations (assuming serial; substrates
    /// model their own parallelism via `seconds_per_op` scaling).
    #[inline]
    pub fn seconds_total(self, count: u64) -> f64 {
        self.seconds_per_op * count as f64
    }
}

/// The contract every substrate dialect implements.
pub trait Substrate: Send + Sync {
    /// Stable identifier, e.g. `"baseline"`, `"photonic"`, `"in-memory"`.
    fn name(&self) -> &str;

    /// Human-readable label for UI.
    fn display_name(&self) -> &str {
        self.name()
    }

    /// Whether this substrate can execute this primitive at all.
    /// Returning false means the dispatcher will skip this pairing.
    fn supports(&self, primitive: &Primitive) -> bool;

    /// First-principles cost model for one invocation of `primitive`
    /// with `profile`.
    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost;

    /// Optional citation block — where the constants came from.
    fn citations(&self) -> &'static [&'static str] {
        &[]
    }
}
