//! Reversible substrate — adiabatic CMOS / S2LAL / Q2LAL physics.
//!
//! Adiabatic logic recovers charge from capacitive nodes by ramping clock
//! voltages slowly enough that very little CV² is dissipated per transition.
//! For the *bijective* (L₀) phase of a computation, energy approaches the
//! Landauer floor (kT·ln(2)) instead of paying the standard L₂ overhead.
//! For the *irreversible projection* — the L₂ phase that erases information,
//! discards intermediate state, or collapses to a smaller output — the
//! circuit pays standard CMOS L₂ cost.
//!
//! For each Bennett-decomposable primitive, the Periodic Stack catalog
//! records its `l0_phase` and `l2_phase`. We assign a *reversible fraction*
//! `f ∈ [0,1]` describing what share of the primitive's atomic-op work is
//! the free L₀ phase. The per-atomic-op energy is then:
//!
//!   E_per_op  =  f · O_rev · E_floor  +  (1−f) · O_l2 · E_floor
//!
//! where `O_rev ≈ 10³` is the adiabatic overhead above Landauer at practical
//! switching speeds, and `O_l2` is the standard L₂ CMOS overhead anchored to
//! the same reference silicon as `tei-d-baseline`.
//!
//! Throughput tradeoff: adiabatic logic runs slower than standard CMOS
//! because the clock ramp must be slow enough that the RC time constant
//! is small compared to the transition. Practical adiabatic chips clock
//! around 0.1–1 GHz (5-20× slower than a same-node CMOS digital baseline).
//! We model 1 GHz sustained atomic-op throughput.
//!
//! Primitives this substrate supports: every primitive that has a Bennett
//! decomposition in the catalog. The dispatcher pre-filters by querying
//! `stack.has_bennett(id)`, so as new Bennett edges are added to the
//! catalog this substrate automatically expands its coverage.
//!
//! Citations:
//!   - DeBenedictis 2020, *arXiv:2009.00448* — S2LAL static 2-level adiabatic.
//!   - Frank 2017, *Computing Communities Consortium* — fundamental adiabatic limits.
//!   - DeBenedictis adiabatic-analysis (aa) ngspice scripts (github.com/erikdebenedictis).
//!   - Bennett 1989, *SIAM J. Comput.* 18(4): 766 — time/space tradeoffs of reversible computation.
//!   - Vaire Computing (2024) — commercial reversible-CMOS demonstration.
//!   - 16-bit adiabatic microprocessor demo on FDSOI 90nm (DeBenedictis group).

use std::sync::Arc;
use tei_ir::OpProfile;
use tei_stack::{Primitive, Stack};
use tei_substrate_traits::{Cost, K_T_LN2_300K, Substrate};

/// Adiabatic overhead above the Landauer floor for the L₀ free phase, at
/// production switching speeds. Lab measurements demonstrate ~10²–10³ above
/// floor for quasi-static and practical-speed adiabatic logic respectively.
/// `1e3` is a defensible production-speed target for the L₀ kernel of any
/// Bennett-decomposable primitive (DeBenedictis aa + Vaire prototypes).
const REVERSIBLE_OVERHEAD_L0: f64 = 1.0e3;

/// L₂ irreversible-projection overhead — same anchor as `tei-d-baseline`'s
/// L₂ constant (Orin Nano-class CMOS at production density).
const REVERSIBLE_OVERHEAD_L2: f64 = 2.0e8;

/// Sustained atomic-op throughput on a production-speed adiabatic chip
/// (post-FDSOI / post-Vaire-class). 1 GS/s as a conservative reference.
const REVERSIBLE_OPS_PER_SEC: f64 = 1.0e9;

/// Per-atomic-op bits erased — taken from the primitive's thermo class.
/// This is the *upper bound* on per-op erasure; the reversible substrate
/// only pays the L₂ overhead on the `(1-fraction)` share that actually
/// erases. The L₀ share charges no bit-erasure cost (just the adiabatic
/// rail + clock energy modeled in `REVERSIBLE_OVERHEAD_L0`).
fn bits_erased_per_atomic(p: &Primitive) -> f64 {
    let thermo = p.thermo.as_str();
    if thermo.starts_with("L0") {
        4.0
    } else if thermo == "L1" {
        32.0
    } else if thermo == "L2max" {
        4096.0
    } else if thermo.starts_with("L2") {
        64.0
    } else {
        64.0
    }
}

/// How many atomic ops one invocation expands into. Matches `tei-d-baseline`'s
/// scaling so the dispatcher compares apples to apples.
fn atomic_ops_per_invocation(p: &Primitive, profile: &OpProfile) -> u64 {
    match p.id {
        // Matmul-class (the LA / TR / TEN entries with MAC structure).
        18 | 19 | 20 | 24 | 48 | 76 | 77 | 78 => profile.matmul_macs().unwrap_or(1),
        // Sampling-class primitives (L₁) don't have Bennett decompositions in
        // the catalog; the substrate won't be asked about them. If asked, fall
        // back to sample_events.
        8 | 38 | 39 | 99 | 245 | 251 | 258 | 274 => profile.sample_events().unwrap_or(1),
        _ => 1,
    }
}

/// Reversible fraction — what share of one invocation's atomic-op work is
/// the bijective L₀ phase that adiabatic logic recovers. The catalog's
/// `l0_phase` / `l2_phase` strings describe the split qualitatively; this
/// table assigns numeric fractions for the heaviest hitters and falls back
/// to a default of 0.5 for any other Bennett-decomposable primitive.
///
/// Returns 0.0 if the primitive has no Bennett decomposition (the substrate
/// will refuse it via `supports()` anyway; this is a safety floor).
fn reversible_fraction(p: &Primitive, has_bennett: bool) -> f64 {
    if !has_bennett {
        return 0.0;
    }
    // Pure-L₀ primitives — every atomic op is in the free phase.
    if p.is_l0() {
        return 1.0;
    }
    match p.id {
        // Linear-algebra core.
        18 => 0.90, // Dense MatMul — multiply phase dominates over k-reduction
        19 => 0.90, // SpMM / SpMV — same
        20 => 0.80, // Attention — QKᵀ and V mats are bijective; softmax is reductive
        24 => 0.90, // Convolution — circular = full L₀; im2col + reduction is L₂
        76 => 0.70, // Matrix decomposition — Givens / Householder are unitary
        77 => 0.70, // Eigendecomposition — QR iteration is unitary
        78 => 0.70, // Krylov subspace — matrix-vector + ortho phase reversible
        // Transforms & elementwise functions.
        79 => 1.00, // DCT — unitary butterfly, only rounding is L₂
        80 => 1.00, // Hilbert transform — FFT + phase shift, unitary
        // Reductive / non-linear blocks.
        26 => 0.10, // Pooling — window extraction is free; max/avg reduces
        34 => 0.20, // Softmax — exp is bijective, normalization is reductive
        35 => 0.30, // Normalization — centering reversible, sigma discarded
        // Other Bennett-decomposable primitives.
        6  => 0.50, // Sort — comparison network is reversible, perm-apply is L₂
        36 => 0.50, // Hash — mixing permutation is bijective, truncation is L₂
        15 => 0.50, // Modular exponentiation
        43 => 0.60, // Relational join — Cartesian product reversible, predicate L₂
        109 => 0.70, // Kalman step — linear predict reversible, innovation L₂
        118 => 0.50, // Convex hull
        132 => 0.70, // Exponential map (diffgeo)
        // Catch-all for Bennett-decomposed primitives without a specific number.
        _ => 0.50,
    }
}

pub struct Reversible {
    stack: Arc<Stack>,
}

impl Reversible {
    pub fn new(stack: Arc<Stack>) -> Self {
        Self { stack }
    }
}

impl Substrate for Reversible {
    fn name(&self) -> &str { "reversible" }
    fn display_name(&self) -> &str { "Reversible (adiabatic CMOS)" }

    fn supports(&self, primitive: &Primitive) -> bool {
        // The catalog says which primitives have a reversible kernel.
        self.stack.has_bennett(primitive.id)
    }

    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        let has_bennett = self.stack.has_bennett(primitive.id);
        let fraction = reversible_fraction(primitive, has_bennett);

        let bits = bits_erased_per_atomic(primitive);
        let floor = bits * K_T_LN2_300K;

        // L₀ free phase pays only the adiabatic rail-and-clock overhead.
        let free_phase = fraction * REVERSIBLE_OVERHEAD_L0 * floor;
        // L₂ irreversible projection still pays full L₂ overhead.
        let paid_phase = (1.0 - fraction) * REVERSIBLE_OVERHEAD_L2 * floor;
        let joules_per_atomic = free_phase + paid_phase;

        let atomic_ops = atomic_ops_per_invocation(primitive, profile);
        let joules_per_op = joules_per_atomic * atomic_ops as f64;
        let seconds_per_op = atomic_ops as f64 / REVERSIBLE_OPS_PER_SEC;

        Cost {
            joules_per_op,
            seconds_per_op,
            accuracy_loss: 0.0,
        }
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "DeBenedictis 2020, arXiv:2009.00448 — S2LAL static 2-level adiabatic.",
            "Frank 2017, Computing Communities Consortium — adiabatic fundamental limits.",
            "DeBenedictis aa (github.com/erikdebenedictis) — ngspice adiabatic-analysis framework.",
            "Bennett 1989, SIAM J. Comput. 18(4): 766 — reversible computation tradeoffs.",
            "Vaire Computing 2024 — commercial reversible-CMOS prototype.",
            "16-bit adiabatic microprocessor on FDSOI 90nm — DeBenedictis group.",
        ]
    }
}
