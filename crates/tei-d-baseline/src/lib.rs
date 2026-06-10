//! Baseline substrate — modern CPU / GPU joule model.
//!
//! This is the comparator the other substrate dialects price against. The
//! model is anchored to a measured anchor point:
//!
//!   **NVIDIA Jetson Orin Nano**, 7 W TDP, ~200 GFLOPS of INT8 inference at
//!   30 fps. Per-MAC: ~3.5 × 10⁻¹¹ J/op delivered at the system level.
//!
//!   Per-MAC at the Landauer floor (~64 bits erased / MAC at 300 K):
//!   `64 × kT·ln(2) = 1.8 × 10⁻¹⁹ J`.
//!
//!   Measured ratio: **~2 × 10⁸ above floor for L₂ ops**. That number is the
//!   anchor for `BASELINE_OVERHEAD_L2`. The same Orin Nano in dense FP32 lands
//!   in the same band; the dtype effect is folded into `bits_erased_per_op`.
//!
//! L₀ (bijective) ops on standard CMOS pay floating-point rounding,
//! rail-to-rail transitions, and clock distribution, but no information
//! erasure. Lab measurements and the broader low-power literature put this
//! at ~10⁶ above the Landauer floor (~3-4 orders of magnitude better than
//! L₂). Used as `BASELINE_OVERHEAD_L0`.
//!
//! Throughput model: 200 GFLOPS at 30 fps ⇒ 6.6 × 10⁹ MAC/s sustained. For
//! `seconds_per_op` we report `1.0 / SUSTAINED_OPS_PER_SEC`. Real GPUs do
//! many ops in parallel — the wall-clock per *one* op is much less than this
//! when there are many of them — but the throughput-divided model is what
//! the dispatcher actually wants when comparing substrate budgets.

use tei_ir::OpProfile;
use tei_stack::Primitive;
use tei_substrate_traits::{Cost, K_T_LN2_300K, Substrate};

/// L₂ overhead above the Landauer floor, calibrated against Orin Nano.
/// Source: Orin Nano datasheet (200 GOPS @ 7 W, INT8); Landauer 1961.
const BASELINE_OVERHEAD_L2: f64 = 2.0e8;

/// L₀ overhead — bijective ops on standard CMOS still pay rail transitions,
/// clock distribution, FP rounding. ~10⁶ above floor at the practical limit.
const BASELINE_OVERHEAD_L0: f64 = 1.0e6;

/// L₁ overhead — sampling on a CPU/GPU is a derived computation, not an
/// intrinsic device property; pays a software-style cost dominated by the
/// PRNG and any rejection-sampling iteration.
const BASELINE_OVERHEAD_L1: f64 = 1.0e9;

/// L₂max overhead — wide reductions, hashes, sorts on a CPU/GPU pay multiple
/// memory-hierarchy levels and an extra factor over plain L₂.
const BASELINE_OVERHEAD_L2MAX: f64 = 5.0e8;

/// Sustained dense-MAC throughput on the Orin Nano anchor (ops/s).
const SUSTAINED_OPS_PER_SEC: f64 = 6.6e9;

/// Bits erased per invocation, by thermo class.
///
/// - L₀: only rounding + rail-transition noise. 4 bits.
/// - L₁: one sampled word per invocation. 32 bits.
/// - L₂: a single MAC-class op. 64 bits (two 32-bit inputs → one output + guard).
/// - L₂max: wide reduction collapsing N inputs to 1. ~4096 bits for typical
///          systolic-array widths.
fn bits_erased_per_op(p: &Primitive) -> f64 {
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

fn overhead_for(p: &Primitive) -> f64 {
    let thermo = p.thermo.as_str();
    if thermo.starts_with("L0") {
        BASELINE_OVERHEAD_L0
    } else if thermo == "L1" {
        BASELINE_OVERHEAD_L1
    } else if thermo == "L2max" {
        BASELINE_OVERHEAD_L2MAX
    } else {
        BASELINE_OVERHEAD_L2
    }
}

/// The baseline (CPU/GPU) substrate.
pub struct Baseline;

impl Substrate for Baseline {
    fn name(&self) -> &str {
        "baseline"
    }
    fn display_name(&self) -> &str {
        "CPU / GPU baseline"
    }

    /// Universal substrate — supports every primitive (general-purpose silicon
    /// can in principle run anything).
    fn supports(&self, _primitive: &Primitive) -> bool {
        true
    }

    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        // bits_erased_per_op is per ATOMIC op (per MAC for matmul-class,
        // per scalar erasure otherwise). Scale by the number of atomic ops
        // inside one invocation so the returned cost is joules-per-invocation.
        let bits = bits_erased_per_op(primitive);
        let floor = bits * K_T_LN2_300K;
        let joules_per_atomic = floor * overhead_for(primitive);

        let atomic_ops_per_invocation: u64 = match primitive.id {
            // Matmul-class: m·k·n MACs per invocation.
            // (Dense MatMul · SpMM/SpMV · Attention · Convolution · Tensor-network ·
            //  Photonic MZI transform · Matrix decomposition · Eigendecomposition ·
            //  Krylov subspace.) Must mirror the matmul-class lists in the
            //  photonic / in-memory / reversible dialects so per-substrate
            //  ratios stay apples-to-apples.
            18 | 19 | 20 | 24 | 48 | 53 | 76 | 77 | 78 => profile.matmul_macs().unwrap_or(1),
            // Sampling-class: sweeps × variables sample events per invocation.
            // (Stochastic rounding · MC accept/reject · MCMC step ·
            //  Bayesian posterior · Discrete-Gaussian sampling ·
            //  Bootstrap resampling · Simulated annealing · Lattice Boltzmann.)
            8 | 38 | 39 | 99 | 245 | 251 | 258 | 274 => profile.sample_events().unwrap_or(1),
            // Spiking-class (NEURO family: LIF neuron · STDP). Dense GPU
            // simulation can't exploit spike sparsity: it pays a MAC for
            // every synapse of every neuron at every timestep, regardless of
            // whether the neuron fired. atomic = neurons × timesteps × fanout.
            50 | 51 => {
                let fanout = profile.reduce_dim.unwrap_or(128) as u64;
                profile.sample_events().unwrap_or(1).saturating_mul(fanout)
            }
            _ => 1,
        };

        let joules_per_op = joules_per_atomic * atomic_ops_per_invocation as f64;
        let seconds_per_op = atomic_ops_per_invocation as f64 / SUSTAINED_OPS_PER_SEC;
        Cost {
            joules_per_op,
            seconds_per_op,
            accuracy_loss: 0.0,
        }
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "NVIDIA Jetson Orin Nano datasheet (7 W TDP, ~200 GOPS INT8 @ 30 fps).",
            "Landauer 1961, IBM J. Res. Dev. 5(3): 183.",
            "Bérut et al. 2012, Nature 483: 187 (experimental verification of kT·ln(2)).",
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tei_ir::{Dtype, TensorShape};

    fn matmul_profile() -> OpProfile {
        OpProfile {
            shape: TensorShape {
                dims: vec![512, 2048],
            },
            reduce_dim: Some(768),
            batch: 1,
            dtype: Dtype::F16,
            sparsity: 0.0,
            sweeps: None,
            variables: None,
        }
    }

    fn dense_matmul_primitive() -> tei_stack::Primitive {
        serde_json::from_str(
            r#"{
            "id": 18, "name": "Dense MatMul", "family": "LA",
            "B": "kernel", "C": "L2", "D": "data-parallel",
            "existing": "HW", "silicon_target": null, "wave": null
        }"#,
        )
        .unwrap()
    }

    /// Anchor: L₂ MAC on Orin Nano-class silicon = 64 bits erased ×
    /// kT·ln(2) × 2×10⁸ overhead = 36.48 pJ/MAC exactly.
    #[test]
    fn l2_mac_anchor_36_48_pj() {
        let s = Baseline;
        let p = dense_matmul_primitive();
        let macs = 512u64 * 768 * 2048;
        let cost = s.cost(&p, &matmul_profile());
        let per_mac = cost.joules_per_op / macs as f64;
        let expected = 64.0 * K_T_LN2_300K * 2.0e8; // 3.648e-11
        assert!(
            (per_mac - expected).abs() / expected < 1e-9,
            "per-MAC {per_mac:.4e} != anchor {expected:.4e}"
        );
    }

    /// Baseline supports every primitive (universal substrate).
    #[test]
    fn universal_support() {
        let s = Baseline;
        assert!(s.supports(&dense_matmul_primitive()));
    }
}
