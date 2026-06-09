//! In-memory substrate — RRAM / memristor crossbar MVM physics.
//!
//! A resistive crossbar performs matrix-vector multiply in O(1) by applying
//! voltages to rows, summing currents along columns, and digitizing the
//! result. Weights are stored as device conductances; the MVM is performed
//! by Ohm's law + Kirchhoff's current law at the array level. Energy per
//! MAC decomposes into:
//!
//!   1. **Crossbar conductance current.** I = G·V on each cell. For typical
//!      RRAM (G ~10⁻⁵ S, V ~0.2 V), each cell dissipates ~10⁻⁶ W during a
//!      pulse, ~10⁻¹⁵ J per 1 ns pulse — ~1 fJ/MAC at the device level.
//!   2. **DAC at the row drivers.** ~1 fJ/bit per row. Amortized across the
//!      reduction dimension `k`.
//!   3. **ADC at the column output.** ~1 pJ/sample at 8-bit precision
//!      (Murmann survey). One sample per output element, so per-MAC ADC
//!      cost scales as `1/k` after amortization across a column.
//!   4. **Write energy.** Programming a weight is ~10 pJ per cell, but only
//!      paid once at deployment, not per inference — ignored in the per-MAC
//!      model.
//!
//! Sandia's CrossSim (Sandia Labs, BSD-3) is the reference simulator. The
//! constants below match the CrossSim default device library to within an
//! order of magnitude.
//!
//! Net per-MAC at the system level for shipping crossbar accelerators
//! (IBM HERMES, Stanford NeuRRAM, Mythic M1076) lands in the
//! **1-30 fJ/MAC range** at moderate (~128-256 row) crossbars. Cost model
//! reflects that band.
//!
//! Citations:
//!   - Wan et al. 2022, *Nature* 608: 504 — NeuRRAM 48-tile compute-in-memory.
//!   - Khaddam-Aljameh et al. 2022, *JSSC* 57: 1027 — IBM HERMES.
//!   - Le Gallo et al. 2023, *Nature Electronics* 6: 680 — PCM crossbar inference.
//!   - Joshi et al. 2020, *Nature Comms* 11: 2473 — PCM analog AI.
//!   - Sandia CrossSim v3.0 documentation (2024).
//!   - Murmann ADC survey 2020-2024.

use serde::{Deserialize, Serialize};
use tei_ir::OpProfile;
use tei_stack::Primitive;
use tei_substrate_traits::{Cost, Substrate};

/// Tunable engineering parameters for the in-memory substrate.
///
/// Defaults match published shipping accelerators (NeuRRAM / HERMES / Mythic)
/// at a ~256×256 crossbar with 1 fJ/MAC device + 1 pJ/sample ADC. Crossbar
/// size is the biggest knob: larger arrays amortize the per-column ADC fixed
/// cost across more multiplies.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InMemoryParams {
    /// Rows × columns. A k × n matmul on a `size × size` crossbar pays
    /// `⌈k/size⌉ · ⌈n/size⌉` array activations, each with one ADC sample
    /// per output column.
    pub crossbar_size: u32,
    /// Per-MAC device-level energy. 1 fJ for production RRAM, ~0.1 fJ for
    /// emerging devices.
    pub device_j_per_mac: f64,
    /// DAC energy per input row bit.
    pub dac_j_per_bit: f64,
    /// ADC energy per output sample. Scales roughly 2× per added bit.
    pub adc_j_per_sample: f64,
}

impl Default for InMemoryParams {
    fn default() -> Self {
        Self {
            crossbar_size:    256,
            device_j_per_mac: 1.0e-15,
            dac_j_per_bit:    1.0e-15,
            adc_j_per_sample: 1.0e-12,
        }
    }
}

/// Typical sustained MAC rate per crossbar at the system level, ops/s.
/// Crossbars themselves are essentially instantaneous; the ADC pipeline
/// throttles end-to-end throughput.
const IN_MEMORY_OPS_PER_SEC: f64 = 1.0e10;

/// In-memory crossbars have device variability (G ~5-10% device-to-device,
/// thermal drift, write asymmetry). Published-system accuracy at 8-bit
/// equivalent precision: ~0.5-1.5% loss vs digital baseline on ImageNet.
const IN_MEMORY_ACCURACY_LOSS: f64 = 0.01;

/// Native primitives — anything that maps to a matrix-vector multiply
/// or a small dense matmul.
fn primitive_is_crossbar_native(p: &Primitive) -> bool {
    matches!(
        p.id,
        18 |  // Dense MatMul — decomposes into a sequence of MVMs    (LA · L2)
        19 |  // SpMM / SpMV — canonical crossbar fit                  (LA · L2)
        20 |  // Attention — Q·Kᵀ and ·V are MVMs                       (LA · L2)
        24 |  // Convolution — im2col → MVM                             (TR · L2)
        48    // Tensor network contraction — matmul sequences           (TEN · L2)
    )
}

/// In-memory crossbar substrate.
#[derive(Default)]
pub struct InMemory {
    pub params: InMemoryParams,
}

impl InMemory {
    pub fn with_params(params: InMemoryParams) -> Self { Self { params } }

    fn matmul_energy(&self, profile: &OpProfile) -> Cost {
        let p = &self.params;
        let macs = profile.matmul_macs().unwrap_or(1) as f64;
        let result_elems = (profile.shape.elements() as u64) * (profile.batch as u64);
        let result_elems_f = result_elems as f64;
        let operand_bits = profile.dtype.bits() as f64;
        let k = profile.reduce_dim.unwrap_or(64) as f64;
        let size = p.crossbar_size.max(1) as f64;

        // Device-level MAC current × pulse duration.
        let crossbar_j = macs * p.device_j_per_mac;

        // DAC: one event per row per cycle. Total row events ≈ macs/k.
        let dac_j = (macs / k) * operand_bits * p.dac_j_per_bit;

        // ADC: one sample per output element per crossbar tile. A k × n
        // matmul on a `size × size` array uses ⌈k/size⌉ × ⌈n/size⌉ tile
        // activations; each tile generates one ADC sample per output column.
        // So total ADC samples = result_elems × ⌈k/size⌉. Larger crossbars
        // amortize the per-column ADC across more multiplies.
        let k_tiles = (k / size).ceil().max(1.0);
        let adc_j = result_elems_f * k_tiles * p.adc_j_per_sample;

        let joules_per_op = crossbar_j + dac_j + adc_j;
        let seconds_per_op = result_elems_f * k_tiles / IN_MEMORY_OPS_PER_SEC;

        Cost {
            joules_per_op,
            seconds_per_op,
            accuracy_loss: IN_MEMORY_ACCURACY_LOSS,
        }
    }

    fn default_cost(&self) -> Cost {
        Cost {
            joules_per_op: 10e-15,
            seconds_per_op: 1.0 / IN_MEMORY_OPS_PER_SEC,
            accuracy_loss: IN_MEMORY_ACCURACY_LOSS,
        }
    }
}

impl Substrate for InMemory {
    fn name(&self) -> &str { "in-memory" }
    fn display_name(&self) -> &str { "In-memory (RRAM crossbar)" }

    fn supports(&self, primitive: &Primitive) -> bool {
        primitive_is_crossbar_native(primitive)
    }

    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        if matches!(primitive.id, 18 | 19 | 20 | 24 | 48)
            && profile.matmul_macs().is_some()
        {
            return self.matmul_energy(profile);
        }
        self.default_cost()
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "Wan et al. 2022, Nature 608: 504 — NeuRRAM 48-tile compute-in-memory.",
            "Khaddam-Aljameh et al. 2022, JSSC 57: 1027 — IBM HERMES.",
            "Le Gallo et al. 2023, Nature Electronics 6: 680 — PCM crossbar inference.",
            "Joshi et al. 2020, Nature Comms 11: 2473 — PCM analog AI.",
            "Sandia CrossSim v3.0 (2024) — reference simulator.",
            "Murmann ADC survey 2020-2024.",
        ]
    }
}
