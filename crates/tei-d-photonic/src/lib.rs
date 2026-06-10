//! Photonic substrate — MZI mesh / silicon-photonic MAC physics.
//!
//! The photonic substrate executes linear-algebra primitives (matmul, conv,
//! attention's QKᵀ + V products, FFT) by modulating amplitudes on coherent
//! light through a mesh of Mach-Zehnder interferometers and integrating on
//! photodetectors. The cost model below sums the four energy contributions
//! that dominate at the system level:
//!
//!   1. **Laser source.** Wall-plug to optical conversion. ~10% efficiency
//!      for telecom DFB / ECL sources.
//!   2. **Modulators.** ~1 fJ/bit (TFLN / thin-film lithium niobate) to
//!      ~10 fJ/bit (silicon ring resonator). Each operand bit gets modulated
//!      once per dot-product.
//!   3. **Photodetectors.** Responsivity ~0.8 A/W; the photocurrent itself
//!      is essentially free of electrical-energy overhead in the analog
//!      domain. Folded into the laser term via efficiency budget.
//!   4. **Data converters (ADC at the column output).** The system-level
//!      energy killer: ~1-10 pJ per sample for 8-bit ADCs at GHz rates
//!      (Murmann ADC survey). This term dominates per-MAC energy for any
//!      modestly-sized matrix.
//!
//! Net per-MAC energy on shipping silicon-photonic tensor cores
//! (Lightmatter Envise, Lightelligence PACE, MIT Englund group, Stanford
//! Solgaard group) lands in the **10-100 fJ/MAC range** at the system level.
//! This crate uses 30 fJ/MAC as the L₂-equivalent point; the actual cost
//! scales with `k` (the reduction dimension) because larger reductions
//! amortize the per-column ADC cost across more multiplies.
//!
//! Citations:
//!   - Shen et al. 2017, *Nature Photonics* 11: 441 (MZI-mesh ONN).
//!   - Wang et al. 2018, *Nature* 562: 101 (TFLN modulators).
//!   - Tait et al. 2017, *Sci. Reports* 7: 7430 (broadcast-and-weight protocol).
//!   - Hamerly et al. 2019, *Phys. Rev. X* 9: 021032 (large-scale ONN scaling).
//!   - Bandyopadhyay et al. 2022, *Optica* 9: 1364 (energy-resolved photonic NN).
//!   - Murmann ADC survey (2020-2024, ISSCC) — ADC FoM trends.

use serde::{Deserialize, Serialize};
use tei_ir::OpProfile;
use tei_stack::Primitive;
use tei_substrate_traits::{Cost, Substrate};

/// Tunable engineering parameters for the photonic substrate.
///
/// Defaults match the literature anchor — TFLN modulators at 1 fJ/bit + 8-bit
/// ADC at 1 pJ/sample, lands ~30 fJ/MAC at the system level. Users can sweep
/// these to ask "what if my modulator were 10× more efficient?" or "what
/// would 8× wavelength multiplexing buy me?"
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhotonicParams {
    /// Number of WDM (wavelength-division-multiplexed) channels carrying
    /// parallel multiplies. Doubling N halves wall-clock per MAC.
    pub wdm_channels: u32,
    /// Per-bit modulator energy. TFLN ≈ 1 fJ/bit, silicon ring ≈ 10 fJ/bit.
    pub modulator_j_per_bit: f64,
    /// Per-sample ADC energy at the column output. 1 pJ for 8-bit; grows
    /// roughly 2× per added bit at the same FoM (Murmann).
    pub adc_j_per_sample: f64,
    /// Wall-plug → coherent-optical conversion efficiency. 10% telecom DFB,
    /// up to 30-50% with VCSEL arrays.
    pub laser_efficiency: f64,
    /// Per-MAC optical-power floor at the laser output. Bounds the energy
    /// delivered to the mesh per multiply before electrical conversion.
    pub optical_j_per_mac: f64,
}

impl Default for PhotonicParams {
    fn default() -> Self {
        Self {
            wdm_channels: 1,
            modulator_j_per_bit: 1.0e-15,
            adc_j_per_sample: 1.0e-12,
            laser_efficiency: 0.10,
            optical_j_per_mac: 0.5e-15,
        }
    }
}

/// Wall-clock per MAC at the system level. Photonic meshes naturally run at
/// the modulator rate (10s of GHz); at the system level the ADC pipeline
/// throttles to ~5-10 GS/s per channel.
const PHOTONIC_OPS_PER_SEC: f64 = 5.0e9;

/// Photonic substrates introduce a small SNR-bounded fidelity loss because
/// analog modulation + photodetection adds shot + thermal noise on top of
/// digital quantization. The published-photonic-NN literature reports
/// inference accuracy losses well under 1% at 8-bit equivalent precision.
const PHOTONIC_ACCURACY_LOSS: f64 = 0.005;

/// Primitives this substrate can execute natively. v0 covers the matmul-class
/// + linear transforms — everything that maps to amplitude × amplitude
/// integration on a coherent mesh.
fn primitive_is_photonic_native(p: &Primitive) -> bool {
    matches!(
        p.id,
        18 |  // Dense MatMul                         (LA · L2)
        19 |  // SpMM / SpMV                          (LA · L2)
        20 |  // Attention                            (LA · L2)
        24 |  // Convolution                          (TR · L2)
        48 |  // Tensor network contraction           (TEN · L2)
        53 |  // Photonic MZI transform               (ANALOG)
        76 |  // Matrix decomposition (QR/LU/SVD …)   (LA · L2)
        77 // Eigendecomposition                   (LA · L2)
    )
}

/// Photonic substrate — MZI mesh and ring-resonator weight banks.
#[derive(Default)]
pub struct Photonic {
    pub params: PhotonicParams,
}

impl Photonic {
    pub fn with_params(params: PhotonicParams) -> Self {
        Self { params }
    }

    /// First-principles cost for a matmul-class primitive.
    ///
    /// Given `m × k × n` dense matmul with batch `B`:
    ///   - total MACs:       `B · m · k · n`
    ///   - modulator events: 2 · operand-bits (one per input bit per MAC, lower
    ///     bound — real meshes amortize broadcast across columns)
    ///   - ADC samples:      `B · m · n` (one per result element)
    ///
    /// WDM channels divide the optical + ADC throughput but the per-energy
    /// totals are largely the same (more lasers + more ADCs at lower duty cycle).
    fn matmul_energy(&self, profile: &OpProfile) -> Cost {
        let p = &self.params;
        let macs = profile.matmul_macs().unwrap_or(1);
        let result_elems = (profile.shape.elements() as u64) * (profile.batch as u64);
        let operand_bits = profile.dtype.bits() as u64;

        let optical_j = (macs as f64) * p.optical_j_per_mac / p.laser_efficiency.max(0.01);
        let modulator_j = 2.0 * (macs as f64) * (operand_bits as f64) * p.modulator_j_per_bit;
        let adc_j = (result_elems as f64) * p.adc_j_per_sample;

        let joules_per_op = optical_j + modulator_j + adc_j;
        // WDM channels parallelize the ADC pipeline, shrinking wall-clock.
        let wdm = (p.wdm_channels.max(1)) as f64;
        let seconds_per_op = result_elems as f64 / (PHOTONIC_OPS_PER_SEC * wdm);

        Cost {
            joules_per_op,
            seconds_per_op,
            accuracy_loss: PHOTONIC_ACCURACY_LOSS,
        }
    }

    fn default_cost(&self) -> Cost {
        Cost {
            joules_per_op: 30e-15,
            seconds_per_op: 1.0 / PHOTONIC_OPS_PER_SEC,
            accuracy_loss: PHOTONIC_ACCURACY_LOSS,
        }
    }
}

impl Substrate for Photonic {
    fn name(&self) -> &str {
        "photonic"
    }
    fn display_name(&self) -> &str {
        "Photonic (MZI mesh)"
    }

    fn supports(&self, primitive: &Primitive) -> bool {
        primitive_is_photonic_native(primitive)
    }

    fn cost(&self, primitive: &Primitive, profile: &OpProfile) -> Cost {
        // Matmul-class primitives — full physics model.
        if matches!(primitive.id, 18 | 19 | 20 | 24 | 48 | 53 | 76 | 77)
            && profile.matmul_macs().is_some()
        {
            return self.matmul_energy(profile);
        }
        self.default_cost()
    }

    fn citations(&self) -> &'static [&'static str] {
        &[
            "Shen et al. 2017, Nature Photonics 11: 441 — MZI-mesh ONN.",
            "Wang et al. 2018, Nature 562: 101 — TFLN modulators (~1 fJ/bit).",
            "Tait et al. 2017, Sci. Reports 7: 7430 — broadcast-and-weight.",
            "Hamerly et al. 2019, Phys. Rev. X 9: 021032 — large-scale ONN scaling.",
            "Bandyopadhyay et al. 2022, Optica 9: 1364 — energy-resolved photonic NN.",
            "Murmann ADC survey 2020-2024 — ADC energy floor.",
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

    fn dense_matmul() -> tei_stack::Primitive {
        serde_json::from_str(
            r#"{
            "id": 18, "name": "Dense MatMul", "family": "LA",
            "B": "kernel", "C": "L2", "D": "data-parallel",
            "existing": "HW", "silicon_target": null, "wave": null
        }"#,
        )
        .unwrap()
    }

    /// Anchor: with default params (TFLN 1 fJ/bit · 1 pJ/sample ADC · 10%
    /// laser · 0.5 fJ/MAC optical) the 512×768×2048 f16 matmul lands at
    /// ~38 fJ/MAC system-level — inside the published Lightmatter /
    /// Lightelligence 10-100 fJ/MAC band.
    #[test]
    fn photonic_mac_in_published_band() {
        let s = Photonic::default();
        let macs = (512u64 * 768 * 2048) as f64;
        let cost = s.cost(&dense_matmul(), &matmul_profile());
        let per_mac = cost.joules_per_op / macs;
        assert!(
            per_mac > 10e-15 && per_mac < 100e-15,
            "photonic per-MAC {per_mac:.3e} outside 10-100 fJ band"
        );
    }

    /// Exact decomposition: optical + modulator + ADC terms.
    #[test]
    fn photonic_term_decomposition() {
        let s = Photonic::default();
        let macs = (512u64 * 768 * 2048) as f64;
        let result_elems = (512u64 * 2048) as f64;
        let expected = macs * 0.5e-15 / 0.10           // optical / laser η
            + 2.0 * macs * 16.0 * 1.0e-15              // modulator (f16 = 16 bits)
            + result_elems * 1.0e-12; // ADC per output
        let cost = s.cost(&dense_matmul(), &matmul_profile());
        assert!((cost.joules_per_op - expected).abs() / expected < 1e-9);
    }

    /// WDM channels shrink wall-clock 1/N but leave energy unchanged.
    #[test]
    fn wdm_parallelizes_time_not_energy() {
        let base = Photonic::default();
        let wide = Photonic::with_params(PhotonicParams {
            wdm_channels: 16,
            ..Default::default()
        });
        let p = dense_matmul();
        let prof = matmul_profile();
        let c1 = base.cost(&p, &prof);
        let c16 = wide.cost(&p, &prof);
        assert!((c1.joules_per_op - c16.joules_per_op).abs() / c1.joules_per_op < 1e-12);
        assert!((c1.seconds_per_op / c16.seconds_per_op - 16.0).abs() < 1e-9);
    }
}
