//! FPGA→ASIC correction — turning a *measured* energy on a rented FPGA
//! stand-in into a *target-substrate* estimate with explicit bounds and cited
//! provenance.
//!
//! This is the honest core of the AWS-FPGA tier. An FPGA (AWS F2: AMD/Xilinx
//! Virtex UltraScale+) is the one rentable fabric where we control the
//! *datapath* and read *real* card power — so it yields `Measured × StandIn`.
//! But an FPGA is not the target ASIC: hardened silicon is markedly more
//! energy-efficient for the same logic. The gap is *bounded and characterized*
//! in the literature, so:
//!
//! ```text
//! target_ASIC_joules  ≈  measured_FPGA_joules × correction.factor   (± bounds)
//! ```
//!
//! That product — measured energy times a published de-rating — is the ownable
//! artifact: a target estimate that says exactly how it was derived and how
//! uncertain it is, never a bare number pretending to be silicon-measured.
//!
//! ## Honesty boundary
//! An FPGA is an honest stand-in only for **digital-datapath** substrates
//! (stochastic/p-bit, reversible, baseline, and the digital control of
//! in-memory / neuromorphic cores). For the **analog-physics** columns —
//! photonic interference, analog RRAM crossbar conductance, analog
//! neuromorphic — the FPGA measures a *digital emulation* whose energy bears
//! no fixed relation to the analog target, so [`fpga_to_asic_default`] returns
//! `None`. Those stay `Modeled` until a real device is on the bench.

use crate::Acquisition;
use serde::Serialize;

/// How a correction factor was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum CorrectionMethod {
    /// A published FPGA→ASIC gap from the literature. Starting point.
    LiteratureDefault,
    /// Calibrated against measured silicon (an analogous hardened block, or our
    /// own tapeout). Tightens the bound; the goal state.
    MeasuredCalibrated,
}

/// An FPGA→ASIC de-rating for one substrate's datapath.
///
/// `target_j ≈ fpga_j × factor`; an ASIC is cheaper than the FPGA running the
/// same logic, so `factor < 1`. `factor_lo`/`factor_hi` give the multiplicative
/// bounds (→ a target-energy interval).
#[derive(Debug, Clone, Serialize)]
pub struct Correction {
    /// Point de-rating: `target_j = fpga_j × factor`.
    pub factor: f64,
    /// Lower bound (smallest target estimate).
    pub factor_lo: f64,
    /// Upper bound (largest target estimate).
    pub factor_hi: f64,
    /// What transformation this models.
    pub basis: String,
    /// Provenance of the numbers (citation, or how calibrated).
    pub source: String,
    /// Literature default vs measured-calibrated.
    pub method: CorrectionMethod,
}

impl Correction {
    /// Apply this correction to a measured FPGA energy, yielding a target
    /// estimate with bounds. `fpga_joules` must be a real `Measured × StandIn`
    /// reading from the FPGA running the target datapath.
    pub fn apply(&self, fpga_joules: f64, target: &str, standin: &str) -> TargetEstimate {
        TargetEstimate {
            target: target.to_string(),
            target_joules: fpga_joules * self.factor,
            target_joules_lo: fpga_joules * self.factor_lo,
            target_joules_hi: fpga_joules * self.factor_hi,
            acquisition: Acquisition::Measured,
            derived: true,
            from_standin: standin.to_string(),
            from_joules: fpga_joules,
            correction: self.clone(),
        }
    }
}

/// A target-substrate energy estimate derived from a measured stand-in reading
/// via a [`Correction`]. The energy *input* was `Measured`; the target number
/// is *derived* (`derived = true`) and carries bounds + the correction that
/// produced it. Honest phrasing: "Measured × Target, via a published
/// FPGA→ASIC correction (± bounds)".
#[derive(Debug, Clone, Serialize)]
pub struct TargetEstimate {
    /// Target substrate name.
    pub target: String,
    /// Point estimate of target energy (joules).
    pub target_joules: f64,
    /// Lower / upper bounds (joules).
    pub target_joules_lo: f64,
    pub target_joules_hi: f64,
    /// Acquisition of the *input* energy (Measured for an FPGA reading).
    pub acquisition: Acquisition,
    /// True ⇒ the target number is derived via a correction, not measured
    /// directly on the target.
    pub derived: bool,
    /// The stand-in the input was measured on.
    pub from_standin: String,
    /// The measured stand-in energy the estimate was derived from (joules).
    pub from_joules: f64,
    /// The correction applied.
    pub correction: Correction,
}

/// Literature-default FPGA→ASIC correction for a substrate, or `None` when an
/// FPGA is not an honest stand-in (analog-physics columns).
///
/// The factors are the FPGA-vs-ASIC **dynamic-power gap** from Kuon & Rose
/// (IEEE TCAD 2007, "Measuring the Gap Between FPGAs and ASICs"): ~7.1× with
/// hard blocks (DSP/RAM), up to ~14× for logic-only soft logic, ~4.6× at the
/// favorable end. It is node-agnostic — it models the FPGA-fabric overhead,
/// not a process-node shrink (that would be a separate, additional factor).
/// Deliberately conservative; meant to be replaced by `MeasuredCalibrated`
/// once we have silicon to calibrate against.
pub fn fpga_to_asic_default(substrate: &str) -> Option<Correction> {
    let digital = Correction {
        factor: 1.0 / 7.1,
        factor_lo: 1.0 / 14.0, // most efficient ASIC ⇒ smallest target energy
        factor_hi: 1.0 / 4.6,  // least efficient ⇒ largest target energy
        basis: "FPGA-fabric → hardened-ASIC dynamic-power gap, digital datapath, \
                node-agnostic (not a process shrink)"
            .to_string(),
        source: "Kuon & Rose 2007, IEEE TCAD — dynamic-power gap 7.1× (with hard \
                 blocks) to 14× (logic-only), ~4.6× favorable"
            .to_string(),
        method: CorrectionMethod::LiteratureDefault,
    };
    match substrate {
        // Digital datapaths — FPGA is an honest stand-in.
        "stochastic" | "reversible" | "baseline" => Some(digital),
        // Analog-physics substrates — FPGA measures a digital emulation whose
        // energy has no fixed relation to the analog target. No stand-in.
        "in-memory" | "neuromorphic" | "photonic" => None,
        _ => None,
    }
}

/// Why an FPGA correction is unavailable for a substrate (for honest error
/// messages). `None` substrate ⇒ analog/unsupported.
pub fn unavailable_reason(substrate: &str) -> &'static str {
    match substrate {
        "in-memory" => "analog RRAM crossbar conductance — FPGA would measure a \
                        digital emulation, not the analog target",
        "neuromorphic" => "analog/mixed-signal neuromorphic cores — FPGA emulation \
                           energy does not de-rate to the analog target",
        "photonic" => "optical interference — no digital FPGA stand-in exists for \
                       the analog physics",
        _ => "no FPGA→ASIC correction defined for this substrate",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn digital_substrates_have_a_correction_bounded_below_one() {
        for s in ["stochastic", "reversible", "baseline"] {
            let c = fpga_to_asic_default(s).expect("digital substrate has a correction");
            assert!(c.factor < 1.0, "ASIC must be cheaper than FPGA");
            assert!(c.factor_lo < c.factor && c.factor < c.factor_hi, "bounds bracket point");
            assert_eq!(c.method, CorrectionMethod::LiteratureDefault);
        }
    }

    #[test]
    fn analog_substrates_have_no_fpga_standin() {
        for s in ["photonic", "in-memory", "neuromorphic"] {
            assert!(fpga_to_asic_default(s).is_none(), "{s} must not get an FPGA correction");
        }
    }

    #[test]
    fn apply_produces_ordered_bounds_and_keeps_provenance() {
        let c = fpga_to_asic_default("stochastic").unwrap();
        let fpga_j = 7.0e-6; // pretend measured on the F2 card
        let est = c.apply(fpga_j, "stochastic", "AWS F2 (Virtex UltraScale+)");
        assert!(est.target_joules_lo < est.target_joules);
        assert!(est.target_joules < est.target_joules_hi);
        assert!(est.target_joules < fpga_j, "target must be below the FPGA measurement");
        assert!(est.derived);
        assert_eq!(est.acquisition, Acquisition::Measured);
        assert_eq!(est.from_joules, fpga_j);
        assert_eq!(est.from_standin, "AWS F2 (Virtex UltraScale+)");
    }
}
