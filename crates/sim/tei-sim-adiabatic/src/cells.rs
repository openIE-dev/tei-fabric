//! Parametric adiabatic cell templates — functions that emit
//! `tei_sim_circuit::Netlist`s, plus the serde-able `CellSpec` the sweep
//! harness and executor consume.
//!
//! # The switch-as-resistor simplification
//!
//! Every template models the transmission gate / pass device of a real
//! adiabatic cell (S2LAL / Q2LAL / ECRL / 2LAL families) as a **fixed linear
//! on-resistance**. This is the canonical first-order model of the
//! adiabatic-logic literature — the (RC/T)·CV² ramp-dissipation law (Athas
//! et al. 1994; DeBenedictis arXiv:2009.00448) is derived for exactly this
//! R–C abstraction, with R the on-resistance of the conducting switch and C
//! the load it (dis)charges. What the simplification leaves out, and the
//! MOSFET-level M2 ladder of `tei-sim-circuit` will add back: threshold
//! drops, the non-adiabatic ½CV_th² residual of real pass transistors,
//! gate-overlap charge injection, and subthreshold leakage during hold.
//! Those effects floor the energy-vs-ramp-time curve in silicon; here the
//! curve follows the ideal −1 log-log slope all the way down, which is the
//! correct *upper bound on recoverable energy* the cost dialect calibrates
//! against.
//!
//! # Templates
//!
//! - [`ramp_charge_cell`] — single-stage R–C charged by a linear ramp; the
//!   validation anchor with an exact closed form (see
//!   [`crate::sweep::ramp_charge_exact`]).
//! - [`charge_recovery_cell`] — 2-phase split-level charge-recovery stage:
//!   a trapezoid power-clock (rise T · hold · fall T) charges the load and
//!   then **recovers the charge back through the switch** on the falling
//!   edge. Per-cycle dissipation ≈ 2·(RC/T)·CV² for slow clocks versus
//!   2·½CV² = CV² for abrupt switching — the energy-recovery demonstration.
//! - [`shift_register_chain`] — N buffered stages driven by an n-phase
//!   trapezoid power-clock, each stage's clock shifted by one phase step:
//!   the DeBenedictis aa shift-register testbench shape. Stages are
//!   electrically independent R–C branches off their own clock phase (the
//!   data path of the real shift register is folded into the per-stage load
//!   C, consistent with the switch-as-resistor abstraction), so total
//!   dissipation must be N × the single-stage value — the linearity check.

use serde::{Deserialize, Serialize};
use tei_sim_circuit::{CircuitError, Netlist, Waveform};

/// Settle tail appended after the last clock edge, in units of RC.
/// e^{−12} ≈ 6×10⁻⁶ — residual stored energy left untracked is negligible
/// against every tolerance in the validation suite.
pub const SETTLE_RC: f64 = 12.0;

/// Canonical single-stage R–C ramp charge: a linear ramp 0 → `v` over
/// `t_ramp` (then held) drives the series on-resistance `r` into the load
/// capacitor `c`. Elements: `vclk` (node 1 → ground), `ron` (1 → 2),
/// `cload` (2 → ground).
pub fn ramp_charge_cell(r: f64, c: f64, v: f64, t_ramp: f64) -> Netlist {
    let mut net = Netlist::new();
    net.vsource("vclk", 1, 0, Waveform::Ramp { v, t_ramp })
        .resistor("ron", 1, 2, r)
        .capacitor("cload", 2, 0, c, 0.0);
    net
}

/// 2-phase split-level charge-recovery stage: same R–C as
/// [`ramp_charge_cell`] but driven by a full trapezoid power-clock pulse
/// (rise `t_ramp` · hold `t_hold` · fall `t_ramp`). On the falling edge the
/// load discharges **back through the resistor into the clock** — the clock
/// recovers ½CV² − E_fall of the stored energy instead of it being dumped
/// to ground, which is the entire adiabatic charge-recovery idea. Per-cycle
/// dissipation is the sum of the rise and fall losses.
pub fn charge_recovery_cell(r: f64, c: f64, v: f64, t_ramp: f64, t_hold: f64) -> Netlist {
    let mut net = Netlist::new();
    net.vsource(
        "vclk",
        1,
        0,
        Waveform::Trapezoid {
            v,
            t_delay: 0.0,
            t_rise: t_ramp,
            t_hold,
            t_fall: t_ramp,
        },
    )
    .resistor("ron", 1, 2, r)
    .capacitor("cload", 2, 0, c, 0.0);
    net
}

/// Shift-register-style chain of `n_stages` buffered stages driven by an
/// `n_phases`-phase trapezoid power-clock. Stage k's clock is delayed by
/// `(k mod n_phases)·(t_ramp + t_hold)` — successive stages switch on
/// successive phases, the aa testbench shape. Stage k occupies nodes
/// `2k+1` (its clock rail) and `2k+2` (its load), with elements
/// `clk{k}`, `ron{k}`, `cload{k}`.
pub fn shift_register_chain(
    n_stages: usize,
    n_phases: usize,
    r: f64,
    c: f64,
    v: f64,
    t_ramp: f64,
    t_hold: f64,
) -> Netlist {
    let phase_step = t_ramp + t_hold;
    let mut net = Netlist::new();
    for k in 0..n_stages {
        let clk_node = 2 * k + 1;
        let load_node = 2 * k + 2;
        let phase = (k % n_phases.max(1)) as f64;
        net.vsource(
            &format!("clk{k}"),
            clk_node,
            0,
            Waveform::Trapezoid {
                v,
                t_delay: phase * phase_step,
                t_rise: t_ramp,
                t_hold,
                t_fall: t_ramp,
            },
        )
        .resistor(&format!("ron{k}"), clk_node, load_node, r)
        .capacitor(&format!("cload{k}"), load_node, 0, c, 0.0);
    }
    net
}

fn default_hold_rc() -> f64 {
    8.0
}
fn default_n_phases() -> usize {
    4
}

/// Serde-able cell selector for the sweep harness (`t_ramp` is *not* part
/// of the spec — the sweep supplies it as `t_over_rc · RC` per point).
///
/// `hold_rc` is the trapezoid hold time in units of RC (default 8 — long
/// enough that the load fully settles to the rail before recovery begins,
/// e^{−8} ≈ 3×10⁻⁴ residual).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CellSpec {
    /// [`ramp_charge_cell`] — abrupt limit ½CV².
    RampCharge { r_ohm: f64, c_f: f64, v: f64 },
    /// [`charge_recovery_cell`] — abrupt limit 2·½CV² = CV² per cycle.
    ChargeRecovery {
        r_ohm: f64,
        c_f: f64,
        v: f64,
        #[serde(default = "default_hold_rc")]
        hold_rc: f64,
    },
    /// [`shift_register_chain`] — abrupt limit N·CV² per clock cycle.
    ShiftRegister {
        r_ohm: f64,
        c_f: f64,
        v: f64,
        n_stages: usize,
        #[serde(default = "default_n_phases")]
        n_phases: usize,
        #[serde(default = "default_hold_rc")]
        hold_rc: f64,
    },
}

impl CellSpec {
    pub fn r_ohm(&self) -> f64 {
        match self {
            CellSpec::RampCharge { r_ohm, .. }
            | CellSpec::ChargeRecovery { r_ohm, .. }
            | CellSpec::ShiftRegister { r_ohm, .. } => *r_ohm,
        }
    }

    pub fn c_f(&self) -> f64 {
        match self {
            CellSpec::RampCharge { c_f, .. }
            | CellSpec::ChargeRecovery { c_f, .. }
            | CellSpec::ShiftRegister { c_f, .. } => *c_f,
        }
    }

    pub fn v(&self) -> f64 {
        match self {
            CellSpec::RampCharge { v, .. }
            | CellSpec::ChargeRecovery { v, .. }
            | CellSpec::ShiftRegister { v, .. } => *v,
        }
    }

    /// The switch time constant RC [s].
    pub fn rc(&self) -> f64 {
        self.r_ohm() * self.c_f()
    }

    /// Abrupt-switching dissipation of the whole cell [J] — the curve's
    /// normalizer: ½CV² for the single charge, CV² for the full
    /// charge/discharge recovery cycle, N·CV² for the N-stage chain cycle.
    pub fn abrupt_limit_j(&self) -> f64 {
        let half_cv2 = 0.5 * self.c_f() * self.v() * self.v();
        match self {
            CellSpec::RampCharge { .. } => half_cv2,
            CellSpec::ChargeRecovery { .. } => 2.0 * half_cv2,
            CellSpec::ShiftRegister { n_stages, .. } => 2.0 * half_cv2 * *n_stages as f64,
        }
    }

    /// Reject non-physical parameters before they reach the solver.
    pub fn validate(&self) -> Result<(), CircuitError> {
        let ok = |x: f64| x.is_finite() && x > 0.0;
        if !(ok(self.r_ohm()) && ok(self.c_f()) && ok(self.v())) {
            return Err(CircuitError::Invalid(format!(
                "cell parameters must be finite and positive: r={} c={} v={}",
                self.r_ohm(),
                self.c_f(),
                self.v()
            )));
        }
        match self {
            CellSpec::RampCharge { .. } => {}
            CellSpec::ChargeRecovery { hold_rc, .. } => {
                if !hold_rc.is_finite() || *hold_rc < 0.0 {
                    return Err(CircuitError::Invalid(format!(
                        "hold_rc must be finite and ≥ 0, got {hold_rc}"
                    )));
                }
            }
            CellSpec::ShiftRegister {
                n_stages,
                n_phases,
                hold_rc,
                ..
            } => {
                if *n_stages == 0 || *n_phases == 0 {
                    return Err(CircuitError::Invalid(
                        "n_stages and n_phases must be ≥ 1".into(),
                    ));
                }
                if !hold_rc.is_finite() || *hold_rc < 0.0 {
                    return Err(CircuitError::Invalid(format!(
                        "hold_rc must be finite and ≥ 0, got {hold_rc}"
                    )));
                }
            }
        }
        Ok(())
    }

    /// Build the netlist for ramp duration `t_ramp` [s] plus the stop time
    /// covering every clock edge and the [`SETTLE_RC`] tail.
    pub fn build(&self, t_ramp: f64) -> (Netlist, f64) {
        let rc = self.rc();
        match self {
            CellSpec::RampCharge { r_ohm, c_f, v } => (
                ramp_charge_cell(*r_ohm, *c_f, *v, t_ramp),
                t_ramp + SETTLE_RC * rc,
            ),
            CellSpec::ChargeRecovery {
                r_ohm,
                c_f,
                v,
                hold_rc,
            } => {
                let t_hold = hold_rc * rc;
                (
                    charge_recovery_cell(*r_ohm, *c_f, *v, t_ramp, t_hold),
                    2.0 * t_ramp + t_hold + SETTLE_RC * rc,
                )
            }
            CellSpec::ShiftRegister {
                r_ohm,
                c_f,
                v,
                n_stages,
                n_phases,
                hold_rc,
            } => {
                let t_hold = hold_rc * rc;
                let phase_step = t_ramp + t_hold;
                let max_delay = ((*n_stages).min(*n_phases) - 1) as f64 * phase_step;
                (
                    shift_register_chain(*n_stages, *n_phases, *r_ohm, *c_f, *v, t_ramp, t_hold),
                    max_delay + 2.0 * t_ramp + t_hold + SETTLE_RC * rc,
                )
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tei_sim_circuit::Element;

    /// Ramp cell: 3 elements, the documented names, nodes contiguous 1..=2.
    #[test]
    fn ramp_cell_shape() {
        let net = ramp_charge_cell(1e3, 1e-9, 1.0, 1e-6);
        assert_eq!(net.elements.len(), 3);
        let names: Vec<&str> = net.elements.iter().map(|e| e.name()).collect();
        assert_eq!(names, ["vclk", "ron", "cload"]);
        let max_node = net
            .elements
            .iter()
            .map(|e| e.nodes())
            .flat_map(|(p, n)| [p, n])
            .max()
            .unwrap();
        assert_eq!(max_node, 2);
    }

    /// Chain: 3N elements, contiguous nodes 1..=2N, phases wrap mod n_phases.
    #[test]
    fn chain_shape_and_phases() {
        let (n_stages, n_phases) = (6, 4);
        let (t_ramp, t_hold) = (2e-6, 1e-6);
        let net = shift_register_chain(n_stages, n_phases, 1e3, 1e-9, 1.0, t_ramp, t_hold);
        assert_eq!(net.elements.len(), 3 * n_stages);
        let mut nodes: Vec<usize> = net
            .elements
            .iter()
            .map(|e| e.nodes())
            .flat_map(|(p, n)| [p, n])
            .filter(|&n| n != 0)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes, (1..=2 * n_stages).collect::<Vec<_>>());
        // Stage 5 is phase 5 mod 4 = 1 → delay = 1·(t_ramp + t_hold).
        let Element::VoltageSource {
            wave: Waveform::Trapezoid { t_delay, .. },
            ..
        } = &net.elements[3 * 5]
        else {
            panic!("expected stage-5 clock");
        };
        assert!((t_delay - (t_ramp + t_hold)).abs() < 1e-18);
    }

    /// CellSpec serde round-trip with documented tags + defaults.
    #[test]
    fn cell_spec_serde_defaults() {
        let spec: CellSpec = serde_json::from_str(
            r#"{"kind":"shift_register","r_ohm":1000.0,"c_f":1e-9,"v":1.0,"n_stages":8}"#,
        )
        .unwrap();
        let CellSpec::ShiftRegister {
            n_phases, hold_rc, ..
        } = &spec
        else {
            panic!()
        };
        assert_eq!(*n_phases, 4);
        assert_eq!(*hold_rc, 8.0);
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"kind\":\"shift_register\""));
        assert_eq!(serde_json::from_str::<CellSpec>(&json).unwrap(), spec);
    }

    /// Validation rejects the obvious non-physical inputs.
    #[test]
    fn cell_spec_validation() {
        let bad = CellSpec::RampCharge {
            r_ohm: -1.0,
            c_f: 1e-9,
            v: 1.0,
        };
        assert!(bad.validate().is_err());
        let bad = CellSpec::ShiftRegister {
            r_ohm: 1e3,
            c_f: 1e-9,
            v: 1.0,
            n_stages: 0,
            n_phases: 4,
            hold_rc: 8.0,
        };
        assert!(bad.validate().is_err());
    }

    /// Abrupt limits: ½CV², CV², N·CV².
    #[test]
    fn abrupt_limits() {
        let (r, c, v) = (1e3, 1e-9, 2.0);
        let half_cv2 = 0.5 * c * v * v;
        assert_eq!(
            CellSpec::RampCharge {
                r_ohm: r,
                c_f: c,
                v
            }
            .abrupt_limit_j(),
            half_cv2
        );
        assert_eq!(
            CellSpec::ChargeRecovery {
                r_ohm: r,
                c_f: c,
                v,
                hold_rc: 8.0
            }
            .abrupt_limit_j(),
            2.0 * half_cv2
        );
        assert_eq!(
            CellSpec::ShiftRegister {
                r_ohm: r,
                c_f: c,
                v,
                n_stages: 5,
                n_phases: 4,
                hold_rc: 8.0
            }
            .abrupt_limit_j(),
            10.0 * half_cv2
        );
    }
}
