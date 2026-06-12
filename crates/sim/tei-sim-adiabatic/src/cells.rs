//! Parametric adiabatic cell templates — functions that emit
//! `tei_sim_circuit::Netlist`s, plus the serde-able `CellSpec` the sweep
//! harness and executor consume.
//!
//! # The switch-as-resistor simplification (R–C templates)
//!
//! The original templates model the transmission gate / pass device of a
//! real adiabatic cell (S2LAL / Q2LAL / ECRL / 2LAL families) as a **fixed
//! linear on-resistance**. This is the canonical first-order model of the
//! adiabatic-logic literature — the (RC/T)·CV² ramp-dissipation law (Athas
//! et al. 1994; DeBenedictis arXiv:2009.00448) is derived for exactly this
//! R–C abstraction, with R the on-resistance of the conducting switch and C
//! the load it (dis)charges. Those templates follow the ideal −1 log-log
//! slope all the way down — the *upper bound on recoverable energy*.
//!
//! # The MOSFET S2LAL templates (M2)
//!
//! With `tei-sim-circuit`'s M2 ladder complete, [`s2lal_chain`] models the
//! switches as real devices: each stage is a CMOS **transmission gate**
//! (NMOS + PMOS, [`transmission_gate`]) between its trapezoid power-clock
//! and the load capacitor, driven by complementary control trapezoids that
//! envelope the clock pulse (the predecessor-phase signal of the real
//! S2LAL pipeline), plus the **complementary off T-gate** from the output
//! to the 0-rail — the opposing data path that is off while the output
//! holds a 1. Charging runs clock → T-gate → load; on the falling edge the
//! charge flows **back through the T-gate into the clock** — the energy
//! recovery, with the unrecovered remainder dissipated in the channels.
//! Level-1 switches add the threshold/triode realism; EKV-lite switches
//! (`leakage: true` in [`CellSpec::S2lalChain`]) add the honest subthreshold
//! leakage through the off T-gate that floors the curve and produces the
//! published interior minimum of E(T).
//!
//! **Kept simplifications** (documented contract): power clocks and the
//! T-gate control signals are *ideal voltage sources* — the point is to
//! measure transistor + wire dissipation per cell, not to design the clock
//! generator (DeBenedictis instruments the same way). No gate capacitance
//! (control sources deliver zero charge), no body terminal, the data path
//! is folded into the per-stage load C, and stages ride their own clock
//! phases so N-stage totals stay linear in N — the property check.
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
//! - [`s2lal_chain`] / [`s2lal_buffer_stage`] — the MOSFET version of the
//!   same chain (see above).

use serde::{Deserialize, Serialize};
use tei_sim_circuit::{CircuitError, MosPolarity, Netlist, Node, VT_300K, Waveform};

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

/// Pass-transistor model for the S2LAL templates: level-1 square-law
/// switches by default, EKV-lite (honest subthreshold leakage) when `ekv`
/// is set. `kp` is the full β [A/V²], `vth` the threshold magnitude [V] —
/// matched NMOS/PMOS (the PMOS is the exact mirror).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SwitchModel {
    pub kp: f64,
    pub vth: f64,
    /// `Some((n, phi_t))` → EKV-lite devices with slope factor n and thermal
    /// voltage phi_t [V]; `None` → level-1 (λ = 0).
    pub ekv: Option<(f64, f64)>,
}

impl SwitchModel {
    /// Characteristic on-resistance 1/(kp·(v − vth)) [Ω] — the R of the
    /// R–C abstraction this switch replaces, used for T/RC normalization
    /// and step sizing. `v` is the clock amplitude.
    pub fn r_on(&self, v: f64) -> f64 {
        1.0 / (self.kp * (v - self.vth))
    }

    /// Append one device of this model.
    #[allow(clippy::too_many_arguments)]
    fn push(&self, net: &mut Netlist, name: &str, d: Node, g: Node, s: Node, pol: MosPolarity) {
        match self.ekv {
            None => {
                net.mosfet(name, d, g, s, pol, self.vth, self.kp, 0.0);
            }
            Some((n, phi_t)) => {
                net.mosfet_ekv(name, d, g, s, pol, self.vth, self.kp, n, phi_t);
            }
        }
    }
}

/// CMOS transmission gate between `a` and `b`: NMOS (gate `ng`) + PMOS
/// (gate `pg`) in parallel — elements `{prefix}n`, `{prefix}p`. Drive `ng`
/// to the rail and `pg` to 0 to pass the full swing in both directions;
/// swap to hold it off.
pub fn transmission_gate(
    net: &mut Netlist,
    prefix: &str,
    a: Node,
    b: Node,
    ng: Node,
    pg: Node,
    sw: &SwitchModel,
) {
    sw.push(net, &format!("{prefix}n"), a, ng, b, MosPolarity::Nmos);
    sw.push(net, &format!("{prefix}p"), a, pg, b, MosPolarity::Pmos);
}

/// One S2LAL buffer stage appended to `net` (see the module docs):
///
/// - power clock `clk{idx}` on node `base`: trapezoid delayed by
///   `t_delay + t_ramp` (rise `t_ramp` · hold `t_hold` · fall `t_ramp`);
/// - pass T-gate `tg{idx}{n,p}` from the clock rail to the load `cload{idx}`
///   on node `base + 1`;
/// - complementary control trapezoids `ctln{idx}` / `ctlp{idx}` on nodes
///   `base + 2` / `base + 3`, on for the clock's active window **plus a
///   6·R_on·C discharge tail** (the predecessor-phase signal, modeled as an
///   ideal source). The tail matters in the abrupt limit: when t_ramp ≪ RC
///   the load can only return its charge through the T-gate *after* the
///   clock has collapsed, and a gate that shuts immediately would strand
///   ½CV² on the load instead of completing the CV² cycle loss;
/// - off T-gate `off{idx}{n,p}` from the output to ground (NMOS gate at
///   ground, PMOS gate at the `rail` node held at `v`) — the opposing
///   S2LAL data path: exactly off under level-1, subthreshold-leaking
///   under EKV.
#[allow(clippy::too_many_arguments)]
pub fn s2lal_buffer_stage(
    net: &mut Netlist,
    idx: usize,
    rail: Node,
    base: Node,
    sw: &SwitchModel,
    c: f64,
    v: f64,
    t_delay: f64,
    t_ramp: f64,
    t_hold: f64,
) {
    let (clk, out, ng, pg) = (base, base + 1, base + 2, base + 3);
    // Clock active window: [t_delay + t_ramp, t_delay + 3·t_ramp + t_hold].
    net.vsource(
        &format!("clk{idx}"),
        clk,
        0,
        Waveform::Trapezoid {
            v,
            t_delay: t_delay + t_ramp,
            t_rise: t_ramp,
            t_hold,
            t_fall: t_ramp,
        },
    );
    // NMOS control: rises over [t_delay, t_delay + t_ramp], holds through
    // the whole clock pulse plus the discharge tail, falls after it.
    let ctl_hold = 2.0 * t_ramp + t_hold + 6.0 * sw.r_on(v) * c;
    net.vsource(
        &format!("ctln{idx}"),
        ng,
        0,
        Waveform::Trapezoid {
            v,
            t_delay,
            t_rise: t_ramp,
            t_hold: ctl_hold,
            t_fall: t_ramp,
        },
    );
    // PMOS control: the exact complement (v − ctln), as a PWL.
    net.vsource(
        &format!("ctlp{idx}"),
        pg,
        0,
        Waveform::Pwl {
            points: vec![
                (t_delay, v),
                (t_delay + t_ramp, 0.0),
                (t_delay + t_ramp + ctl_hold, 0.0),
                (t_delay + 2.0 * t_ramp + ctl_hold, v),
            ],
        },
    );
    transmission_gate(net, &format!("tg{idx}"), clk, out, ng, pg, sw);
    // Opposing (off) T-gate to the 0-rail: NMOS gate grounded, PMOS gate at v.
    sw.push(net, &format!("off{idx}n"), out, 0, 0, MosPolarity::Nmos);
    sw.push(net, &format!("off{idx}p"), out, rail, 0, MosPolarity::Pmos);
    net.capacitor(&format!("cload{idx}"), out, 0, c, 0.0);
}

/// S2LAL shift-register-style chain of `n_stages` buffer stages on an
/// `n_phases`-phase power clock — the MOSFET version of
/// [`shift_register_chain`], same interface shape. Stage k's clock is
/// delayed by `(k mod n_phases)·(t_ramp + t_hold)`. Node 1 is the shared
/// PMOS-off control rail (DC at `v`); stage k occupies nodes
/// `4k+2 ..= 4k+5` (clock, load, NMOS control, PMOS control).
#[allow(clippy::too_many_arguments)]
pub fn s2lal_chain(
    n_stages: usize,
    n_phases: usize,
    sw: &SwitchModel,
    c: f64,
    v: f64,
    t_ramp: f64,
    t_hold: f64,
) -> Netlist {
    let phase_step = t_ramp + t_hold;
    let mut net = Netlist::new();
    net.vsource("voff", 1, 0, Waveform::Dc { v });
    for k in 0..n_stages {
        let phase = (k % n_phases.max(1)) as f64;
        s2lal_buffer_stage(
            &mut net,
            k,
            1,
            4 * k + 2,
            sw,
            c,
            v,
            phase * phase_step,
            t_ramp,
            t_hold,
        );
    }
    net
}

fn default_hold_rc() -> f64 {
    8.0
}
fn default_n_phases() -> usize {
    4
}
fn default_kp() -> f64 {
    1e-4
}
fn default_vth() -> f64 {
    0.3
}
fn default_ekv_n() -> f64 {
    1.3
}
fn default_phi_t() -> f64 {
    VT_300K
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
    /// [`s2lal_chain`] — MOSFET transmission-gate stages (M2). Abrupt limit
    /// N·CV² per clock cycle, like [`CellSpec::ShiftRegister`]. The
    /// characteristic R of the T/RC normalization is the switch
    /// on-resistance 1/(kp·(v − vth)). `leakage: true` swaps the level-1
    /// switches for EKV-lite ones (slope factor `ekv_n`, thermal voltage
    /// `phi_t`), whose subthreshold leakage through the off T-gate floors
    /// the energy curve — E(T) develops the published interior minimum.
    S2lalChain {
        c_f: f64,
        v: f64,
        n_stages: usize,
        #[serde(default = "default_n_phases")]
        n_phases: usize,
        #[serde(default = "default_hold_rc")]
        hold_rc: f64,
        /// Switch transconductance β [A/V²].
        #[serde(default = "default_kp")]
        kp: f64,
        /// Switch threshold magnitude [V]; must be < `v`.
        #[serde(default = "default_vth")]
        vth: f64,
        /// EKV-lite switches with honest subthreshold leakage.
        #[serde(default)]
        leakage: bool,
        /// EKV slope factor (used only with `leakage: true`).
        #[serde(default = "default_ekv_n")]
        ekv_n: f64,
        /// EKV thermal voltage [V] (used only with `leakage: true`).
        #[serde(default = "default_phi_t")]
        phi_t: f64,
    },
}

impl CellSpec {
    /// The characteristic switch resistance [Ω]: the explicit `r_ohm` of the
    /// R–C templates, the T-gate on-resistance 1/(kp·(v − vth)) of the
    /// MOSFET templates.
    pub fn r_ohm(&self) -> f64 {
        match self {
            CellSpec::RampCharge { r_ohm, .. }
            | CellSpec::ChargeRecovery { r_ohm, .. }
            | CellSpec::ShiftRegister { r_ohm, .. } => *r_ohm,
            CellSpec::S2lalChain { kp, vth, v, .. } => 1.0 / (kp * (v - vth)),
        }
    }

    pub fn c_f(&self) -> f64 {
        match self {
            CellSpec::RampCharge { c_f, .. }
            | CellSpec::ChargeRecovery { c_f, .. }
            | CellSpec::ShiftRegister { c_f, .. }
            | CellSpec::S2lalChain { c_f, .. } => *c_f,
        }
    }

    pub fn v(&self) -> f64 {
        match self {
            CellSpec::RampCharge { v, .. }
            | CellSpec::ChargeRecovery { v, .. }
            | CellSpec::ShiftRegister { v, .. }
            | CellSpec::S2lalChain { v, .. } => *v,
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
            CellSpec::ShiftRegister { n_stages, .. } | CellSpec::S2lalChain { n_stages, .. } => {
                2.0 * half_cv2 * *n_stages as f64
            }
        }
    }

    /// Reject non-physical parameters before they reach the solver.
    pub fn validate(&self) -> Result<(), CircuitError> {
        let ok = |x: f64| x.is_finite() && x > 0.0;
        let hold_ok = |hold_rc: &f64| -> Result<(), CircuitError> {
            if !hold_rc.is_finite() || *hold_rc < 0.0 {
                return Err(CircuitError::Invalid(format!(
                    "hold_rc must be finite and ≥ 0, got {hold_rc}"
                )));
            }
            Ok(())
        };
        let stages_ok = |n_stages: &usize, n_phases: &usize| -> Result<(), CircuitError> {
            if *n_stages == 0 || *n_phases == 0 {
                return Err(CircuitError::Invalid(
                    "n_stages and n_phases must be ≥ 1".into(),
                ));
            }
            Ok(())
        };
        // S2lalChain's r_ohm() is derived from (kp, vth, v) — check those
        // first so a bad vth surfaces as its own message, not as a weird r.
        if let CellSpec::S2lalChain {
            v,
            kp,
            vth,
            leakage,
            ekv_n,
            phi_t,
            ..
        } = self
        {
            if !ok(*kp) {
                return Err(CircuitError::Invalid(format!(
                    "kp must be finite and positive, got {kp}"
                )));
            }
            if !vth.is_finite() || *vth < 0.0 || *vth >= *v {
                return Err(CircuitError::Invalid(format!(
                    "vth must satisfy 0 ≤ vth < v (no switch overdrive otherwise), got vth={vth} v={v}"
                )));
            }
            if *leakage && !(ok(*ekv_n) && ok(*phi_t)) {
                return Err(CircuitError::Invalid(format!(
                    "ekv_n and phi_t must be finite and positive, got n={ekv_n} phi_t={phi_t}"
                )));
            }
        }
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
            CellSpec::ChargeRecovery { hold_rc, .. } => hold_ok(hold_rc)?,
            CellSpec::ShiftRegister {
                n_stages,
                n_phases,
                hold_rc,
                ..
            }
            | CellSpec::S2lalChain {
                n_stages,
                n_phases,
                hold_rc,
                ..
            } => {
                stages_ok(n_stages, n_phases)?;
                hold_ok(hold_rc)?;
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
            CellSpec::S2lalChain {
                c_f,
                v,
                n_stages,
                n_phases,
                hold_rc,
                kp,
                vth,
                leakage,
                ekv_n,
                phi_t,
            } => {
                let sw = SwitchModel {
                    kp: *kp,
                    vth: *vth,
                    ekv: leakage.then_some((*ekv_n, *phi_t)),
                };
                let t_hold = hold_rc * rc;
                let phase_step = t_ramp + t_hold;
                let max_delay = ((*n_stages).min(*n_phases) - 1) as f64 * phase_step;
                // Stage span: control rise (t_ramp) + clock pulse
                // (2·t_ramp + t_hold) + control fall (t_ramp).
                (
                    s2lal_chain(*n_stages, *n_phases, &sw, *c_f, *v, t_ramp, t_hold),
                    max_delay + 4.0 * t_ramp + t_hold + SETTLE_RC * rc,
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

    /// S2LAL chain: 1 rail source + 8 elements/stage, contiguous nodes
    /// 1..=4N+1, phases wrap mod n_phases, T-gate polarities paired.
    #[test]
    fn s2lal_chain_shape_and_phases() {
        let (n_stages, n_phases) = (6, 4);
        let (t_ramp, t_hold) = (2e-6, 1e-6);
        let sw = SwitchModel {
            kp: 1e-4,
            vth: 0.3,
            ekv: None,
        };
        let net = s2lal_chain(n_stages, n_phases, &sw, 1e-12, 1.0, t_ramp, t_hold);
        assert_eq!(net.elements.len(), 1 + 8 * n_stages);
        let mut nodes: Vec<usize> = net
            .elements
            .iter()
            .flat_map(|e| {
                let (p, n) = e.nodes();
                [Some(p), Some(n), e.gate()]
            })
            .flatten()
            .filter(|&n| n != 0)
            .collect();
        nodes.sort_unstable();
        nodes.dedup();
        assert_eq!(nodes, (1..=4 * n_stages + 1).collect::<Vec<_>>());
        // Stage 5 is phase 5 mod 4 = 1 → its clock is delayed by
        // 1·(t_ramp + t_hold) + t_ramp (the control-envelope shift).
        let Element::VoltageSource {
            wave: Waveform::Trapezoid { t_delay, .. },
            ..
        } = &net.elements[1 + 8 * 5]
        else {
            panic!("expected stage-5 clock");
        };
        assert!((t_delay - ((t_ramp + t_hold) + t_ramp)).abs() < 1e-18);
        // EKV flavor swaps every switch for the leaky model.
        let sw_ekv = SwitchModel {
            ekv: Some((1.3, VT_300K)),
            ..sw
        };
        let net = s2lal_chain(2, 2, &sw_ekv, 1e-12, 1.0, t_ramp, t_hold);
        let n_ekv = net
            .elements
            .iter()
            .filter(|e| matches!(e, Element::MosfetEkv { .. }))
            .count();
        assert_eq!(n_ekv, 8); // 4 switches per stage × 2 stages
    }

    /// S2lalChain CellSpec: serde defaults, derived r_ohm, abrupt limit,
    /// and validation of the vth < v requirement.
    #[test]
    fn s2lal_spec_serde_defaults_and_validation() {
        let spec: CellSpec =
            serde_json::from_str(r#"{"kind":"s2lal_chain","c_f":1e-12,"v":1.0,"n_stages":4}"#)
                .unwrap();
        let CellSpec::S2lalChain {
            n_phases,
            hold_rc,
            kp,
            vth,
            leakage,
            ekv_n,
            phi_t,
            ..
        } = &spec
        else {
            panic!()
        };
        assert_eq!(*n_phases, 4);
        assert_eq!(*hold_rc, 8.0);
        assert_eq!(*kp, 1e-4);
        assert_eq!(*vth, 0.3);
        assert!(!leakage);
        assert_eq!(*ekv_n, 1.3);
        assert_eq!(*phi_t, VT_300K);
        assert!((spec.r_ohm() - 1.0 / (1e-4 * 0.7)).abs() < 1e-9);
        assert_eq!(spec.abrupt_limit_j(), 4.0 * 1e-12);
        assert!(spec.validate().is_ok());
        let json = serde_json::to_string(&spec).unwrap();
        assert!(json.contains("\"kind\":\"s2lal_chain\""));
        assert_eq!(serde_json::from_str::<CellSpec>(&json).unwrap(), spec);

        // vth ≥ v leaves the switch without overdrive — rejected.
        let bad: CellSpec = serde_json::from_str(
            r#"{"kind":"s2lal_chain","c_f":1e-12,"v":1.0,"n_stages":1,"vth":1.0}"#,
        )
        .unwrap();
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
