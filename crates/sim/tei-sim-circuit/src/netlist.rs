//! Netlist model — nodes (node 0 = ground), elements, and source waveforms.
//!
//! Elements carry user-chosen unique names so the per-element energy map in
//! the transient result is addressable. Polarity convention throughout the
//! crate: the element voltage is v = v(p) − v(n) and the element current i
//! flows from terminal `p` through the element to terminal `n`, so the
//! instantaneous power **absorbed** by the element is p = i·v. Sources
//! delivering energy therefore show negative absorbed energy; the transient
//! result reports `source_energy` with the sign flipped (delivered).

use serde::{Deserialize, Serialize};

/// Node identifier. Node `0` is ground (the reference); MNA unknowns are the
/// potentials of nodes `1..=N`.
pub type Node = usize;

/// Thermal voltage kT/q at 300 K, in volts (≈ 25.852 mV).
pub const VT_300K: f64 = 1.380_649e-23 * 300.0 / 1.602_176_634e-19;

/// Source waveform v(t) (or i(t) for current sources).
///
/// `Trapezoid` is the adiabatic power-clock shape: flat at 0 until `t_delay`,
/// linear rise over `t_rise`, hold at `v` for `t_hold`, linear fall over
/// `t_fall`, then flat at 0 again. Periodic clock trains can be composed
/// with `Pwl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "shape", rename_all = "snake_case")]
pub enum Waveform {
    /// Constant value for all t.
    Dc { v: f64 },
    /// Linear ramp 0 → `v` over `t_ramp`, then hold at `v`.
    Ramp { v: f64, t_ramp: f64 },
    /// Single power-clock pulse: delay / rise / hold / fall / 0.
    Trapezoid {
        v: f64,
        #[serde(default)]
        t_delay: f64,
        t_rise: f64,
        t_hold: f64,
        t_fall: f64,
    },
    /// offset + amplitude·sin(2π·freq_hz·t + phase_rad).
    Sine {
        #[serde(default)]
        offset: f64,
        amplitude: f64,
        freq_hz: f64,
        #[serde(default)]
        phase_rad: f64,
    },
    /// Piecewise-linear (t, value) points; clamped outside the time range.
    Pwl { points: Vec<(f64, f64)> },
}

impl Waveform {
    /// Evaluate the waveform at time `t`.
    pub fn at(&self, t: f64) -> f64 {
        match self {
            Waveform::Dc { v } => *v,
            Waveform::Ramp { v, t_ramp } => {
                if *t_ramp <= 0.0 {
                    // Degenerate ramp = step at t = 0.
                    if t > 0.0 { *v } else { 0.0 }
                } else if t <= 0.0 {
                    0.0
                } else if t >= *t_ramp {
                    *v
                } else {
                    v * t / t_ramp
                }
            }
            Waveform::Trapezoid {
                v,
                t_delay,
                t_rise,
                t_hold,
                t_fall,
            } => {
                let tt = t - t_delay;
                if tt <= 0.0 {
                    0.0
                } else if tt < *t_rise {
                    v * tt / t_rise
                } else if tt < t_rise + t_hold {
                    *v
                } else if tt < t_rise + t_hold + t_fall {
                    v * (1.0 - (tt - t_rise - t_hold) / t_fall)
                } else {
                    0.0
                }
            }
            Waveform::Sine {
                offset,
                amplitude,
                freq_hz,
                phase_rad,
            } => offset + amplitude * (std::f64::consts::TAU * freq_hz * t + phase_rad).sin(),
            Waveform::Pwl { points } => {
                if points.is_empty() {
                    return 0.0;
                }
                if t <= points[0].0 {
                    return points[0].1;
                }
                let last = points[points.len() - 1];
                if t >= last.0 {
                    return last.1;
                }
                for w in points.windows(2) {
                    let (t0, v0) = w[0];
                    let (t1, v1) = w[1];
                    if t >= t0 && t <= t1 {
                        if t1 <= t0 {
                            return v1; // vertical segment: take the later value
                        }
                        return v0 + (v1 - v0) * (t - t0) / (t1 - t0);
                    }
                }
                last.1
            }
        }
    }
}

/// MOSFET channel polarity. A `Pmos` device is the **exact mirror** of the
/// `Nmos` device with the same parameters: i_P(v_g, v_d, v_s) =
/// −i_N(−v_g, −v_d, −v_s), with `vth` the (positive) threshold magnitude for
/// both. The mirror is exact in floating point, which is what the PMOS
/// symmetry validation pins to 1e-12.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MosPolarity {
    Nmos,
    Pmos,
}

/// A circuit element. `p`/`n` are the positive/negative terminals; current
/// through the element is oriented p → n (see module docs). For the
/// three-terminal MOSFETs the (p, n) pair is (drain, source) — the gate is a
/// pure high-impedance control terminal (no DC gate current in either model,
/// and no gate capacitance: gate dynamics are deliberately out of scope, the
/// channel dissipation is what the adiabatic energy analysis needs).
///
/// `ic` on reactive elements is the initial condition used to start the
/// transient (capacitor voltage / inductor current); it defaults to 0, which
/// is the SPICE `UIC` behavior the validation suite relies on (a DC source
/// hitting an uncharged capacitor at t = 0 is the canonical RC step).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Element {
    Resistor {
        name: String,
        p: Node,
        n: Node,
        r: f64,
    },
    Capacitor {
        name: String,
        p: Node,
        n: Node,
        c: f64,
        #[serde(default)]
        ic: f64,
    },
    Inductor {
        name: String,
        p: Node,
        n: Node,
        l: f64,
        #[serde(default)]
        ic: f64,
    },
    VoltageSource {
        name: String,
        p: Node,
        n: Node,
        wave: Waveform,
    },
    CurrentSource {
        name: String,
        p: Node,
        n: Node,
        wave: Waveform,
    },
    /// Shockley diode i = I_s·(e^{v/(n·V_T)} − 1), anode = p, cathode = n.
    Diode {
        name: String,
        p: Node,
        n: Node,
        #[serde(default = "default_i_s")]
        i_s: f64,
        #[serde(default = "default_ideality")]
        n_ideality: f64,
    },
    /// Level-1 (Shichman–Hodges) MOSFET — the M2 square-law device.
    ///
    /// NMOS with v_ds ≥ 0 (the device is source/drain symmetric; reversed
    /// bias swaps the roles, which is what lets it model transmission gates):
    ///
    /// ```text
    /// cutoff     v_gs ≤ vth:        i_d = 0
    /// triode     v_ds < v_gs − vth: i_d = kp·[(v_gs − vth)·v_ds − v_ds²/2]·(1 + λ·v_ds)
    /// saturation otherwise:         i_d = (kp/2)·(v_gs − vth)²·(1 + λ·v_ds)
    /// ```
    ///
    /// `kp` is the full transconductance factor β = k′·W/L [A/V²] (W/L is
    /// folded in — no geometry parsing, per the roadmap's deliberately-out
    /// list). The (1 + λ·v_ds) factor applies in both conducting regions,
    /// keeping i_d and its derivatives continuous at the triode/saturation
    /// boundary. `vth` is the threshold magnitude (positive for both
    /// polarities); PMOS is the exact mirror (see [`MosPolarity`]). No body
    /// terminal: potentials enter bulk-referenced at ground (NMOS) / its
    /// mirror (PMOS), i.e. a body-effect-free device. No gate current, no
    /// gate capacitance.
    Mosfet {
        name: String,
        d: Node,
        g: Node,
        s: Node,
        polarity: MosPolarity,
        vth: f64,
        kp: f64,
        #[serde(default)]
        lambda: f64,
    },
    /// EKV-lite MOSFET — single-expression all-region model with honest
    /// near/sub-threshold behavior (the adiabatic energy floor):
    ///
    /// ```text
    /// i_d = I_S·[F(v_p − v_s) − F(v_p − v_d)]
    /// F(u) = ln²(1 + e^{u/(2·φ_t)})        I_S = 2·n·kp·φ_t²
    /// v_p  = (v_g − vth)/n                  (pinch-off voltage)
    /// ```
    ///
    /// Limits: deep subthreshold i_d ∝ e^{(v_gs − vth)/(n·φ_t)} — slope
    /// n·φ_t·ln10 per decade; strong-inversion saturation
    /// i_d → kp/(2n)·(v_gs − vth)². The forward−reverse form is
    /// source/drain symmetric by construction (transmission gates work
    /// without case analysis) and never fully shuts off — the subthreshold
    /// leakage that puts the interior minimum into E(T). Same conventions as
    /// [`Element::Mosfet`]: `vth` is the threshold magnitude, PMOS is the
    /// exact mirror, no body terminal / gate current / gate capacitance.
    MosfetEkv {
        name: String,
        d: Node,
        g: Node,
        s: Node,
        polarity: MosPolarity,
        vth: f64,
        kp: f64,
        /// Slope factor (subthreshold swing = n·φ_t·ln10 per decade).
        #[serde(default = "default_ekv_n")]
        n: f64,
        /// Thermal voltage kT/q [V]; defaults to [`VT_300K`] ≈ 25.85 mV.
        #[serde(default = "default_phi_t")]
        phi_t: f64,
    },
}

fn default_i_s() -> f64 {
    1e-14
}
fn default_ideality() -> f64 {
    1.0
}
fn default_ekv_n() -> f64 {
    1.3
}
fn default_phi_t() -> f64 {
    VT_300K
}

impl Element {
    pub fn name(&self) -> &str {
        match self {
            Element::Resistor { name, .. }
            | Element::Capacitor { name, .. }
            | Element::Inductor { name, .. }
            | Element::VoltageSource { name, .. }
            | Element::CurrentSource { name, .. }
            | Element::Diode { name, .. }
            | Element::Mosfet { name, .. }
            | Element::MosfetEkv { name, .. } => name,
        }
    }

    /// The (p, n) current-carrying terminal pair. For MOSFETs this is
    /// (drain, source) — the channel branch; the gate carries no current
    /// (reach it via [`Element::gate`]).
    pub fn nodes(&self) -> (Node, Node) {
        match self {
            Element::Resistor { p, n, .. }
            | Element::Capacitor { p, n, .. }
            | Element::Inductor { p, n, .. }
            | Element::VoltageSource { p, n, .. }
            | Element::CurrentSource { p, n, .. }
            | Element::Diode { p, n, .. } => (*p, *n),
            Element::Mosfet { d, s, .. } | Element::MosfetEkv { d, s, .. } => (*d, *s),
        }
    }

    /// The gate (control) terminal of a MOSFET, `None` for everything else.
    pub fn gate(&self) -> Option<Node> {
        match self {
            Element::Mosfet { g, .. } | Element::MosfetEkv { g, .. } => Some(*g),
            _ => None,
        }
    }

    /// Whether the element makes the MNA matrix depend on the iterate
    /// (Newton-Raphson inner loop; rules the element out of
    /// [`crate::LinearDcSolver`]).
    pub fn is_nonlinear(&self) -> bool {
        matches!(
            self,
            Element::Diode { .. } | Element::Mosfet { .. } | Element::MosfetEkv { .. }
        )
    }
}

/// A flat netlist: just the element list. Node ids are implicit (0 = ground;
/// non-ground ids must form a contiguous 1..=N — validation enforces it,
/// because a skipped id would be an all-zero MNA row).
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Netlist {
    pub elements: Vec<Element>,
}

impl Netlist {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn resistor(&mut self, name: &str, p: Node, n: Node, r: f64) -> &mut Self {
        self.elements.push(Element::Resistor {
            name: name.into(),
            p,
            n,
            r,
        });
        self
    }

    pub fn capacitor(&mut self, name: &str, p: Node, n: Node, c: f64, ic: f64) -> &mut Self {
        self.elements.push(Element::Capacitor {
            name: name.into(),
            p,
            n,
            c,
            ic,
        });
        self
    }

    pub fn inductor(&mut self, name: &str, p: Node, n: Node, l: f64, ic: f64) -> &mut Self {
        self.elements.push(Element::Inductor {
            name: name.into(),
            p,
            n,
            l,
            ic,
        });
        self
    }

    pub fn vsource(&mut self, name: &str, p: Node, n: Node, wave: Waveform) -> &mut Self {
        self.elements.push(Element::VoltageSource {
            name: name.into(),
            p,
            n,
            wave,
        });
        self
    }

    pub fn isource(&mut self, name: &str, p: Node, n: Node, wave: Waveform) -> &mut Self {
        self.elements.push(Element::CurrentSource {
            name: name.into(),
            p,
            n,
            wave,
        });
        self
    }

    pub fn diode(&mut self, name: &str, p: Node, n: Node, i_s: f64, n_ideality: f64) -> &mut Self {
        self.elements.push(Element::Diode {
            name: name.into(),
            p,
            n,
            i_s,
            n_ideality,
        });
        self
    }

    /// Level-1 MOSFET (drain, gate, source). `vth` is the threshold
    /// magnitude, `kp` the full β = k′·W/L [A/V²], `lambda` the
    /// channel-length-modulation coefficient [1/V].
    #[allow(clippy::too_many_arguments)]
    pub fn mosfet(
        &mut self,
        name: &str,
        d: Node,
        g: Node,
        s: Node,
        polarity: MosPolarity,
        vth: f64,
        kp: f64,
        lambda: f64,
    ) -> &mut Self {
        self.elements.push(Element::Mosfet {
            name: name.into(),
            d,
            g,
            s,
            polarity,
            vth,
            kp,
            lambda,
        });
        self
    }

    /// EKV-lite MOSFET (drain, gate, source). `n` is the slope factor,
    /// `phi_t` the thermal voltage [V].
    #[allow(clippy::too_many_arguments)]
    pub fn mosfet_ekv(
        &mut self,
        name: &str,
        d: Node,
        g: Node,
        s: Node,
        polarity: MosPolarity,
        vth: f64,
        kp: f64,
        n: f64,
        phi_t: f64,
    ) -> &mut Self {
        self.elements.push(Element::MosfetEkv {
            name: name.into(),
            d,
            g,
            s,
            polarity,
            vth,
            kp,
            n,
            phi_t,
        });
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Ramp: 0 before t=0, linear to v at t_ramp, held after.
    #[test]
    fn ramp_shape() {
        let w = Waveform::Ramp {
            v: 2.0,
            t_ramp: 4.0,
        };
        assert_eq!(w.at(-1.0), 0.0);
        assert!((w.at(1.0) - 0.5).abs() < 1e-15);
        assert!((w.at(4.0) - 2.0).abs() < 1e-15);
        assert!((w.at(100.0) - 2.0).abs() < 1e-15);
    }

    /// Trapezoid: delay / rise / hold / fall / zero — the power-clock pulse.
    #[test]
    fn trapezoid_shape() {
        let w = Waveform::Trapezoid {
            v: 1.0,
            t_delay: 1.0,
            t_rise: 2.0,
            t_hold: 3.0,
            t_fall: 2.0,
        };
        assert_eq!(w.at(0.5), 0.0);
        assert!((w.at(2.0) - 0.5).abs() < 1e-15); // mid-rise
        assert!((w.at(4.0) - 1.0).abs() < 1e-15); // hold
        assert!((w.at(7.0) - 0.5).abs() < 1e-15); // mid-fall
        assert_eq!(w.at(9.0), 0.0);
    }

    /// PWL: clamped ends, linear interior.
    #[test]
    fn pwl_shape() {
        let w = Waveform::Pwl {
            points: vec![(0.0, 0.0), (1.0, 1.0), (3.0, 1.0), (4.0, 0.0)],
        };
        assert_eq!(w.at(-5.0), 0.0);
        assert!((w.at(0.5) - 0.5).abs() < 1e-15);
        assert!((w.at(2.0) - 1.0).abs() < 1e-15);
        assert!((w.at(3.5) - 0.5).abs() < 1e-15);
        assert_eq!(w.at(10.0), 0.0);
    }

    /// Element/waveform serde round-trip with the documented tags.
    #[test]
    fn serde_round_trip() {
        let mut net = Netlist::new();
        net.vsource("vs", 1, 0, Waveform::Dc { v: 1.0 })
            .resistor("r1", 1, 2, 1e3)
            .capacitor("c1", 2, 0, 1e-9, 0.0);
        let json = serde_json::to_string(&net).unwrap();
        assert!(json.contains("\"kind\":\"voltage_source\""));
        assert!(json.contains("\"shape\":\"dc\""));
        let back: Netlist = serde_json::from_str(&json).unwrap();
        assert_eq!(back, net);
    }

    /// MOSFET serde: documented tags, snake_case polarity, and the M2
    /// defaults (lambda = 0; EKV n = 1.3, phi_t = VT_300K).
    #[test]
    fn mosfet_serde_round_trip_and_defaults() {
        let m: Element = serde_json::from_str(
            r#"{"kind":"mosfet","name":"m1","d":2,"g":1,"s":0,
                "polarity":"nmos","vth":0.5,"kp":1e-4}"#,
        )
        .unwrap();
        let Element::Mosfet {
            polarity, lambda, ..
        } = &m
        else {
            panic!("expected level-1 mosfet");
        };
        assert_eq!(*polarity, MosPolarity::Nmos);
        assert_eq!(*lambda, 0.0);
        let json = serde_json::to_string(&m).unwrap();
        assert!(json.contains("\"kind\":\"mosfet\""));
        assert!(json.contains("\"polarity\":\"nmos\""));
        assert_eq!(serde_json::from_str::<Element>(&json).unwrap(), m);

        let e: Element = serde_json::from_str(
            r#"{"kind":"mosfet_ekv","name":"m2","d":3,"g":1,"s":2,
                "polarity":"pmos","vth":0.4,"kp":2e-4}"#,
        )
        .unwrap();
        let Element::MosfetEkv { n, phi_t, .. } = &e else {
            panic!("expected ekv mosfet");
        };
        assert_eq!(*n, 1.3);
        assert_eq!(*phi_t, VT_300K);
        assert_eq!(e.nodes(), (3, 2));
        assert_eq!(e.gate(), Some(1));
        assert!(e.is_nonlinear());
        let json = serde_json::to_string(&e).unwrap();
        assert!(json.contains("\"kind\":\"mosfet_ekv\""));
        assert_eq!(serde_json::from_str::<Element>(&json).unwrap(), e);
    }
}
