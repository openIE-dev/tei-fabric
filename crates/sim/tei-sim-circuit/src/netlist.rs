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

/// A circuit element. `p`/`n` are the positive/negative terminals; current
/// through the element is oriented p → n (see module docs).
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
}

fn default_i_s() -> f64 {
    1e-14
}
fn default_ideality() -> f64 {
    1.0
}

impl Element {
    pub fn name(&self) -> &str {
        match self {
            Element::Resistor { name, .. }
            | Element::Capacitor { name, .. }
            | Element::Inductor { name, .. }
            | Element::VoltageSource { name, .. }
            | Element::CurrentSource { name, .. }
            | Element::Diode { name, .. } => name,
        }
    }

    pub fn nodes(&self) -> (Node, Node) {
        match self {
            Element::Resistor { p, n, .. }
            | Element::Capacitor { p, n, .. }
            | Element::Inductor { p, n, .. }
            | Element::VoltageSource { p, n, .. }
            | Element::CurrentSource { p, n, .. }
            | Element::Diode { p, n, .. } => (*p, *n),
        }
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
}
