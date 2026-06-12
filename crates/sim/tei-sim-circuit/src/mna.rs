//! Modified nodal analysis — stamp derivations, assembly, Newton solve.
//!
//! # Unknown vector
//!
//! `x = [v₁ … v_N | i_b1 … i_bB]` — the potentials of the non-ground nodes
//! followed by the branch currents of the elements that need an extra MNA
//! row (voltage sources always; inductors during transient steps and DC;
//! capacitors only during the t = 0 initialization, where they are imposed
//! as voltage constraints at their initial condition).
//!
//! Each node row is KCL written as Σ(currents leaving the node) = 0; constant
//! current contributions move to the right-hand side `b` with flipped sign,
//! so an injection *into* a node appears as `+` in `b`.
//!
//! # Stamps (derivations)
//!
//! **Resistor** `R` between p and n — current p→n is `i = (v_p − v_n)/R = G·v`
//! with `G = 1/R`. It leaves p and enters n, so:
//!
//! ```text
//! A[p,p] += G   A[n,n] += G   A[p,n] −= G   A[n,p] −= G
//! ```
//!
//! **Capacitor companion** — discretize `i = C dv/dt` over a step of size h.
//!
//! *Trapezoidal*: `(i_new + i_old)/2 = C·(v_new − v_old)/h`, i.e.
//!
//! ```text
//! i_new = G_eq·v_new − I_hist,   G_eq = 2C/h,   I_hist = G_eq·v_old + i_old
//! ```
//!
//! *Backward Euler*: `i_new = C·(v_new − v_old)/h`, i.e. `G_eq = C/h`,
//! `I_hist = G_eq·v_old`.
//!
//! Both are a conductance `G_eq` stamped like a resistor plus a history
//! current source `I_hist` directed n→p (it cancels the constant part of
//! `i_new`): `b[p] += I_hist`, `b[n] −= I_hist`. After the solve the element
//! current is reconstructed as `i_new = G_eq·v_new − I_hist` — *exactly* the
//! current the stamp imposed, which is what makes the discrete Tellegen
//! identity (Σ i·v = 0 over all elements at every accepted time point) hold
//! to LU precision and the energy ledger close.
//!
//! **Inductor companion** — discretize `v = L di/dt`, keeping the branch
//! current `i` as an unknown (row `r`):
//!
//! *Trapezoidal*: `(v_new + v_old)/2 = L·(i_new − i_old)/h` ⇒
//!
//! ```text
//! i_new − (h/2L)·(v_p − v_n) = i_old + (h/2L)·v_old
//! ```
//!
//! *Backward Euler*: `i_new − (h/L)·(v_p − v_n) = i_old`.
//!
//! Row `r`: `A[r,r] = 1`, `A[r,p] −= coef`, `A[r,n] += coef`, `b[r] = rhs`;
//! KCL columns: `A[p,r] += 1`, `A[n,r] −= 1` (the branch current leaves p).
//!
//! **Voltage source** — extended-MNA branch row `r` imposing the constraint,
//! with the branch current i (p→n through the source) entering KCL:
//!
//! ```text
//! A[r,p] = 1   A[r,n] = −1   b[r] = V(t)        (constraint row)
//! A[p,r] += 1  A[n,r] −= 1                       (KCL columns)
//! ```
//!
//! **Current source** `J(t)` p→n: pure RHS, `b[p] −= J`, `b[n] += J`.
//!
//! **Shockley diode** (M2) — `i(v) = I_s·(e^{v/(nV_T)} − 1)`. Newton-Raphson
//! linearizes about the previous iterate `v*`:
//!
//! ```text
//! g  = di/dv|_{v*} = I_s/(nV_T)·e^{v*/(nV_T)} + g_min
//! i* = I_s·(e^{v*/(nV_T)} − 1) + g_min·v*
//! i(v) ≈ g·v + I_eq,   I_eq = i* − g·v*
//! ```
//!
//! `g` stamps like a resistor and `I_eq` like a constant current p→n
//! (`b[p] −= I_eq`, `b[n] += I_eq`). `g_min = 1e-12 S` keeps the Jacobian
//! nonsingular when the diode is fully off. Iterates are damped with the
//! SPICE junction limiter `pnjlim` (Nagel) so the exponential cannot
//! overflow; convergence requires every unknown to move less than
//! `abstol + reltol·|x|` *and* the limiter to have been inactive.
//!
//! **MOSFET** (M2, level-1 and EKV-lite) — the channel current i_ds (d→s)
//! is a function of the three terminal potentials, i(v_g, v_d, v_s).
//! Newton-Raphson linearizes about the previous iterate (v_g*, v_d*, v_s*):
//!
//! ```text
//! i(v) ≈ a_g·v_g + a_d·v_d + a_s·v_s + I_eq        a_x = ∂i/∂v_x|_*
//! I_eq = i* − a_g·v_g* − a_d·v_d* − a_s·v_s*
//! ```
//!
//! The current enters KCL only at the drain and source rows (no DC gate
//! current), so the stamp is two rows × three columns plus the RHS pair:
//!
//! ```text
//! A[d,x] += a_x   A[s,x] −= a_x   (x ∈ {g, d, s})
//! b[d] −= I_eq    b[s] += I_eq
//! ```
//!
//! For level-1, (a_g, a_d, a_s) = (gm, gds, −gm−gds) in the forward
//! orientation, with source/drain roles swapped when v_ds < 0 (the
//! transmission-gate case). For EKV-lite the three partials are independent
//! (the pinch-off voltage divides v_g by the slope factor n — the implicit
//! bulk at ground absorbs the difference). GMIN between d and s is folded
//! into the evaluation, exactly as for the diode. Both models are polynomial
//! / softplus-smooth (no exponential blow-up like the diode), so no junction
//! limiter is applied to MOSFET iterates; stiff DC points fall back to the
//! same source stepping.
//!
//! The system is rebuilt every step and every Newton iteration; *how* it is
//! solved is the [`crate::solver::SystemSolver`]'s decision (M4): small
//! systems assemble dense and LU-factor from scratch (O(n³) is microseconds
//! at cell scale), large ones assemble triplets and go through the sparse
//! Markowitz LU in `tei-sim-core`, refactoring the cached pivot order and
//! fill pattern with each step's new values. The stamps below are generic
//! over a [`MatrixSink`] so both paths share one assembly, entry for entry.

use crate::netlist::{Element, MosPolarity, Netlist, Node, VT_300K};
use crate::solver::SystemSolver;
use crate::{CircuitError, Method};
use tei_sim_core::linalg::Mat;

/// Accumulating sink for MNA matrix stamps — a dense [`Mat`] on the dense
/// path, a triplet list on the sparse path. Every stamp is `+=`.
pub(crate) trait MatrixSink {
    fn add(&mut self, i: usize, j: usize, v: f64);
}

impl MatrixSink for Mat {
    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        self[(i, j)] += v;
    }
}

impl MatrixSink for Vec<(u32, u32, f64)> {
    #[inline]
    fn add(&mut self, i: usize, j: usize, v: f64) {
        self.push((i as u32, j as u32, v));
    }
}

/// Conductance keeping an off diode's Jacobian nonsingular (SPICE GMIN).
pub(crate) const GMIN: f64 = 1e-12;
const NR_MAX_ITERS: usize = 200;
const NR_ABSTOL: f64 = 1e-9;
const NR_RELTOL: f64 = 1e-6;

/// What the assembly is for — branch-row allocation differs per mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Mode {
    /// DC operating point: capacitors open, inductors short (branch row
    /// imposing v_p − v_n = 0, current as unknown).
    DcOp,
    /// t = 0 transient initialization: capacitors imposed as voltage
    /// constraints at their `ic` (branch current = the exact i_C(0⁺)),
    /// inductors as current sources at their `ic`.
    Init,
    /// A transient step: companion models for C and L.
    Step,
}

/// Per-element instantaneous state at an accepted solution point:
/// `v = v_p − v_n`, `i` = current p→n through the element.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct ElemState {
    pub v: f64,
    pub i: f64,
}

/// Per-element Newton linearization point. Diodes use `v1` (the junction
/// voltage); MOSFETs use the full terminal triple `(v1, v2, v3) =
/// (v_g, v_d, v_s)` — EKV currents depend on the potentials individually,
/// not just on differences (implicit bulk at ground). Linear elements
/// ignore it.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct NlLin {
    pub v1: f64,
    pub v2: f64,
    pub v3: f64,
}

/// Static analysis of a netlist for one assembly mode: node count, total
/// system dimension, and the branch-row index (absolute into `x`) per element.
pub(crate) struct Topology {
    pub n_nodes: usize,
    pub dim: usize,
    pub branch: Vec<Option<usize>>,
}

impl Topology {
    pub fn build(net: &Netlist, mode: Mode) -> Result<Self, CircuitError> {
        let n_nodes = validate(net)?;
        let mut branch = vec![None; net.elements.len()];
        let mut nb = 0;
        for (k, el) in net.elements.iter().enumerate() {
            let needs = match el {
                Element::VoltageSource { .. } => true,
                Element::Inductor { .. } => matches!(mode, Mode::DcOp | Mode::Step),
                Element::Capacitor { .. } => mode == Mode::Init,
                _ => false,
            };
            if needs {
                branch[k] = Some(n_nodes + nb);
                nb += 1;
            }
        }
        Ok(Self {
            n_nodes,
            dim: n_nodes + nb,
            branch,
        })
    }
}

/// Structural validation. Returns the non-ground node count N (= max id).
///
/// Checks: non-empty, unique non-empty names, p ≠ n, positive element
/// values, ground referenced somewhere, and node ids contiguous in 1..=N
/// (a skipped id would produce an all-zero KCL row → guaranteed-singular).
pub(crate) fn validate(net: &Netlist) -> Result<usize, CircuitError> {
    if net.elements.is_empty() {
        return Err(CircuitError::Invalid("empty netlist".into()));
    }
    let mut names = std::collections::BTreeSet::new();
    let mut max_node: Node = 0;
    let mut touches_ground = false;
    for el in &net.elements {
        let name = el.name();
        if name.is_empty() {
            return Err(CircuitError::Invalid("element with empty name".into()));
        }
        if !names.insert(name.to_string()) {
            return Err(CircuitError::Invalid(format!(
                "duplicate element name '{name}'"
            )));
        }
        let (p, n) = el.nodes();
        if p == n {
            return Err(CircuitError::Invalid(format!(
                "element '{name}': both terminals on node {p}"
            )));
        }
        max_node = max_node.max(p).max(n);
        touches_ground |= p == 0 || n == 0;
        if let Some(g) = el.gate() {
            // The gate may coincide with drain or source (diode-connected);
            // only d == s is degenerate, and that is the p == n check above.
            max_node = max_node.max(g);
            touches_ground |= g == 0;
        }
        let positive = |v: f64, what: &str| -> Result<(), CircuitError> {
            if v > 0.0 && v.is_finite() {
                Ok(())
            } else {
                Err(CircuitError::Invalid(format!(
                    "element '{name}': {what} must be positive and finite, got {v}"
                )))
            }
        };
        match el {
            Element::Resistor { r, .. } => positive(*r, "R")?,
            Element::Capacitor { c, .. } => positive(*c, "C")?,
            Element::Inductor { l, .. } => positive(*l, "L")?,
            Element::Diode {
                i_s, n_ideality, ..
            } => {
                positive(*i_s, "I_s")?;
                positive(*n_ideality, "n")?;
            }
            Element::Mosfet {
                vth, kp, lambda, ..
            } => {
                positive(*kp, "kp")?;
                if !vth.is_finite() || *vth < 0.0 {
                    return Err(CircuitError::Invalid(format!(
                        "element '{name}': vth must be finite and ≥ 0 (threshold magnitude), got {vth}"
                    )));
                }
                if !lambda.is_finite() || *lambda < 0.0 {
                    return Err(CircuitError::Invalid(format!(
                        "element '{name}': lambda must be finite and ≥ 0, got {lambda}"
                    )));
                }
            }
            Element::MosfetEkv {
                vth, kp, n, phi_t, ..
            } => {
                positive(*kp, "kp")?;
                positive(*n, "n")?;
                positive(*phi_t, "phi_t")?;
                if !vth.is_finite() || *vth < 0.0 {
                    return Err(CircuitError::Invalid(format!(
                        "element '{name}': vth must be finite and ≥ 0 (threshold magnitude), got {vth}"
                    )));
                }
            }
            Element::VoltageSource { .. } | Element::CurrentSource { .. } => {}
        }
    }
    if !touches_ground {
        return Err(CircuitError::Invalid(
            "no element references ground (node 0)".into(),
        ));
    }
    let mut seen = vec![false; max_node + 1];
    for el in &net.elements {
        let (p, n) = el.nodes();
        seen[p] = true;
        seen[n] = true;
        if let Some(g) = el.gate() {
            seen[g] = true;
        }
    }
    if let Some(gap) = (1..=max_node).find(|&k| !seen[k]) {
        return Err(CircuitError::Invalid(format!(
            "node ids must be contiguous: node {gap} is unused (max node is {max_node})"
        )));
    }
    Ok(max_node)
}

#[inline]
fn node_idx(k: Node) -> Option<usize> {
    if k == 0 { None } else { Some(k - 1) }
}

/// Resistor-style conductance stamp between (possibly grounded) terminals.
fn stamp_g<S: MatrixSink>(a: &mut S, ip: Option<usize>, inn: Option<usize>, g: f64) {
    if let Some(i) = ip {
        a.add(i, i, g);
    }
    if let Some(j) = inn {
        a.add(j, j, g);
    }
    if let (Some(i), Some(j)) = (ip, inn) {
        a.add(i, j, -g);
        a.add(j, i, -g);
    }
}

/// RHS stamp for a constant current `j_pn` flowing p→n *through* the element:
/// it leaves node p and enters node n.
fn inject(b: &mut [f64], ip: Option<usize>, inn: Option<usize>, j_pn: f64) {
    if let Some(i) = ip {
        b[i] -= j_pn;
    }
    if let Some(j) = inn {
        b[j] += j_pn;
    }
}

/// Extended-MNA voltage-constraint row `rb`: v_p − v_n = val, with the branch
/// current (p→n) entering both KCL columns.
fn stamp_vrow<S: MatrixSink>(
    a: &mut S,
    b: &mut [f64],
    ip: Option<usize>,
    inn: Option<usize>,
    rb: usize,
    val: f64,
) {
    if let Some(i) = ip {
        a.add(rb, i, 1.0);
        a.add(i, rb, 1.0);
    }
    if let Some(j) = inn {
        a.add(rb, j, -1.0);
        a.add(j, rb, -1.0);
    }
    b[rb] = val;
}

/// Diode small-signal companion (g, I_eq) about linearization voltage `v*`,
/// GMIN folded in. See module docs for the derivation.
fn diode_companion(i_s: f64, n_ideality: f64, v_star: f64) -> (f64, f64) {
    let nvt = n_ideality * VT_300K;
    let e = (v_star / nvt).exp();
    let g = i_s / nvt * e + GMIN;
    let i_star = i_s * (e - 1.0) + GMIN * v_star;
    (g, i_star - g * v_star)
}

/// SPICE3 junction limiter (Nagel): damp the Newton update of a pn-junction
/// voltage so the exponential stays evaluable.
fn pnjlim(vnew: f64, vold: f64, vt: f64, vcrit: f64) -> f64 {
    if vnew > vcrit && (vnew - vold).abs() > 2.0 * vt {
        if vold > 0.0 {
            let arg = 1.0 + (vnew - vold) / vt;
            if arg > 0.0 {
                vold + vt * arg.ln()
            } else {
                vcrit
            }
        } else {
            vt * (vnew / vt).ln()
        }
    } else {
        vnew
    }
}

/// Critical voltage for `pnjlim`: the knee beyond which steps are damped.
fn vcrit(i_s: f64, n_ideality: f64) -> f64 {
    let nvt = n_ideality * VT_300K;
    nvt * (nvt / (std::f64::consts::SQRT_2 * i_s)).ln()
}

/// Initial Newton linearization points: diodes start at vcrit (the SPICE
/// start), MOSFETs at all-zero terminal potentials (cutoff — GMIN keeps the
/// Jacobian regular), everything else at zero.
pub(crate) fn initial_lin(net: &Netlist) -> Vec<NlLin> {
    net.elements
        .iter()
        .map(|el| match el {
            Element::Diode {
                i_s, n_ideality, ..
            } => NlLin {
                v1: vcrit(*i_s, *n_ideality),
                ..NlLin::default()
            },
            _ => NlLin::default(),
        })
        .collect()
}

/// Level-1 core in the forward orientation (v_ds ≥ 0): returns
/// (i_d, ∂i/∂v_gs, ∂i/∂v_ds). The (1 + λ·v_ds) factor applies in both
/// conducting regions, so i and both partials are continuous at the
/// triode/saturation boundary v_ds = v_gs − vth.
fn mos1_core(vth: f64, kp: f64, lambda: f64, vgs: f64, vds: f64) -> (f64, f64, f64) {
    let vov = vgs - vth;
    if vov <= 0.0 {
        return (0.0, 0.0, 0.0); // cutoff
    }
    let cl = 1.0 + lambda * vds;
    if vds < vov {
        // Triode.
        let q = vov * vds - 0.5 * vds * vds;
        (
            kp * q * cl,
            kp * vds * cl,
            kp * (vov - vds) * cl + kp * q * lambda,
        )
    } else {
        // Saturation.
        let half = 0.5 * kp * vov * vov;
        (half * cl, kp * vov * cl, half * lambda)
    }
}

/// Level-1 NMOS at absolute terminal potentials: (i_ds, ∂i/∂v_g, ∂i/∂v_d,
/// ∂i/∂v_s), current oriented d→s. The device is source/drain symmetric:
/// v_d < v_s swaps the roles (i = −f(v_gd, v_sd)), which is what makes
/// transmission gates work without case analysis at the call site.
fn mos1_nmos(vth: f64, kp: f64, lambda: f64, vg: f64, vd: f64, vs: f64) -> (f64, f64, f64, f64) {
    if vd >= vs {
        let (i, gm, gds) = mos1_core(vth, kp, lambda, vg - vs, vd - vs);
        (i, gm, gds, -(gm + gds))
    } else {
        let (i, gm, gds) = mos1_core(vth, kp, lambda, vg - vd, vs - vd);
        (-i, -gm, gm + gds, -gds)
    }
}

/// Numerically stable softplus ln(1 + e^x) and the logistic sigmoid.
fn softplus(x: f64) -> f64 {
    if x > 0.0 {
        x + (-x).exp().ln_1p()
    } else {
        x.exp().ln_1p()
    }
}
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// EKV interpolation function F(u) = ln²(1 + e^{u/(2φt)}) and its derivative
/// F′(u) = ln(1 + e^{u/(2φt)})·σ(u/(2φt))/φt.
fn ekv_f(u: f64, phi_t: f64) -> (f64, f64) {
    let x = u / (2.0 * phi_t);
    let sp = softplus(x);
    (sp * sp, sp * sigmoid(x) / phi_t)
}

/// EKV-lite NMOS at absolute terminal potentials: (i_ds, ∂i/∂v_g, ∂i/∂v_d,
/// ∂i/∂v_s). The forward−reverse form is d/s symmetric by construction.
/// Note Σ ∂i/∂v_x ≠ 0 for n ≠ 1 — the implicit bulk at ground absorbs the
/// difference, so the companion I_eq must use absolute potentials.
fn ekv_nmos(
    vth: f64,
    kp: f64,
    n: f64,
    phi_t: f64,
    vg: f64,
    vd: f64,
    vs: f64,
) -> (f64, f64, f64, f64) {
    let i_spec = 2.0 * n * kp * phi_t * phi_t;
    let vp = (vg - vth) / n;
    let (ff, dff) = ekv_f(vp - vs, phi_t);
    let (fr, dfr) = ekv_f(vp - vd, phi_t);
    (
        i_spec * (ff - fr),
        i_spec * (dff - dfr) / n,
        i_spec * dfr,
        -i_spec * dff,
    )
}

/// Channel current + partials of either MOSFET model at absolute terminal
/// potentials, with the polarity mirror and GMIN (d↔s) folded in. PMOS is
/// the exact mirror i_P(v) = −i_N(−v); by the chain rule the partials carry
/// over unnegated. Used identically by the Newton companion stamps and the
/// post-solve state reconstruction, which is what keeps the discrete
/// Tellegen identity intact through the nonlinear elements.
fn mos_eval(el: &Element, vg: f64, vd: f64, vs: f64) -> (f64, f64, f64, f64) {
    let pol = match el {
        Element::Mosfet { polarity, .. } | Element::MosfetEkv { polarity, .. } => *polarity,
        _ => unreachable!("mos_eval called on a non-MOSFET element"),
    };
    let (xg, xd, xs) = match pol {
        MosPolarity::Nmos => (vg, vd, vs),
        MosPolarity::Pmos => (-vg, -vd, -vs),
    };
    let (i, ag, ad, asrc) = match el {
        Element::Mosfet {
            vth, kp, lambda, ..
        } => mos1_nmos(*vth, *kp, *lambda, xg, xd, xs),
        Element::MosfetEkv {
            vth, kp, n, phi_t, ..
        } => ekv_nmos(*vth, *kp, *n, *phi_t, xg, xd, xs),
        _ => unreachable!(),
    };
    let i = match pol {
        MosPolarity::Nmos => i,
        MosPolarity::Pmos => -i,
    };
    (i + GMIN * (vd - vs), ag, ad + GMIN, asrc - GMIN)
}

/// Assemble the dense MNA system `A·x = b` for one solve (the small-circuit
/// path; see [`assemble_into`] for the shared stamping).
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    net: &Netlist,
    topo: &Topology,
    mode: Mode,
    method: Method,
    t: f64,
    h: f64,
    src_scale: f64,
    hist: &[ElemState],
    lin: &[NlLin],
) -> (Mat, Vec<f64>) {
    let mut a = Mat::zeros(topo.dim, topo.dim);
    let mut b = vec![0.0; topo.dim];
    assemble_into(
        net, topo, mode, method, t, h, src_scale, hist, lin, &mut a, &mut b,
    );
    (a, b)
}

/// Assemble the MNA system `A·x = b` for one solve into a caller-provided
/// [`MatrixSink`] and RHS.
///
/// `t` — time at which sources are evaluated (the *new* time point; the old
/// source value enters through the companion history, which is exactly what
/// makes the overall scheme trapezoidal in the input as well).
/// `h` — step size (Step mode only). `src_scale` — source-stepping λ.
/// `hist` — previous-step element states (Step mode only).
/// `lin` — per-element nonlinear linearization points (Newton iterate).
///
/// The element walk is deterministic, so re-assembling the same
/// (netlist, topology, mode) emits stamps in an identical sequence — the
/// invariant the sparse path's triplet→slot map relies on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble_into<S: MatrixSink>(
    net: &Netlist,
    topo: &Topology,
    mode: Mode,
    method: Method,
    t: f64,
    h: f64,
    src_scale: f64,
    hist: &[ElemState],
    lin: &[NlLin],
    a: &mut S,
    b: &mut [f64],
) {
    for (k, el) in net.elements.iter().enumerate() {
        let (p, n) = el.nodes();
        let (ip, inn) = (node_idx(p), node_idx(n));
        match el {
            Element::Resistor { r, .. } => stamp_g(a, ip, inn, 1.0 / r),
            Element::Capacitor { c, ic, .. } => match mode {
                Mode::DcOp => {} // open circuit
                Mode::Init => {
                    stamp_vrow(a, b, ip, inn, topo.branch[k].unwrap(), *ic);
                }
                Mode::Step => {
                    let geq = match method {
                        Method::Trapezoidal => 2.0 * c / h,
                        Method::BackwardEuler => c / h,
                    };
                    let i_hist = match method {
                        Method::Trapezoidal => geq * hist[k].v + hist[k].i,
                        Method::BackwardEuler => geq * hist[k].v,
                    };
                    stamp_g(a, ip, inn, geq);
                    // i_new = G_eq·v − I_hist: the constant −I_hist is a
                    // current source p→n of value −I_hist.
                    inject(b, ip, inn, -i_hist);
                }
            },
            Element::Inductor { l, ic, .. } => match mode {
                Mode::DcOp => {
                    // Short: v_p − v_n = 0, branch current unknown.
                    stamp_vrow(a, b, ip, inn, topo.branch[k].unwrap(), 0.0);
                }
                Mode::Init => inject(b, ip, inn, *ic),
                Mode::Step => {
                    let rb = topo.branch[k].unwrap();
                    let coef = match method {
                        Method::Trapezoidal => h / (2.0 * l),
                        Method::BackwardEuler => h / l,
                    };
                    a.add(rb, rb, 1.0);
                    if let Some(i) = ip {
                        a.add(rb, i, -coef);
                        a.add(i, rb, 1.0);
                    }
                    if let Some(j) = inn {
                        a.add(rb, j, coef);
                        a.add(j, rb, -1.0);
                    }
                    b[rb] = hist[k].i
                        + match method {
                            Method::Trapezoidal => coef * hist[k].v,
                            Method::BackwardEuler => 0.0,
                        };
                }
            },
            Element::VoltageSource { wave, .. } => {
                stamp_vrow(
                    a,
                    b,
                    ip,
                    inn,
                    topo.branch[k].unwrap(),
                    src_scale * wave.at(t),
                );
            }
            Element::CurrentSource { wave, .. } => {
                inject(b, ip, inn, src_scale * wave.at(t));
            }
            Element::Diode {
                i_s, n_ideality, ..
            } => {
                let (g, i_eq) = diode_companion(*i_s, *n_ideality, lin[k].v1);
                stamp_g(a, ip, inn, g);
                inject(b, ip, inn, i_eq);
            }
            Element::Mosfet { g, .. } | Element::MosfetEkv { g, .. } => {
                // Channel current d→s linearized about the previous iterate
                // (see module docs). ip/inn are the drain/source rows; the
                // gate contributes a column only. The emitted stamp sequence
                // is the same every call (entries may be 0.0 in cutoff) —
                // the invariant the sparse triplet→slot map relies on.
                let l = &lin[k];
                let (i, ag, ad, asrc) = mos_eval(el, l.v1, l.v2, l.v3);
                let ig = node_idx(*g);
                for (col, coef) in [(ig, ag), (ip, ad), (inn, asrc)] {
                    if let Some(c) = col {
                        if let Some(r) = ip {
                            a.add(r, c, coef);
                        }
                        if let Some(r) = inn {
                            a.add(r, c, -coef);
                        }
                    }
                }
                let i_eq = i - ag * l.v1 - ad * l.v2 - asrc * l.v3;
                inject(b, ip, inn, i_eq);
            }
        }
    }
}

/// Solve one MNA system. Linear netlists are a single assemble + solve;
/// netlists with nonlinear devices (diodes, MOSFETs) run Newton-Raphson,
/// warm-started from (and updating) `lin` so consecutive transient steps
/// converge in a couple of iterations. Diode iterates are damped with
/// `pnjlim`; MOSFET iterates are taken raw (both models are polynomial /
/// softplus-smooth). The `solver` carries the dense-vs-sparse decision and,
/// on the sparse path, the cached pivot order + fill pattern reused across
/// calls — every Newton iteration is a numeric-only refactor.
#[allow(clippy::too_many_arguments)]
pub(crate) fn solve(
    net: &Netlist,
    topo: &Topology,
    mode: Mode,
    method: Method,
    t: f64,
    h: f64,
    src_scale: f64,
    hist: &[ElemState],
    lin: &mut [NlLin],
    solver: &mut SystemSolver,
) -> Result<Vec<f64>, CircuitError> {
    let nonlinear = net.elements.iter().any(Element::is_nonlinear);
    if !nonlinear {
        return solver.solve_system(net, topo, mode, method, t, h, src_scale, hist, lin);
    }
    let vn = |x: &[f64], node: Node| if node == 0 { 0.0 } else { x[node - 1] };
    let mut x_prev: Option<Vec<f64>> = None;
    for _ in 0..NR_MAX_ITERS {
        let x = solver.solve_system(net, topo, mode, method, t, h, src_scale, hist, lin)?;
        let mut limited = false;
        for (k, el) in net.elements.iter().enumerate() {
            match el {
                Element::Diode {
                    i_s, n_ideality, ..
                } => {
                    let (p, n) = el.nodes();
                    let vd = vn(&x, p) - vn(&x, n);
                    let nvt = n_ideality * VT_300K;
                    let vlim = pnjlim(vd, lin[k].v1, nvt, vcrit(*i_s, *n_ideality));
                    if (vlim - vd).abs() > 1e-15 {
                        limited = true;
                    }
                    lin[k].v1 = vlim;
                }
                Element::Mosfet { .. } | Element::MosfetEkv { .. } => {
                    let (d, s) = el.nodes();
                    let g = el.gate().unwrap();
                    lin[k] = NlLin {
                        v1: vn(&x, g),
                        v2: vn(&x, d),
                        v3: vn(&x, s),
                    };
                }
                _ => {}
            }
        }
        let converged = x_prev.as_ref().is_some_and(|xp| {
            x.iter()
                .zip(xp)
                .all(|(a, b)| (a - b).abs() <= NR_ABSTOL + NR_RELTOL * a.abs().max(b.abs()))
        });
        if converged && !limited {
            return Ok(x);
        }
        x_prev = Some(x);
    }
    Err(CircuitError::NoConvergence)
}

/// Reconstruct every element's (v, i) from a solved `x`. The reconstruction
/// uses exactly the currents the stamps imposed (companion relations, branch
/// unknowns, source waveform values), which is what makes Σ i·v vanish to LU
/// precision at every accepted point — the discrete Tellegen identity the
/// energy ledger is built on.
#[allow(clippy::too_many_arguments)]
pub(crate) fn element_states(
    net: &Netlist,
    topo: &Topology,
    mode: Mode,
    method: Method,
    t: f64,
    h: f64,
    src_scale: f64,
    hist: &[ElemState],
    x: &[f64],
) -> Vec<ElemState> {
    let vn = |node: Node| if node == 0 { 0.0 } else { x[node - 1] };
    net.elements
        .iter()
        .enumerate()
        .map(|(k, el)| {
            let (p, n) = el.nodes();
            let v = vn(p) - vn(n);
            let i = match el {
                Element::Resistor { r, .. } => v / r,
                Element::Capacitor { c, .. } => match mode {
                    Mode::DcOp => 0.0,
                    Mode::Init => x[topo.branch[k].unwrap()],
                    Mode::Step => {
                        let geq = match method {
                            Method::Trapezoidal => 2.0 * c / h,
                            Method::BackwardEuler => c / h,
                        };
                        let i_hist = match method {
                            Method::Trapezoidal => geq * hist[k].v + hist[k].i,
                            Method::BackwardEuler => geq * hist[k].v,
                        };
                        geq * v - i_hist
                    }
                },
                Element::Inductor { ic, .. } => match mode {
                    Mode::Init => *ic,
                    _ => x[topo.branch[k].unwrap()],
                },
                Element::VoltageSource { .. } => x[topo.branch[k].unwrap()],
                Element::CurrentSource { wave, .. } => src_scale * wave.at(t),
                Element::Diode {
                    i_s, n_ideality, ..
                } => i_s * ((v / (n_ideality * VT_300K)).exp() - 1.0) + GMIN * v,
                Element::Mosfet { g, .. } | Element::MosfetEkv { g, .. } => {
                    // Channel current at the converged potentials. The
                    // converged companion current differs from this by
                    // O(Δx²) ≪ the Tellegen tolerance; the gate carries
                    // none, so p = i·v_ds is the device's full power.
                    mos_eval(el, vn(*g), vn(p), vn(n)).0
                }
            };
            ElemState { v, i }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::netlist::Waveform;

    /// Duplicate names and node-id gaps are rejected up front.
    #[test]
    fn validation_rejects_bad_netlists() {
        let mut net = Netlist::new();
        net.resistor("r", 1, 0, 1.0).resistor("r", 1, 0, 2.0);
        assert!(matches!(validate(&net), Err(CircuitError::Invalid(_))));

        let mut net = Netlist::new();
        net.vsource("v", 1, 0, Waveform::Dc { v: 1.0 })
            .resistor("r", 1, 3, 1.0)
            .resistor("r2", 3, 0, 1.0);
        assert!(matches!(validate(&net), Err(CircuitError::Invalid(_))));

        let mut net = Netlist::new();
        net.resistor("r", 1, 2, 1.0); // never touches ground
        assert!(matches!(validate(&net), Err(CircuitError::Invalid(_))));
    }
}
