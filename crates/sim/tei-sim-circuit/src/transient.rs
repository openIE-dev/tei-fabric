//! DC operating point and the fixed-step transient loop with per-element
//! energy instrumentation (the M3 contract).
//!
//! # Initialization
//!
//! The transient starts from the elements' initial conditions (SPICE `UIC`
//! semantics): at t = 0 the network is solved once with every capacitor
//! imposed as a voltage constraint at its `ic` and every inductor as a
//! current source at its `ic`. The capacitor constraint rows return the
//! exact t = 0⁺ capacitor currents as branch unknowns — precisely the history
//! the first trapezoidal companion needs, and the t = 0 sample the energy
//! trapezoid-rule integration starts from.
//!
//! # Energy accounting
//!
//! Each element accumulates absorbed energy by the trapezoid rule on its
//! instantaneous power p(t) = i(t)·v(t):
//!
//! ```text
//! E_k += h/2 · (p_k(t) + p_k(t+h))
//! ```
//!
//! Because every reconstructed element current is exactly the current its MNA
//! stamp imposed, KCL holds exactly at every accepted point and Tellegen's
//! theorem applies discretely: Σ_k p_k(t) = 0 to LU precision, hence
//! Σ_k E_k = 0 — sources deliver exactly what resistors dissipate plus what
//! reactive elements absorb, with no leakage from the integrator. The result
//! reports the per-step worst-case residual (`tellegen_max`) so the identity
//! is observable, plus the physically grouped totals.

use crate::mna::{self, ElemState, Mode, Topology};
use crate::netlist::{Element, Netlist, Node};
use crate::solver::{SolverChoice, SolverKind, SystemSolver};
use crate::{CircuitError, Method};
use serde::Serialize;
use tei_sim_core::exec::Progress;

/// Options for a transient run. `t_stop` is snapped to the step grid
/// (`n = round(t_stop/dt)` steps of exactly `dt`). `store_stride` downsamples
/// the stored traces (energy accumulation always uses every step).
/// `solver` picks the linear-system path (default `Auto`: dense for small
/// cells, sparse Markowitz LU with pattern-reuse refactor for large ones).
#[derive(Debug, Clone)]
pub struct TransientOpts {
    pub t_stop: f64,
    pub dt: f64,
    pub method: Method,
    pub store_stride: usize,
    pub solver: SolverChoice,
}

impl TransientOpts {
    pub fn new(t_stop: f64, dt: f64) -> Self {
        Self {
            t_stop,
            dt,
            method: Method::default(),
            store_stride: 1,
            solver: SolverChoice::default(),
        }
    }
}

/// Result of a transient run.
#[derive(Debug, Clone, Serialize)]
pub struct TransientResult {
    /// Stored sample times (downsampled by `store_stride`; always includes
    /// t = 0 and the final point).
    pub t: Vec<f64>,
    /// Node-voltage traces: `v[id − 1][sample]` for nodes 1..=N.
    pub v: Vec<Vec<f64>>,
    /// Absorbed energy ∫ i·v dt per element, in netlist order.
    pub element_energy: Vec<(String, f64)>,
    /// Energy delivered by all sources (= −Σ absorbed over V/I sources) [J].
    pub source_energy: f64,
    /// Energy dissipated in resistors and diodes [J].
    pub dissipated_energy: f64,
    /// ∫ i·v dt absorbed by capacitors and inductors (trapezoid rule) [J].
    /// Equals `source_energy − dissipated_energy` to LU precision (Tellegen).
    pub reactive_absorbed_energy: f64,
    /// Δ(½CV² + ½LI²) between the final and initial states [J]. Agrees with
    /// `reactive_absorbed_energy` to the integrator's O(h²).
    pub delta_stored_energy: f64,
    /// max_t |Σ_k i_k·v_k| — the discrete-Tellegen residual [W]; ~LU epsilon.
    pub tellegen_max: f64,
    /// Steps taken.
    pub steps: u64,
    /// Step size actually used.
    pub dt: f64,
    /// Linear-solver path the stepping ran through (dense vs sparse).
    pub solver: SolverKind,
}

impl TransientResult {
    /// Absorbed energy of the named element, if present.
    pub fn energy(&self, name: &str) -> Option<f64> {
        self.element_energy
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, e)| *e)
    }
}

/// DC operating point: node voltages plus the branch currents of voltage
/// sources and inductors (capacitors open, inductors short).
#[derive(Debug, Clone, Serialize)]
pub struct DcSolution {
    /// Node potentials, `v[id − 1]` for nodes 1..=N.
    pub v: Vec<f64>,
    /// (name, current p→n) for every voltage source and inductor.
    pub branch_currents: Vec<(String, f64)>,
}

impl DcSolution {
    pub fn node(&self, id: Node) -> f64 {
        if id == 0 { 0.0 } else { self.v[id - 1] }
    }

    pub fn branch_current(&self, name: &str) -> Option<f64> {
        self.branch_currents
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, i)| *i)
    }
}

/// Solve the DC operating point (sources at their t = 0 values).
///
/// Linear circuits solve in one LU; circuits with nonlinear devices (diodes,
/// MOSFETs) run Newton-Raphson — `pnjlim`-damped on the diode junctions —
/// falling back to source stepping (λ swept 0.1 → 1, nonlinear
/// linearizations warm-started across λ) if plain Newton stalls — the
/// roadmap's day-one mitigation for stiff operating points.
pub fn solve_dc(net: &Netlist) -> Result<DcSolution, CircuitError> {
    let topo = Topology::build(net, Mode::DcOp)?;
    let hist = vec![ElemState::default(); net.elements.len()];
    let mut lin = mna::initial_lin(net);
    let mut solver = SystemSolver::new(topo.n_nodes, SolverChoice::Auto);
    let solve_at = |scale: f64, lin: &mut [mna::NlLin], solver: &mut SystemSolver| {
        mna::solve(
            net,
            &topo,
            Mode::DcOp,
            Method::Trapezoidal,
            0.0,
            1.0,
            scale,
            &hist,
            lin,
            solver,
        )
    };
    let x = match solve_at(1.0, &mut lin, &mut solver) {
        Ok(x) => x,
        Err(CircuitError::NoConvergence) => {
            let mut lin = mna::initial_lin(net);
            let mut last = Err(CircuitError::NoConvergence);
            for step in 1..=10 {
                last = solve_at(step as f64 / 10.0, &mut lin, &mut solver);
                if last.is_err() {
                    break;
                }
            }
            last?
        }
        Err(e) => return Err(e),
    };
    let mut branch_currents = Vec::new();
    for (k, el) in net.elements.iter().enumerate() {
        if let (Some(rb), Element::VoltageSource { .. } | Element::Inductor { .. }) =
            (topo.branch[k], el)
        {
            branch_currents.push((el.name().to_string(), x[rb]));
        }
    }
    Ok(DcSolution {
        v: x[..topo.n_nodes].to_vec(),
        branch_currents,
    })
}

/// Repeated-DC solver for **linear** netlists with a fixed matrix and varying
/// voltage-source values — the factor-once / solve-many fast path behind the
/// crossbar's exact IR-drop mode (roadmap §2 cross-crate flow:
/// "`tei-sim-circuit` provides the exact IR-drop mode for `tei-sim-crossbar`").
///
/// In MNA, element values enter the **matrix** while voltage-source values
/// enter only the **RHS** (the constraint row reads `v_p − v_n = V`). So for
/// a netlist whose elements never change between solves — a programmed
/// crossbar tile, where only the row drive voltages vary per query — the
/// sparse LU is factored once at construction and every later [`Self::solve`]
/// just rewrites the source rows of the cached base RHS and runs the two
/// triangular substitutions. No reassembly, no refactorization.
///
/// Restrictions (deliberate, keeps the export minimal): linear elements only —
/// a diode would make the matrix iterate-dependent. Capacitors are open and
/// inductors short, exactly as in [`solve_dc`]. The sparse path is used
/// unconditionally: the dense path's whole reason to exist is bit-stability
/// of the tiny pre-M4 validation cells, which never come through here.
#[derive(Debug, Clone)]
pub struct LinearDcSolver {
    n_nodes: usize,
    lu: tei_sim_core::sparse::SparseLu,
    /// Base RHS assembled once from the netlist's own t = 0 source values.
    b0: Vec<f64>,
    /// Branch-row index of each voltage source, in element order.
    vrows: Vec<usize>,
    /// (name, branch row) of every element carrying a branch unknown.
    branches: Vec<(String, usize)>,
}

impl LinearDcSolver {
    /// Validate, assemble, and factor. Errors mirror [`solve_dc`]:
    /// `Invalid` for malformed/nonlinear netlists, `Singular` if the MNA
    /// matrix cannot be factored.
    pub fn new(net: &Netlist) -> Result<Self, CircuitError> {
        if net.elements.iter().any(Element::is_nonlinear) {
            return Err(CircuitError::Invalid(
                "LinearDcSolver requires a linear netlist (no diodes or MOSFETs)".into(),
            ));
        }
        let topo = Topology::build(net, Mode::DcOp)?;
        let hist = vec![ElemState::default(); net.elements.len()];
        let lin = mna::initial_lin(net);
        let mut triplets: Vec<(u32, u32, f64)> = Vec::new();
        let mut b0 = vec![0.0; topo.dim];
        mna::assemble_into(
            net,
            &topo,
            Mode::DcOp,
            Method::Trapezoidal,
            0.0,
            1.0,
            1.0,
            &hist,
            &lin,
            &mut triplets,
            &mut b0,
        );
        let (csr, _map) =
            tei_sim_core::sparse::Csr::from_triplets_with_map(topo.dim, topo.dim, &triplets);
        let lu =
            tei_sim_core::sparse::SparseLu::factor(&csr).map_err(|_| CircuitError::Singular)?;
        let mut vrows = Vec::new();
        let mut branches = Vec::new();
        for (k, el) in net.elements.iter().enumerate() {
            if let Some(rb) = topo.branch[k] {
                branches.push((el.name().to_string(), rb));
                if matches!(el, Element::VoltageSource { .. }) {
                    vrows.push(rb);
                }
            }
        }
        Ok(Self {
            n_nodes: topo.n_nodes,
            lu,
            b0,
            vrows,
            branches,
        })
    }

    /// Non-ground node count N (solutions index nodes 1..=N).
    pub fn n_nodes(&self) -> usize {
        self.n_nodes
    }

    /// Number of voltage sources = the length [`Self::solve`] expects.
    pub fn n_vsources(&self) -> usize {
        self.vrows.len()
    }

    /// Solve the operating point with the k-th voltage source (in netlist
    /// element order) set to `v_src[k]`, reusing the cached factorization.
    /// Cost per call: one RHS rewrite + two sparse triangular substitutions.
    pub fn solve(&self, v_src: &[f64]) -> DcSolution {
        assert_eq!(
            v_src.len(),
            self.vrows.len(),
            "one value per voltage source"
        );
        let mut b = self.b0.clone();
        for (&rb, &v) in self.vrows.iter().zip(v_src) {
            b[rb] = v;
        }
        let mut x = self.lu.solve(&b);
        let branch_currents = self
            .branches
            .iter()
            .map(|(n, rb)| (n.clone(), x[*rb]))
            .collect();
        x.truncate(self.n_nodes);
        DcSolution {
            v: x,
            branch_currents,
        }
    }
}

/// Σ ½CV² + ½LI² over the reactive elements at a state snapshot.
fn stored_energy(net: &Netlist, states: &[ElemState]) -> f64 {
    net.elements
        .iter()
        .zip(states)
        .map(|(el, s)| match el {
            Element::Capacitor { c, .. } => 0.5 * c * s.v * s.v,
            Element::Inductor { l, .. } => 0.5 * l * s.i * s.i,
            _ => 0.0,
        })
        .sum()
}

/// Run a transient analysis (no progress reporting).
pub fn transient(net: &Netlist, opts: &TransientOpts) -> Result<TransientResult, CircuitError> {
    transient_with_progress(net, opts, &mut |_| {})
}

/// Run a transient analysis, reporting progress roughly every 1% of steps.
pub fn transient_with_progress(
    net: &Netlist,
    opts: &TransientOpts,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<TransientResult, CircuitError> {
    if !opts.dt.is_finite() || opts.dt <= 0.0 {
        return Err(CircuitError::Invalid(format!(
            "dt must be positive and finite, got {}",
            opts.dt
        )));
    }
    if !opts.t_stop.is_finite() || opts.t_stop < opts.dt {
        return Err(CircuitError::Invalid(format!(
            "t_stop must be finite and ≥ dt, got {}",
            opts.t_stop
        )));
    }
    let n_steps_f = (opts.t_stop / opts.dt).round();
    if n_steps_f > 2e8 {
        return Err(CircuitError::Invalid(format!(
            "{n_steps_f:.0} steps exceeds the fixed-step budget; raise dt"
        )));
    }
    let n_steps = (n_steps_f as u64).max(1);
    let h = opts.dt;
    let stride = opts.store_stride.max(1) as u64;
    let method = opts.method;
    let n_elems = net.elements.len();

    let topo = Topology::build(net, Mode::Step)?;
    let topo_init = Topology::build(net, Mode::Init)?;
    let mut lin = mna::initial_lin(net);
    // One solver per assembly mode: Init and Step systems differ in
    // dimension and sparsity pattern, so each caches its own factorization.
    let mut init_solver = SystemSolver::new(topo_init.n_nodes, opts.solver);
    let mut step_solver = SystemSolver::new(topo.n_nodes, opts.solver);

    // t = 0: ICs imposed exactly; capacitor branch currents = i_C(0⁺).
    let dummy = vec![ElemState::default(); n_elems];
    let x0 = mna::solve(
        net,
        &topo_init,
        Mode::Init,
        method,
        0.0,
        h,
        1.0,
        &dummy,
        &mut lin,
        &mut init_solver,
    )?;
    let mut states = mna::element_states(
        net,
        &topo_init,
        Mode::Init,
        method,
        0.0,
        h,
        1.0,
        &dummy,
        &x0,
    );
    let stored0 = stored_energy(net, &states);

    let mut t_samples = vec![0.0];
    let mut v_traces: Vec<Vec<f64>> = (0..topo.n_nodes).map(|i| vec![x0[i]]).collect();
    let mut energy = vec![0.0f64; n_elems];
    let mut p_old: Vec<f64> = states.iter().map(|s| s.v * s.i).collect();
    let mut tellegen_max = 0.0f64;
    let tick = (n_steps / 100).max(1);

    for k in 1..=n_steps {
        let t = k as f64 * h;
        let x = mna::solve(
            net,
            &topo,
            Mode::Step,
            method,
            t,
            h,
            1.0,
            &states,
            &mut lin,
            &mut step_solver,
        )?;
        let new_states =
            mna::element_states(net, &topo, Mode::Step, method, t, h, 1.0, &states, &x);
        let mut p_sum = 0.0;
        for (e, ns) in new_states.iter().enumerate() {
            let p_new = ns.v * ns.i;
            energy[e] += 0.5 * h * (p_old[e] + p_new);
            p_old[e] = p_new;
            p_sum += p_new;
        }
        tellegen_max = tellegen_max.max(p_sum.abs());
        states = new_states;
        if k % stride == 0 || k == n_steps {
            t_samples.push(t);
            for (i, trace) in v_traces.iter_mut().enumerate() {
                trace.push(x[i]);
            }
        }
        if k % tick == 0 || k == n_steps {
            on_progress(Progress {
                fraction: k as f64 / n_steps as f64,
                metrics: serde_json::json!({ "step": k, "t": t }),
            });
        }
    }

    let mut element_energy = Vec::with_capacity(n_elems);
    let (mut dissipated, mut src_absorbed, mut reactive) = (0.0, 0.0, 0.0);
    for (el, &e) in net.elements.iter().zip(&energy) {
        match el {
            Element::Resistor { .. }
            | Element::Diode { .. }
            | Element::Mosfet { .. }
            | Element::MosfetEkv { .. } => dissipated += e,
            Element::VoltageSource { .. } | Element::CurrentSource { .. } => src_absorbed += e,
            Element::Capacitor { .. } | Element::Inductor { .. } => reactive += e,
        }
        element_energy.push((el.name().to_string(), e));
    }

    Ok(TransientResult {
        t: t_samples,
        v: v_traces,
        element_energy,
        source_energy: -src_absorbed,
        dissipated_energy: dissipated,
        reactive_absorbed_energy: reactive,
        delta_stored_energy: stored_energy(net, &states) - stored0,
        tellegen_max,
        steps: n_steps,
        dt: h,
        solver: step_solver.kind(),
    })
}
