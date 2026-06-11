//! M4 validation — the sparse Markowitz-LU path against the dense path and
//! against closed-form ground truth, plus the roadmap §6 performance budget
//! as ignored benches (repo convention: `cargo test --release -- --ignored
//! bench`, as in tei-sim-stochastic).
//!
//! Coverage:
//!   - `sparse_transient_matches_dense_on_rc_ladders` (10/100/1000 nodes,
//!     ≤ 1e-10 agreement; 1000 release-only — the dense Init system there is
//!     dim ≈ 2N and O(n³))
//!   - `thousand_node_ladder_charges_monotonically_to_source` (analytic DC
//!     steady state + monotone charging, asserted through the sparse path)
//!   - `floating_island_is_singular_not_panic`
//!   - `bench_adiabatic_cell_budget_100_nodes_1e6_steps` (§6: < 10 s)
//!   - `bench_dense_vs_sparse_rc_ladders` (crossover + 500/1000-node wins;
//!     the numbers behind `SPARSE_NODE_THRESHOLD`)
//!
//! The power-clock chain generator is a local copy of the aa shift-register
//! testbench shape from tei-sim-adiabatic — circuit sits *below* adiabatic
//! in the dependency order and must not depend on it.

use tei_sim_circuit::{
    CircuitError, Method, Netlist, SolverChoice, SolverKind, TransientOpts, Waveform, transient,
};

const R: f64 = 1e3;
const C: f64 = 1e-9;
const V: f64 = 1.0;

/// N-node RC ladder: source at node 1, then a chain of series R with a C to
/// ground at every interior node — the classic diffusion line.
fn rc_ladder(n_nodes: usize) -> Netlist {
    assert!(n_nodes >= 2);
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: V });
    for k in 1..n_nodes {
        net.resistor(&format!("r{k}"), k, k + 1, R);
        net.capacitor(&format!("c{k}"), k + 1, 0, C, 0.0);
    }
    net
}

/// N-stage power-clock chain (2 nodes/stage): local copy of the
/// tei-sim-adiabatic shift-register testbench shape (4-phase trapezoid
/// clocks, switch-as-resistor stages).
fn power_clock_chain(n_stages: usize, t_ramp: f64, t_hold: f64) -> Netlist {
    let mut net = Netlist::new();
    for k in 0..n_stages {
        let phase = (k % 4) as f64;
        net.vsource(
            &format!("clk{k}"),
            2 * k + 1,
            0,
            Waveform::Trapezoid {
                v: V,
                t_delay: phase * (t_ramp + t_hold),
                t_rise: t_ramp,
                t_hold,
                t_fall: t_ramp,
            },
        )
        .resistor(&format!("ron{k}"), 2 * k + 1, 2 * k + 2, R)
        .capacitor(&format!("cload{k}"), 2 * k + 2, 0, C, 0.0);
    }
    net
}

fn run_ladder(n: usize, steps: usize, choice: SolverChoice) -> tei_sim_circuit::TransientResult {
    let dt = R * C / 10.0;
    let mut opts = TransientOpts::new(steps as f64 * dt, dt);
    opts.solver = choice;
    transient(&rc_ladder(n), &opts).unwrap()
}

/// Sparse solve matches dense solve to 1e-10 on the same MNA systems,
/// across RC ladders of 10/100/1000 nodes (every stored node sample of a
/// transient run, which exercises Init + Step assemblies and the
/// pattern-reuse refactor on every step after the first).
#[test]
fn sparse_transient_matches_dense_on_rc_ladders() {
    // The 1000-node *dense* reference runs an O(n³) factor on a dim ≈ 2N
    // Init system; keep it to the release gate (`cargo test --release`).
    let sizes: &[(usize, usize)] = if cfg!(debug_assertions) {
        &[(10, 200), (100, 20)]
    } else {
        &[(10, 200), (100, 20), (1000, 2)]
    };
    for &(n, steps) in sizes {
        let dense = run_ladder(n, steps, SolverChoice::Dense);
        let sparse = run_ladder(n, steps, SolverChoice::Sparse);
        assert_eq!(dense.solver, SolverKind::Dense);
        assert_eq!(sparse.solver, SolverKind::Sparse);
        assert_eq!(dense.t, sparse.t);
        for (node, (td, ts)) in dense.v.iter().zip(&sparse.v).enumerate() {
            for (vd, vs) in td.iter().zip(ts) {
                assert!(
                    (vd - vs).abs() < 1e-10,
                    "n={n} node {}: dense {vd:.15e} vs sparse {vs:.15e}",
                    node + 1
                );
            }
        }
        // Energy bookkeeping agrees too (same stamps, same Tellegen closure).
        assert!(
            (dense.source_energy - sparse.source_energy).abs()
                <= 1e-10 * dense.source_energy.abs().max(1e-30),
            "n={n}: source energy dense {:.15e} vs sparse {:.15e}",
            dense.source_energy,
            sparse.source_energy
        );
    }
}

/// 1000-node RC ladder transient through the sparse path (auto-selected):
/// DC steady state sends every node to the source voltage, and the charge-up
/// is monotone at every node (backward Euler is L-stable, so the diffusive
/// ladder must charge without overshoot).
#[test]
fn thousand_node_ladder_charges_monotonically_to_source() {
    let n = 1000;
    // Slowest diffusion mode of the line ≈ (4/π²)·n²·RC; run 20× that.
    let tau = 4.0 / (std::f64::consts::PI * std::f64::consts::PI) * (n as f64) * (n as f64) * R * C;
    let t_stop = 20.0 * tau;
    let steps = 4000usize;
    let mut opts = TransientOpts::new(t_stop, t_stop / steps as f64);
    opts.method = Method::BackwardEuler;
    opts.store_stride = 8;
    let res = transient(&rc_ladder(n), &opts).unwrap();
    assert_eq!(
        res.solver,
        SolverKind::Sparse,
        "1000 nodes must route to the sparse path under Auto"
    );
    for (i, trace) in res.v.iter().enumerate() {
        let v_end = *trace.last().unwrap();
        assert!(
            (v_end - V).abs() < 1e-6,
            "node {} ended at {v_end} (≠ source {V})",
            i + 1
        );
        for w in trace.windows(2) {
            assert!(
                w[1] >= w[0] - 1e-9,
                "node {} charge-up not monotone: {} → {}",
                i + 1,
                w[0],
                w[1]
            );
        }
    }
}

/// A resistor island floating off ground produces a singular MNA system:
/// the sparse path must report `CircuitError::Singular` cleanly, no panic.
#[test]
fn floating_island_is_singular_not_panic() {
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: V })
        .resistor("ra", 2, 3, R)
        .resistor("rb", 2, 3, 2.0 * R);
    let mut opts = TransientOpts::new(1e-6, 1e-8);
    opts.solver = SolverChoice::Sparse;
    assert_eq!(transient(&net, &opts).unwrap_err(), CircuitError::Singular);
    // Dense path agrees on the diagnosis.
    opts.solver = SolverChoice::Dense;
    assert_eq!(transient(&net, &opts).unwrap_err(), CircuitError::Singular);
}

/// Roadmap §6 budget: 100-node adiabatic cell, 10⁶ timesteps, < 10 s.
/// Ignored by default — run with `cargo test --release -- --ignored bench`.
#[test]
#[ignore]
fn bench_adiabatic_cell_budget_100_nodes_1e6_steps() {
    let rc = R * C;
    let (t_ramp, t_hold) = (20.0 * rc, 8.0 * rc);
    let net = power_clock_chain(50, t_ramp, t_hold); // 50 stages → 100 nodes
    let t_stop = 3.0 * (t_ramp + t_hold) + 2.0 * t_ramp + t_hold + 12.0 * rc;
    let mut opts = TransientOpts::new(t_stop, t_stop / 1e6);
    opts.store_stride = usize::MAX;
    let t0 = std::time::Instant::now();
    let res = transient(&net, &opts).unwrap();
    let wall = t0.elapsed().as_secs_f64();
    assert_eq!(res.steps, 1_000_000);
    assert_eq!(res.solver, SolverKind::Sparse);
    eprintln!(
        "100-node adiabatic cell, 10⁶ steps: {wall:.2} s ({:.2} µs/step, budget 10 s)",
        wall * 1e6 / res.steps as f64
    );
    assert!(wall < 10.0, "budget blown: {wall:.2} s ≥ 10 s");
}

/// Dense vs sparse per-step cost on RC ladders — the measurement behind
/// `SPARSE_NODE_THRESHOLD` and the demonstration that sparse wins at 500+.
/// Ignored by default — run with `cargo test --release -- --ignored bench`.
#[test]
#[ignore]
fn bench_dense_vs_sparse_rc_ladders() {
    let time_per_step = |n: usize, steps: usize, choice: SolverChoice| {
        let t0 = std::time::Instant::now();
        let res = run_ladder(n, steps, choice);
        assert_eq!(res.steps, steps as u64);
        t0.elapsed().as_secs_f64() / steps as f64 * 1e6 // µs/step incl. init
    };
    eprintln!("  nodes |  dense µs/step | sparse µs/step | speedup");
    let mut at = std::collections::BTreeMap::new();
    for &(n, steps) in &[
        (4usize, 50_000usize),
        (8, 50_000),
        (16, 20_000),
        (32, 10_000),
        (48, 10_000),
        (64, 5_000),
        (100, 2_000),
        (500, 200),
        (1000, 50),
    ] {
        let d = time_per_step(n, steps, SolverChoice::Dense);
        let s = time_per_step(n, steps, SolverChoice::Sparse);
        eprintln!("  {n:>5} | {d:>14.2} | {s:>14.2} | {:>6.1}×", d / s);
        at.insert(n, (d, s));
    }
    let (d500, s500) = at[&500];
    let (d1000, s1000) = at[&1000];
    assert!(s500 < d500, "sparse must beat dense at 500 nodes");
    assert!(s1000 < d1000, "sparse must beat dense at 1000 nodes");
}
