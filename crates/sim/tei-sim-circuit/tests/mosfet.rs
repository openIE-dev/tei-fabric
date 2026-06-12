//! M2 MOSFET validation — closed-form ground truth and properties only, per
//! the roadmap's binding validation policy (no foreign-tool fixtures).
//!
//! Coverage:
//!   - level-1 diode-connected square law at three bias points (1e-9 rel)
//!   - level-1 triode at small v_ds ≈ kp·(v_gs − vth)·v_ds (1e-6 rel)
//!   - level-1 saturation flat at λ = 0 (1e-12 after the GMIN term), and
//!     exactly ∝ (1 + λ·v_ds) at λ > 0
//!   - PMOS = exact NMOS mirror (1e-12)
//!   - EKV subthreshold slope = n·φ_t·ln10 per decade (1%), for n = 1.0
//!     and n = 1.3
//!   - EKV strong inversion → square law kp/(2n)·(v_gs − vth)² (2%)
//!   - CMOS inverter rail-to-rail DC transfer, crossover at v_dd/2
//!   - transient inverter driving a load cap: discrete Tellegen residual
//!     < 1e-9 W through the nonlinear stamps
//!   - 22-node inverter cascade under `SolverChoice::Auto` routes to the
//!     sparse path (per-Newton-iteration numeric refactor) and matches the
//!     dense path to 1e-9
//!   - LinearDcSolver rejects MOSFET netlists; validation rejects bad params
//!
//! The expected currents include the GMIN (= 1e-12 S) drain–source term the
//! stamps fold in; where it matters it is written out explicitly.

use tei_sim_circuit::{
    LinearDcSolver, MosPolarity, Netlist, SolverChoice, SolverKind, TransientOpts, VT_300K,
    Waveform, solve_dc, transient,
};

const KP: f64 = 1e-3;
const VTH: f64 = 0.5;
const GMIN: f64 = 1e-12;

fn dc(v: f64) -> Waveform {
    Waveform::Dc { v }
}

/// Drain current of an NMOS/PMOS with all three terminals pinned by sources:
/// node 1 = gate, node 2 = drain, node 3 = source. Returns the current d→s
/// through the channel (= −branch current of the drain source).
fn pinned_mos_current(pol: MosPolarity, lambda: f64, vg: f64, vd: f64, vs: f64) -> f64 {
    let mut net = Netlist::new();
    net.vsource("vg", 1, 0, dc(vg))
        .vsource("vd", 2, 0, dc(vd))
        .vsource("vs", 3, 0, dc(vs))
        .mosfet("m1", 2, 1, 3, pol, VTH, KP, lambda);
    let sol = solve_dc(&net).unwrap();
    -sol.branch_current("vd").unwrap()
}

/// Same, for the EKV-lite model.
fn pinned_ekv_current(pol: MosPolarity, n: f64, kp: f64, vg: f64, vd: f64, vs: f64) -> f64 {
    let mut net = Netlist::new();
    net.vsource("vg", 1, 0, dc(vg))
        .vsource("vd", 2, 0, dc(vd))
        .vsource("vs", 3, 0, dc(vs))
        .mosfet_ekv("m1", 2, 1, 3, pol, VTH, kp, n, VT_300K);
    let sol = solve_dc(&net).unwrap();
    -sol.branch_current("vd").unwrap()
}

/// Level-1 diode-connected (gate tied to drain — same node) square law:
/// i = kp/2·(v − vth)² + GMIN·v at three bias points, 1e-9 relative.
#[test]
fn level1_diode_connected_square_law() {
    for vb in [0.8, 1.0, 1.3] {
        let mut net = Netlist::new();
        net.vsource("vb", 1, 0, dc(vb))
            .mosfet("m1", 1, 1, 0, MosPolarity::Nmos, VTH, KP, 0.0);
        let sol = solve_dc(&net).unwrap();
        let i = -sol.branch_current("vb").unwrap();
        let vov = vb - VTH;
        let expected = 0.5 * KP * vov * vov + GMIN * vb;
        let rel = (i - expected).abs() / expected;
        assert!(
            rel < 1e-9,
            "vb={vb}: i={i:.12e} vs square law {expected:.12e} (rel {rel:.2e})"
        );
    }
}

/// Level-1 deep triode: at v_ds = 1e-7 V the current matches the full triode
/// expression to 1e-9 and the linear approximation kp·(v_gs − vth)·v_ds to
/// 1e-6 relative (the dropped v_ds²/2 term is 1e-7 of the linear part).
#[test]
fn level1_triode_small_vds_is_linear() {
    let (vg, vds) = (1.0, 1e-7);
    let i = pinned_mos_current(MosPolarity::Nmos, 0.0, vg, vds, 0.0);
    let vov = vg - VTH;
    let full = KP * (vov * vds - 0.5 * vds * vds) + GMIN * vds;
    assert!(
        (i - full).abs() / full < 1e-9,
        "i={i:.12e} vs full triode {full:.12e}"
    );
    let linear = KP * vov * vds;
    assert!(
        (i - linear).abs() / linear < 1e-6,
        "i={i:.12e} vs kp·vov·vds {linear:.12e}"
    );
}

/// Level-1 saturation at λ = 0 is flat in v_ds: after removing the GMIN
/// term the current is identical (1e-12 rel) across v_ds, and equals
/// kp/2·(v_gs − vth)². With λ > 0 the current is exactly
/// kp/2·(v_gs − vth)²·(1 + λ·v_ds).
#[test]
fn level1_saturation_flat_at_zero_lambda_and_exact_lambda_slope() {
    let vg = 1.0;
    let vov = vg - VTH;
    let isat = 0.5 * KP * vov * vov;
    let mut flat = Vec::new();
    for vds in [0.6, 1.2, 1.8] {
        let i = pinned_mos_current(MosPolarity::Nmos, 0.0, vg, vds, 0.0) - GMIN * vds;
        assert!(
            (i - isat).abs() / isat < 1e-12,
            "λ=0 vds={vds}: i={i:.15e} vs {isat:.15e}"
        );
        flat.push(i);
    }
    assert!((flat[0] - flat[2]).abs() / flat[0] < 1e-12, "not flat");

    let lambda = 0.05;
    for vds in [0.6, 1.8] {
        let i = pinned_mos_current(MosPolarity::Nmos, lambda, vg, vds, 0.0);
        let expected = isat * (1.0 + lambda * vds) + GMIN * vds;
        assert!(
            (i - expected).abs() / expected < 1e-9,
            "λ vds={vds}: i={i:.12e} vs {expected:.12e}"
        );
    }
}

/// PMOS is the exact mirror of NMOS: i_P(v_g, v_d, v_s) = −i_N(−v_g, −v_d,
/// −v_s) to 1e-12 relative, across cutoff / triode / saturation / reversed
/// (v_ds < 0) bias points, level-1 and EKV alike.
#[test]
fn pmos_is_exact_nmos_mirror() {
    let biases = [
        (1.0, 0.05, 0.0), // triode
        (1.0, 1.5, 0.0),  // saturation
        (0.2, 1.0, 0.0),  // cutoff (GMIN only)
        (1.0, 0.0, 0.8),  // reversed: source above drain
        (1.3, 0.4, -0.2), // triode, lifted source
    ];
    for &(vg, vd, vs) in &biases {
        let i_n = pinned_mos_current(MosPolarity::Nmos, 0.02, vg, vd, vs);
        let i_p = pinned_mos_current(MosPolarity::Pmos, 0.02, -vg, -vd, -vs);
        assert!(
            (i_p + i_n).abs() <= 1e-12 * i_n.abs().max(1e-30),
            "level-1 ({vg},{vd},{vs}): i_n={i_n:.15e}, i_p={i_p:.15e}"
        );
        let i_n = pinned_ekv_current(MosPolarity::Nmos, 1.3, KP, vg, vd, vs);
        let i_p = pinned_ekv_current(MosPolarity::Pmos, 1.3, KP, -vg, -vd, -vs);
        assert!(
            (i_p + i_n).abs() <= 1e-12 * i_n.abs().max(1e-30),
            "EKV ({vg},{vd},{vs}): i_n={i_n:.15e}, i_p={i_p:.15e}"
        );
    }
}

/// EKV subthreshold slope: d(v_gs)/d(log₁₀ i) = n·φ_t·ln10 per decade,
/// within 1%, measured well below threshold (saturated v_ds ≫ φ_t so the
/// reverse term vanishes), for n = 1.0 and n = 1.3.
#[test]
fn ekv_subthreshold_slope_is_n_phi_t_ln10() {
    // (n, two gate underdrives, v_ds) — depths chosen so the device current
    // dwarfs both the GMIN term and the ln²(1+x) ≈ x² correction.
    let cases = [(1.0, -0.20, -0.30, 0.2), (1.3, -0.35, -0.45, 0.3)];
    for &(n, dv1, dv2, vds) in &cases {
        let kp = 1e-2;
        let i1 = pinned_ekv_current(MosPolarity::Nmos, n, kp, VTH + dv1, vds, 0.0);
        let i2 = pinned_ekv_current(MosPolarity::Nmos, n, kp, VTH + dv2, vds, 0.0);
        let slope = (dv1 - dv2) / (i1 / i2).log10(); // V per decade
        let expected = n * VT_300K * std::f64::consts::LN_10;
        let rel = (slope - expected).abs() / expected;
        eprintln!(
            "EKV n={n}: subthreshold slope {:.4} mV/dec (n·φt·ln10 = {:.4} mV/dec, rel {rel:.2e})",
            slope * 1e3,
            expected * 1e3
        );
        assert!(
            rel < 0.01,
            "n={n}: slope {:.3} mV/dec vs n·φt·ln10 = {:.3} mV/dec (rel {rel:.2e})",
            slope * 1e3,
            expected * 1e3
        );
    }
}

/// EKV strong inversion recovers the square law: diode-connected (saturated)
/// i → kp/(2n)·(v_gs − vth)² within 2% for overdrives ≫ φ_t.
#[test]
fn ekv_strong_inversion_recovers_square_law() {
    let (n, kp) = (1.3, KP);
    for vov in [0.7, 1.0] {
        let vb = VTH + vov;
        let mut net = Netlist::new();
        net.vsource("vb", 1, 0, dc(vb)).mosfet_ekv(
            "m1",
            1,
            1,
            0,
            MosPolarity::Nmos,
            VTH,
            kp,
            n,
            VT_300K,
        );
        let sol = solve_dc(&net).unwrap();
        let i = -sol.branch_current("vb").unwrap();
        let square = kp / (2.0 * n) * vov * vov;
        let rel = (i - square).abs() / square;
        assert!(
            rel < 0.02,
            "vov={vov}: i={i:.6e} vs kp/(2n)·vov² = {square:.6e} (rel {rel:.2e})"
        );
    }
}

/// CMOS inverter (matched kp/vth, λ = 0.1 for a well-conditioned high-gain
/// point): rail-to-rail DC transfer, monotone falling, and the crossover at
/// exactly v_dd/2 by symmetry.
#[test]
fn cmos_inverter_dc_transfer_rail_to_rail() {
    let vdd = 1.0;
    let (vth, kp, lambda) = (0.3, 1e-3, 0.1);
    let run = |vin: f64| -> f64 {
        let mut net = Netlist::new();
        net.vsource("vdd", 1, 0, dc(vdd))
            .vsource("vin", 2, 0, dc(vin))
            .mosfet("mn", 3, 2, 0, MosPolarity::Nmos, vth, kp, lambda)
            .mosfet("mp", 3, 2, 1, MosPolarity::Pmos, vth, kp, lambda);
        solve_dc(&net).unwrap().node(3)
    };
    let vouts: Vec<f64> = (0..=10).map(|k| run(vdd * k as f64 / 10.0)).collect();
    assert!(vouts[0] > 0.999 * vdd, "vout(0) = {} not at rail", vouts[0]);
    assert!(
        vouts[10] < 1e-3 * vdd,
        "vout(vdd) = {} not at rail",
        vouts[10]
    );
    for w in vouts.windows(2) {
        assert!(w[1] <= w[0] + 1e-12, "transfer not monotone: {vouts:?}");
    }
    // Matched devices: NMOS and PMOS currents balance at exactly v_dd/2
    // (the (1 + λ·v) factors mirror around the midpoint, GMIN included).
    let mid = run(vdd / 2.0);
    assert!(
        (mid - vdd / 2.0).abs() < 1e-9,
        "crossover vout(vdd/2) = {mid}, expected {}",
        vdd / 2.0
    );
}

/// Transient CMOS inverter driving a load cap through an input pulse:
/// the output swings to both rails and the discrete Tellegen identity holds
/// through the MOSFET stamps — per-step power residual < 1e-9 W and
/// source = dissipated + reactive-absorbed to 1e-10 relative.
#[test]
fn inverter_transient_tellegen_through_mosfets() {
    let vdd = 1.0;
    let (vth, kp) = (0.3, 1e-3);
    let c = 1e-12;
    let mut net = Netlist::new();
    net.vsource("vdd", 1, 0, dc(vdd))
        .vsource(
            "vin",
            2,
            0,
            Waveform::Trapezoid {
                v: vdd,
                t_delay: 10e-9,
                t_rise: 5e-9,
                t_hold: 15e-9,
                t_fall: 5e-9,
            },
        )
        .mosfet("mn", 3, 2, 0, MosPolarity::Nmos, vth, kp, 0.0)
        .mosfet("mp", 3, 2, 1, MosPolarity::Pmos, vth, kp, 0.0)
        .capacitor("cl", 3, 0, c, 0.0);
    let res = transient(&net, &TransientOpts::new(60e-9, 0.05e-9)).unwrap();

    assert!(
        res.tellegen_max < 1e-9,
        "per-step Tellegen residual {} W through MOSFET stamps",
        res.tellegen_max
    );
    let identity =
        (res.source_energy - (res.dissipated_energy + res.reactive_absorbed_energy)).abs();
    assert!(
        identity < 1e-10 * res.source_energy.abs().max(1e-30),
        "energy identity residual {identity:.3e} J"
    );
    // Rail-to-rail dynamics: the output reaches both logic levels.
    let out = &res.v[2];
    let t_idx = |t: f64| (t / res.dt).round() as usize;
    assert!(out[t_idx(9e-9)] > 0.99 * vdd, "out not high before pulse");
    assert!(out[t_idx(28e-9)] < 0.01 * vdd, "out not low mid-pulse");
    assert!(
        *out.last().unwrap() > 0.99 * vdd,
        "out not recovered after pulse"
    );
}

/// 20-stage inverter cascade (22 nodes > SPARSE_NODE_THRESHOLD): under
/// `SolverChoice::Auto` the transient must route to the sparse Markowitz-LU
/// path — exercising the per-Newton-iteration numeric refactor on a MOSFET
/// circuit — and agree with the forced-dense run to 1e-9 on every stored
/// sample and on the energy ledger.
#[test]
fn mosfet_cascade_auto_routes_sparse_and_matches_dense() {
    let vdd = 1.0;
    let (vth, kp) = (0.3, 4e-3); // stage RC ≈ 0.36 ns — the 20-stage wave settles well inside t_stop
    let c = 1e-12;
    let n_stages = 20;
    let build = || {
        let mut net = Netlist::new();
        net.vsource("vdd", 1, 0, dc(vdd)).vsource(
            "vin",
            2,
            0,
            Waveform::Trapezoid {
                v: vdd,
                t_delay: 10e-9,
                t_rise: 5e-9,
                t_hold: 20e-9,
                t_fall: 5e-9,
            },
        );
        for k in 0..n_stages {
            let (input, out) = (2 + k, 3 + k);
            net.mosfet(
                &format!("mn{k}"),
                out,
                input,
                0,
                MosPolarity::Nmos,
                vth,
                kp,
                0.0,
            )
            .mosfet(
                &format!("mp{k}"),
                out,
                input,
                1,
                MosPolarity::Pmos,
                vth,
                kp,
                0.0,
            )
            .capacitor(&format!("cl{k}"), out, 0, c, 0.0);
        }
        net
    };
    let run = |choice: SolverChoice| {
        let mut opts = TransientOpts::new(60e-9, 0.05e-9);
        opts.solver = choice;
        opts.store_stride = 10;
        transient(&build(), &opts).unwrap()
    };
    let auto = run(SolverChoice::Auto);
    assert_eq!(
        auto.solver,
        SolverKind::Sparse,
        "22-node MOSFET cascade must route to sparse under Auto"
    );
    let dense = run(SolverChoice::Dense);
    assert_eq!(dense.solver, SolverKind::Dense);
    assert_eq!(auto.t, dense.t);
    for (node, (ta, td)) in auto.v.iter().zip(&dense.v).enumerate() {
        for (va, vd) in ta.iter().zip(td) {
            assert!(
                (va - vd).abs() < 1e-9,
                "node {}: sparse {va:.12e} vs dense {vd:.12e}",
                node + 1
            );
        }
    }
    assert!(
        (auto.dissipated_energy - dense.dissipated_energy).abs()
            <= 1e-9 * dense.dissipated_energy.abs().max(1e-30),
        "dissipation: sparse {:.12e} vs dense {:.12e}",
        auto.dissipated_energy,
        dense.dissipated_energy
    );
    assert!(auto.tellegen_max < 1e-9 && dense.tellegen_max < 1e-9);
    // The cascade actually settled to alternating logic levels (vin = 0 at
    // the end: odd-indexed stage outputs low, even-indexed high).
    assert!(*auto.v[2 + n_stages - 1].last().unwrap() < 0.01 * vdd); // out19
    assert!(*auto.v[2 + n_stages - 2].last().unwrap() > 0.99 * vdd); // out18
}

/// MOSFETs make the matrix iterate-dependent: LinearDcSolver must reject
/// them, and validation must reject non-physical parameters up front.
#[test]
fn mosfet_rejection_paths() {
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, dc(1.0))
        .mosfet("m1", 1, 1, 0, MosPolarity::Nmos, VTH, KP, 0.0);
    assert!(LinearDcSolver::new(&net).is_err());

    // kp ≤ 0 rejected.
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, dc(1.0))
        .mosfet("m1", 1, 1, 0, MosPolarity::Nmos, VTH, -1.0, 0.0);
    assert!(solve_dc(&net).is_err());

    // λ < 0 rejected.
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, dc(1.0))
        .mosfet("m1", 1, 1, 0, MosPolarity::Nmos, VTH, KP, -0.1);
    assert!(solve_dc(&net).is_err());

    // d == s degenerate channel rejected.
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, dc(1.0))
        .mosfet("m1", 1, 1, 1, MosPolarity::Nmos, VTH, KP, 0.0);
    assert!(solve_dc(&net).is_err());

    // EKV: n and phi_t must be positive.
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, dc(1.0)).mosfet_ekv(
        "m1",
        1,
        1,
        0,
        MosPolarity::Nmos,
        VTH,
        KP,
        0.0,
        VT_300K,
    );
    assert!(solve_dc(&net).is_err());
}
