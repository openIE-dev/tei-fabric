//! Analytic validation of the MNA transient solver — closed-form ground
//! truth only, per the roadmap's binding validation policy.
//!
//! Coverage map (spec items a–f plus extras):
//!   a) `rc_step_response_matches_closed_form` + the convergence tests
//!   b) `dc_voltage_divider_exact`
//!   c) `rc_energy_conservation`
//!   d) `adiabatic_ramp_scaling_law`  (the marquee)
//!   e) `trapezoidal_second_order_convergence` (+ BE first-order control)
//!   f) `two_step_staircase_halves_dissipation`
//!   +) RLC underdamped closed form (inductor stamps), diode DC vs bisection
//!      (M2), executor end-to-end with progress + ledger.

use tei_sim_circuit::{
    CircuitExecutor, CircuitJob, Method, Netlist, TransientOpts, Waveform, solve_dc, transient,
};
use tei_sim_core::exec::Executor;

/// Series source → R → C to ground; v_C is node 2, cap starts at 0.
fn rc_net(r: f64, c: f64, wave: Waveform) -> Netlist {
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, wave)
        .resistor("r1", 1, 2, r)
        .capacitor("c1", 2, 0, c, 0.0);
    net
}

const R: f64 = 1e3;
const C: f64 = 1e-9;
const RC: f64 = R * C;
const V: f64 = 1.0;

/// |v_C(t_check) − analytic| for an RC step at the given dt/method.
/// t_check must be an integer multiple of dt.
fn rc_step_error(dt: f64, method: Method, t_check: f64) -> f64 {
    let net = rc_net(R, C, Waveform::Dc { v: V });
    let mut opts = TransientOpts::new(t_check, dt);
    opts.method = method;
    let res = transient(&net, &opts).unwrap();
    let idx = (t_check / dt).round() as usize;
    assert!((res.t[idx] - t_check).abs() < 1e-12 * t_check.max(1e-30));
    let v_sim = res.v[1][idx]; // node 2
    let v_exact = V * (1.0 - (-t_check / RC).exp());
    (v_sim - v_exact).abs()
}

/// (b) DC voltage divider — exact ratio to 1e-12.
#[test]
fn dc_voltage_divider_exact() {
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: 1.0 })
        .resistor("r1", 1, 2, 1e3)
        .resistor("r2", 2, 0, 2e3);
    let dc = solve_dc(&net).unwrap();
    assert!(
        (dc.node(2) - 2.0 / 3.0).abs() < 1e-12,
        "divider voltage {} ≠ 2/3",
        dc.node(2)
    );
    // Source branch current: V/(R1+R2) flowing 0→1 inside the source,
    // i.e. −1/3 mA p→n.
    let i = dc.branch_current("vs").unwrap();
    assert!((i + 1.0 / 3000.0).abs() < 1e-15);
}

/// DC operating point semantics: capacitor open (v_C = V), inductor short
/// (i_L = V/R).
#[test]
fn dc_reactive_semantics() {
    let dc = solve_dc(&rc_net(R, C, Waveform::Dc { v: V })).unwrap();
    assert!((dc.node(2) - V).abs() < 1e-12, "cap not open at DC");

    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: V })
        .resistor("r1", 1, 2, R)
        .inductor("l1", 2, 0, 1e-3, 0.0);
    let dc = solve_dc(&net).unwrap();
    assert!(dc.node(2).abs() < 1e-12, "inductor not short at DC");
    assert!((dc.branch_current("l1").unwrap() - V / R).abs() < 1e-15);
}

/// (a) RC step response v_C(t) = V(1 − e^{−t/RC}) at multiple times, fine dt.
#[test]
fn rc_step_response_matches_closed_form() {
    let dt = RC / 1000.0;
    let net = rc_net(R, C, Waveform::Dc { v: V });
    let res = transient(&net, &TransientOpts::new(5.0 * RC, dt)).unwrap();
    for t_over_rc in [0.5f64, 1.0, 2.0, 5.0] {
        let idx = (t_over_rc * 1000.0).round() as usize;
        let v_exact = V * (1.0 - (-t_over_rc).exp());
        let err = (res.v[1][idx] - v_exact).abs();
        assert!(err < 1e-6 * V, "t={t_over_rc}RC: |err|={err:.3e}");
    }
}

/// (a)+(e) Trapezoidal is second order: halving dt divides the error by ≈ 4.
#[test]
fn trapezoidal_second_order_convergence() {
    let t_check = 2.0 * RC;
    let e1 = rc_step_error(RC / 20.0, Method::Trapezoidal, t_check);
    let e2 = rc_step_error(RC / 40.0, Method::Trapezoidal, t_check);
    let e3 = rc_step_error(RC / 80.0, Method::Trapezoidal, t_check);
    let (r12, r23) = (e1 / e2, e2 / e3);
    assert!((3.3..=4.7).contains(&r12), "ratio dt→dt/2 = {r12:.3}");
    assert!((3.3..=4.7).contains(&r23), "ratio dt/2→dt/4 = {r23:.3}");
    // Absolute accuracy at fine dt (item a).
    assert!(rc_step_error(RC / 1000.0, Method::Trapezoidal, t_check) < 1e-6 * V);
}

/// (e) Backward Euler is first order (ratio ≈ 2) — the methods really differ.
#[test]
fn backward_euler_first_order_convergence() {
    let t_check = 2.0 * RC;
    let e1 = rc_step_error(RC / 20.0, Method::BackwardEuler, t_check);
    let e2 = rc_step_error(RC / 40.0, Method::BackwardEuler, t_check);
    let r = e1 / e2;
    assert!((1.7..=2.4).contains(&r), "BE ratio = {r:.3}");
}

/// (c) Energy conservation in an RC charge-up:
/// E_source = E_R + ½CV² within 0.1%, the discrete Tellegen identity
/// (source = dissipated + reactive-absorbed) to ~LU precision, and the
/// analytic E_R(t) = ½CV²(1 − e^{−2t/RC}).
#[test]
fn rc_energy_conservation() {
    let net = rc_net(R, C, Waveform::Dc { v: V });
    let res = transient(&net, &TransientOpts::new(12.0 * RC, RC / 500.0)).unwrap();

    let e_src = res.source_energy;
    let e_r = res.energy("r1").unwrap();
    let v_end = *res.v[1].last().unwrap();
    let stored = 0.5 * C * v_end * v_end;

    // Physical closure at fine dt.
    let closure = (e_src - (e_r + stored)).abs() / e_src;
    assert!(closure < 1e-3, "energy closure off by {closure:.2e}");

    // Discrete Tellegen: exact by construction, independent of dt.
    let exact = (e_src - (res.dissipated_energy + res.reactive_absorbed_energy)).abs();
    assert!(
        exact < 1e-10 * e_src,
        "discrete Tellegen residual {exact:.3e} J"
    );
    assert!(
        res.tellegen_max < 1e-12,
        "per-step power residual {} W",
        res.tellegen_max
    );

    // Closed forms.
    let e_r_exact = 0.5 * C * V * V * (1.0 - (-2.0 * 12.0f64).exp());
    assert!((e_r - e_r_exact).abs() / e_r_exact < 1e-3);
    let e_src_exact = C * V * V * (1.0 - (-12.0f64).exp());
    assert!((e_src - e_src_exact).abs() / e_src_exact < 1e-3);

    // ∫i·v over the cap ≈ Δ(½Cv²).
    assert!((res.reactive_absorbed_energy - res.delta_stored_energy).abs() < 1e-3 * stored);
}

/// (d) THE ADIABATIC LAW. C charged through R by a linear ramp of duration T:
///
///   E_R(T) = (RC/T)·CV²·[1 − (RC/T)(1 − e^{−T/RC})]
///
/// so T ≫ RC ⇒ E_R → (RC/T)·CV² (log-log slope −1) and T ≪ RC ⇒ E_R → ½CV².
/// Sweep T over five decades (10⁻²…10³ × RC): abrupt limit within 2%,
/// monotone decrease, slope −1 within 5% in the adiabatic regime, and every
/// point within 2% of the full closed form.
#[test]
fn adiabatic_ramp_scaling_law() {
    let decades: [f64; 6] = [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0];
    let cv2 = C * V * V;
    let mut measured = Vec::new();
    for &k in &decades {
        let t_ramp = RC * 10f64.powf(k);
        let dt = (t_ramp / 100.0).min(RC / 100.0);
        let net = rc_net(R, C, Waveform::Ramp { v: V, t_ramp });
        let mut opts = TransientOpts::new(t_ramp + 10.0 * RC, dt);
        opts.store_stride = 1 << 20; // energies only; skip trace storage
        let res = transient(&net, &opts).unwrap();
        let e_r = res.energy("r1").unwrap();

        // Full closed form (charge run to completion; settle residual ~e^{−20}).
        let x = RC / t_ramp;
        let exact = x * cv2 * (1.0 - x * (1.0 - (-1.0 / x).exp()));
        assert!(
            (e_r - exact).abs() / exact < 0.02,
            "T=10^{k}·RC: E_R={e_r:.4e} vs closed form {exact:.4e}"
        );
        measured.push((t_ramp, e_r));
    }

    // (ii) abrupt limit: T = RC/100 recovers ½CV² within 2%.
    let e_abrupt = measured[0].1;
    assert!(
        (e_abrupt - 0.5 * cv2).abs() / (0.5 * cv2) < 0.02,
        "abrupt limit E_R = {e_abrupt:.4e}, ½CV² = {:.4e}",
        0.5 * cv2
    );

    // (iii) strictly monotone decrease across the sweep.
    for w in measured.windows(2) {
        assert!(
            w[1].1 < w[0].1,
            "E_R not monotone: {:.4e} → {:.4e}",
            w[0].1,
            w[1].1
        );
    }

    // (i) log-log slope → −1 in the adiabatic regime (last two decade pairs).
    for w in measured[3..].windows(2) {
        let slope = (w[1].1 / w[0].1).log10() / (w[1].0 / w[0].0).log10();
        assert!(
            (slope + 1.0).abs() < 0.05,
            "adiabatic slope {slope:.4} (T = {:.1e} → {:.1e})",
            w[0].0,
            w[1].0
        );
    }
}

/// (f) Mini power-clock: charging in N voltage steps dissipates ½CV²/N.
/// Two steps of V/2 each lose 2·½C(V/2)² = ¼CV² — half the single-step loss.
/// Assert E(2-step) < 0.6·E(1-step) and the ¼CV² value within 2%.
#[test]
fn two_step_staircase_halves_dissipation() {
    let tr = RC / 100.0; // step edge, fast w.r.t. RC
    let dt = RC / 1000.0;
    let run = |points: Vec<(f64, f64)>, t_stop: f64| {
        let net = rc_net(R, C, Waveform::Pwl { points });
        let mut opts = TransientOpts::new(t_stop, dt);
        opts.store_stride = 1 << 20;
        transient(&net, &opts).unwrap().energy("r1").unwrap()
    };

    let e1 = run(vec![(0.0, 0.0), (tr, V)], 10.0 * RC);
    let e2 = run(
        vec![
            (0.0, 0.0),
            (tr, V / 2.0),
            (10.0 * RC, V / 2.0),
            (10.0 * RC + tr, V),
        ],
        20.0 * RC,
    );

    let half_cv2 = 0.5 * C * V * V;
    assert!(
        (e1 - half_cv2).abs() / half_cv2 < 0.02,
        "1-step E_R = {e1:.4e}"
    );
    assert!(
        e2 < 0.6 * e1,
        "2-step staircase did not halve dissipation: {e2:.4e} vs {e1:.4e}"
    );
    assert!(
        (e2 - half_cv2 / 2.0).abs() / (half_cv2 / 2.0) < 0.02,
        "2-step E_R = {e2:.4e}, expected ½CV²/2 = {:.4e}",
        half_cv2 / 2.0
    );
}

/// Bonus: series RLC underdamped step — validates the inductor companion.
/// v_C(t) = V[1 − e^{−αt}(cos ω_d t + (α/ω_d) sin ω_d t)].
#[test]
fn rlc_underdamped_step_matches_closed_form() {
    let (rr, ll, cc) = (10.0f64, 1e-3f64, 1e-6f64);
    let alpha = rr / (2.0 * ll);
    let w0sq = 1.0 / (ll * cc);
    let wd = (w0sq - alpha * alpha).sqrt();

    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: V })
        .resistor("r1", 1, 2, rr)
        .inductor("l1", 2, 3, ll, 0.0)
        .capacitor("c1", 3, 0, cc, 0.0);
    let dt = 2e-7;
    let res = transient(&net, &TransientOpts::new(5e-4, dt)).unwrap();
    for t in [1e-4, 2e-4, 4e-4] {
        let idx = (t / dt).round() as usize;
        let v_exact =
            V * (1.0 - (-alpha * t).exp() * ((wd * t).cos() + alpha / wd * (wd * t).sin()));
        let err = (res.v[2][idx] - v_exact).abs();
        assert!(err < 5e-4, "t={t:e}: |err|={err:.3e}");
    }
}

/// M2: Shockley diode + R divider DC point vs an independent bisection of
/// (V − v_d)/R = I_s(e^{v_d/V_T} − 1).
#[test]
fn diode_resistor_dc_matches_bisection() {
    let (i_s, n_id) = (1e-14, 1.0);
    let mut net = Netlist::new();
    net.vsource("vs", 1, 0, Waveform::Dc { v: V })
        .resistor("r1", 1, 2, R)
        .diode("d1", 2, 0, i_s, n_id);
    let dc = solve_dc(&net).unwrap();
    let vd = dc.node(2);

    let vt = tei_sim_circuit::VT_300K * n_id;
    let f = |v: f64| (V - v) / R - i_s * ((v / vt).exp() - 1.0);
    let (mut lo, mut hi) = (0.0, 1.0);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let vd_ref = 0.5 * (lo + hi);
    assert!(
        (vd - vd_ref).abs() < 1e-9,
        "Newton v_d = {vd:.12} vs bisection {vd_ref:.12}"
    );
    // The solution is in the physically sensible window for I_s = 1e-14.
    assert!((0.4..0.8).contains(&vd));
}

/// Executor end-to-end: serde job → run → ledger.joules ≈ ½CV², downsampled
/// traces, per-element energy map, ≥ 50 progress ticks.
#[test]
fn executor_runs_rc_job_with_progress_and_ledger() {
    let job: CircuitJob = serde_json::from_str(
        r#"{
            "netlist": { "elements": [
                {"kind":"voltage_source","name":"vs","p":1,"n":0,"wave":{"shape":"dc","v":1.0}},
                {"kind":"resistor","name":"r1","p":1,"n":2,"r":1000.0},
                {"kind":"capacitor","name":"c1","p":2,"n":0,"c":1e-9}
            ]},
            "t_stop": 1e-5,
            "dt": 1e-9,
            "method": "trapezoidal"
        }"#,
    )
    .unwrap();

    let mut ticks = 0usize;
    let res = CircuitExecutor.execute(&job, &mut |p| {
        assert!((0.0..=1.0).contains(&p.fraction));
        ticks += 1;
    });
    assert!(ticks >= 50, "only {ticks} progress ticks");

    let half_cv2 = 0.5 * C * V * V;
    assert!(
        (res.ledger.joules - half_cv2).abs() / half_cv2 < 0.01,
        "ledger.joules = {:.4e}",
        res.ledger.joules
    );
    let out = &res.outputs;
    assert!(out["t"].as_array().unwrap().len() <= 1100); // downsampled
    assert_eq!(out["nodes"].as_array().unwrap().len(), 2);
    assert!(out["element_energy_j"]["r1"].as_f64().unwrap() > 0.0);
    assert!(out["source_energy_j"].as_f64().unwrap() > 0.0);
    assert!(res.ledger.wall_seconds.is_some());
}
