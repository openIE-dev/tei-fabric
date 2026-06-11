//! F3 validation per docs/SIM-ROADMAP.md §3.7 — closed-form and property
//! ground truth only, no foreign-tool fixtures.
//!
//! Coverage: (a) PEC box-cavity resonances vs the exact f_mnp (the F3
//! anchor), (b) 3D numerical dispersion vs the Yee axis relation, (c) 3D
//! CPML reflection floor by reference subtraction, (d) the 3D CFL
//! stability boundary, (e) Drude below/above-plasma transmission
//! brackets, (f) the Lorentz resonance-absorption bracket, (g) the
//! bit-exact ADE vacuum limit, (h) determinism across runs and rayon
//! thread counts, plus the job-schema defaults and an `#[ignore]`
//! throughput bench.

use std::f64::consts::PI;

use tei_sim_core::exec::{Executor, Progress};
use tei_sim_field::{
    Axis, Comp, CpmlParams, Dft3Monitor, Dipole3, EpsSpec3, Field3Executor, Field3Job, Grid3Spec,
    Grid3d, Probe3, Probe3Spec, TimeProfile, run_job3, yee_axis_wavenumber,
};

const DT: f64 = 0.5 / 1.7320508075688772; // default courant 0.5, Δt = S/√3

fn vacuum_grid3(nx: usize, ny: usize, nz: usize, npml: usize, courant: f64) -> Grid3d {
    let spec = Grid3Spec {
        nx,
        ny,
        nz,
        courant,
        npml,
        cpml: CpmlParams::default(),
    };
    Grid3d::new(&spec, EpsSpec3::Uniform { eps_r: 1.0 }.build(nx, ny, nz))
}

fn gaussian_dipole(i: usize, j: usize, k: usize, t0: f64, tau: f64) -> Dipole3 {
    Dipole3 {
        i,
        j,
        k,
        axis: Axis::Z,
        time: TimeProfile::Gaussian { t0, tau },
        amplitude: 1.0,
    }
}

/// Hann-windowed DFT amplitude of a (possibly strided) probe trace whose
/// m-th sample sits at t = (m·stride + 1)·Δt.
fn hann_dft(trace: &[f64], stride: usize, dt: f64, omega: f64) -> f64 {
    let n = trace.len();
    let (mut re, mut im) = (0.0f64, 0.0f64);
    for (m, &v) in trace.iter().enumerate() {
        let t = (m * stride + 1) as f64 * dt;
        let w = 0.5 * (1.0 - (2.0 * PI * m as f64 / (n as f64 - 1.0)).cos());
        re += v * w * (omega * t).cos();
        im -= v * w * (omega * t).sin();
    }
    re.hypot(im)
}

/// (a) THE F3 anchor: a vacuum box with PEC walls (npml = 0 — the outer
/// tangential-E ring is exactly PEC) resonates at
/// f_mnp = ½·√((m/Lx)² + (n/Ly)² + (p/Lz)²) (c = 1). An Ez dipole couples
/// only to the TM_mnp family (Ez ≠ 0 requires m, n ≥ 1), so the band
/// 0.16 < ω < 0.32 holds exactly three lines: TM110, TM111, TM210. We
/// ring the cavity with a broadband modulated-Gaussian dipole, Hann-DFT
/// the probe trace, and match every located peak to the closed form
/// within 1%. At ~20–30 cells per wavelength and S = 0.5 the Yee grid
/// dispersion shifts these resonances by < 0.1%, far inside the
/// tolerance; the dominant measurement error is the scan/window
/// resolution (Δω_scan = 5·10⁻⁴, Hann mainlobe ≈ 7·10⁻³ over the
/// T ≈ 1732 window).
#[test]
fn cavity_modes_match_analytic() {
    let job: Field3Job = serde_json::from_value(serde_json::json!({
        "nx": 25, "ny": 21, "nz": 17, "steps": 6000,
        "npml": 0,
        "eps": { "kind": "uniform" },
        "source": { "i": 7, "j": 6, "k": 5, "axis": "z",
                    "time": { "type": "modulated_gaussian",
                              "omega": 0.26, "t0": 60.0, "tau": 15.0 } },
        "probes": [ { "i": 15, "j": 9, "k": 11, "component": "ez" } ]
    }))
    .expect("Field3Job JSON schema");
    let r = run_job3(&job, &mut |_| {});
    let dt = r.outputs["dt"].as_f64().unwrap();
    let stride = r.outputs["probes"][0]["stride"].as_u64().unwrap() as usize;
    let trace: Vec<f64> = r.outputs["probes"][0]["trace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    let (lx, ly, lz) = (24.0f64, 20.0, 16.0);
    let mode = |m: f64, n: f64, p: f64| {
        PI * ((m / lx).powi(2) + (n / ly).powi(2) + (p / lz).powi(2)).sqrt()
    };
    let expected = [
        mode(1.0, 1.0, 0.0),
        mode(1.0, 1.0, 1.0),
        mode(2.0, 1.0, 0.0),
    ];

    let dw = 5e-4;
    let omegas: Vec<f64> = (0..=320).map(|m| 0.16 + dw * m as f64).collect();
    let amp: Vec<f64> = omegas
        .iter()
        .map(|&w| hann_dft(&trace, stride, dt, w))
        .collect();
    let max = amp.iter().fold(0.0f64, |a, &b| a.max(b));
    let mut peaks = Vec::new();
    for m in 1..amp.len() - 1 {
        if amp[m] > amp[m - 1] && amp[m] >= amp[m + 1] && amp[m] > 0.2 * max {
            // Parabolic sub-bin refinement.
            let d = 0.5 * (amp[m - 1] - amp[m + 1]) / (amp[m - 1] - 2.0 * amp[m] + amp[m + 1]);
            peaks.push(omegas[m] + d * dw);
        }
    }
    eprintln!("cavity peaks: measured {peaks:?}, analytic {expected:?}");
    assert_eq!(
        peaks.len(),
        3,
        "expected exactly TM110/TM111/TM210 in band, got {peaks:?}"
    );
    for (got, want) in peaks.iter().zip(&expected) {
        let rel = (got - want).abs() / want;
        assert!(
            rel < 0.01,
            "cavity mode {got:.6} vs analytic {want:.6} (rel err {rel:.4})"
        );
    }
}

/// (b) 3D numerical dispersion: a CW Ez dipole radiates a spherical wave
/// whose on-axis (equatorial-plane) phase is exactly k·r; DFT monitors on
/// the +x axis give the phase gradient, least-squares fitted and compared
/// to the axis-aligned 3D Yee relation — which is the same closed form as
/// 2D, sin(kΔ/2) = (Δ/cΔt)·sin(ωΔt/2), since the transverse sin² terms
/// vanish for axis propagation. λ ≈ 6.9 cells is deliberately coarse so
/// the grid dispersion (≈ 3.5%) well exceeds the 1% test tolerance.
#[test]
fn dispersion_matches_yee_relation_3d() {
    let (nx, ny, nz, npml) = (120usize, 51usize, 51usize, 8usize);
    let mut g = vacuum_grid3(nx, ny, nz, npml, 0.5);
    let dt = g.dt;

    let period_steps = 24usize;
    let omega = 2.0 * PI / (period_steps as f64 * dt);
    let src = Dipole3 {
        i: 30,
        j: 25,
        k: 25,
        axis: Axis::Z,
        time: TimeProfile::Cw {
            omega,
            ramp: 4.0 * period_steps as f64 * dt,
        },
        amplitude: 1.0,
    };

    let xs: Vec<usize> = (60..=90).step_by(2).collect();
    let mut mons: Vec<Dft3Monitor> = xs
        .iter()
        .map(|&i| Dft3Monitor::new(Comp::Ez, i, 25, 25, vec![omega]))
        .collect();

    let warmup = 450usize;
    let accumulate = 14 * period_steps;
    for n in 0..warmup + accumulate {
        g.step();
        let t = (n + 1) as f64 * dt;
        src.inject(&mut g, t);
        if n >= warmup {
            for m in &mut mons {
                m.record(&g, t);
            }
        }
    }

    let phases: Vec<f64> = mons
        .iter()
        .map(|m| {
            let a = m.accum()[0];
            a.im.atan2(a.re)
        })
        .collect();
    let mut unwrapped = vec![phases[0]];
    for w in phases.windows(2) {
        let mut d = w[1] - w[0];
        while d > PI {
            d -= 2.0 * PI;
        }
        while d < -PI {
            d += 2.0 * PI;
        }
        unwrapped.push(unwrapped.last().unwrap() + d);
    }
    let n = xs.len() as f64;
    let xm = xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let ym = unwrapped.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx) = (0.0, 0.0);
    for (&x, &y) in xs.iter().zip(&unwrapped) {
        sxy += (x as f64 - xm) * (y - ym);
        sxx += (x as f64 - xm) * (x as f64 - xm);
    }
    let k_measured = -(sxy / sxx); // outgoing wave: φ(x) = −k·x + const

    let k_theory = yee_axis_wavenumber(omega, dt);
    assert!(k_theory.is_finite());
    let rel = (k_measured - k_theory).abs() / k_theory;
    eprintln!("3D dispersion: k_measured {k_measured:.6}, Yee {k_theory:.6}, rel {rel:.5}");
    assert!(
        rel < 0.01,
        "measured k = {k_measured:.6}, Yee k = {k_theory:.6}, rel err {rel:.4}"
    );
    // Sharpness: grid dispersion itself must exceed the tolerance.
    assert!((k_theory - omega).abs() / omega > 0.015);
}

/// Gaussian dipole pulse on a vacuum 3D grid; returns the Ez probe trace.
fn pulse_probe_trace_3d(
    (nx, ny, nz): (usize, usize, usize),
    src: (usize, usize, usize),
    prb: (usize, usize, usize),
    steps: usize,
    (t0, tau): (f64, f64),
) -> Vec<f64> {
    let mut g = vacuum_grid3(nx, ny, nz, 10, 0.5);
    let dt = g.dt;
    let src = gaussian_dipole(src.0, src.1, src.2, t0, tau);
    let mut probe = Probe3::new(Probe3Spec {
        i: prb.0,
        j: prb.1,
        k: prb.2,
        component: Comp::Ez,
    });
    for n in 0..steps {
        g.step();
        src.inject(&mut g, (n + 1) as f64 * dt);
        probe.record(&g);
    }
    probe.trace
}

/// (c) 3D CPML reflection floor, by the same reference-subtraction
/// technique as F1: the reference grid is identical except extended along
/// +x, so reflections from the five common faces are **identical in both
/// runs and cancel exactly** in the difference; what remains at the probe
/// is purely the small grid's +x-face reflection (the reference's own +x
/// face is too far to contribute inside the window: earliest return
/// t = 231 vs window end t ≈ 173). Roadmap floor for F1 was −50 dB at
/// strict normal incidence; the 3D spherical wavefront hits the face over
/// a cone of angles, so the F3 requirement is < −40 dB (achieved value
/// printed and asserted; measured ≈ −59 dB at npml = 10).
#[test]
fn cpml3_reflection_below_minus_40_db() {
    let steps = 600;
    let pulse = (24.0, 8.0);
    let small = pulse_probe_trace_3d((60, 44, 44), (20, 22, 22), (45, 22, 22), steps, pulse);
    let reference = pulse_probe_trace_3d((160, 44, 44), (20, 22, 22), (45, 22, 22), steps, pulse);

    let incident_peak = reference.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let reflected_peak = small
        .iter()
        .zip(&reference)
        .fold(0.0f64, |a, (&x, &y)| a.max((x - y).abs()));
    assert!(incident_peak > 0.0);
    let ratio = reflected_peak / incident_peak;
    let db = 20.0 * ratio.log10();
    eprintln!("3D CPML reflection: {ratio:.2e} ({db:.1} dB)");
    assert!(
        ratio < 1e-2,
        "CPML reflection {ratio:.2e} ({db:.1} dB), need < 1e-2 (−40 dB)"
    );
}

/// (d) 3D CFL stability boundary: Δt = S·Δ/√3 is stable iff S ≤ 1
/// (Taflove & Hagness §4.7; the worst mode at the band corner grows
/// ≈ ×(S + √(S²−1))² ≈ ×1.49 per step at S = 1.02). S = 1.02 must blow
/// up; S = 0.98 must stay bounded.
#[test]
fn courant_stability_boundary_3d() {
    let run = |courant: f64, steps: usize| -> f64 {
        let mut g = vacuum_grid3(40, 40, 40, 8, courant);
        let dt = g.dt;
        let src = gaussian_dipole(20, 20, 20, 10.0, 3.0);
        for n in 0..steps {
            g.step();
            src.inject(&mut g, (n + 1) as f64 * dt);
        }
        g.max_abs_e()
    };

    let unstable = run(1.02, 400);
    assert!(
        unstable > 1e6,
        "S = 1.02 should diverge, max = {unstable:e}"
    );
    let stable = run(0.98, 1200);
    assert!(
        stable.is_finite() && stable < 1e2,
        "S = 0.98 should stay bounded, max = {stable:e}"
    );
}

/// Amplitude transmission |E(ω_c)|with-slab / |E(ω_c)|vacuum through a
/// full-cross-section material slab: two executor runs (device and
/// reference) on identical geometry, DFT-probed past the slab at the
/// pulse carrier. Spherical spreading and grid dispersion cancel in the
/// ratio.
fn slab_transmission(
    materials: serde_json::Value,
    slab: (usize, usize),
    omega_c: f64,
    (t0, tau): (f64, f64),
    steps: usize,
) -> f64 {
    let run = |mats: serde_json::Value| -> f64 {
        let job: Field3Job = serde_json::from_value(serde_json::json!({
            "nx": 78, "ny": 36, "nz": 36, "steps": steps,
            "npml": 8,
            "eps": { "kind": "uniform" },
            "materials": mats,
            "source": { "i": 12, "j": 18, "k": 18, "axis": "z",
                        "time": { "type": "modulated_gaussian",
                                  "omega": omega_c, "t0": t0, "tau": tau } },
            "probes": [ { "i": 60, "j": 18, "k": 18, "component": "ez" } ],
            "frequencies": [omega_c]
        }))
        .expect("Field3Job JSON schema");
        let r = run_job3(&job, &mut |_| {});
        assert!(
            r.outputs.get("error").is_none(),
            "job error: {}",
            r.outputs["error"]
        );
        r.outputs["probes"][0]["dft"][0]["abs"].as_f64().unwrap()
    };
    let _ = slab; // slab bounds live inside `materials`; kept for call-site clarity
    let with = run(materials);
    let without = run(serde_json::json!([]));
    assert!(without > 0.0);
    with / without
}

/// (e₁) Drude below the plasma frequency: a lossless (γ = 0) plasma slab
/// admits only evanescent fields below ωp — at ω = ωp/2 and 12 cells
/// thickness the analytic decay alone is e^{−√(ωp²−ω²)·d} ≈ 5.5·10⁻³ in
/// amplitude, so transmitted **power** ≪ 1%; with γ = 0 nothing is
/// absorbed, hence ≥ 99% of the power reflects.
#[test]
fn drude_below_plasma_reflects() {
    let t = slab_transmission(
        serde_json::json!([{
            "model": { "kind": "drude", "omega_p": 0.5, "gamma": 0.0 },
            "i0": 34, "i1": 46, "j0": 0, "j1": 36, "k0": 0, "k1": 36
        }]),
        (34, 46),
        0.25,
        (120.0, 30.0),
        950,
    );
    eprintln!("Drude below ωp: power transmission {:.2e}", t * t);
    assert!(
        t * t < 0.01,
        "below ωp the slab must reflect ≥ 99% power, transmitted {:.2e}",
        t * t
    );
}

/// (e₂) Drude well above the plasma frequency: at ω = 2ωp the slab is a
/// transparent dielectric with n = √(1 − ωp²/ω²) ≈ 0.866 (interface
/// reflectance < 1%), so most of the power transmits — the property
/// bracket complementing (e₁).
#[test]
fn drude_above_plasma_transmits() {
    let t = slab_transmission(
        serde_json::json!([{
            "model": { "kind": "drude", "omega_p": 0.5, "gamma": 0.0 },
            "i0": 34, "i1": 46, "j0": 0, "j1": 36, "k0": 0, "k1": 36
        }]),
        (34, 46),
        1.0,
        (60.0, 15.0),
        600,
    );
    eprintln!("Drude above ωp: power transmission {:.3}", t * t);
    assert!(
        t * t > 0.5,
        "above ωp the slab must transmit, got power {:.3}",
        t * t
    );
}

/// (f) Lorentz bracket: on resonance (ω = ω0) the single-pole medium has
/// ε ≈ ε∞ + i·ωp²/(γω0) = 1 + 6i — Im(n) ≈ 1.6 absorbs the pulse inside
/// a 10-cell slab (analytic bulk decay e^{−ω·Im(n)·d} ≈ 8·10⁻³); off
/// resonance at ω = ω0/2 the same slab is a nearly lossless dielectric
/// (ε ≈ 2.32, Im(n) ≈ 0.05) and transmits.
#[test]
fn lorentz_resonance_absorption_bracket() {
    let slab = serde_json::json!([{
        "model": { "kind": "lorentz", "omega_p": 0.3, "omega0": 0.3, "gamma": 0.05 },
        "i0": 34, "i1": 44, "j0": 0, "j1": 36, "k0": 0, "k1": 36
    }]);
    let t_on = slab_transmission(slab.clone(), (34, 44), 0.3, (200.0, 50.0), 1400);
    let t_off = slab_transmission(slab, (34, 44), 0.15, (200.0, 50.0), 1450);
    eprintln!("Lorentz: on-resonance |T| {t_on:.3e}, off-resonance |T| {t_off:.3}");
    assert!(
        t_on < 0.05,
        "on-resonance amplitude transmission {t_on:.3} should be < 0.05"
    );
    assert!(
        t_off > 0.6,
        "off-resonance amplitude transmission {t_off:.3} should be > 0.6"
    );
    assert!(t_on < 0.1 * t_off, "bracket: {t_on:.3e} !< 0.1·{t_off:.3}");
}

/// (g) ADE vacuum limit: with ωp = 0 both recurrences keep their
/// auxiliary state at exactly 0.0 and the post-pass subtracts literal
/// zeros, so a "dispersive" run must reproduce the vacuum run **bit for
/// bit** (the test allows 1e-12 but expects 0).
#[test]
fn ade_vacuum_limit_matches_vacuum() {
    let base = serde_json::json!({
        "nx": 30, "ny": 26, "nz": 22, "steps": 400,
        "npml": 6,
        "eps": { "kind": "uniform" },
        "source": { "i": 9, "j": 13, "k": 11, "axis": "z",
                    "time": { "type": "gaussian", "t0": 20.0, "tau": 6.0 } },
        "probes": [ { "i": 21, "j": 13, "k": 11, "component": "ez" } ]
    });
    let mut with = base.clone();
    with["materials"] = serde_json::json!([
        { "model": { "kind": "drude", "omega_p": 0.0, "gamma": 0.3 },
          "i0": 13, "i1": 17, "j0": 4, "j1": 22, "k0": 4, "k1": 18 },
        { "model": { "kind": "lorentz", "omega_p": 0.0, "omega0": 0.4, "gamma": 0.1 },
          "i0": 17, "i1": 19, "j0": 4, "j1": 22, "k0": 4, "k1": 18 }
    ]);
    let job_vac: Field3Job = serde_json::from_value(base).unwrap();
    let job_disp: Field3Job = serde_json::from_value(with).unwrap();
    let r_vac = run_job3(&job_vac, &mut |_| {});
    let r_disp = run_job3(&job_disp, &mut |_| {});
    let tr = |r: &tei_sim_core::exec::ExecutionResult| -> Vec<f64> {
        r.outputs["probes"][0]["trace"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };
    let (a, b) = (tr(&r_vac), tr(&r_disp));
    assert_eq!(a.len(), b.len());
    let max_diff = a
        .iter()
        .zip(&b)
        .fold(0.0f64, |m, (&x, &y)| m.max((x - y).abs()));
    assert!(
        max_diff <= 1e-12,
        "ωp → 0 must reduce to vacuum, max trace diff {max_diff:e}"
    );
}

/// (h₁) Determinism: identical jobs are bit-identical (no RNG anywhere in
/// the field column). Also exercises the full Field3Job schema end to end
/// — dielectric box + Drude region + probe + DFT + e_mag snapshot — and
/// pins the ledger convention: macs = 6 component updates × cells ×
/// steps, adc_samples = one read per monitor per step.
#[test]
fn executor3_is_deterministic() {
    let job: Field3Job = serde_json::from_str(
        r#"{
            "nx": 36, "ny": 30, "nz": 26, "steps": 300,
            "eps": { "kind": "box", "eps_r": 4.0,
                     "i0": 20, "i1": 26, "j0": 8, "j1": 22, "k0": 6, "k1": 20 },
            "materials": [
                { "model": { "kind": "drude", "omega_p": 0.6, "gamma": 0.02 },
                  "i0": 6, "i1": 10, "j0": 8, "j1": 22, "k0": 6, "k1": 20 }
            ],
            "source": { "i": 13, "j": 15, "k": 13, "axis": "z",
                        "time": { "type": "gaussian", "t0": 20.0, "tau": 6.0 } },
            "probes": [ { "i": 30, "j": 15, "k": 13 } ],
            "frequencies": [0.3, 0.5],
            "snapshot": { "axis": "z", "index": 13, "field": "e_mag" }
        }"#,
    )
    .expect("Field3Job JSON schema");

    let exec = Field3Executor;
    let mut last_fraction = 0.0;
    let mut ticks = 0u32;
    let mut on_progress = |p: Progress| {
        last_fraction = p.fraction;
        ticks += 1;
    };
    let r1 = exec.execute(&job, &mut on_progress);
    let r2 = exec.execute(&job, &mut |_| {});

    assert_eq!(r1.outputs, r2.outputs, "runs must be bit-identical");
    assert!((last_fraction - 1.0).abs() < 1e-12);
    assert!(ticks >= 100, "~1% progress cadence");

    assert_eq!(r1.ledger.macs, 6 * 36 * 30 * 26 * 300);
    // probe reads + DFT reads, one per step per monitor
    assert_eq!(r1.ledger.adc_samples, 300 + 300);

    let dft = r1.outputs["probes"][0]["dft"].as_array().unwrap();
    assert_eq!(dft.len(), 2);
    assert!(dft[0]["abs"].as_f64().unwrap() > 0.0);
    let trace = r1.outputs["probes"][0]["trace"].as_array().unwrap();
    assert_eq!(trace.len(), 300);
    let snap = &r1.outputs["snapshot"];
    assert_eq!(snap["n0"].as_u64().unwrap(), 36);
    assert_eq!(snap["n1"].as_u64().unwrap(), 30);
    let data = snap["data"].as_array().unwrap();
    assert_eq!(data.len(), 36 * 30);
    assert!(data.iter().all(|v| v.as_f64().unwrap().is_finite()));
}

/// (h₂) Thread-count invariance: the slab decomposition has fixed
/// boundaries and no reductions, so a 1-thread rayon pool and a 4-thread
/// pool must produce bit-identical outputs (grid chosen above the
/// serial-fallback threshold so the parallel path actually runs).
#[test]
fn rayon_thread_count_invariant() {
    let job: Field3Job = serde_json::from_value(serde_json::json!({
        "nx": 48, "ny": 40, "nz": 40, "steps": 150,
        "eps": { "kind": "box", "eps_r": 2.25,
                 "i0": 28, "i1": 36, "j0": 10, "j1": 30, "k0": 10, "k1": 30 },
        "source": { "i": 14, "j": 20, "k": 20, "axis": "y",
                    "time": { "type": "gaussian", "t0": 16.0, "tau": 5.0 } },
        "probes": [ { "i": 40, "j": 20, "k": 20, "component": "ey" } ],
        "frequencies": [0.4],
        "snapshot": { "axis": "x", "index": 24, "field": "e_mag" }
    }))
    .unwrap();
    assert!(48 * 40 * 40 >= tei_sim_field::grid3::PAR_MIN_CELLS);

    let run_in_pool = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| run_job3(&job, &mut |_| {}))
    };
    let r1 = run_in_pool(1);
    let r4 = run_in_pool(4);
    assert_eq!(
        r1.outputs, r4.outputs,
        "1-thread and 4-thread runs must be bit-identical"
    );
}

/// Schema defaults and backward-compat shape: a minimal job (only grid,
/// steps, eps, source) gets courant 0.5 (Δt = S/√3), npml 10, a
/// grid-center Ez probe, no DFT lines, no snapshot.
#[test]
fn job3_schema_defaults() {
    let job: Field3Job = serde_json::from_str(
        r#"{
            "nx": 30, "ny": 30, "nz": 30, "steps": 50,
            "eps": { "kind": "uniform" },
            "source": { "i": 15, "j": 15, "k": 15,
                        "time": { "type": "gaussian", "t0": 12.0, "tau": 4.0 } }
        }"#,
    )
    .expect("minimal Field3Job");
    assert_eq!(job.courant, 0.5);
    assert_eq!(job.npml, 10);
    assert!(job.materials.is_empty() && job.probes.is_empty());
    assert!(job.frequencies.is_empty() && job.snapshot.is_none());
    assert!(matches!(job.source.axis, Axis::Z), "axis defaults to z");

    let r = run_job3(&job, &mut |_| {});
    assert!(r.outputs.get("error").is_none());
    let dt = r.outputs["dt"].as_f64().unwrap();
    assert!((dt - DT).abs() < 1e-15);
    let p = &r.outputs["probes"][0];
    assert_eq!(
        (p["i"].as_u64(), p["j"].as_u64(), p["k"].as_u64()),
        (Some(15), Some(15), Some(15)),
        "default probe at grid center"
    );
    assert_eq!(p["component"].as_str(), Some("ez"));
    assert!(r.outputs["snapshot"].is_null());

    // Error convention: invalid jobs report instead of panicking.
    let bad = Field3Job {
        probes: vec![Probe3Spec {
            i: 999,
            j: 0,
            k: 0,
            component: Comp::Ez,
        }],
        ..job
    };
    let r = run_job3(&bad, &mut |_| {});
    assert!(r.outputs["error"].as_str().unwrap().contains("probe"));
    assert_eq!(r.ledger.macs, 0);
}

/// Throughput report (roadmap §6 — informational, no hard assert): cell
/// updates per second on a 64³ vacuum grid with CPML, 500 steps, single
/// rayon thread vs all cores. Run with
/// `cargo test -p tei-sim-field --release -- --ignored --nocapture`.
#[test]
#[ignore]
fn bench_3d_throughput() {
    let steps = 500usize;
    let cells = 64usize.pow(3);
    let run_in_pool = |threads: usize| -> f64 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| {
                let mut g = vacuum_grid3(64, 64, 64, 10, 0.5);
                let dt = g.dt;
                let src = gaussian_dipole(32, 32, 32, 24.0, 8.0);
                for n in 0..20 {
                    g.step();
                    src.inject(&mut g, (n + 1) as f64 * dt);
                } // warm-up
                let t0 = std::time::Instant::now();
                for n in 20..20 + steps {
                    g.step();
                    src.inject(&mut g, (n + 1) as f64 * dt);
                }
                let secs = t0.elapsed().as_secs_f64();
                (cells * steps) as f64 / secs / 1e6
            })
    };
    let single = run_in_pool(1);
    let multi = run_in_pool(rayon::current_num_threads().max(2));
    eprintln!(
        "3D FDTD 64³×{steps}: {single:.1} MCells/s single-thread, {multi:.1} MCells/s multi-thread"
    );
    assert!(single > 0.0 && multi > 0.0);
}
