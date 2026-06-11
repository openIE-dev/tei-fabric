//! F2 validation per docs/SIM-ROADMAP.md §3.7 — analytic ground truth and
//! property tests only, no foreign-tool fixtures.
//!
//! Coverage: (a) slab-waveguide effective index vs closed-form anchors of
//! the analytic transcendental dispersion relation, (b) dispersion-relation
//! and mode-profile properties, (c) straight-guide |S₂₁| ≈ 1 and
//! FDTD-propagated n_eff vs the analytic value, (d) two-port passivity,
//! (e) Fabry-Pérot etalon vs the exact Airy formula (the uniform-Δ device
//! section makes the 1D reduction exact — see `DeviceSection` docs),
//! (f) the photonic handoff: extracted S-parameters star-composed with
//! analytic `tei-sim-photonic` components, (g) job-schema backward
//! compatibility and the `outputs.sparams` shape.

use std::f64::consts::{FRAC_PI_2, PI};
use std::sync::OnceLock;

use tei_sim_core::exec::ExecutionResult;
use tei_sim_core::linalg::{C64, CMat};
use tei_sim_field::{ExtractedSparams, FieldJob, SlabWaveguide, run_job, sparams};
use tei_sim_photonic::Sparams;

// ---------------------------------------------------------------- anchors

/// (a) Closed-form anchors: at u = κa = π/4 the TE₀ relation v = u·tan(u)
/// gives v = u, V = π/(2√2); at u = π/3, v = √3·u, V = 2π/3; for TE₁ at
/// u = 3π/4, v = −u·cot(u) = u, V = 3π/(2√2). Choosing parameters that hit
/// these V values makes n_eff exact in closed form.
#[test]
fn te0_effective_index_closed_form_anchors() {
    // u = π/4: a = 1, ε = 2/1, ω = π/(2√2) ⇒ β² = 2ω² − π²/16 = 3π²/16,
    // n_eff = β/ω = √6/2.
    let wg = SlabWaveguide {
        eps_core: 2.0,
        eps_clad: 1.0,
        half_width: 1.0,
    };
    let omega = PI / (2.0 * 2f64.sqrt());
    let m = wg.solve(omega, 0).expect("TE0 always guided");
    assert!(
        (m.n_eff - 6f64.sqrt() / 2.0).abs() < 1e-9,
        "TE0 anchor: n_eff = {:.12}, closed form {:.12}",
        m.n_eff,
        6f64.sqrt() / 2.0
    );
    assert!((m.kappa - PI / 4.0).abs() < 1e-9);
    assert!((m.gamma - PI / 4.0).abs() < 1e-9);

    // u = π/3: v = √3·u, V = 2π/3. With a = 1, ε = 3.25/2.25, ω = 2π/3:
    // β² = 3.25·ω² − π²/9 = 4π²/3, n_eff = √3.
    let wg2 = SlabWaveguide {
        eps_core: 3.25,
        eps_clad: 2.25,
        half_width: 1.0,
    };
    let m2 = wg2.solve(2.0 * PI / 3.0, 0).expect("guided");
    assert!(
        (m2.n_eff - 3f64.sqrt()).abs() < 1e-9,
        "second TE0 anchor: n_eff = {:.12}, closed form {:.12}",
        m2.n_eff,
        3f64.sqrt()
    );

    // TE₁ anchor, u = 3π/4: a = 1, ε = 2/1, ω = 3π/(2√2) ⇒ n_eff = √6/2.
    let m3 = wg
        .solve(3.0 * PI / (2.0 * 2f64.sqrt()), 1)
        .expect("TE1 guided above V = π/2");
    assert!(
        (m3.n_eff - 6f64.sqrt() / 2.0).abs() < 1e-9,
        "TE1 anchor: n_eff = {:.12}",
        m3.n_eff
    );
}

/// (b) Dispersion-relation properties: bounded residual at the root,
/// n_eff ∈ (n_clad, n_core), monotone n_eff(ω) for TE₀, and the V-number
/// cutoff ladder (TE_m guided iff V > mπ/2).
#[test]
fn mode_solver_dispersion_properties() {
    let cases = [
        (4.0, 1.0, 3.0),
        (2.25, 1.0, 5.0),
        (12.0, 2.0, 1.5),
        (4.0, 3.9, 8.0),
    ];
    for &(eps_core, eps_clad, a) in &cases {
        let wg = SlabWaveguide {
            eps_core,
            eps_clad,
            half_width: a,
        };
        for &omega in &[0.08, 0.2, 0.5, 1.1] {
            let count = wg.mode_count(omega);
            assert!(count >= 1, "TE0 has no cutoff");
            for m in 0..count {
                let md = wg.solve(omega, m).expect("guided below count");
                // Bounded residual form of v = u·tan(u − mπ/2):
                // u·sin(δ) − v·cos(δ) = 0 with δ = u − mπ/2.
                let (u, v) = (md.kappa * a, md.gamma * a);
                let d = u - m as f64 * FRAC_PI_2;
                let resid = u * d.sin() - v * d.cos();
                assert!(
                    resid.abs() < 1e-9,
                    "dispersion residual {resid:e} (eps {eps_core}/{eps_clad}, a {a}, ω {omega}, m {m})"
                );
                assert!(md.n_eff > eps_clad.sqrt() && md.n_eff < eps_core.sqrt());
            }
            // First non-guided order must return None.
            assert!(wg.solve(omega, count).is_none(), "cutoff ladder broken");
        }
        // n_eff(ω) is strictly increasing for TE0.
        let mut prev = 0.0;
        for k in 1..=20 {
            let n = wg.solve(0.05 * k as f64, 0).unwrap().n_eff;
            assert!(n > prev, "n_eff(ω) must increase");
            prev = n;
        }
    }
}

/// (b') Mode-profile properties: continuity at the core interface, parity,
/// unit-L2 sampling, and exact TE₀ ⊥ TE₁ orthogonality on a symmetric grid.
#[test]
fn mode_profile_properties() {
    let wg = SlabWaveguide {
        eps_core: 4.0,
        eps_clad: 1.0,
        half_width: 3.0,
    };
    let omega = 0.6; // V = 3.12: TE0 + TE1 guided, TE2 not
    assert_eq!(wg.mode_count(omega), 2);
    for m in 0..2 {
        let md = wg.solve(omega, m).unwrap();
        let a = md.half_width;
        for s in [a, -a] {
            let inside = md.profile(s * (1.0 - 1e-9));
            let outside = md.profile(s * (1.0 + 1e-9));
            assert!(
                (inside - outside).abs() < 1e-6,
                "profile discontinuous at the interface: {inside} vs {outside}"
            );
        }
        // Parity about the center.
        let sign = if m % 2 == 0 { 1.0 } else { -1.0 };
        for s in [0.7, 1.9, 4.2] {
            assert!((md.profile(s) - sign * md.profile(-s)).abs() < 1e-12);
        }
    }

    let (ny, yc) = (80usize, 39.5);
    let p0 = wg.solve(omega, 0).unwrap().sample(ny, yc);
    let p1 = wg.solve(omega, 1).unwrap().sample(ny, yc);
    let l2 = |p: &[f64]| p.iter().map(|x| x * x).sum::<f64>();
    assert!((l2(&p0) - 1.0).abs() < 1e-12, "unit L2 norm");
    assert!((l2(&p1) - 1.0).abs() < 1e-12, "unit L2 norm");
    let dot: f64 = p0.iter().zip(&p1).map(|(a, b)| a * b).sum();
    assert!(dot.abs() < 1e-12, "TE0 ⊥ TE1 (even × odd), got {dot:e}");
}

// ------------------------------------------------ straight-guide fixture

/// Straight single-mode guide: ε 4/1, core 6 cells (a = 3), band centred on
/// ω = 0.25 (V = 1.30 — single-mode for the whole test band), ports 100
/// cells apart. Shared by the |S₂₁| / n_eff / passivity / handoff / schema
/// tests so the two FDTD runs happen once.
fn straight_job() -> FieldJob {
    serde_json::from_value(serde_json::json!({
        "nx": 240, "ny": 80, "steps": 4200,
        "eps": {"kind": "waveguide_x", "eps_r": 4.0, "j0": 37, "j1": 43},
        "mode_source": {
            "i": 30, "omega": 0.25,
            "time": {"type": "modulated_gaussian", "omega": 0.25, "t0": 130.0, "tau": 40.0}
        },
        "frequencies": [0.22, 0.25, 0.28],
        "ports": [{"i": 60}, {"i": 160}]
    }))
    .expect("straight-guide job schema")
}

fn straight_wg() -> SlabWaveguide {
    SlabWaveguide {
        eps_core: 4.0,
        eps_clad: 1.0,
        half_width: 3.0,
    }
}

fn straight() -> &'static (ExtractedSparams, ExecutionResult) {
    static FIXTURE: OnceLock<(ExtractedSparams, ExecutionResult)> = OnceLock::new();
    FIXTURE.get_or_init(|| sparams::extract(&straight_job(), &mut |_| {}).expect("extraction"))
}

/// (c) Straight lossless guide: |S₂₁| ≈ 1 at every requested frequency.
/// The deviation budget is mode-profile discretization (the continuum
/// profile sampled on the Yee rows sheds a little radiation at launch) plus
/// the Riemann DFT; both are sub-percent at this resolution.
#[test]
fn straight_waveguide_s21_near_unity() {
    let (ex, _) = straight();
    for (fi, &w) in ex.frequencies.iter().enumerate() {
        let s21 = ex.s_col[1][fi].abs();
        println!("straight guide: omega = {w}, |S21| = {s21:.6}");
        assert!(
            (s21 - 1.0).abs() < 0.03,
            "|S21| = {s21:.5} at omega = {w} (want within 3% of 1)"
        );
    }
}

/// (c') FDTD-propagated effective index: arg S₂₁ = β·L over the L = 100
/// cell port spacing. Unwrapped with the analytic branch (safe — the
/// margin is π/(β·L) ≈ 7%, well above the gap), the measured n_eff must
/// land within 2% of the analytic transcendental solution; the residual is
/// Yee grid dispersion (λ_core ≈ 12.5 cells here).
#[test]
fn straight_waveguide_fdtd_neff_matches_analytic() {
    let (ex, _) = straight();
    let wg = straight_wg();
    let l = 100.0;
    for (fi, &w) in ex.frequencies.iter().enumerate() {
        let analytic = wg.solve(w, 0).unwrap();
        let s21 = ex.s_col[1][fi];
        let phase = s21.im.atan2(s21.re);
        let m = ((analytic.beta * l - phase) / (2.0 * PI)).round();
        let beta_meas = (phase + 2.0 * PI * m) / l;
        let n_meas = beta_meas / w;
        let rel = (n_meas - analytic.n_eff).abs() / analytic.n_eff;
        println!(
            "omega = {w}: n_eff analytic = {:.6}, FDTD = {n_meas:.6}, rel err = {rel:.5}",
            analytic.n_eff
        );
        assert!(
            rel < 0.02,
            "n_eff: FDTD {n_meas:.5} vs analytic {:.5} (rel {rel:.4}) at omega = {w}",
            analytic.n_eff
        );
    }
}

/// (d) Two-port passivity: |S₁₁|² + |S₂₁|² ≤ 1 + tolerance for the lossless
/// guide. S₁₁ here is *identically* zero by construction — the device equals
/// the reference, so the subtraction cancels bit-exactly; non-trivial
/// reflection extraction is validated by the etalon test below.
#[test]
fn straight_waveguide_two_port_passivity() {
    let (ex, _) = straight();
    for (fi, &w) in ex.frequencies.iter().enumerate() {
        let (s11, s21) = (ex.s_col[0][fi], ex.s_col[1][fi]);
        assert!(
            s11.abs() < 1e-12,
            "straight-guide S11 must cancel exactly, got {:e}",
            s11.abs()
        );
        let sum = s11.norm_sq() + s21.norm_sq();
        println!("omega = {w}: |S11|^2 + |S21|^2 = {sum:.6}");
        assert!(
            sum <= 1.0 + 0.05,
            "passivity violated at omega = {w}: {sum:.5}"
        );
    }
}

// ----------------------------------------------------------- etalon test

/// (e) Fabry-Pérot etalon vs the exact Airy formula. The device section
/// adds Δ = 1.85 to ε over every row of a 16-column span, which keeps the
/// transverse mode profile identical (the shift cancels out of the
/// transverse eigenproblem) and makes the structure an exact 1D etalon for
/// the modal amplitude: sections with β₁ and β₂ = √(β₁² + Δ·k₀²), Fresnel
/// coefficient r = (β₁−β₂)/(β₁+β₂), and
///
/// ```text
///   S11 = r·(1 − e^{2iβ₂d})/(1 − r²·e^{2iβ₂d})
///   |S21| = |1 − r²| / |1 − r²·e^{2iβ₂d}|
/// ```
///
/// β₁ from the analytic transcendental solution — closed-form ground truth
/// end to end. The grid leaves ≈ 17 cells per medium wavelength inside the
/// section, so Yee dispersion shifts the fringe phase by well under the
/// test tolerance.
#[test]
fn etalon_matches_airy_formula() {
    let job: FieldJob = serde_json::from_value(serde_json::json!({
        "nx": 220, "ny": 110, "steps": 5200,
        "eps": {
            "kind": "waveguide_x", "eps_r": 4.0, "j0": 45, "j1": 55,
            "device": {"delta": 1.85, "i0": 110, "i1": 126}
        },
        "mode_source": {
            "i": 30, "omega": 0.15,
            "time": {"type": "modulated_gaussian", "omega": 0.15, "t0": 230.0, "tau": 70.0}
        },
        "frequencies": [0.138, 0.15, 0.162],
        "ports": [{"i": 60}, {"i": 170}]
    }))
    .expect("etalon job schema");
    let (ex, _) = sparams::extract(&job, &mut |_| {}).expect("extraction");

    let wg = SlabWaveguide {
        eps_core: 4.0,
        eps_clad: 1.0,
        half_width: 5.0,
    };
    let (delta, d) = (1.85, 16.0);
    for (fi, &w) in ex.frequencies.iter().enumerate() {
        let beta1 = wg.solve(w, 0).unwrap().beta;
        let beta2 = (beta1 * beta1 + delta * w * w).sqrt();
        let r = (beta1 - beta2) / (beta1 + beta2);
        let ph = C64::from_polar(1.0, 2.0 * beta2 * d);
        let den = C64::ONE - ph * (r * r);
        let s11_an = (C64::ONE - ph) * r / den;
        let s21_an_mag = (1.0 - r * r) / den.abs();

        let (s11, s21) = (ex.s_col[0][fi], ex.s_col[1][fi]);
        let sum = s11.norm_sq() + s21.norm_sq();
        println!(
            "etalon omega = {w}: |S11|^2 = {:.5} (Airy {:.5}), |S21|^2 = {:.5} (Airy {:.5}), sum = {sum:.5}",
            s11.norm_sq(),
            s11_an.norm_sq(),
            s21.norm_sq(),
            s21_an_mag * s21_an_mag
        );
        assert!(
            (s11.norm_sq() - s11_an.norm_sq()).abs() < 0.01,
            "|S11|² {:.5} vs Airy {:.5} at omega = {w}",
            s11.norm_sq(),
            s11_an.norm_sq()
        );
        assert!(
            (s21.norm_sq() - s21_an_mag * s21_an_mag).abs() < 0.02,
            "|S21|² {:.5} vs Airy {:.5} at omega = {w}",
            s21.norm_sq(),
            s21_an_mag * s21_an_mag
        );
        // Lossless two-port energy property.
        assert!(
            (sum - 1.0).abs() < 0.03,
            "energy sum {sum:.5} at omega = {w}"
        );
    }
}

// ------------------------------------------------------- photonic handoff

/// (f) The device→circuit closed loop: FDTD-extracted straight guide →
/// `tei_sim_photonic::Sparams` → Redheffer star with analytic photonic
/// components. With a reflectionless analytic waveguide downstream the
/// composite must factor exactly (S₂₁ = t·S₂₁ᶠᵈᵗᵈ, S₁₁ = S₁₁ᶠᵈᵗᵈ); with a
/// reflective interface the full star-product formula applies and the
/// composite must stay passive.
#[test]
fn extracted_two_port_feeds_photonic_circuit() {
    let (ex, _) = straight();
    let fi = 1; // band centre ω = 0.25
    let dev = ex.two_port(fi);
    let (s11_f, s21_f) = (ex.s_col[0][fi], ex.s_col[1][fi]);

    // (i) Compose with an analytic lossless waveguide section.
    let wg = tei_sim_photonic::Waveguide {
        length_um: 20.0,
        n_eff: 1.8,
        loss_db_per_cm: 0.0,
    };
    let lambda_um = 1.55;
    let t = wg.transfer(lambda_um);
    let comp = dev.star(&Sparams::from_full(&wg.s_matrix(lambda_um), 1));
    let expect_s21 = t * s21_f;
    assert!(
        (comp.s21[(0, 0)] - expect_s21).abs() < 1e-12,
        "reflectionless cascade must factor exactly"
    );
    assert!((comp.s11[(0, 0)] - s11_f).abs() < 1e-12);
    assert!(
        (comp.s21[(0, 0)].abs() - s21_f.abs()).abs() < 1e-12,
        "lossless section must preserve |S21|"
    );

    // (ii) Compose with a reflective interface S = [[r, t'],[t', −r]]
    // (real, unitary) and check the resolvent path of the star product.
    let r = 0.3f64;
    let tp = (1.0 - r * r).sqrt();
    let mut b = CMat::zeros(2, 2);
    b[(0, 0)] = C64::new(r, 0.0);
    b[(0, 1)] = C64::new(tp, 0.0);
    b[(1, 0)] = C64::new(tp, 0.0);
    b[(1, 1)] = C64::new(-r, 0.0);
    let comp2 = dev.star(&Sparams::from_full(&b, 1));
    // Closed-form 2-port star (A22 = S11ᶠ by mirror symmetry):
    let denom = C64::ONE - s11_f * r;
    let expect2_s21 = C64::new(tp, 0.0) * s21_f / denom;
    let expect2_s11 = s11_f + s21_f * s21_f * r / denom;
    assert!((comp2.s21[(0, 0)] - expect2_s21).abs() < 1e-12);
    assert!((comp2.s11[(0, 0)] - expect2_s11).abs() < 1e-12);
    let sum = comp2.s11[(0, 0)].norm_sq() + comp2.s21[(0, 0)].norm_sq();
    println!("composite (guide ⋆ interface): |S11|^2 + |S21|^2 = {sum:.6}");
    // The analytic interface is exactly unitary; the budget is entirely the
    // FDTD extraction noise on |S21ᶠᵈᵗᵈ| (≈ 2·10⁻⁴ here).
    assert!(sum <= 1.0 + 1e-2, "composite must stay passive, got {sum}");
}

// -------------------------------------------------------- schema & serde

/// (g) `outputs.sparams` JSON shape: frequencies echo the job, entries are
/// ports-major × frequencies-minor with from/to/omega/re/im/abs/phase, and
/// the redundant abs/phase fields agree with re/im.
#[test]
fn sparams_output_schema() {
    let (_, result) = straight();
    let sp = &result.outputs["sparams"];
    assert_eq!(
        sp["frequencies"].as_array().unwrap().len(),
        3,
        "frequencies echoed"
    );
    assert_eq!(sp["source_port"].as_u64(), Some(1));
    let entries = sp["entries"].as_array().unwrap();
    assert_eq!(entries.len(), 2 * 3, "ports × frequencies");
    for e in entries {
        let (re, im) = (e["re"].as_f64().unwrap(), e["im"].as_f64().unwrap());
        let abs = e["abs"].as_f64().unwrap();
        let phase = e["phase"].as_f64().unwrap();
        assert!((abs - (re * re + im * im).sqrt()).abs() < 1e-12);
        assert!((phase - im.atan2(re)).abs() < 1e-12);
        assert_eq!(e["from"].as_u64(), Some(1));
        let to = e["to"].as_u64().unwrap();
        assert!(to == 1 || to == 2);
    }
    // Entry order: port 1 (3 freqs) then port 2 (3 freqs).
    assert_eq!(entries[0]["to"].as_u64(), Some(1));
    assert_eq!(entries[3]["to"].as_u64(), Some(2));
    // The ledger counts both extraction runs.
    let job = straight_job();
    assert_eq!(
        result.ledger.macs,
        2 * 3 * (job.nx * job.ny * job.steps) as u64
    );
}

/// (g') Schema backward compatibility: an F1-era job (no F2 fields)
/// deserializes with the new optional fields defaulted, and the F2 error
/// paths report through `outputs.error` instead of panicking.
#[test]
fn field_job_serde_backward_compat() {
    // Verbatim F1 job from the F1 validation suite.
    let f1: FieldJob = serde_json::from_str(
        r#"{
            "nx": 80, "ny": 80, "steps": 400,
            "eps": { "kind": "slab", "eps_r": 4.0, "i0": 50, "i1": 58 },
            "source": {
                "shape": { "shape": "point", "i": 25, "j": 40 },
                "time": { "type": "gaussian", "t0": 24.0, "tau": 8.0 }
            },
            "frequencies": [0.2, 0.4],
            "snapshot": true
        }"#,
    )
    .expect("F1 jobs must keep deserializing");
    assert!(f1.source.is_some());
    assert!(f1.mode_source.is_none());
    assert!(f1.ports.is_empty());
    assert!(f1.reference_eps.is_none());
    // Round-trip: the F2 fields stay omitted for F1 jobs.
    let back = serde_json::to_value(&f1).unwrap();
    assert!(back.get("mode_source").is_none());
    assert!(back.get("ports").is_none());

    // Error paths (cheap — they fail validation before any FDTD stepping).
    let no_source: FieldJob = serde_json::from_value(serde_json::json!({
        "nx": 40, "ny": 40, "steps": 10,
        "eps": {"kind": "uniform"}
    }))
    .unwrap();
    let r = run_job(&no_source, &mut |_| {});
    assert!(r.outputs["error"].as_str().unwrap().contains("source"));

    let ports_no_freq: FieldJob = serde_json::from_value(serde_json::json!({
        "nx": 60, "ny": 40, "steps": 10,
        "eps": {"kind": "waveguide_x", "eps_r": 4.0, "j0": 18, "j1": 22},
        "mode_source": {
            "i": 15, "omega": 0.3,
            "time": {"type": "modulated_gaussian", "omega": 0.3, "t0": 40.0, "tau": 12.0}
        },
        "ports": [{"i": 25}, {"i": 45}]
    }))
    .unwrap();
    let r = run_job(&ports_no_freq, &mut |_| {});
    assert!(r.outputs["error"].as_str().unwrap().contains("frequencies"));

    let wrong_eps: FieldJob = serde_json::from_value(serde_json::json!({
        "nx": 60, "ny": 40, "steps": 10,
        "eps": {"kind": "uniform"},
        "frequencies": [0.3],
        "mode_source": {
            "i": 15, "omega": 0.3,
            "time": {"type": "modulated_gaussian", "omega": 0.3, "t0": 40.0, "tau": 12.0}
        },
        "ports": [{"i": 25}, {"i": 45}]
    }))
    .unwrap();
    let r = run_job(&wrong_eps, &mut |_| {});
    assert!(r.outputs["error"].as_str().unwrap().contains("waveguide_x"));
}
