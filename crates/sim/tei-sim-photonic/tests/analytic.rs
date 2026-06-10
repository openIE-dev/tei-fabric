//! Analytic validation for tei-sim-photonic — closed-form math and published
//! results only (docs/SIM-ROADMAP.md §3.4, validation policy §2).
//!
//! (a) component unitarity · (b) MZI closed-form transfer · (c) ring
//! resonances, FSR, critical-coupling extinction · (d) Redheffer identity,
//! phase cascade, associativity · (e) Clements round-trip (Haar U →
//! decompose → rebuild) · (f) MVM correctness + exact ledger counts ·
//! (g) determinism.

use tei_sim_core::exec::Executor;
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::linalg::{C64, CMat};
use tei_sim_core::rng::Rng;
use tei_sim_photonic::{
    AllPassRing, ClementsMesh, OpticalMvm, PhotonicExecutor, PhotonicJob, Sparams, UnitarySpec,
    Waveguide, add_drop_transfer, all_pass_transfer, cascade_2port, directional_coupler,
    haar_unitary, mzi_transfer, phase_shifter,
};

const TAU: f64 = std::f64::consts::TAU;
const PI: f64 = std::f64::consts::PI;

/// ‖S†S − I‖_F.
fn unitarity_err(s: &CMat) -> f64 {
    s.dagger()
        .matmul(s)
        .frobenius_distance(&CMat::identity(s.rows))
}

/// Max-modulus elementwise difference.
fn max_diff(a: &CMat, b: &CMat) -> f64 {
    assert!(a.rows == b.rows && a.cols == b.cols);
    a.data
        .iter()
        .zip(&b.data)
        .map(|(x, y)| (*x - *y).abs())
        .fold(0.0, f64::max)
}

// ───────────────────────── (a) component unitarity ─────────────────────────

/// Every lossless component satisfies S†S = I to 1e-12 across its parameter
/// space (frequency for the waveguide, coupling/phase grids for the rest).
#[test]
fn a_lossless_components_unitary() {
    // Waveguide S-matrix across a wavelength sweep.
    let wg = Waveguide {
        length_um: 137.0,
        n_eff: 2.4,
        loss_db_per_cm: 0.0,
    };
    for k in 0..50 {
        let lambda = 1.50 + 0.002 * k as f64;
        assert!(unitarity_err(&wg.s_matrix(lambda)) < 1e-12);
    }
    // Directional coupler across the full coupling range.
    for k in 0..=20 {
        let s = directional_coupler(k as f64 / 20.0);
        assert!(unitarity_err(&s) < 1e-12);
    }
    // Phase shifter and MZI across (θ, φ) grids.
    for i in 0..24 {
        let theta = TAU * i as f64 / 24.0;
        assert!(unitarity_err(&phase_shifter(theta)) < 1e-12);
        for j in 0..8 {
            let phi = TAU * j as f64 / 8.0;
            assert!(unitarity_err(&mzi_transfer(theta, phi)) < 1e-12);
        }
    }
    // Lossless all-pass ring is all-pass: |H| = 1 for all θ.
    for i in 0..200 {
        let theta = TAU * i as f64 / 100.0;
        let h = all_pass_transfer(0.9, 1.0, theta);
        assert!(
            (h.abs() - 1.0).abs() < 1e-12,
            "|H| = {} at θ = {theta}",
            h.abs()
        );
    }
    // Lossless add-drop conserves power: |t|² + |d|² = 1.
    for i in 0..200 {
        let theta = TAU * i as f64 / 100.0;
        let (t, d) = add_drop_transfer(0.95, 0.9, 1.0, theta);
        assert!((t.norm_sq() + d.norm_sq() - 1.0).abs() < 1e-12);
    }
}

// ───────────────────────── (b) MZI closed form ─────────────────────────

/// Bar transmission |M₀₀(θ)|² = sin²(θ/2) (stated convention), cross
/// |M₁₀|² = cos²(θ/2); θ = 0 ⇒ full cross, θ = π ⇒ full bar.
#[test]
fn b_mzi_transfer_closed_form() {
    for i in 0..=96 {
        let theta = TAU * i as f64 / 96.0;
        for j in 0..6 {
            let phi = TAU * j as f64 / 6.0;
            let m = mzi_transfer(theta, phi);
            let bar = m[(0, 0)].norm_sq();
            let cross = m[(1, 0)].norm_sq();
            let s2 = (theta / 2.0).sin().powi(2);
            assert!((bar - s2).abs() < 1e-12, "θ={theta}: bar {bar} vs {s2}");
            assert!((cross - (1.0 - s2)).abs() < 1e-12);
        }
    }
    // Extremes.
    let m0 = mzi_transfer(0.0, 0.0);
    assert!(m0[(0, 0)].norm_sq() < 1e-12, "θ=0 must be full cross");
    assert!((m0[(1, 0)].norm_sq() - 1.0).abs() < 1e-12);
    let mpi = mzi_transfer(PI, 0.0);
    assert!(
        (mpi[(0, 0)].norm_sq() - 1.0).abs() < 1e-12,
        "θ=π must be full bar"
    );
    assert!(mpi[(1, 0)].norm_sq() < 1e-12);
}

// ───────────────────────── (c) ring resonator ─────────────────────────

/// Resonance dips at θ = 2πm; critical coupling (t = a) extinguishes the
/// through port (< −40 dB); FSR matches λ²/(n_g·L) with n_g = n_eff
/// (dispersion-free) within the sweep resolution.
#[test]
fn c_ring_resonance_fsr_critical_coupling() {
    // Dips exactly at θ = 2πm in the θ domain, critical coupling t = a.
    let (t, a) = (0.9, 0.9);
    for m in 0..5 {
        let h = all_pass_transfer(t, a, TAU * m as f64);
        assert!(
            h.norm_sq() < 1e-20,
            "critical-coupling resonance power {}",
            h.norm_sq()
        );
    }
    // Off resonance the dip recovers: T(θ=π) = (t+a)²/(1+ta)².
    let h_anti = all_pass_transfer(t, a, PI);
    let expect = ((t + a) / (1.0 + t * a)).powi(2);
    assert!((h_anti.norm_sq() - expect).abs() < 1e-12);

    // Wavelength domain: R = 10 µm ring, n_eff = 2.4, critical coupling.
    let ring = AllPassRing {
        circumference_um: TAU * 10.0,
        n_eff: 2.4,
        t: 0.9,
        loss_db_per_cm: 0.0,
    };
    // Round-trip amplitude is 1 (lossless); use a = t via explicit transfer
    // so the dip is a true critical-coupling extinction.
    let a = ring.t;
    let opl = ring.n_eff * ring.circumference_um; // n_eff·L

    // < −40 dB extinction evaluated at an exact resonance λ_m = n_eff·L/m.
    let m_res = (opl / 1.55).round();
    let lambda_res = opl / m_res;
    let h_res = all_pass_transfer(ring.t, a, ring.round_trip_phase(lambda_res));
    assert!(
        h_res.norm_sq() < 1e-4,
        "extinction {} dB",
        10.0 * h_res.norm_sq().log10()
    );

    // Fine sweep 1.50–1.60 µm: find dips, check spacing against closed form.
    let (lo, hi, npts) = (1.50, 1.60, 20_001usize);
    let step = (hi - lo) / (npts - 1) as f64;
    let power: Vec<f64> = (0..npts)
        .map(|i| {
            let lambda = lo + step * i as f64;
            all_pass_transfer(ring.t, a, ring.round_trip_phase(lambda)).norm_sq()
        })
        .collect();
    let mut dips = Vec::new();
    for i in 1..npts - 1 {
        if power[i] < power[i - 1] && power[i] <= power[i + 1] && power[i] < 0.2 {
            dips.push(lo + step * i as f64);
        }
    }
    assert!(
        dips.len() >= 4,
        "expected several dips, found {}",
        dips.len()
    );
    for w in dips.windows(2) {
        let measured_fsr = w[1] - w[0];
        // Closed form λ²/(n_g L) at the geometric-mean wavelength — exact
        // for the dispersion-free comb λ_m = n_eff·L/m, since
        // λ_m·λ_{m+1}/(n_eff·L) = n_eff·L/(m(m+1)) = λ_m − λ_{m+1}.
        let lambda_mid = (w[0] * w[1]).sqrt();
        let analytic_fsr = lambda_mid * lambda_mid / opl;
        assert!(
            (measured_fsr - analytic_fsr).abs() < 3.0 * step,
            "FSR {measured_fsr} vs analytic {analytic_fsr} (step {step})"
        );
    }
}

// ───────────────────────── (d) Redheffer star product ─────────────────────────

/// Star with the through-connection is the identity (to 1e-14); cascading
/// two waveguides multiplies their phases exactly; the product is
/// associative on random passive S-matrices (to 1e-12).
#[test]
fn d_redheffer_identity_cascade_associativity() {
    // Identity: S ⋆ through = through ⋆ S = S, on a lossy waveguide 2-port.
    let wg = Waveguide {
        length_um: 73.0,
        n_eff: 2.21,
        loss_db_per_cm: 2.5,
    };
    let s = Sparams::from_full(&wg.s_matrix(1.55), 1);
    let thru = Sparams::through(1);
    assert!(max_diff(&s.star(&thru).full(), &s.full()) < 1e-14);
    assert!(max_diff(&thru.star(&s).full(), &s.full()) < 1e-14);

    // Phase cascade: wg(φ₁) ⋆ wg(φ₂) has s21 = e^{i(φ₁+φ₂)}.
    let wg1 = Waveguide {
        length_um: 11.0,
        n_eff: 2.4,
        loss_db_per_cm: 0.0,
    };
    let wg2 = Waveguide {
        length_um: 29.0,
        n_eff: 2.4,
        loss_db_per_cm: 0.0,
    };
    let lambda = 1.55;
    let total = cascade_2port(&wg1.s_matrix(lambda), &wg2.s_matrix(lambda));
    let phase_sum = TAU * wg1.n_eff / lambda * (wg1.length_um + wg2.length_um);
    let expect = C64::from_polar(1.0, phase_sum);
    assert!((total[(1, 0)] - expect).abs() < 1e-12, "cascaded phase");
    assert!(total[(0, 0)].abs() < 1e-14, "matched cascade stays matched");

    // Associativity on random passive (contractive) 4-ports, 2 left/2 right.
    let mut rng = Rng::new(404);
    for trial in 0..10 {
        let passive = |rng: &mut Rng| {
            let mut u = haar_unitary(4, rng);
            for v in u.data.iter_mut() {
                *v = *v * 0.93; // strictly passive ⇒ star loops nonsingular
            }
            Sparams::from_full(&u, 2)
        };
        let a = passive(&mut rng);
        let b = passive(&mut rng);
        let c = passive(&mut rng);
        let left = a.star(&b).star(&c).full();
        let right = a.star(&b.star(&c)).full();
        assert!(
            max_diff(&left, &right) < 1e-12,
            "associativity trial {trial}: {}",
            max_diff(&left, &right)
        );
    }
}

// ───────────────────────── (e) Clements round-trip ─────────────────────────

/// Haar-random U → Clements decomposition → mesh rebuild: ‖U′ − U‖_max
/// < 1e-10 for N ∈ {2, 4, 8}, several seeds. Validates the nulling
/// procedure and the diagonal push-through in one shot (Clements et al.,
/// Optica 2016 — the decomposition is constructive, so the bound is
/// numerical only).
#[test]
fn e_clements_round_trip() {
    for &n in &[2usize, 4, 8] {
        for seed in 1..=3u64 {
            let u = haar_unitary(n, &mut Rng::new(seed));
            let mesh = ClementsMesh::decompose(&u).expect("haar input is unitary");
            assert_eq!(mesh.n_mzis(), n * (n - 1) / 2, "MZI count");
            let rebuilt = mesh.unitary();
            let err = max_diff(&rebuilt, &u);
            assert!(err < 1e-10, "N={n} seed={seed}: ‖U′−U‖_max = {err:.3e}");
        }
    }
}

/// The rectangular constructor yields a unitary mesh for arbitrary phase
/// settings, with the canonical n(n−1)/2 MZI count.
#[test]
fn e2_rectangular_mesh_unitary() {
    let n = 6;
    let mut rng = Rng::new(7);
    let settings: Vec<(f64, f64)> = (0..n * (n - 1) / 2)
        .map(|_| (rng.f64() * PI / 2.0, rng.f64() * TAU))
        .collect();
    let phases: Vec<f64> = (0..n).map(|_| rng.f64() * TAU).collect();
    let mesh = ClementsMesh::rectangular(n, &settings, phases);
    assert!(unitarity_err(&mesh.unitary()) < 1e-12);
    // apply() agrees with the assembled unitary on a random vector.
    let x: Vec<C64> = (0..n)
        .map(|_| C64::new(rng.normal(), rng.normal()))
        .collect();
    let u = mesh.unitary();
    let direct: Vec<C64> = (0..n)
        .map(|i| {
            let mut acc = C64::ZERO;
            for j in 0..n {
                acc = acc + u[(i, j)] * x[j];
            }
            acc
        })
        .collect();
    let applied = mesh.apply(&x);
    for (a, b) in applied.iter().zip(&direct) {
        assert!((*a - *b).abs() < 1e-12);
    }
}

// ───────────────────────── (f) optical MVM ─────────────────────────

/// Noiseless mesh MVM matches |U·x|² to 1e-10; ledger counts are exact:
/// macs = N² and adc_samples = N per query, modulator_events = N² per load.
#[test]
fn f_mvm_correctness_and_ledger() {
    let n = 8;
    let u = haar_unitary(n, &mut Rng::new(11));
    let mvm = OpticalMvm::from_unitary(&u).unwrap();
    let mut ledger = EventLedger::default();
    mvm.program(&mut ledger);
    assert_eq!(ledger.modulator_events, (n * n) as u64); // 2·n(n−1)/2 + n

    let mut rng = Rng::new(12);
    let n_queries = 5u64;
    for _ in 0..n_queries {
        let x: Vec<C64> = (0..n)
            .map(|_| C64::new(rng.normal(), rng.normal()))
            .collect();
        let detected = mvm.forward(&x, &mut ledger);
        for i in 0..n {
            let mut acc = C64::ZERO;
            for j in 0..n {
                acc = acc + u[(i, j)] * x[j];
            }
            assert!(
                (detected[i] - acc.norm_sq()).abs() < 1e-10,
                "output {i}: {} vs {}",
                detected[i],
                acc.norm_sq()
            );
        }
    }
    assert_eq!(ledger.macs, n_queries * (n * n) as u64);
    assert_eq!(ledger.adc_samples, n_queries * n as u64);
}

// ───────────────────────── (g) determinism ─────────────────────────

/// Same seed ⇒ bit-identical Haar unitary and identical executor results.
#[test]
fn g_determinism() {
    // Haar: bit-exact reproducibility.
    let u1 = haar_unitary(8, &mut Rng::new(2026));
    let u2 = haar_unitary(8, &mut Rng::new(2026));
    for (a, b) in u1.data.iter().zip(&u2.data) {
        assert_eq!(a.re.to_bits(), b.re.to_bits());
        assert_eq!(a.im.to_bits(), b.im.to_bits());
    }

    // Executor: identical outputs and ledger counters across runs.
    let job = PhotonicJob {
        n: 8,
        unitary: UnitarySpec::RandomHaar { seed: 42 },
        n_queries: 16,
        seed: 7,
    };
    let run = || {
        let mut ticks = 0usize;
        let r = PhotonicExecutor.execute(&job, &mut |_p| ticks += 1);
        (r, ticks)
    };
    let (r1, ticks1) = run();
    let (r2, ticks2) = run();
    assert_eq!(ticks1, ticks2);
    assert_eq!(
        serde_json::to_string(&r1.outputs).unwrap(),
        serde_json::to_string(&r2.outputs).unwrap()
    );
    assert_eq!(r1.ledger.macs, r2.ledger.macs);
    assert_eq!(r1.ledger.adc_samples, r2.ledger.adc_samples);
    assert_eq!(r1.ledger.modulator_events, r2.ledger.modulator_events);

    // And the executor's own quality gates hold.
    let out = &r1.outputs;
    assert!(out["reconstruction_max_error"].as_f64().unwrap() < 1e-10);
    assert!(out["mvm_rms_error"].as_f64().unwrap() < 1e-10);
    assert_eq!(r1.ledger.macs, 16 * 64);
    assert_eq!(r1.ledger.adc_samples, 16 * 8);
    assert_eq!(r1.ledger.modulator_events, 64);
}
