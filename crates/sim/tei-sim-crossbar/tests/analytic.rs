//! Analytic + published-number validation per docs/SIM-ROADMAP.md §3.3.
//! No foreign-tool fixtures — every expected value below is computed in
//! closed form: independent-noise variance propagation, full-scale-sinusoid
//! quantization SNR (Bennett 1948), the PCM drift power law, exact tiling
//! algebra, and the lognormal mean exp(μ + σ²/2).

use tei_sim_core::exec::{Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;
use tei_sim_crossbar::{
    AdcParams, CrossbarArray, CrossbarExecutor, CrossbarJob, DeviceParams, IrDropMode,
};

/// (a) σ-propagation: with ONLY per-device read noise (σᵢ = σ_read·|Gᵢ|,
/// independent across devices and reads), the output variance must follow
/// the closed form σ_y² = Σᵢ xᵢ²·σᵢ² — in weight units, σᵢ = σ_read·|wᵢ|.
#[test]
fn read_noise_variance_propagation() {
    let k = 16;
    let sigma_read = 0.05;
    let mut rng = Rng::new(101);
    let w: Vec<f64> = (0..k).map(|_| 0.2 + 0.8 * rng.f64()).collect();
    let x: Vec<f64> = (0..k).map(|_| 2.0 * rng.f64() - 1.0).collect();

    let params = DeviceParams {
        sigma_read,
        ..Default::default()
    };
    let arr = CrossbarArray::program(&w, k, 1, 64, params, &mut rng);

    // Closed form: σ_y² = Σ xᵢ²·(σ_read·wᵢ)².
    let var_expected: f64 = x
        .iter()
        .zip(&w)
        .map(|(xi, wi)| (xi * sigma_read * wi).powi(2))
        .sum();

    let trials = 40_000;
    let mut ledger = EventLedger::default();
    let (mut sum, mut sumsq) = (0.0, 0.0);
    for _ in 0..trials {
        let y = arr.mvm(&x, &mut rng, &mut ledger)[0];
        sum += y;
        sumsq += y * y;
    }
    let mean = sum / trials as f64;
    let var = sumsq / trials as f64 - mean * mean;

    // Sample variance has relative sd ≈ √(2/N) ≈ 0.7% here; assert 5%.
    let rel = (var - var_expected).abs() / var_expected;
    assert!(
        rel < 0.05,
        "empirical σ_y² {var:.3e} vs closed form {var_expected:.3e} (rel {rel:.3})"
    );
    // The noise is zero-mean: ⟨y⟩ must match the ideal output.
    let y_ideal = arr.ideal_mvm(&x)[0];
    assert!(
        (mean - y_ideal).abs() < 5.0 * (var_expected / trials as f64).sqrt(),
        "mean {mean} vs ideal {y_ideal}"
    );
}

/// (b) ADC quantization SNR: noiseless devices, full-scale sinusoidal drive,
/// SNR ≈ 6.02·b + 1.76 dB (the classic full-scale-sine result, Bennett
/// 1948 / Widrow) within ±1.5 dB for b ∈ {4, 6, 8}. Phase is randomized per
/// sample so the quantization error decorrelates from the signal.
#[test]
fn adc_quantization_snr() {
    let k = 8;
    let mut rng = Rng::new(202);
    let w: Vec<f64> = (0..k).map(|_| 0.1 + 0.9 * rng.f64()).collect();
    let d: f64 = w.iter().sum(); // output amplitude when x = 1⃗·sin θ

    for bits in [4u32, 6, 8] {
        let params = DeviceParams {
            adc: Some(AdcParams {
                bits,
                range: d, // sinusoid exactly fills the ADC full scale
                inl_lsb: 0.0,
            }),
            ..Default::default()
        };
        let arr = CrossbarArray::program(&w, k, 1, 64, params, &mut Rng::new(1));

        let n = 20_000;
        let mut ledger = EventLedger::default();
        let (mut sig2, mut err2) = (0.0, 0.0);
        for _ in 0..n {
            let s = (std::f64::consts::TAU * rng.f64()).sin();
            let x = vec![s; k];
            let y_ideal = arr.ideal_mvm(&x)[0];
            let y_q = arr.mvm(&x, &mut Rng::new(0), &mut ledger)[0];
            sig2 += y_ideal * y_ideal;
            err2 += (y_q - y_ideal) * (y_q - y_ideal);
        }
        let snr = 10.0 * (sig2 / err2).log10();
        let expected = 6.02 * bits as f64 + 1.76;
        assert!(
            (snr - expected).abs() < 1.5,
            "b={bits}: SNR {snr:.2} dB vs 6.02b+1.76 = {expected:.2} dB"
        );
    }
}

/// (c) Drift law: program at G0, age to t = 10^a·t0 for a ∈ {1,2,3}; the
/// log-log slope of mean conductance vs age must recover −ν to 1e-6 — the
/// power law G(t) = G0·(t/t0)^(−ν) is applied deterministically.
#[test]
fn drift_exponent_recovery() {
    let nu = 0.05; // typical PCM drift exponent scale
    let (rows, cols) = (4, 4);
    let w = vec![0.8; rows * cols];

    let mut pts = Vec::new(); // (ln t, ln Ḡ)
    for a in [1i32, 2, 3] {
        let age = 10f64.powi(a);
        let params = DeviceParams {
            drift_nu: nu,
            age,
            ..Default::default()
        };
        let arr = CrossbarArray::program(&w, rows, cols, 64, params, &mut Rng::new(1));
        let mut g_sum = 0.0;
        for i in 0..rows {
            for j in 0..cols {
                g_sum += arr.conductance(i, j);
            }
        }
        let g_mean = g_sum / (rows * cols) as f64;
        pts.push((age.ln(), g_mean.ln()));
    }

    // Least-squares slope of ln Ḡ vs ln t.
    let n = pts.len() as f64;
    let (sx, sy): (f64, f64) = pts.iter().fold((0.0, 0.0), |(a, b), p| (a + p.0, b + p.1));
    let (mx, my) = (sx / n, sy / n);
    let num: f64 = pts.iter().map(|p| (p.0 - mx) * (p.1 - my)).sum();
    let den: f64 = pts.iter().map(|p| (p.0 - mx) * (p.0 - mx)).sum();
    let slope = num / den;
    assert!(
        (slope - (-nu)).abs() < 1e-6,
        "recovered exponent {slope} vs −ν = {}",
        -nu
    );
}

/// (d) Tiling exactness: all non-idealities OFF, a 700×900 matrix on a
/// 256-side array must equal the dense reference product, and the ledger
/// must show adc_samples = 900·⌈700/256⌉ and macs = 700·900.
///
/// Inputs and weights are dyadic rationals (11-bit mantissas) and g_max = 1
/// with max|w| = 1, so every product and partial sum is exact in f64 and the
/// 1e-12 bound is met with zero headroom needed.
#[test]
fn tiling_exactness_and_ledger() {
    let (rows, cols, size) = (700usize, 900usize, 256usize);
    let mut rng = Rng::new(303);
    let dyadic = |rng: &mut Rng| (rng.below(2049) as f64 - 1024.0) / 1024.0;
    let mut w: Vec<f64> = (0..rows * cols).map(|_| dyadic(&mut rng)).collect();
    w[0] = 1.0; // pin max|w| = 1 → g_scale = g_max exactly
    let x: Vec<f64> = (0..rows).map(|_| dyadic(&mut rng)).collect();

    let params = DeviceParams {
        g_max: 1.0,
        ..Default::default()
    };
    let arr = CrossbarArray::program(&w, rows, cols, size, params, &mut Rng::new(1));

    // Dense reference y_j = Σᵢ xᵢ·w[i][j].
    let mut y_ref = vec![0.0f64; cols];
    for (i, &xi) in x.iter().enumerate() {
        for (j, yj) in y_ref.iter_mut().enumerate() {
            *yj += xi * w[i * cols + j];
        }
    }

    let mut ledger = EventLedger::default();
    let y = arr.mvm(&x, &mut Rng::new(2), &mut ledger);
    for (j, (a, b)) in y.iter().zip(&y_ref).enumerate() {
        assert!((a - b).abs() <= 1e-12, "col {j}: tiled {a} vs dense {b}");
    }
    let row_tiles = rows.div_ceil(size) as u64; // ⌈700/256⌉ = 3
    assert_eq!(ledger.adc_samples, cols as u64 * row_tiles, "adc_samples");
    assert_eq!(ledger.macs, (rows * cols) as u64, "macs");
}

/// (e) Lognormal programming: with G_prog = G_target·exp(σZ) (μ = 0), the
/// empirical mean over many devices must match the lognormal mean
/// E[G]/G_target = exp(μ + σ²/2) = exp(σ²/2).
#[test]
fn lognormal_programming_mean() {
    let (rows, cols) = (256, 256);
    let sigma = 0.3;
    let w = vec![0.5; rows * cols];
    let params = DeviceParams {
        sigma_prog: sigma,
        ..Default::default()
    };
    let g_max = params.g_max;
    let arr = CrossbarArray::program(&w, rows, cols, 128, params, &mut Rng::new(404));

    // g_scale = g_max / max|w| = 2·g_max, so the per-device target is
    // 0.5·g_scale = g_max exactly.
    let mut sum = 0.0;
    for i in 0..rows {
        for j in 0..cols {
            sum += arr.conductance(i, j);
        }
    }
    let mean_factor = sum / (rows * cols) as f64 / g_max;
    let expected = (sigma * sigma / 2.0).exp(); // exp(μ + σ²/2), μ = 0
    // se of the mean ≈ 0.32/√65536 ≈ 0.00125 → 1% is an 8σ band.
    assert!(
        (mean_factor - expected).abs() < 0.01 * expected,
        "mean factor {mean_factor:.5} vs exp(σ²/2) = {expected:.5}"
    );
}

/// (f) Determinism: identical seeds for programming and reading produce
/// bit-identical noisy outputs with the full non-ideality stack enabled.
#[test]
fn seed_determinism() {
    let (rows, cols) = (100, 80);
    let mut wrng = Rng::new(7);
    let w: Vec<f64> = (0..rows * cols).map(|_| 2.0 * wrng.f64() - 1.0).collect();
    let x: Vec<f64> = (0..rows).map(|_| 2.0 * wrng.f64() - 1.0).collect();
    let params = DeviceParams {
        sigma_prog: 0.1,
        sigma_read: 0.05,
        drift_nu: 0.05,
        age: 100.0,
        dac_bits: Some(6),
        input_range: 1.0,
        adc: Some(AdcParams {
            bits: 8,
            range: 40.0,
            inl_lsb: 0.5,
        }),
        ir_drop: IrDropMode::FirstOrder { r_wire: 1.0 },
        ..Default::default()
    };

    let run = || {
        let arr = CrossbarArray::program(&w, rows, cols, 32, params.clone(), &mut Rng::new(11));
        let mut ledger = EventLedger::default();
        arr.mvm(&x, &mut Rng::new(13), &mut ledger)
    };
    let (y1, y2) = (run(), run());
    assert_eq!(y1, y2, "same seeds must give bit-identical outputs");
}

/// Exact-mesh IR drop is explicitly deferred to tei-sim-circuit M1.
#[test]
#[should_panic(expected = "tei-sim-circuit")]
fn exact_mesh_is_deferred() {
    let params = DeviceParams {
        ir_drop: IrDropMode::ExactMesh,
        ..Default::default()
    };
    let arr = CrossbarArray::program(&[1.0], 1, 1, 8, params, &mut Rng::new(1));
    arr.mvm(&[1.0], &mut Rng::new(2), &mut EventLedger::default());
}

/// CrossbarExecutor end-to-end: a noisy job reports finite RMS/SNR, the
/// ledger counts match the tiling algebra, and progress monotonically
/// reaches 1.
#[test]
fn executor_end_to_end() {
    let job = CrossbarJob {
        rows: 300,
        cols: 200,
        array_size: 128,
        device: DeviceParams {
            sigma_read: 0.02,
            ..Default::default()
        },
        n_queries: 20,
        seed: 42,
    };
    let mut fractions = Vec::new();
    let result = CrossbarExecutor.execute(&job, &mut |p: Progress| fractions.push(p.fraction));

    assert_eq!(result.ledger.macs, 300 * 200 * 20);
    // ⌈300/128⌉ = 3 row tiles → 200·3 ADC samples per MVM.
    assert_eq!(result.ledger.adc_samples, 200 * 3 * 20);
    assert!(result.ledger.wall_seconds.is_some());
    assert!(fractions.windows(2).all(|w| w[0] <= w[1]));
    assert_eq!(*fractions.last().unwrap(), 1.0);

    let rms = result.outputs["rms_error"].as_f64().unwrap();
    let snr = result.outputs["snr_db"].as_f64().unwrap();
    assert!(rms > 0.0 && rms.is_finite());
    assert!(snr > 0.0 && snr.is_finite());
}
