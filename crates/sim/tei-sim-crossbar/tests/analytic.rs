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

// ───────────────────── exact-mesh IR drop (ExactMesh) ─────────────────────
//
// The coupled resistive-mesh solve via tei-sim-circuit — validation per the
// roadmap §3.3 table row "IR-drop first-order mode vs exact mesh mode
// convergence | internal consistency", plus analytic single-device and
// vanishing-wire limits and the IR-drop monotonicity signature.

/// Noiseless MVM through the given IR-drop mode (all other non-idealities
/// off, so outputs are deterministic functions of the wire model alone).
fn ir_mvm(
    w: &[f64],
    rows: usize,
    cols: usize,
    size: usize,
    ir_drop: IrDropMode,
    x: &[f64],
) -> Vec<f64> {
    let params = DeviceParams {
        ir_drop,
        ..Default::default()
    };
    let arr = CrossbarArray::program(w, rows, cols, size, params, &mut Rng::new(1));
    arr.mvm(x, &mut Rng::new(2), &mut EventLedger::default())
}

/// (g) r_wire → 0 limit: the exact mesh must reproduce the ideal (lossless
/// wire) outputs to 1e-9, with mixed-sign weights and inputs so both
/// polarity wires of the differential column pairs are exercised.
#[test]
fn exact_mesh_matches_ideal_as_r_wire_vanishes() {
    let (rows, cols) = (8, 8);
    let mut rng = Rng::new(909);
    let w: Vec<f64> = (0..rows * cols).map(|_| 2.0 * rng.f64() - 1.0).collect();
    let x: Vec<f64> = (0..rows).map(|_| 2.0 * rng.f64() - 1.0).collect();

    let y_ideal = ir_mvm(&w, rows, cols, 8, IrDropMode::Ideal, &x);
    let y_mesh = ir_mvm(
        &w,
        rows,
        cols,
        8,
        IrDropMode::ExactMesh { r_wire: 1e-9 },
        &x,
    );
    for (j, (a, b)) in y_mesh.iter().zip(&y_ideal).enumerate() {
        assert!((a - b).abs() < 1e-9, "col {j}: mesh {a} vs ideal {b}");
    }
}

/// (h) Single device with its series wire is an exact voltage divider:
/// I = v·G/(1 + G·R_total) with R_total = 2·r_wire (one row segment from the
/// driver + one column segment to the sense ground) — matched to 1e-12 for
/// both weight signs (the negative device lives on the − polarity wire).
#[test]
fn exact_mesh_single_device_matches_voltage_divider() {
    let (r_wire, x) = (5.0, 0.7);
    let params = DeviceParams::default();
    let g = params.g_max; // |w| = 1 → g_scale = g_max → |G| = g_max
    for w in [1.0f64, -1.0] {
        let y = ir_mvm(&[w], 1, 1, 8, IrDropMode::ExactMesh { r_wire }, &[x])[0];
        // In weight·input units: y = x·w / (1 + |G|·2·r_wire).
        let expected = x * w / (1.0 + g * 2.0 * r_wire);
        assert!(
            (y - expected).abs() < 1e-12,
            "w={w}: mesh {y} vs divider closed form {expected}"
        );
    }
}

/// (i) THE §3.3 table row — first-order vs exact mesh convergence (internal
/// consistency): on a 32×32 tile with realistic conductances, the
/// first-order closed form must sit strictly closer to the exact mesh than
/// the ideal model does, both errors must shrink as r_wire decreases, and
/// the first-order error must shrink *faster* (it is exact to O(R²)).
#[test]
fn first_order_vs_exact_mesh_convergence() {
    let (rows, cols) = (32, 32);
    let mut rng = Rng::new(515);
    // All-positive weights and inputs: the fully-active worst case where the
    // shared-wire coupling the first-order model ignores is largest.
    let w: Vec<f64> = (0..rows * cols).map(|_| 0.2 + 0.8 * rng.f64()).collect();
    let x: Vec<f64> = (0..rows).map(|_| rng.f64()).collect();

    let rms = |a: &[f64], b: &[f64]| {
        (a.iter().zip(b).map(|(p, q)| (p - q) * (p - q)).sum::<f64>() / a.len() as f64).sqrt()
    };

    let y_ideal = ir_mvm(&w, rows, cols, 32, IrDropMode::Ideal, &x);
    let mut errs = Vec::new(); // (err_first_order, err_ideal) per r_wire
    for r_wire in [2.0, 0.5] {
        let y_fo = ir_mvm(&w, rows, cols, 32, IrDropMode::FirstOrder { r_wire }, &x);
        let y_em = ir_mvm(&w, rows, cols, 32, IrDropMode::ExactMesh { r_wire }, &x);
        let (e_fo, e_id) = (rms(&y_fo, &y_em), rms(&y_ideal, &y_em));
        println!(
            "r_wire={r_wire}: |FirstOrder−Exact| rms = {e_fo:.3e}, |Ideal−Exact| rms = {e_id:.3e}"
        );
        assert!(
            e_fo < e_id,
            "r_wire={r_wire}: first-order error {e_fo:.3e} !< ideal error {e_id:.3e}"
        );
        errs.push((e_fo, e_id));
    }
    let (coarse, fine) = (errs[0], errs[1]);
    assert!(
        fine.0 < coarse.0,
        "first-order error must shrink with r_wire"
    );
    assert!(fine.1 < coarse.1, "ideal error must shrink with r_wire");

    // Rate separation, shown where the first-order premise (independent
    // current paths) holds: a diagonal weight matrix has one device per row
    // and per column, so no wire segment carries another device's current —
    // FirstOrder coincides with the exact mesh to LU precision while Ideal
    // still misses the O(r_wire) self-path drop. (In the fully-active case
    // above, both errors are instead dominated by the same shared-wire
    // coupling term, which FirstOrder ignores by construction, so it
    // improves on Ideal by a constant factor rather than by an order.)
    let mut w_diag = vec![0.0; rows * cols];
    for i in 0..rows {
        w_diag[i * cols + i] = 0.2 + 0.8 * rng.f64();
    }
    let y_id = ir_mvm(&w_diag, rows, cols, 32, IrDropMode::Ideal, &x);
    for r_wire in [2.0, 0.5] {
        let y_fo = ir_mvm(
            &w_diag,
            rows,
            cols,
            32,
            IrDropMode::FirstOrder { r_wire },
            &x,
        );
        let y_em = ir_mvm(
            &w_diag,
            rows,
            cols,
            32,
            IrDropMode::ExactMesh { r_wire },
            &x,
        );
        let (e_fo, e_id) = (rms(&y_fo, &y_em), rms(&y_id, &y_em));
        println!(
            "diagonal r_wire={r_wire}: |FirstOrder−Exact| rms = {e_fo:.3e}, |Ideal−Exact| rms = {e_id:.3e}"
        );
        assert!(
            e_fo < 1e-12,
            "single device per wire: first-order must be exact, got {e_fo:.3e}"
        );
        assert!(
            e_id > 1e3 * e_fo,
            "ideal must keep its O(r_wire) error: {e_id:.3e} vs {e_fo:.3e}"
        );
    }
}

/// (j) IR-drop monotonicity signature: a far-corner crosspoint (long row +
/// column path) delivers less current than a near-corner one, and with a
/// uniform fully-on array the column outputs decay with distance from the
/// drivers; everything stays below the ideal value.
#[test]
fn exact_mesh_ir_drop_monotonic() {
    let (rows, cols, r_wire) = (4usize, 4usize, 200.0);
    // Single device at the near corner (last row, first column: 1 row + 1
    // column segment) vs the far corner (first row, last column).
    let mut w_near = vec![0.0; rows * cols];
    w_near[(rows - 1) * cols] = 1.0;
    let mut w_far = vec![0.0; rows * cols];
    w_far[cols - 1] = 1.0;
    let x = vec![1.0; rows];
    let mode = IrDropMode::ExactMesh { r_wire };
    let y_near = ir_mvm(&w_near, rows, cols, 8, mode.clone(), &x)[0];
    let y_far = ir_mvm(&w_far, rows, cols, 8, mode.clone(), &x)[cols - 1];
    assert!(
        y_far < y_near && y_near < 1.0,
        "far-corner {y_far} !< near-corner {y_near} !< ideal 1"
    );

    // Uniform 8×8: outputs strictly decrease with column distance.
    let (rows, cols) = (8usize, 8usize);
    let y = ir_mvm(
        &vec![1.0; rows * cols],
        rows,
        cols,
        8,
        IrDropMode::ExactMesh { r_wire },
        &vec![1.0; rows],
    );
    assert!(y[0] < rows as f64, "every column must sit below ideal");
    for j in 1..cols {
        assert!(
            y[j] < y[j - 1],
            "col {j}: {} !< {} — IR drop must grow with row-wire distance",
            y[j],
            y[j - 1]
        );
    }
}

/// (k) ExactMesh determinism + ledger: identical seeds with the full
/// non-ideality stack give bit-identical outputs across a tiled matrix, and
/// the ledger counts one mesh solve per (tile, MVM) with the macs /
/// adc_samples conventions unchanged.
#[test]
fn exact_mesh_seed_determinism_and_ledger() {
    let (rows, cols, size) = (40usize, 20usize, 16usize);
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
        ir_drop: IrDropMode::ExactMesh { r_wire: 1.0 },
        ..Default::default()
    };

    let run = || {
        let arr = CrossbarArray::program(&w, rows, cols, size, params.clone(), &mut Rng::new(11));
        let mut ledger = EventLedger::default();
        let y = arr.mvm(&x, &mut Rng::new(13), &mut ledger);
        (y, ledger)
    };
    let ((y1, l1), (y2, _)) = (run(), run());
    assert_eq!(y1, y2, "same seeds must give bit-identical outputs");
    // ⌈40/16⌉·⌈20/16⌉ = 3·2 = 6 tiles → 6 mesh solves for one MVM.
    assert_eq!(l1.mesh_solves, 6, "one mesh solve per tile per MVM");
    assert_eq!(l1.macs, (rows * cols) as u64, "macs convention unchanged");
    assert_eq!(
        l1.adc_samples,
        (cols * 3) as u64,
        "adc convention unchanged"
    );
}

/// (l) ExactMesh through the executor: the job round-trips from JSON
/// (serde-tagged like FirstOrder), runs deterministically (same job →
/// bit-identical outputs), and reports the mesh-solve count.
#[test]
fn exact_mesh_executor_job_deterministic() {
    let job: CrossbarJob = serde_json::from_str(
        r#"{
            "rows": 48, "cols": 24, "array_size": 32,
            "device": { "sigma_read": 0.02, "ir_drop": { "exact_mesh": { "r_wire": 1.0 } } },
            "n_queries": 3, "seed": 9
        }"#,
    )
    .unwrap();
    let r1 = CrossbarExecutor.execute(&job, &mut |_| {});
    let r2 = CrossbarExecutor.execute(&job, &mut |_| {});
    assert_eq!(r1.outputs["rms_error"], r2.outputs["rms_error"]);
    assert_eq!(r1.outputs["snr_db"], r2.outputs["snr_db"]);
    // ⌈48/32⌉·⌈24/32⌉ = 2 tiles × 3 queries.
    assert_eq!(r1.ledger.mesh_solves, 6);
    assert_eq!(r1.outputs["mesh_solves"], 6);
}

/// Factor-once / solve-many cost of the largest supported tile (64×64):
/// programming pays one MNA factorization, each query one cached-LU
/// re-solve. Run with `cargo test --release -- --ignored --nocapture`.
#[test]
#[ignore = "timing benchmark, run explicitly in release"]
fn bench_exact_mesh_factor_and_solve_64() {
    let (rows, cols) = (64usize, 64usize);
    let mut rng = Rng::new(99);
    let w: Vec<f64> = (0..rows * cols).map(|_| 2.0 * rng.f64() - 1.0).collect();
    let params = DeviceParams {
        ir_drop: IrDropMode::ExactMesh { r_wire: 1.0 },
        ..Default::default()
    };

    let t0 = std::time::Instant::now();
    let arr = CrossbarArray::program(&w, rows, cols, 64, params, &mut Rng::new(1));
    let t_program = t0.elapsed();

    let n = 100;
    let mut ledger = EventLedger::default();
    let mut sink = 0.0;
    let t1 = std::time::Instant::now();
    for q in 0..n {
        let x: Vec<f64> = (0..rows).map(|_| (q as f64 / n as f64) - 0.5).collect();
        sink += arr.mvm(&x, &mut Rng::new(2), &mut ledger)[0];
    }
    let t_query = t1.elapsed();
    println!(
        "64×64 ExactMesh: program+factor {:.1} ms, {n} queries in {:.1} ms \
         ({:.3} ms/solve, sink {sink:.3e})",
        t_program.as_secs_f64() * 1e3,
        t_query.as_secs_f64() * 1e3,
        t_query.as_secs_f64() * 1e3 / n as f64
    );
    assert_eq!(ledger.mesh_solves, n as u64);
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
