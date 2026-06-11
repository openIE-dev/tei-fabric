//! MNIST-in-pure-Rust crossbar demo validation (docs/SIM-ROADMAP.md §3.3).
//!
//! Non-ignored tests need no data: IDX parsing from hand-built bytes, an
//! f64 finite-difference gradient check against the f32 backprop, synthetic
//! separable training, determinism, and TEIMLP01 persistence.
//!
//! `#[ignore = "needs MNIST data"]` tests require the four IDX files in
//! `$MNIST_DIR` (default `~/.cache/tei-fabric/mnist`) — run
//! `scripts/fetch-mnist.sh`, then
//! `cargo test -p tei-sim-crossbar --release -- --ignored --nocapture`.
//!
//! Published anchor (validation policy: analytic/property/published only):
//! NeuRRAM — Wan et al., "A compute-in-memory chip based on resistive
//! random-access memory", Nature 608, 2022 — MNIST ~99% in software vs
//! ~97–98% measured on-chip: a small (≲2%) loss at realistic device noise,
//! decaying toward the 10% chance floor as noise grows. The ignored tests
//! check the same direction and magnitude.

use tei_sim_core::rng::Rng;
use tei_sim_crossbar::idx::{Idx, parse_idx};
use tei_sim_crossbar::mlp::{Grads, Mlp, TrainConfig};
use tei_sim_crossbar::mnist::{CrossbarMlp, Pipeline, SWEEP_SIGMAS, noise_sweep};

// ───────────────────────────── IDX parsing ─────────────────────────────

/// Hand-built IDX bytes parse to the right dims/data and round-trip.
#[test]
fn idx_round_trip() {
    // magic [0,0,0x08,3], dims 2×2×3, 12 payload bytes.
    let mut bytes = vec![0u8, 0, 0x08, 3];
    for d in [2u32, 2, 3] {
        bytes.extend_from_slice(&d.to_be_bytes());
    }
    let payload: Vec<u8> = (1..=12).collect();
    bytes.extend_from_slice(&payload);

    let idx = parse_idx(&bytes).unwrap();
    assert_eq!(idx.dims, vec![2, 2, 3]);
    assert_eq!(idx.data, payload);
    assert_eq!(idx.to_bytes(), bytes);

    // 1-d labels file shape too.
    let idx1 = Idx {
        dims: vec![5],
        data: vec![0, 1, 2, 3, 4],
    };
    assert_eq!(parse_idx(&idx1.to_bytes()).unwrap(), idx1);
}

/// Gzip magic gives the explicit "run scripts/fetch-mnist.sh" error; other
/// malformed inputs error without panicking.
#[test]
fn idx_rejects_gzip_and_garbage() {
    let err = parse_idx(&[0x1f, 0x8b, 0x08, 0x00, 0, 0, 0, 0]).unwrap_err();
    assert!(err.contains("fetch-mnist.sh"), "{err}");

    assert!(parse_idx(&[]).is_err());
    assert!(parse_idx(&[0, 0, 0x0d, 1, 0, 0, 0, 1, 9]).is_err()); // f32 dtype
    let mut short = vec![0u8, 0, 0x08, 1];
    short.extend_from_slice(&10u32.to_be_bytes());
    short.extend_from_slice(&[1, 2, 3]); // payload too small
    assert!(parse_idx(&short).is_err());
}

// ───────────────────────────── MLP analytics ─────────────────────────────

/// f64 replica of the f32 forward + cross-entropy, for finite differences.
fn loss_f64(m: &Mlp, x: &[f32], label: usize) -> f64 {
    let w1: Vec<f64> = m.w1.iter().map(|&v| v as f64).collect();
    let w2: Vec<f64> = m.w2.iter().map(|&v| v as f64).collect();
    let mut h = vec![0.0f64; m.n_hidden];
    for (j, hj) in h.iter_mut().enumerate() {
        let mut a = m.b1[j] as f64;
        for (i, &xi) in x.iter().enumerate() {
            a += xi as f64 * w1[i * m.n_hidden + j];
        }
        *hj = a.max(0.0);
    }
    let mut z = vec![0.0f64; m.n_out];
    for (k, zk) in z.iter_mut().enumerate() {
        let mut a = m.b2[k] as f64;
        for (j, &hj) in h.iter().enumerate() {
            a += hj * w2[j * m.n_out + k];
        }
        *zk = a;
    }
    let zmax = z.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let lse = zmax + z.iter().map(|&v| (v - zmax).exp()).sum::<f64>().ln();
    lse - z[label]
}

/// Central finite differences in f64 agree with the f32 backprop gradients
/// to < 1e-4 relative error on a tiny synthetic net.
#[test]
fn gradient_check_finite_difference() {
    let mut rng = Rng::new(7);
    let mlp = Mlp::new(8, 6, 4, &mut rng);
    let x: Vec<f32> = (0..8).map(|_| rng.f64() as f32).collect();
    let label = 2usize;

    let mut g = Grads::zeros(&mlp);
    mlp.backprop(&x, label, &mut g);

    let eps = 1e-4f32;
    let check = |get: &dyn Fn(&Mlp) -> &Vec<f32>,
                 get_mut: &dyn Fn(&mut Mlp) -> &mut Vec<f32>,
                 grad: &[f32],
                 name: &str| {
        for p in 0..get(&mlp).len() {
            let mut plus = mlp.clone();
            get_mut(&mut plus)[p] += eps;
            let mut minus = mlp.clone();
            get_mut(&mut minus)[p] -= eps;
            // f64 loss on the perturbed f32 weights. Divide by the step the
            // f32 parameters actually realized (w ± eps rounds in f32, which
            // would otherwise inject ~3e-4 relative step error); remaining
            // FD truncation is O(eps²) ≈ 1e-8, well under the 1e-4 gate.
            let step = get(&plus)[p] as f64 - get(&minus)[p] as f64;
            let fd = (loss_f64(&plus, &x, label) - loss_f64(&minus, &x, label)) / step;
            let bp = grad[p] as f64;
            let denom = fd.abs().max(bp.abs()).max(1e-3);
            assert!(
                (fd - bp).abs() / denom < 1e-4,
                "{name}[{p}]: fd {fd} vs backprop {bp}"
            );
        }
    };
    check(&|m| &m.w1, &|m| &mut m.w1, &g.w1, "w1");
    check(&|m| &m.b1, &|m| &mut m.b1, &g.b1, "b1");
    check(&|m| &m.w2, &|m| &mut m.w2, &g.w2, "w2");
    check(&|m| &m.b2, &|m| &mut m.b2, &g.b2, "b2");
}

/// Synthetic linearly separable classes train to 100% accuracy.
#[test]
fn separable_synthetic_trains_to_100() {
    // 3 well-separated Gaussian blobs in 4-d: centers 5·e_c, σ = 0.3.
    let mut rng = Rng::new(11);
    let (n, dim, classes) = (120usize, 4usize, 3usize);
    let mut images = Vec::with_capacity(n * dim);
    let mut labels = Vec::with_capacity(n);
    for s in 0..n {
        let c = s % classes;
        for d in 0..dim {
            let center = if d == c { 5.0 } else { 0.0 };
            images.push((center + 0.3 * rng.normal()) as f32);
        }
        labels.push(c as u8);
    }

    let mut mlp = Mlp::new(dim, 16, classes, &mut Rng::new(1));
    let cfg = TrainConfig {
        epochs: 100,
        batch: 8,
        lr: 0.05,
        momentum: 0.9,
        seed: 3,
    };
    mlp.train(&images, &labels, &cfg);

    let correct = (0..n)
        .filter(|&s| mlp.predict(&images[s * dim..(s + 1) * dim]) == labels[s] as usize)
        .count();
    assert_eq!(correct, n, "separable data must reach 100% train accuracy");
}

/// Same seed + data → bit-identical weights (He init, Fisher–Yates shuffle,
/// and sequential accumulation are all deterministic).
#[test]
fn training_is_deterministic() {
    let mut rng = Rng::new(5);
    let (n, dim) = (64usize, 6usize);
    let images: Vec<f32> = (0..n * dim).map(|_| rng.f64() as f32).collect();
    let labels: Vec<u8> = (0..n).map(|s| (s % 3) as u8).collect();
    let cfg = TrainConfig {
        epochs: 4,
        batch: 16,
        lr: 0.1,
        momentum: 0.9,
        seed: 21,
    };

    let train = || {
        let mut m = Mlp::new(dim, 10, 3, &mut Rng::new(cfg.seed));
        m.train(&images, &labels, &cfg);
        m
    };
    let (a, b) = (train(), train());
    assert_eq!(a.w1, b.w1);
    assert_eq!(a.b1, b.b1);
    assert_eq!(a.w2, b.w2);
    assert_eq!(a.b2, b.b2);
}

/// TEIMLP01 save/load round-trips bitwise, and rejects bad magic.
#[test]
fn save_load_round_trip() {
    let mlp = Mlp::new(12, 7, 5, &mut Rng::new(99));
    let back = Mlp::from_bytes(&mlp.to_bytes()).unwrap();
    assert_eq!(mlp, back);

    let path = std::env::temp_dir().join(format!("teimlp-test-{}.bin", std::process::id()));
    mlp.save(&path).unwrap();
    let loaded = Mlp::load(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(mlp, loaded);

    assert!(Mlp::from_bytes(b"NOTMLP01rest").is_err());
    let mut truncated = mlp.to_bytes();
    truncated.truncate(truncated.len() - 4);
    assert!(Mlp::from_bytes(&truncated).is_err());
}

// ──────────────────────── MNIST (needs data) ────────────────────────

fn pipeline() -> Pipeline {
    let p = Pipeline::train_or_load(TrainConfig::default()).expect("MNIST data");
    if let Some(s) = p.train_seconds {
        println!(
            "trained 784→128→10 (epochs {}, batch {}, lr {}, momentum {}, seed {}) in {s:.1} s",
            p.config.epochs, p.config.batch, p.config.lr, p.config.momentum, p.config.seed
        );
    } else {
        println!("loaded cached TEIMLP01 weights");
    }
    p
}

/// Headline: digital f32 MLP ≥ 96% on the 10k test set, and the crossbar at
/// σ = 0 (8-bit DAC/ADC quantization only) within 0.5% of digital — the
/// quantization-tolerant regime NeuRRAM (Wan 2022) demonstrates on-chip.
#[test]
#[ignore = "needs MNIST data"]
fn mnist_headline_accuracy() {
    let p = pipeline();
    let digital = p.digital_accuracy();
    println!(
        "digital f32 accuracy (10k):          {:.2}%",
        digital * 100.0
    );
    assert!(digital >= 0.96, "digital accuracy {digital} < 0.96");

    let clean = CrossbarMlp::program(&p.mlp, &p.data, 0.0, 0.0, 256, 7);
    let acc_clean = clean.accuracy(&p.data, 10_000);
    println!(
        "crossbar σ=0 accuracy (10k):         {:.2}%",
        acc_clean * 100.0
    );
    assert!(
        (digital - acc_clean).abs() <= 0.005,
        "crossbar σ=0 {acc_clean} vs digital {digital}"
    );

    // Default realistic noise point (σ_prog = σ_read = 0.03 — within the
    // device-variability range CrossSim/NeuRRAM-class parts report).
    let noisy = CrossbarMlp::program(&p.mlp, &p.data, 0.03, 0.03, 256, 7);
    let acc_noisy = noisy.accuracy(&p.data, 10_000);
    println!(
        "crossbar σ=0.03 accuracy (10k):      {:.2}%",
        acc_noisy * 100.0
    );
    // NeuRRAM direction/magnitude: a few-percent loss at realistic noise,
    // nowhere near chance.
    assert!(acc_noisy >= digital - 0.05, "σ=0.03 lost more than 5%");
}

/// Accuracy degrades monotonically (≤ 0.5% jitter) as σ grows — the
/// direction NeuRRAM (Wan 2022) reports going from software to noisy
/// on-chip inference.
#[test]
#[ignore = "needs MNIST data"]
fn mnist_noise_degradation_monotone() {
    let p = pipeline();
    let table = noise_sweep(&p, &SWEEP_SIGMAS, 2000);
    println!("σ (prog=read)  accuracy (2000-image subset)");
    for &(s, a) in &table {
        println!("{s:>12.3}  {:.2}%", a * 100.0);
    }
    for w in table.windows(2) {
        assert!(
            w[1].1 <= w[0].1 + 0.005,
            "accuracy rose with noise beyond jitter: σ {} → {}: {} → {}",
            w[0].0,
            w[1].0,
            w[0].1,
            w[1].1
        );
    }
}

/// Very high device noise collapses accuracy to the 10-class chance floor.
#[test]
#[ignore = "needs MNIST data"]
fn mnist_high_noise_reaches_chance() {
    let p = pipeline();
    let xb = CrossbarMlp::program(&p.mlp, &p.data, 3.0, 3.0, 256, 7);
    let acc = xb.accuracy(&p.data, 2000);
    println!("crossbar σ=3.0 accuracy (2000): {:.2}%", acc * 100.0);
    // Chance is 10%; allow generous statistical + class-imbalance slack.
    assert!((0.05..=0.18).contains(&acc), "expected ~chance, got {acc}");
}

/// Crossbar evaluation is deterministic under rayon: per-image RNG streams
/// make two runs produce identical predictions.
#[test]
#[ignore = "needs MNIST data"]
fn mnist_crossbar_deterministic() {
    let p = pipeline();
    let run = || {
        let xb = CrossbarMlp::program(&p.mlp, &p.data, 0.05, 0.05, 256, 7);
        xb.predictions(&p.data, 500)
    };
    assert_eq!(run(), run());
}
