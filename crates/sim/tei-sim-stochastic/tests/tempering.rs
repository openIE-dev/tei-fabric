//! Parallel-tempering validation — analytic + property ground truth only
//! (docs/SIM-ROADMAP.md §2 validation policy). Exact enumeration ≤ 20 spins,
//! Metropolis-rule properties, determinism across thread counts.

use tei_sim_core::exec::Executor;
use tei_sim_core::rng::Rng;
use tei_sim_stochastic::tempering::{Ladder, Tempering, TemperingSpec, parallel_temper};
use tei_sim_stochastic::{IsingModel, Schedule, StochasticExecutor, StochasticJob, anneal};

// ───────────────────────────── helpers ─────────────────────────────

/// Small random Ising model (ring + chords), same recipe as tests/analytic.rs.
fn random_model(n: usize, seed: u64) -> IsingModel {
    let mut rng = Rng::new(seed);
    let h: Vec<f64> = (0..n).map(|_| 0.5 * rng.normal()).collect();
    let mut edges: Vec<(u32, u32, f64)> = (0..n)
        .map(|i| (i as u32, ((i + 1) % n) as u32, 0.6 * rng.normal()))
        .collect();
    for _ in 0..n / 2 {
        let a = rng.below(n) as u32;
        let b = rng.below(n) as u32;
        if a != b
            && !edges
                .iter()
                .any(|&(x, y, _)| (x, y) == (a, b) || (x, y) == (b, a))
        {
            edges.push((a, b, 0.6 * rng.normal()));
        }
    }
    IsingModel::new(h, &edges)
}

/// ±J Sherrington-Kirkpatrick instance: complete graph, J_ij ∈ {−1, +1}.
fn sk_instance(n: usize, seed: u64) -> IsingModel {
    let mut rng = Rng::new(seed);
    let mut edges = Vec::with_capacity(n * (n - 1) / 2);
    for i in 0..n {
        for j in (i + 1)..n {
            let w = if rng.bernoulli(0.5) { 1.0 } else { -1.0 };
            edges.push((i as u32, j as u32, w));
        }
    }
    IsingModel::new(vec![0.0; n], &edges)
}

/// Exact ground-state energy by full 2^n enumeration (n ≤ 20).
fn exact_ground_energy(model: &IsingModel) -> f64 {
    let n = model.n;
    assert!(n <= 20);
    let mut s = vec![1i8; n];
    let mut best = f64::INFINITY;
    for mask in 0u64..(1 << n) {
        for (i, si) in s.iter_mut().enumerate() {
            *si = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        best = best.min(model.energy(&s));
    }
    best
}

/// Exact Boltzmann probabilities over all 2^n states at inverse temp β.
fn exact_probs(model: &IsingModel, beta: f64) -> Vec<f64> {
    let n = model.n;
    assert!(n <= 20);
    let mut s = vec![1i8; n];
    let mut probs = vec![0.0f64; 1 << n];
    let mut z = 0.0;
    for (mask, p) in probs.iter_mut().enumerate() {
        for (i, si) in s.iter_mut().enumerate() {
            *si = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        *p = (-beta * model.energy(&s)).exp();
        z += *p;
    }
    for p in probs.iter_mut() {
        *p /= z;
    }
    probs
}

fn state_mask(s: &[i8]) -> usize {
    let mut mask = 0usize;
    for (i, &si) in s.iter().enumerate() {
        if si == 1 {
            mask |= 1 << i;
        }
    }
    mask
}

// ───────────────────────────── unit checks ─────────────────────────────

#[test]
fn ladder_shapes() {
    let geo = TemperingSpec {
        replicas: 8,
        beta_min: 0.1,
        beta_max: 6.0,
        swap_interval: 10,
        ladder: Ladder::Geometric,
    };
    let b = geo.betas();
    assert_eq!(b.len(), 8);
    assert!((b[0] - 0.1).abs() < 1e-12 && (b[7] - 6.0).abs() < 1e-12);
    assert!(b.windows(2).all(|w| w[1] > w[0]), "ladder must ascend");
    // Geometric: constant ratio.
    let r0 = b[1] / b[0];
    for w in b.windows(2) {
        assert!((w[1] / w[0] - r0).abs() < 1e-9, "ratio drift in {b:?}");
    }

    let lin = TemperingSpec {
        ladder: Ladder::Linear,
        replicas: 5,
        beta_min: 0.5,
        beta_max: 2.5,
        swap_interval: 10,
    };
    let b = lin.betas();
    assert_eq!(b.len(), 5);
    assert!((b[0] - 0.5).abs() < 1e-12 && (b[4] - 2.5).abs() < 1e-12);
    // Linear: constant difference.
    let d0 = b[1] - b[0];
    for w in b.windows(2) {
        assert!((w[1] - w[0] - d0).abs() < 1e-12, "spacing drift in {b:?}");
    }

    // Degenerate single rung sits at the cold end.
    let one = TemperingSpec {
        replicas: 1,
        ..TemperingSpec::default()
    };
    assert_eq!(one.betas(), vec![6.0]);
}

#[test]
fn spec_serde() {
    // The documented API shape parses field-for-field.
    let json = r#"{"replicas": 8, "beta_min": 0.1, "beta_max": 6.0,
                   "swap_interval": 10, "ladder": "geometric"}"#;
    let spec: TemperingSpec = serde_json::from_str(json).unwrap();
    assert_eq!(spec.replicas, 8);
    assert_eq!(spec.beta_min, 0.1);
    assert_eq!(spec.beta_max, 6.0);
    assert_eq!(spec.swap_interval, 10);
    assert_eq!(spec.ladder, Ladder::Geometric);

    // Every field defaults.
    let dflt: TemperingSpec = serde_json::from_str("{}").unwrap();
    assert_eq!(dflt.replicas, 8);
    assert_eq!(dflt.beta_min, 0.1);
    assert_eq!(dflt.beta_max, 6.0);
    assert_eq!(dflt.swap_interval, 10);
    assert_eq!(dflt.ladder, Ladder::Geometric);

    // Linear variant in snake_case; round-trip preserves it.
    let lin: TemperingSpec = serde_json::from_str(r#"{"ladder": "linear"}"#).unwrap();
    assert_eq!(lin.ladder, Ladder::Linear);
    let back: TemperingSpec = serde_json::from_str(&serde_json::to_string(&lin).unwrap()).unwrap();
    assert_eq!(back.ladder, Ladder::Linear);
    assert!(
        serde_json::to_string(&spec)
            .unwrap()
            .contains("\"geometric\"")
    );
}

// ─────────────────────── detailed-balance anchor ───────────────────────

/// THE correctness anchor: replica exchange must leave every rung's
/// stationary distribution exactly Boltzmann at that rung's β. N=4 model,
/// exact 2⁴ enumeration per rung, χ² per rung. swap_interval = 1 maximizes
/// exchange traffic, so a broken Metropolis rule cannot hide.
#[test]
fn pt_preserves_per_beta_marginals() {
    let model = random_model(4, 3);
    let spec = TemperingSpec {
        replicas: 3,
        beta_min: 0.4,
        beta_max: 1.2,
        swap_interval: 1,
        ladder: Ladder::Geometric,
    };
    let betas = spec.betas();
    let exact: Vec<Vec<f64>> = betas.iter().map(|&b| exact_probs(&model, b)).collect();

    let mut pt = Tempering::new(&model, &spec, 11);
    for _ in 0..1_000 {
        pt.advance(&model, 1);
        pt.swap_round();
    }
    let n_samples = 200_000u64;
    let mut counts = vec![[0u64; 16]; 3];
    for _ in 0..n_samples {
        pt.advance(&model, 1);
        pt.swap_round();
        for (k, c) in counts.iter_mut().enumerate() {
            c[state_mask(pt.state(k))] += 1;
        }
    }
    // Plenty of exchange traffic, or the test proves nothing.
    assert!(
        pt.stats.acceptance_overall() > 0.2,
        "exchange acceptance too low ({:.3}) to exercise detailed balance",
        pt.stats.acceptance_overall()
    );
    // χ², 15 dof per rung; consecutive Gibbs samples are correlated which
    // inflates χ² above the iid quantile — same generous bound as
    // tests/analytic.rs boltzmann_distribution, still catches a distorted
    // marginal by orders of magnitude.
    for (k, c) in counts.iter().enumerate() {
        let chi2: f64 = (0..16)
            .map(|m| {
                let expected = exact[k][m] * n_samples as f64;
                (c[m] as f64 - expected).powi(2) / expected
            })
            .sum();
        assert!(
            chi2 < 600.0,
            "rung {k} (β={:.3}): χ² = {chi2:.1} — swaps distorted the marginal",
            betas[k]
        );
    }
}

// ─────────────────────── Metropolis swap properties ───────────────────────

/// Recorded swap decisions obey the Metropolis rule: x ≥ 0 always accepted;
/// for x < 0 each accept is Bernoulli(eˣ) with an independent uniform, so the
/// total accept count concentrates at Σp within 5√(Σp(1−p)).
#[test]
fn swap_acceptance_matches_metropolis() {
    let model = random_model(12, 11);
    let spec = TemperingSpec {
        replicas: 4,
        beta_min: 0.2,
        beta_max: 2.0,
        swap_interval: 2,
        ladder: Ladder::Geometric,
    };
    let out = parallel_temper(&model, &spec, 4_000, 21, u64::MAX, None);

    let mut sum_p = 0.0;
    let mut var = 0.0;
    let mut observed = 0u64;
    let mut n_neg = 0u64;
    for d in &out.stats.decisions {
        if d.log_ratio >= 0.0 {
            assert!(d.accepted, "x = {} ≥ 0 must always accept", d.log_ratio);
        } else {
            let p = d.log_ratio.exp();
            sum_p += p;
            var += p * (1.0 - p);
            n_neg += 1;
            if d.accepted {
                observed += 1;
            }
        }
    }
    assert!(
        n_neg >= 200,
        "only {n_neg} downhill decisions — run too short"
    );
    let dev = (observed as f64 - sum_p).abs();
    let bound = 5.0 * var.sqrt();
    assert!(
        dev <= bound,
        "downhill accepts {observed} vs Σp {sum_p:.1} (|Δ| = {dev:.1} > 5σ = {bound:.1})"
    );
}

/// Fewer rungs over the same β range ⇒ wider Δβ gaps ⇒ lower acceptance.
#[test]
fn acceptance_decreases_with_wider_ladder() {
    let model = random_model(16, 5);
    let acc = |replicas: usize| {
        let spec = TemperingSpec {
            replicas,
            beta_min: 0.1,
            beta_max: 6.0,
            swap_interval: 10,
            ladder: Ladder::Geometric,
        };
        let out = parallel_temper(&model, &spec, 2_000, 9, u64::MAX, None);
        assert!(out.stats.attempts > 0);
        out.stats.acceptance_overall()
    };
    let dense = acc(6);
    let sparse = acc(3);
    assert!(
        dense > sparse,
        "K=6 acceptance {dense:.3} should exceed K=3 acceptance {sparse:.3}"
    );
}

// ─────────────────────── PT beats plain annealing ───────────────────────

/// On a frustrated ±J SK instance (exact ground state by 2¹⁶ enumeration),
/// parallel tempering with a matched total sweep budget (K·S) is never worse
/// than simulated annealing on any seed and strictly better on at least one.
#[test]
fn pt_beats_anneal_on_frustrated_sk() {
    let model = sk_instance(16, 918);
    let e0 = exact_ground_energy(&model);

    let replicas = 8usize;
    let sweeps_per_replica = 20u64;
    let (beta_min, beta_max) = (0.1, 6.0);
    let spec = TemperingSpec {
        replicas,
        beta_min,
        beta_max,
        swap_interval: 5,
        ladder: Ladder::Geometric,
    };
    // Matched budget: anneal gets all K·S sweeps in one chain.
    let schedule = Schedule {
        sweeps: replicas as u64 * sweeps_per_replica,
        beta0: beta_min,
        beta1: beta_max,
        kind: "geometric".into(),
    };

    let mut strict_wins = 0;
    for seed in 0..10u64 {
        let pt = parallel_temper(&model, &spec, sweeps_per_replica, seed, u64::MAX, None);
        let an = anneal(&model, &schedule, seed, u64::MAX, None);
        assert!(pt.best_energy >= e0 - 1e-9 && an.best_energy >= e0 - 1e-9);
        assert!(
            pt.best_energy <= an.best_energy + 1e-9,
            "seed {seed}: PT {} worse than anneal {} (ground {e0})",
            pt.best_energy,
            an.best_energy
        );
        if pt.best_energy < an.best_energy - 1e-9 {
            strict_wins += 1;
        }
        eprintln!(
            "seed {seed}: PT {} | anneal {} | ground {e0}",
            pt.best_energy, an.best_energy
        );
    }
    assert!(
        strict_wins >= 1,
        "matched budget never separated PT from annealing"
    );
}

// ─────────────────────────── determinism ───────────────────────────

/// Bit-identical results at any rayon thread count (and across runs).
#[test]
fn pt_determinism_single_vs_multi_thread() {
    // Big enough (64 spins × 20-sweep chunks × 6 replicas) that advance()
    // takes the rayon path — the parallel schedule itself is under test.
    let model = random_model(64, 8);
    let spec = TemperingSpec {
        replicas: 6,
        beta_min: 0.2,
        beta_max: 4.0,
        swap_interval: 20,
        ladder: Ladder::Geometric,
    };
    let run = |threads: usize| {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        pool.install(|| parallel_temper(&model, &spec, 500, 42, 100, None))
    };
    let a = run(1);
    let b = run(6);

    assert_eq!(a.best_energy.to_bits(), b.best_energy.to_bits());
    assert_eq!(a.best_state, b.best_state);
    assert_eq!(a.stats.attempts, b.stats.attempts);
    assert_eq!(a.stats.accepts, b.stats.accepts);
    assert_eq!(a.stats.per_pair_accepts, b.stats.per_pair_accepts);
    assert_eq!(a.stats.decisions.len(), b.stats.decisions.len());
    for (da, db) in a.stats.decisions.iter().zip(&b.stats.decisions) {
        assert_eq!(da.log_ratio.to_bits(), db.log_ratio.to_bits());
        assert_eq!(da.accepted, db.accepted);
    }
    assert_eq!(a.ledger.sweeps, b.ledger.sweeps);
    assert_eq!(a.ledger.spin_updates, b.ledger.spin_updates);
    assert_eq!(a.ledger.flips, b.ledger.flips);
    assert_eq!(a.trace.len(), b.trace.len());
    for (ta, tb) in a.trace.iter().zip(&b.trace) {
        assert_eq!(ta.energy.to_bits(), tb.energy.to_bits());
        assert_eq!(ta.best_energy.to_bits(), tb.best_energy.to_bits());
    }
    // Per-replica budget accounting: K replicas × S sweeps.
    assert_eq!(a.ledger.sweeps, 6 * 500);
}

// ─────────────────────────── executor surface ───────────────────────────

/// Tempering job through the executor: swap stats + progress metrics out;
/// tempering-absent JSON deserializes to None and follows the anneal path.
#[test]
fn executor_round_trip() {
    let json = r#"{
        "problem": {"kind": "petersen"},
        "schedule": {"sweeps": 1000, "beta0": 0.1, "beta1": 6.0},
        "seed": 7,
        "tempering": {"replicas": 4, "beta_min": 0.1, "beta_max": 6.0,
                      "swap_interval": 10, "ladder": "geometric"}
    }"#;
    let job: StochasticJob = serde_json::from_str(json).unwrap();
    assert!(job.tempering.is_some());

    let mut ticks = Vec::new();
    let res = StochasticExecutor.execute(&job, &mut |p| ticks.push(p));
    // Petersen optimum is 12 (published) — PT at β→6 must reach it.
    assert_eq!(res.outputs["best_cut"], 12.0);
    assert_eq!(res.outputs["replica_count"], 4);
    assert_eq!(
        res.outputs["per_pair_acceptance"].as_array().unwrap().len(),
        3
    );
    let attempts = res.outputs["swap_attempts"].as_u64().unwrap();
    let accepts = res.outputs["swap_accepts"].as_u64().unwrap();
    assert!(attempts > 0 && accepts <= attempts);
    let overall = res.outputs["swap_acceptance_overall"].as_f64().unwrap();
    assert!((overall - accepts as f64 / attempts as f64).abs() < 1e-12);
    // Merged ledger: 4 replicas × 1000 sweeps.
    assert_eq!(res.ledger.sweeps, 4 * 1000);
    // Progress metrics carry the tempering extras plus the cut conversion.
    assert!(!ticks.is_empty());
    let m = &ticks.last().unwrap().metrics;
    assert!(m.get("best_cut").is_some());
    assert!(m.get("swap_acceptance_rate").is_some());
    assert_eq!(m["replica_betas"].as_array().unwrap().len(), 4);

    // Absent field → None → anneal path, no swap stats in outputs.
    let plain = r#"{
        "problem": {"kind": "cycle", "n": 10},
        "schedule": {"sweeps": 500, "beta0": 0.1, "beta1": 6.0},
        "seed": 2
    }"#;
    let job2: StochasticJob = serde_json::from_str(plain).unwrap();
    assert!(job2.tempering.is_none());
    let res2 = StochasticExecutor.execute(&job2, &mut |_| {});
    assert_eq!(res2.outputs["best_cut"], 10.0);
    assert!(res2.outputs.get("swap_attempts").is_none());
    assert_eq!(res2.ledger.sweeps, 500);
}
