//! Analytic + published-number validation per docs/SIM-ROADMAP.md §3.1.
//! No foreign-tool fixtures — every expected value below is computed in
//! closed form or taken from the printed literature.

use tei_sim_core::rng::Rng;
use tei_sim_stochastic::{
    GibbsSampler, IsingModel, Schedule, anneal, graphs, maxcut, sample_observables,
};

/// Exact ⟨E⟩ and ⟨|m|⟩ by full 2^n enumeration.
fn exact_observables(model: &IsingModel, beta: f64) -> (f64, f64) {
    let n = model.n;
    assert!(n <= 20);
    let mut z = 0.0;
    let mut e_acc = 0.0;
    let mut m_acc = 0.0;
    let mut s = vec![1i8; n];
    for mask in 0u64..(1 << n) {
        for (i, si) in s.iter_mut().enumerate() {
            *si = if mask >> i & 1 == 1 { 1 } else { -1 };
        }
        let e = model.energy(&s);
        let w = (-beta * e).exp();
        let m: i64 = s.iter().map(|&x| x as i64).sum();
        z += w;
        e_acc += w * e;
        m_acc += w * (m as f64 / n as f64).abs();
    }
    (e_acc / z, m_acc / z)
}

fn random_model(n: usize, seed: u64) -> IsingModel {
    let mut rng = Rng::new(seed);
    let h: Vec<f64> = (0..n).map(|_| 0.5 * rng.normal()).collect();
    // Ring + random chords for nontrivial structure.
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

/// Sampler ⟨E⟩, ⟨|m|⟩ match exact enumeration on a 12-spin model.
#[test]
fn matches_exact_enumeration() {
    let model = random_model(12, 11);
    let beta = 0.7;
    let (e_exact, m_exact) = exact_observables(&model, beta);
    let (e_mc, m_mc) = sample_observables(&model, beta, 2_000, 60_000, 5);
    assert!(
        (e_mc - e_exact).abs() < 0.05 * e_exact.abs().max(1.0),
        "⟨E⟩ MC {e_mc:.4} vs exact {e_exact:.4}"
    );
    assert!(
        (m_mc - m_exact).abs() < 0.05,
        "⟨|m|⟩ MC {m_mc:.4} vs exact {m_exact:.4}"
    );
}

/// Empirical state distribution matches Boltzmann on a 4-spin model (χ²).
#[test]
fn boltzmann_distribution() {
    let model = random_model(4, 3);
    let beta = 0.8;
    // Exact probabilities.
    let mut probs = [0.0f64; 16];
    let mut s = vec![1i8; 4];
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
    // Empirical counts.
    let mut rng = Rng::new(17);
    let sampler = GibbsSampler::new(&model);
    let mut ledger = Default::default();
    let mut state = vec![1i8; 4];
    for _ in 0..1_000 {
        sampler.sweep(&model, &mut state, beta, &mut rng, &mut ledger);
    }
    let n_samples = 200_000u64;
    let mut counts = [0u64; 16];
    for _ in 0..n_samples {
        sampler.sweep(&model, &mut state, beta, &mut rng, &mut ledger);
        let mut mask = 0usize;
        for (i, &si) in state.iter().enumerate() {
            if si == 1 {
                mask |= 1 << i;
            }
        }
        counts[mask] += 1;
    }
    // χ² with 15 dof; consecutive Gibbs samples are correlated, which
    // inflates χ² above the iid quantile — use a generous bound that still
    // catches a wrong distribution by orders of magnitude.
    let chi2: f64 = (0..16)
        .map(|k| {
            let expected = probs[k] * n_samples as f64;
            (counts[k] as f64 - expected).powi(2) / expected
        })
        .sum();
    assert!(chi2 < 600.0, "χ² = {chi2:.1} — sampler distribution is off");
}

/// Onsager 1944: the 2D square-lattice Ising critical temperature is
/// T_c = 2/ln(1+√2) ≈ 2.269. Below T_c the lattice magnetizes; above it
/// doesn't. We bracket T_c from both sides on a 16×16 periodic lattice.
#[test]
fn onsager_critical_bracket() {
    let l = 16;
    let n = l * l;
    let idx = |x: usize, y: usize| (y * l + x) as u32;
    let mut edges = Vec::new();
    for y in 0..l {
        for x in 0..l {
            edges.push((idx(x, y), idx((x + 1) % l, y), 1.0));
            edges.push((idx(x, y), idx(x, (y + 1) % l), 1.0));
        }
    }
    let model = IsingModel::new(vec![0.0; n], &edges);

    let (_, m_cold) = sample_observables(&model, 1.0 / 1.5, 3_000, 8_000, 21); // T = 1.5 < T_c
    let (_, m_hot) = sample_observables(&model, 1.0 / 3.5, 3_000, 8_000, 22); // T = 3.5 > T_c
    assert!(
        m_cold > 0.85,
        "T=1.5 should be deeply ordered, ⟨|m|⟩ = {m_cold:.3}"
    );
    assert!(
        m_hot < 0.35,
        "T=3.5 should be disordered, ⟨|m|⟩ = {m_hot:.3}"
    );
    assert!(
        m_cold - m_hot > 0.5,
        "order parameter must drop across T_c = 2/ln(1+√2): {m_cold:.3} → {m_hot:.3}"
    );
}

fn solve_maxcut(spec: maxcut::ProblemSpec, sweeps: u64, seed: u64) -> (f64, Option<f64>) {
    let g = spec.build();
    let model = maxcut::to_ising(&g);
    let total_w: f64 = g.edges.iter().map(|e| e.2).sum();
    let schedule = Schedule {
        sweeps,
        beta0: 0.1,
        beta1: 6.0,
        kind: "geometric".into(),
    };
    let out = anneal(&model, &schedule, seed, sweeps.max(1), None);
    ((total_w - out.best_energy) / 2.0, spec.known_optimum())
}

/// K₁₀: optimum = ⌊10/2⌋·⌈10/2⌉ = 25 (closed form).
#[test]
fn maxcut_complete_k10() {
    let (cut, opt) = solve_maxcut(maxcut::ProblemSpec::Complete { n: 10 }, 2_000, 1);
    assert_eq!(
        cut,
        opt.unwrap(),
        "K10 cut {cut} vs optimum {}",
        opt.unwrap()
    );
}

/// Even cycle C₂₀: optimum = 20 (bipartite — every edge cut).
#[test]
fn maxcut_even_cycle() {
    let (cut, opt) = solve_maxcut(maxcut::ProblemSpec::Cycle { n: 20 }, 3_000, 2);
    assert_eq!(cut, opt.unwrap());
}

/// Odd cycle C₂₁: optimum = 20 (one edge must survive).
#[test]
fn maxcut_odd_cycle() {
    let (cut, opt) = solve_maxcut(maxcut::ProblemSpec::Cycle { n: 21 }, 3_000, 3);
    assert_eq!(cut, opt.unwrap());
}

/// K_{6,7}: optimum = 42 (cut every edge of the bipartition).
#[test]
fn maxcut_complete_bipartite() {
    let (cut, opt) = solve_maxcut(
        maxcut::ProblemSpec::CompleteBipartite { a: 6, b: 7 },
        3_000,
        4,
    );
    assert_eq!(cut, opt.unwrap());
}

/// Petersen graph: optimum = 12 (classic published value).
#[test]
fn maxcut_petersen() {
    let (cut, opt) = solve_maxcut(maxcut::ProblemSpec::Petersen, 3_000, 5);
    assert_eq!(cut, opt.unwrap());
}

/// Throughput budget (roadmap §6): ≥ 10⁸ spin-updates/s/core in release.
/// Ignored by default — run with `cargo test --release -- --ignored bench`.
#[test]
#[ignore]
fn bench_spin_update_rate() {
    let g = graphs::random_regular(10_000, 4, 42);
    let model = maxcut::to_ising(&g);
    let schedule = Schedule {
        sweeps: 2_000,
        beta0: 0.5,
        beta1: 0.5,
        kind: "linear".into(),
    };
    let t0 = std::time::Instant::now();
    let out = anneal(&model, &schedule, 7, u64::MAX, None);
    let dt = t0.elapsed().as_secs_f64();
    let rate = out.ledger.spin_updates as f64 / dt;
    eprintln!(
        "spin-update rate: {rate:.3e}/s ({} updates in {dt:.2}s)",
        out.ledger.spin_updates
    );
    assert!(rate > 1.0e7, "rate {rate:.3e} below floor");
}
