//! Analytic validation per docs/SIM-ROADMAP.md §3.7 — closed-form ground
//! truth only, no foreign-tool fixtures.
//!
//! Coverage: (a) numerical dispersion vs the exact Yee relation,
//! (b) CPML reflection floor, (c) CFL stability boundary, (d) dielectric
//! slab group delay, (e) post-source energy decay, (f) determinism.

use std::f64::consts::PI;

use tei_sim_core::exec::{Executor, Progress};
use tei_sim_field::{
    CpmlParams, DftMonitor, EpsSpec, FieldExecutor, FieldJob, Grid2d, GridSpec, Probe, Source,
    SourceShape, TimeProfile, yee_axis_wavenumber,
};

fn vacuum_grid(nx: usize, ny: usize, npml: usize, courant: f64) -> Grid2d {
    let spec = GridSpec {
        nx,
        ny,
        courant,
        npml,
        cpml: CpmlParams::default(),
    };
    Grid2d::new(&spec, vec![1.0; nx * ny])
}

fn gaussian_point(i: usize, j: usize, t0: f64, tau: f64) -> Source {
    Source {
        shape: SourceShape::Point { i, j },
        time: TimeProfile::Gaussian { t0, tau },
        amplitude: 1.0,
    }
}

/// (a) Numerical dispersion: a CW point source radiates a cylindrical wave;
/// on the +x axis the asymptotic phase gradient equals the axis-aligned Yee
/// wavenumber k(ω) (the Hankel-function curvature correction is
/// O(1/(kr)²) ≈ 2·10⁻⁴ at the monitor radii used here). We DFT the steady
/// state at a row of on-axis monitors over an integer number of periods
/// that is also an integer number of steps — the conjugate −ω line then
/// cancels exactly — and least-squares fit the unwrapped phase slope.
#[test]
fn numerical_dispersion_matches_yee_relation() {
    let (nx, ny, npml) = (140usize, 100usize, 10usize);
    let s = 0.5;
    let mut g = vacuum_grid(nx, ny, npml, s);
    let dt = g.dt;

    // Period locked to an integer number of steps (24) so the DFT window
    // can span whole periods exactly. λ ≈ 8.3 cells — deliberately coarse
    // so grid dispersion (≈ 2.2%) well exceeds the 1% test tolerance.
    let period_steps = 24usize;
    let omega = 2.0 * PI / (period_steps as f64 * dt);
    let src = Source {
        shape: SourceShape::Point { i: 40, j: ny / 2 },
        time: TimeProfile::Cw {
            omega,
            ramp: 4.0 * period_steps as f64 * dt,
        },
        amplitude: 1.0,
    };

    let xs: Vec<usize> = (70..=100).step_by(2).collect();
    let mut mons: Vec<DftMonitor> = xs
        .iter()
        .map(|&i| DftMonitor::new(i, ny / 2, vec![omega]))
        .collect();

    let warmup = 450usize; // wavefront + ramp transient clears the monitors
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

    // Unwrap DFT phases along the axis, then least-squares the slope.
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
    assert!(
        rel < 0.01,
        "measured k = {k_measured:.6}, Yee k = {k_theory:.6}, rel err {rel:.4}"
    );
    // Sharpness: the grid dispersion itself must exceed the tolerance,
    // otherwise the test could not distinguish Yee from vacuum physics.
    assert!((k_theory - omega).abs() / omega > 0.015);
}

/// Run a Gaussian point pulse and return the probe trace.
fn pulse_probe_trace(
    nx: usize,
    ny: usize,
    src_ij: (usize, usize),
    probe_ij: (usize, usize),
    steps: usize,
    eps: Vec<f64>,
    (t0, tau): (f64, f64),
) -> Vec<f64> {
    let spec = GridSpec {
        nx,
        ny,
        courant: 0.5,
        npml: 10,
        cpml: CpmlParams::default(),
    };
    let mut g = Grid2d::new(&spec, eps);
    let dt = g.dt;
    let src = gaussian_point(src_ij.0, src_ij.1, t0, tau);
    let mut probe = Probe::new(probe_ij.0, probe_ij.1);
    for n in 0..steps {
        g.step();
        src.inject(&mut g, (n + 1) as f64 * dt);
        probe.record(&g);
    }
    probe.trace
}

/// (b) CPML reflection floor. A 2D pulse has a long-lived wake (the 2D
/// Green's function tail), so raw late-time amplitude cannot isolate
/// boundary reflections. Standard measurement instead: run the identical
/// source/probe geometry on a 2× larger reference grid whose boundaries
/// are too far for any reflection to reach the probe inside the time
/// window, and subtract the traces. The difference is purely the small
/// grid's boundary reflection; its peak must sit < −50 dB (< 3·10⁻³)
/// below the incident peak.
#[test]
fn cpml_reflection_below_minus_50_db() {
    // Probe 12 cells from the CPML interface (interface at j = 10); window
    // chosen so the reference grid's own reflections (earliest at t = 152)
    // never reach its probe (window ends at t ≈ 145).
    let steps = 410;
    let pulse = (24.0, 8.0);
    let small = pulse_probe_trace(
        100,
        100,
        (50, 50),
        (50, 22),
        steps,
        vec![1.0; 100 * 100],
        pulse,
    );
    let reference = pulse_probe_trace(
        200,
        200,
        (100, 100),
        (100, 72),
        steps,
        vec![1.0; 200 * 200],
        pulse,
    );

    let incident_peak = reference.iter().fold(0.0f64, |a, &b| a.max(b.abs()));
    let reflected_peak = small
        .iter()
        .zip(&reference)
        .fold(0.0f64, |a, (&x, &y)| a.max((x - y).abs()));
    assert!(incident_peak > 0.0);
    let ratio = reflected_peak / incident_peak;
    assert!(
        ratio < 3e-3,
        "CPML reflection {ratio:.2e} ({:.1} dB), need < 3e-3 (−50 dB)",
        20.0 * ratio.log10()
    );
}

/// (c) CFL stability boundary: Δt = S·Δ/√2 is stable iff S ≤ 1
/// (Taflove & Hagness §4.7). S = 1.02 must blow up (worst mode grows
/// ≈ ×1.49 per step); S = 0.98 must stay bounded.
#[test]
fn courant_stability_boundary() {
    let run = |courant: f64, steps: usize| -> f64 {
        let spec = GridSpec {
            nx: 60,
            ny: 60,
            courant,
            npml: 8,
            cpml: CpmlParams::default(),
        };
        let mut g = Grid2d::new(&spec, vec![1.0; 60 * 60]);
        let dt = g.dt;
        let src = gaussian_point(30, 30, 10.0, 3.0);
        for n in 0..steps {
            g.step();
            src.inject(&mut g, (n + 1) as f64 * dt);
        }
        g.ez_max_abs()
    };

    let unstable = run(1.02, 500);
    assert!(
        unstable > 1e6,
        "S = 1.02 should diverge, max = {unstable:e}"
    );
    let stable = run(0.98, 1500);
    assert!(
        stable.is_finite() && stable < 1e2,
        "S = 0.98 should stay bounded, max = {stable:e}"
    );
}

/// (d) Dielectric slab delay: a baseband pulse crossing a slab of ε_r = 4
/// (n = 2), thickness d, arrives later than in vacuum by (n−1)·d/c — here
/// 16 cells of time = 45.25 steps. Measured as the cross-correlation peak
/// between the vacuum and slab probe traces (parabolic sub-step
/// refinement); a baseband Gaussian keeps group-velocity dispersion
/// negligible. Fabry-Pérot echoes (≈ 0.1 amplitude) trail the main pulse
/// by 2nd = 64 time units, far outside the correlation peak.
#[test]
fn dielectric_slab_group_delay() {
    let (nx, ny) = (200usize, 80usize);
    let (i0, i1) = (95usize, 111usize); // d = 16 cells
    let steps = 640;
    let pulse = (45.0, 15.0);
    let src = (50, ny / 2);
    let prb = (150, ny / 2);

    let vacuum = pulse_probe_trace(nx, ny, src, prb, steps, vec![1.0; nx * ny], pulse);
    let slab_eps = EpsSpec::Slab {
        eps_r: 4.0,
        i0,
        i1,
        background: 1.0,
    }
    .build(nx, ny);
    let slab = pulse_probe_trace(nx, ny, src, prb, steps, slab_eps, pulse);

    // c(l) = Σ_t vacuum[t]·slab[t+l], maximized where the slab trace,
    // shifted back by l steps, best matches the vacuum trace.
    let max_lag = 90usize;
    let corr: Vec<f64> = (0..=max_lag)
        .map(|l| {
            vacuum[..steps - l]
                .iter()
                .zip(&slab[l..])
                .map(|(a, b)| a * b)
                .sum()
        })
        .collect();
    let lmax = corr
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .unwrap()
        .0;
    assert!(lmax > 0 && lmax < max_lag, "correlation peak at edge");
    // Parabolic interpolation around the discrete peak.
    let (c0, c1, c2) = (corr[lmax - 1], corr[lmax], corr[lmax + 1]);
    let lag = lmax as f64 + 0.5 * (c0 - c2) / (c0 - 2.0 * c1 + c2);

    let dt = 0.5 / 2f64.sqrt();
    let expected = (2.0 - 1.0) * (i1 - i0) as f64 / dt; // (n−1)·d/c in steps
    let rel = (lag - expected).abs() / expected;
    assert!(
        rel < 0.05,
        "slab delay {lag:.2} steps, expected {expected:.2} (rel err {rel:.3})"
    );
}

/// Time-centered field energy ½[Σ ε·Ez² + Σ Hx⁻·Hx⁺ + Σ Hy⁻·Hy⁺] with the
/// H product bracketing Ez's time level — the discrete quantity that is
/// exactly conserved by the lossless Yee leapfrog under CFL (the plainly
/// sampled ½Σ(E² + H²) oscillates at O(ωΔt) because of the half-step
/// stagger). Vacuum here, so ε ≡ 1. Computed by stepping a clone to get
/// H at the next half level.
fn staggered_energy(g: &Grid2d) -> f64 {
    let mut next = g.clone();
    next.step();
    let e: f64 = g.ez.iter().map(|&v| v * v).sum();
    let hx: f64 = g.hx.iter().zip(&next.hx).map(|(a, b)| a * b).sum();
    let hy: f64 = g.hy.iter().zip(&next.hy).map(|(a, b)| a * b).sum();
    0.5 * (e + hx + hy)
}

/// (e) Energy boundedness: after the source turns off, total field energy
/// must decay monotonically (CPML absorption) and never grow.
#[test]
fn energy_decays_monotonically_after_source_off() {
    let mut g = vacuum_grid(90, 90, 10, 0.5);
    let dt = g.dt;
    let src = gaussian_point(45, 45, 24.0, 8.0);
    let drive = 160usize; // source amplitude < 1e-7 by t = 160·dt ≈ 57
    for n in 0..drive {
        g.step();
        src.inject(&mut g, (n + 1) as f64 * dt);
    }

    let mut energies = vec![staggered_energy(&g)];
    for _ in 0..28 {
        for _ in 0..25 {
            g.step();
        }
        energies.push(staggered_energy(&g));
    }

    let e_first = energies[0];
    assert!(e_first > 0.0);
    for w in energies.windows(2) {
        assert!(
            w[1] <= w[0] * (1.0 + 1e-9) + 1e-30,
            "energy grew: {} -> {}",
            w[0],
            w[1]
        );
    }
    let e_last = *energies.last().unwrap();
    assert!(
        e_last < 1e-4 * e_first,
        "CPML should have drained the grid: {e_last:e} vs {e_first:e}"
    );
}

/// (f) Determinism: identical jobs produce bit-identical outputs (there is
/// no RNG anywhere in the field column). Also exercises the serde job
/// schema end-to-end and checks the ledger arithmetic.
#[test]
fn executor_is_deterministic() {
    let job: FieldJob = serde_json::from_str(
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
    .expect("FieldJob JSON schema");

    let exec = FieldExecutor;
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

    assert_eq!(r1.ledger.macs, 3 * 80 * 80 * 400);
    // probe reads + DFT reads, one per step per monitor
    assert_eq!(r1.ledger.adc_samples, 400 + 400);

    let dft = r1.outputs["dft"].as_array().unwrap();
    assert_eq!(dft.len(), 2);
    assert!(dft[0]["abs"].as_f64().unwrap() > 0.0);
    let trace = r1.outputs["probe"]["trace"].as_array().unwrap();
    assert_eq!(trace.len(), 400);
    let snap = r1.outputs["snapshot"]["ez"].as_array().unwrap();
    assert_eq!(snap.len(), 80 * 80);
}
