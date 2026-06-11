//! F4 validation: the WGSL GPU kernels against the validated f64 CPU core
//! (roadmap §2 — cross-check vs our own validated implementation, plus the
//! direct CPML dB measurement and the energy-decay property re-run on the
//! GPU). Every test SKIPS gracefully (eprintln + return) when no wgpu
//! adapter is present, so GPU-less CI stays green.
#![cfg(feature = "gpu")]

use tei_sim_field::gpu::{GpuSim, run_probe};
use tei_sim_field::{
    CpmlParams, EpsSpec, FieldJob, Grid2d, GridSpec, Probe, Source, SourceShape, TimeProfile,
};

/// Skip helper: false (with a message) when no adapter is available.
fn gpu() -> bool {
    if tei_sim_field::gpu_available() {
        true
    } else {
        eprintln!("SKIP: no wgpu adapter available on this machine");
        false
    }
}

fn gaussian_point(i: usize, j: usize, t0: f64, tau: f64) -> Source {
    Source {
        shape: SourceShape::Point { i, j },
        time: TimeProfile::Gaussian { t0, tau },
        amplitude: 1.0,
    }
}

/// Minimal job builder (no F2 machinery).
fn job(
    nx: usize,
    ny: usize,
    steps: usize,
    eps: EpsSpec,
    source: Option<Source>,
    probe: Option<[usize; 2]>,
) -> FieldJob {
    FieldJob {
        nx,
        ny,
        steps,
        courant: 0.5,
        npml: 10,
        cpml: CpmlParams::default(),
        eps,
        source,
        mode_source: None,
        ports: Vec::new(),
        reference_eps: None,
        frequencies: Vec::new(),
        probe,
        snapshot: false,
    }
}

/// The CPU reference: the same loop the executor runs (step → inject →
/// record), in f64, on the validated [`Grid2d`] core.
fn cpu_probe_trace(job: &FieldJob) -> Vec<f64> {
    let spec = GridSpec {
        nx: job.nx,
        ny: job.ny,
        courant: job.courant,
        npml: job.npml,
        cpml: job.cpml.clone(),
    };
    let mut g = Grid2d::new(&spec, job.eps.build(job.nx, job.ny));
    let dt = g.dt;
    let (pi, pj) = job
        .probe
        .map(|p| (p[0], p[1]))
        .unwrap_or((job.nx / 2, job.ny / 2));
    let mut probe = Probe::new(pi, pj);
    for n in 0..job.steps {
        g.step();
        if let Some(src) = &job.source {
            src.inject(&mut g, (n + 1) as f64 * dt);
        }
        probe.record(&g);
    }
    probe.trace
}

/// (relative L2 error, max |diff| / peak |cpu|) of a GPU trace vs CPU.
fn trace_errors(gpu: &[f32], cpu: &[f64]) -> (f64, f64) {
    assert_eq!(gpu.len(), cpu.len());
    let mut num = 0.0f64;
    let mut den = 0.0f64;
    let mut max_diff = 0.0f64;
    let mut peak = 0.0f64;
    for (&g, &c) in gpu.iter().zip(cpu) {
        let d = g as f64 - c;
        num += d * d;
        den += c * c;
        max_diff = max_diff.max(d.abs());
        peak = peak.max(c.abs());
    }
    assert!(den > 0.0 && peak > 0.0, "degenerate CPU reference trace");
    ((num / den).sqrt(), max_diff / peak)
}

/// Vacuum Gaussian pulse, ~1000 steps of f32 accumulation: GPU probe trace
/// must match the f64 CPU trace to rel-L2 < 1e-3 and max-abs < 1e-3 of the
/// incident peak.
#[test]
fn vacuum_pulse_matches_cpu() {
    if !gpu() {
        return;
    }
    let j = job(
        160,
        160,
        1000,
        EpsSpec::Uniform { eps_r: 1.0 },
        Some(gaussian_point(80, 80, 30.0, 10.0)),
        Some([80, 40]),
    );
    let gpu_trace = run_probe(&j).unwrap();
    let cpu_trace = cpu_probe_trace(&j);
    let (l2, max_rel) = trace_errors(&gpu_trace, &cpu_trace);
    eprintln!("vacuum 160x160x1000: rel L2 = {l2:.3e}, max/peak = {max_rel:.3e}");
    assert!(l2 < 1e-3, "rel L2 {l2:.3e} >= 1e-3");
    assert!(max_rel < 1e-3, "max/peak {max_rel:.3e} >= 1e-3");
}

/// Dielectric slab (ε_r = 4 block in the propagation path): same GPU-vs-CPU
/// agreement through refraction, reflection and the Fabry-Pérot echoes.
#[test]
fn dielectric_slab_matches_cpu() {
    if !gpu() {
        return;
    }
    let j = job(
        200,
        80,
        800,
        EpsSpec::Slab {
            eps_r: 4.0,
            i0: 95,
            i1: 111,
            background: 1.0,
        },
        Some(gaussian_point(50, 40, 45.0, 15.0)),
        Some([150, 40]),
    );
    let gpu_trace = run_probe(&j).unwrap();
    let cpu_trace = cpu_probe_trace(&j);
    let (l2, max_rel) = trace_errors(&gpu_trace, &cpu_trace);
    eprintln!("slab 200x80x800: rel L2 = {l2:.3e}, max/peak = {max_rel:.3e}");
    assert!(l2 < 1e-3, "rel L2 {l2:.3e} >= 1e-3");
    assert!(max_rel < 1e-3, "max/peak {max_rel:.3e} >= 1e-3");
}

/// Direct CPML reflection measurement on the GPU (the stronger claim than
/// GPU-vs-CPU agreement): identical source/probe geometry on a 100² grid
/// and a 200² reference whose boundaries cannot reach the probe inside the
/// window; the trace difference is purely the small grid's boundary
/// reflection. Must sit below −40 dB of the incident peak (the f64 CPU
/// core achieves ≈ −54 dB on this exact geometry).
#[test]
fn gpu_cpml_reflection_below_minus_40_db() {
    if !gpu() {
        return;
    }
    let steps = 410;
    let small = run_probe(&job(
        100,
        100,
        steps,
        EpsSpec::Uniform { eps_r: 1.0 },
        Some(gaussian_point(50, 50, 24.0, 8.0)),
        Some([50, 22]),
    ))
    .unwrap();
    let reference = run_probe(&job(
        200,
        200,
        steps,
        EpsSpec::Uniform { eps_r: 1.0 },
        Some(gaussian_point(100, 100, 24.0, 8.0)),
        Some([100, 72]),
    ))
    .unwrap();

    let incident_peak = reference
        .iter()
        .fold(0.0f64, |a, &b| a.max((b as f64).abs()));
    let reflected_peak = small
        .iter()
        .zip(&reference)
        .fold(0.0f64, |a, (&x, &y)| a.max((x as f64 - y as f64).abs()));
    assert!(incident_peak > 0.0);
    let ratio = reflected_peak / incident_peak;
    eprintln!(
        "GPU CPML reflection: {ratio:.2e} ({:.1} dB)",
        20.0 * ratio.log10()
    );
    assert!(
        ratio < 1e-2,
        "GPU CPML reflection {ratio:.2e} ({:.1} dB), need < 1e-2 (−40 dB)",
        20.0 * ratio.log10()
    );
}

/// Staggered (time-centered) field energy on the GPU — the discrete
/// invariant of the lossless leapfrog: ½[Σ ε·Ez² + Σ Hx⁻·Hx⁺ + Σ Hy⁻·Hy⁺].
/// Costs one extra leapfrog step to sample H at the next half level.
fn gpu_staggered_energy(sim: &mut GpuSim) -> f64 {
    let ez = sim.read_ez().unwrap();
    let hx = sim.read_hx().unwrap();
    let hy = sim.read_hy().unwrap();
    sim.step(1).unwrap();
    let hx2 = sim.read_hx().unwrap();
    let hy2 = sim.read_hy().unwrap();
    let e: f64 = ez.iter().map(|&v| (v as f64) * (v as f64)).sum();
    let sx: f64 = hx
        .iter()
        .zip(&hx2)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    let sy: f64 = hy
        .iter()
        .zip(&hy2)
        .map(|(&a, &b)| a as f64 * b as f64)
        .sum();
    0.5 * (e + sx + sy)
}

/// Energy decay property on the GPU: after the source dies off, staggered
/// energy must decay monotonically (CPML absorption) and drain the grid —
/// the same property the CPU suite asserts in tests/analytic.rs. Vacuum,
/// so ε ≡ 1 in the energy sum.
#[test]
fn gpu_energy_decays_monotonically_after_source_off() {
    if !gpu() {
        return;
    }
    let drive = 160usize; // source amplitude < 1e-7 by t = 160·dt ≈ 57
    let windows = 28usize;
    let j = job(
        90,
        90,
        drive + 1 + windows * 25,
        EpsSpec::Uniform { eps_r: 1.0 },
        Some(gaussian_point(45, 45, 24.0, 8.0)),
        None,
    );
    let mut sim = GpuSim::new(&j).unwrap();
    sim.step(drive).unwrap();

    let mut energies = vec![gpu_staggered_energy(&mut sim)];
    for _ in 0..windows {
        sim.step(24).unwrap();
        energies.push(gpu_staggered_energy(&mut sim)); // +1 step inside
    }

    let e_first = energies[0];
    assert!(e_first > 0.0);
    for w in energies.windows(2) {
        assert!(
            w[1] <= w[0] * (1.0 + 1e-6) + 1e-30,
            "energy grew: {} -> {}",
            w[0],
            w[1]
        );
    }
    let e_last = *energies.last().unwrap();
    eprintln!("GPU energy decay: {e_first:.3e} -> {e_last:.3e}");
    assert!(
        e_last < 1e-4 * e_first,
        "CPML should have drained the grid: {e_last:e} vs {e_first:e}"
    );
}

/// Determinism: two independent GPU runs of the same job on the same
/// adapter are bit-identical — no atomics, fixed dispatch order.
#[test]
fn gpu_runs_are_bit_identical() {
    if !gpu() {
        return;
    }
    let j = job(
        120,
        120,
        500,
        EpsSpec::Slab {
            eps_r: 4.0,
            i0: 70,
            i1: 80,
            background: 1.0,
        },
        Some(gaussian_point(40, 60, 24.0, 8.0)),
        Some([90, 60]),
    );
    let run = || {
        let mut sim = GpuSim::new(&j).unwrap();
        sim.step(j.steps).unwrap();
        (sim.read_trace().unwrap(), sim.read_ez().unwrap())
    };
    let (t1, e1) = run();
    let (t2, e2) = run();
    assert!(
        t1.iter().zip(&t2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "probe traces differ between identical GPU runs"
    );
    assert!(
        e1.iter().zip(&e2).all(|(a, b)| a.to_bits() == b.to_bits()),
        "final Ez fields differ between identical GPU runs"
    );
}

/// Throughput report, GPU vs CPU, in MCells/s (= nx·ny·steps / wall / 1e6).
/// `cargo test -p tei-sim-field --release --features gpu -- --ignored bench`
#[test]
#[ignore]
fn bench_gpu_vs_cpu() {
    if !gpu() {
        return;
    }
    let bench = |nx: usize, ny: usize, steps: usize| {
        let j = job(
            nx,
            ny,
            steps,
            EpsSpec::Uniform { eps_r: 1.0 },
            Some(gaussian_point(nx / 2, ny / 2, 30.0, 10.0)),
            None,
        );
        let cells = (nx * ny * steps) as f64;

        // GPU: setup excluded; read_trace at the end forces full sync.
        let mut sim = GpuSim::new(&j).unwrap();
        let t0 = std::time::Instant::now();
        sim.step(steps).unwrap();
        let trace = sim.read_trace().unwrap();
        let gpu_s = t0.elapsed().as_secs_f64();
        assert_eq!(trace.len(), steps);

        // CPU: the same loop on the f64 core.
        let t0 = std::time::Instant::now();
        let cpu_trace = cpu_probe_trace(&j);
        let cpu_s = t0.elapsed().as_secs_f64();
        assert_eq!(cpu_trace.len(), steps);

        eprintln!(
            "bench {nx}x{ny} x {steps} steps: GPU {:.1} MCells/s ({gpu_s:.3} s) | CPU {:.1} MCells/s ({cpu_s:.3} s) | speedup {:.1}x",
            cells / gpu_s / 1e6,
            cells / cpu_s / 1e6,
            cpu_s / gpu_s
        );
    };
    bench(512, 512, 2000);
    bench(1024, 1024, 1000);
}
