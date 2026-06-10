//! tei-sim-field — MEEP-class (MIT) Yee-grid FDTD, stage F1.
//!
//! Finite-difference time-domain solver for the field substrate column:
//! a 2D Yee grid evolving the out-of-plane-E component set (Ez, Hx, Hy)
//! with CPML absorbing boundaries, per-cell dielectric media, soft
//! point/line sources (Gaussian-pulse and CW time signatures), time-series
//! probes and on-the-fly DFT monitors.
//!
//! **Polarization naming.** We evolve (Ez, Hx, Hy) — electric field normal
//! to the simulation plane. Taflove & Hagness call this mode **TMz**; the
//! integrated-photonics / slab-waveguide convention calls it **TE**
//! (E parallel to the slab interfaces). This crate uses the photonics name
//! "TEz", always accompanied by the explicit component list.
//!
//! **Normalized units**: c = ε₀ = μ₀ = 1 and Δx = Δy = Δ = 1, so lengths
//! are in cells, times in cell-traversal times, and Δt = S·Δ/√2 with
//! Courant number S < 1 (the 2D CFL bound). Angular frequencies ω are in
//! rad per unit time; free-space wavelength λ = 2π/ω cells.
//!
//! **Validation** (tests/analytic.rs, roadmap §3.7 — analytic ground truth
//! only): numerical dispersion vs the closed-form Yee relation, CPML
//! reflection floor, CFL stability boundary, dielectric-slab group delay,
//! post-source energy decay, determinism.
//!
//! Deliberately out at F1 (the contract of roadmap §3.7): dispersive and
//! conductive media, 3D, waveguide mode sources, port monitors /
//! S-parameter extraction, far-field transforms. Those are F2/F3.
//!
//! Lineage: MEEP-class — A.F. Oskooi et al., "MEEP: A flexible free-software
//! package for electromagnetic simulations by the FDTD method", Comput.
//! Phys. Commun. 181, 687 (2010). Core references: K.S. Yee, IEEE Trans.
//! Antennas Propag. 14, 302 (1966); A. Taflove & S.C. Hagness,
//! *Computational Electrodynamics: The Finite-Difference Time-Domain
//! Method*, 3rd ed., Artech House (2005); J.A. Roden & S.D. Gedney,
//! "Convolutional PML (CPML)", Microw. Opt. Technol. Lett. 27, 334 (2000).

pub mod grid;
pub mod monitor;
pub mod source;

pub use grid::{CpmlParams, Grid2d, GridSpec, yee_axis_wavenumber};
pub use monitor::{DftMonitor, Probe};
pub use source::{Source, SourceShape, TimeProfile};

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;

/// Maximum number of points in the downsampled probe trace returned by the
/// executor (longer traces are strided down to fit).
pub const MAX_TRACE_POINTS: usize = 1024;

fn one() -> f64 {
    1.0
}

/// Permittivity layout: per-cell relative permittivity ε_r over the grid.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EpsSpec {
    /// Uniform ε_r everywhere.
    Uniform {
        #[serde(default = "one")]
        eps_r: f64,
    },
    /// A vertical dielectric slab: ε_r in columns i0 ≤ i < i1 (all rows),
    /// `background` elsewhere.
    Slab {
        eps_r: f64,
        i0: usize,
        i1: usize,
        #[serde(default = "one")]
        background: f64,
    },
}

impl EpsSpec {
    /// Materialize the per-cell ε_r vector (row-major, `eps[i*ny + j]`).
    pub fn build(&self, nx: usize, ny: usize) -> Vec<f64> {
        match *self {
            EpsSpec::Uniform { eps_r } => vec![eps_r; nx * ny],
            EpsSpec::Slab {
                eps_r,
                i0,
                i1,
                background,
            } => {
                let mut v = vec![background; nx * ny];
                for i in i0..i1.min(nx) {
                    v[i * ny..(i + 1) * ny].fill(eps_r);
                }
                v
            }
        }
    }
}

/// Job spec accepted by the field executor (mirrors /api/execute).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldJob {
    /// Ez points along x.
    pub nx: usize,
    /// Ez points along y.
    pub ny: usize,
    /// Number of leapfrog steps.
    pub steps: usize,
    /// Courant number S (Δt = S·Δ/√2); stable for S < 1.
    #[serde(default = "grid::default_courant")]
    pub courant: f64,
    /// CPML thickness in cells on each side.
    #[serde(default = "grid::default_npml")]
    pub npml: usize,
    #[serde(default)]
    pub cpml: CpmlParams,
    pub eps: EpsSpec,
    pub source: Source,
    /// Angular frequencies ω for the DFT monitor at the probe cell.
    #[serde(default)]
    pub frequencies: Vec<f64>,
    /// Probe cell [i, j]; defaults to the grid center.
    #[serde(default)]
    pub probe: Option<[usize; 2]>,
    /// Include a final flat Ez snapshot in the outputs (for rendering).
    #[serde(default)]
    pub snapshot: bool,
}

/// Run a [`FieldJob`] to completion, streaming ~1% progress ticks.
///
/// Ledger accounting: `macs` ≈ 3 component updates × cells × steps (the Hx,
/// Hy, Ez sweeps each touch every cell once per step; CPML strip work is
/// folded into the same estimate); `adc_samples` = one analog field read
/// per monitor per step (probe + DFT monitor).
pub fn run_job(job: &FieldJob, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
    let spec = GridSpec {
        nx: job.nx,
        ny: job.ny,
        courant: job.courant,
        npml: job.npml,
        cpml: job.cpml.clone(),
    };
    let mut grid = Grid2d::new(&spec, job.eps.build(job.nx, job.ny));
    let dt = grid.dt;

    let (pi, pj) = job
        .probe
        .map(|p| (p[0], p[1]))
        .unwrap_or((job.nx / 2, job.ny / 2));
    let mut probe = Probe::new(pi, pj);
    let mut dft = DftMonitor::new(pi, pj, job.frequencies.clone());

    let t0 = std::time::Instant::now();
    let tick = (job.steps / 100).max(1);
    for n in 0..job.steps {
        grid.step();
        let t = (n + 1) as f64 * dt;
        job.source.inject(&mut grid, t);
        probe.record(&grid);
        dft.record(&grid, t);
        if (n + 1) % tick == 0 || n + 1 == job.steps {
            on_progress(Progress {
                fraction: (n + 1) as f64 / job.steps as f64,
                metrics: serde_json::json!({
                    "step": n + 1,
                    "t": t,
                    "ez_probe": probe.last(),
                }),
            });
        }
    }

    let mut ledger = EventLedger::default();
    ledger.macs = 3 * (job.nx * job.ny) as u64 * job.steps as u64;
    ledger.adc_samples = probe.trace.len() as u64 + dft.samples();
    ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

    let stride = probe.trace.len().div_ceil(MAX_TRACE_POINTS).max(1);
    let trace: Vec<f64> = probe.trace.iter().step_by(stride).copied().collect();
    let spectra = dft.spectra(dt);
    let dft_out: Vec<serde_json::Value> = job
        .frequencies
        .iter()
        .zip(&spectra)
        .map(|(&w, a)| {
            serde_json::json!({
                "omega": w,
                "re": a.re,
                "im": a.im,
                "abs": a.abs(),
            })
        })
        .collect();
    let snapshot = job.snapshot.then(|| {
        serde_json::json!({
            "nx": job.nx,
            "ny": job.ny,
            "ez": grid.snapshot_ez(),
        })
    });

    ExecutionResult {
        ledger,
        outputs: serde_json::json!({
            "dt": dt,
            "steps": job.steps,
            "probe": { "i": pi, "j": pj, "stride": stride, "trace": trace },
            "dft": dft_out,
            "snapshot": snapshot,
        }),
    }
}

/// Executor for the field column (registered alongside `tei-d-photonic`'s
/// cost dialect once F2 lands the device→circuit handoff).
pub struct FieldExecutor;

impl Executor for FieldExecutor {
    type Job = FieldJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        run_job(job, on_progress)
    }
}
