//! tei-sim-field — MEEP-class (MIT) Yee-grid FDTD, stages F1 + F2.
//!
//! Finite-difference time-domain solver for the field substrate column:
//! a 2D Yee grid evolving the out-of-plane-E component set (Ez, Hx, Hy)
//! with CPML absorbing boundaries, per-cell dielectric media, soft
//! point/line sources (Gaussian-pulse, modulated-Gaussian and CW time
//! signatures), time-series probes and on-the-fly DFT monitors (F1) —
//! plus, at F2, slab-waveguide **mode sources**, **port monitors** and
//! **S-parameter extraction** feeding `tei-sim-photonic` component models:
//! the device→circuit closed loop of docs/SIM-ROADMAP.md §2.
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
//! **Validation** (tests/, roadmap §3.7 — analytic ground truth only):
//! numerical dispersion vs the closed-form Yee relation, CPML reflection
//! floor, CFL stability boundary, dielectric-slab group delay, post-source
//! energy decay, determinism (F1); slab-waveguide effective index against
//! the analytic transcendental solution and closed-form anchors,
//! straight-guide |S₂₁| and FDTD-propagated n_eff, two-port passivity, the
//! Fabry-Pérot etalon vs the exact Airy formula, and the star-composed
//! photonic handoff (F2).
//!
//! Deliberately out at F2 (the contract of roadmap §3.7): dispersive and
//! conductive media, 3D, far-field transforms, multi-column excitation
//! (S-columns beyond the source port come from re-running with the source
//! on another port). Those are F3+.
//!
//! Lineage: MEEP-class — A.F. Oskooi et al., "MEEP: A flexible free-software
//! package for electromagnetic simulations by the FDTD method", Comput.
//! Phys. Commun. 181, 687 (2010). Core references: K.S. Yee, IEEE Trans.
//! Antennas Propag. 14, 302 (1966); A. Taflove & S.C. Hagness,
//! *Computational Electrodynamics: The Finite-Difference Time-Domain
//! Method*, 3rd ed., Artech House (2005); J.A. Roden & S.D. Gedney,
//! "Convolutional PML (CPML)", Microw. Opt. Technol. Lett. 27, 334 (2000);
//! D. Marcuse, *Theory of Dielectric Optical Waveguides*, Academic Press
//! (1991) — slab mode relations.

pub mod grid;
pub mod mode;
pub mod monitor;
pub mod port;
pub mod source;
pub mod sparams;

pub use grid::{CpmlParams, Grid2d, GridSpec, yee_axis_wavenumber};
pub use mode::{SlabMode, SlabWaveguide};
pub use monitor::{DftMonitor, Probe};
pub use port::{PortMonitor, PortSpec};
pub use source::{Source, SourceShape, TimeProfile};
pub use sparams::ExtractedSparams;

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;

/// Maximum number of points in the downsampled probe trace returned by the
/// executor (longer traces are strided down to fit).
pub const MAX_TRACE_POINTS: usize = 1024;

fn one() -> f64 {
    1.0
}

/// A device section inside a [`EpsSpec::WaveguideX`]: ε_r += `delta` for
/// **every row** of columns `i0 ≤ i < i1`. The uniform transverse shift is
/// deliberate physics, not a simplification: with ε(x, y) = ε_wg(y) +
/// Δ·χ_[i0,i1)(x) the transverse mode profile is *identical* in both
/// sections (the shift cancels out of the transverse eigenproblem) and the
/// propagation constant shifts to β₂ = √(β₁² + Δ·k₀²) — so the junction is
/// exactly profile-matched and the device is an *exact* 1D Fabry-Pérot
/// etalon with Fresnel coefficient r = (β₁−β₂)/(β₁+β₂), giving the F2
/// validation suite a closed-form Airy ground truth.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeviceSection {
    /// Permittivity shift Δ added in the section (may be negative as long
    /// as ε stays positive).
    pub delta: f64,
    /// First column of the section.
    pub i0: usize,
    /// One past the last column of the section.
    pub i1: usize,
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
    /// A horizontal slab waveguide guiding along x: core ε_r in rows
    /// j0 ≤ j < j1 (all columns), `background` cladding elsewhere, plus an
    /// optional [`DeviceSection`]. This is the geometry mode sources and
    /// ports operate on (F2).
    WaveguideX {
        eps_r: f64,
        j0: usize,
        j1: usize,
        #[serde(default = "one")]
        background: f64,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        device: Option<DeviceSection>,
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
            EpsSpec::WaveguideX {
                eps_r,
                j0,
                j1,
                background,
                ref device,
            } => {
                let mut v = vec![background; nx * ny];
                for i in 0..nx {
                    for j in j0..j1.min(ny) {
                        v[i * ny + j] = eps_r;
                    }
                }
                if let Some(d) = device {
                    for i in d.i0..d.i1.min(nx) {
                        for j in 0..ny {
                            v[i * ny + j] += d.delta;
                        }
                    }
                }
                v
            }
        }
    }

    /// The reference (device-removed) layout used for S-parameter
    /// extraction: a `WaveguideX` with its [`DeviceSection`] stripped;
    /// every other variant is its own reference.
    pub fn reference(&self) -> EpsSpec {
        match self {
            EpsSpec::WaveguideX {
                eps_r,
                j0,
                j1,
                background,
                ..
            } => EpsSpec::WaveguideX {
                eps_r: *eps_r,
                j0: *j0,
                j1: *j1,
                background: *background,
                device: None,
            },
            other => other.clone(),
        }
    }

    /// The transverse slab-waveguide problem of a `WaveguideX` layout plus
    /// its mode-profile center line: core rows j0..j1 hold Ez points, so
    /// the continuum core spans [j0 − ½, j1 − ½] — half-width
    /// a = (j1 − j0)/2 about y_center = (j0 + j1 − 1)/2. `None` for
    /// non-waveguide layouts.
    pub fn slab_waveguide(&self) -> Option<(SlabWaveguide, f64)> {
        match *self {
            EpsSpec::WaveguideX {
                eps_r,
                j0,
                j1,
                background,
                ..
            } if j1 > j0 && eps_r > background => Some((
                SlabWaveguide {
                    eps_core: eps_r,
                    eps_clad: background,
                    half_width: (j1 - j0) as f64 / 2.0,
                },
                (j0 + j1) as f64 / 2.0 - 0.5,
            )),
            _ => None,
        }
    }
}

fn default_amplitude() -> f64 {
    1.0
}

/// A guided-mode line source (F2): the spatial profile is the solved TE_m
/// slab mode of the job's `waveguide_x` layout at `omega`, injected softly
/// (additively) on column `i` with the given time signature. The launch is
/// **bidirectional** — both ±x lobes are excited with equal amplitude; the
/// backward lobe exits into the left CPML and S-parameter extraction
/// normalizes against the forward lobe measured at port 1, so no
/// unidirectional (TF/SF) machinery is required at F2.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModeSourceSpec {
    /// Grid column of the source line.
    pub i: usize,
    /// TE mode order m to launch (0 = fundamental).
    #[serde(default)]
    pub mode: usize,
    /// Frequency at which the spatial profile is solved (use the band
    /// center of the time signature).
    pub omega: f64,
    /// Time signature (typically `modulated_gaussian` centred on `omega`).
    pub time: TimeProfile,
    #[serde(default = "default_amplitude")]
    pub amplitude: f64,
}

/// A materialized mode source: sampled profile × time signature.
#[derive(Debug, Clone)]
pub struct ModeInjector {
    i: usize,
    profile: Vec<f64>,
    time: TimeProfile,
    amplitude: f64,
}

impl ModeInjector {
    /// Solve the requested mode and sample its profile on the grid rows.
    pub(crate) fn build(
        spec: &ModeSourceSpec,
        wg: &SlabWaveguide,
        y_center: f64,
        ny: usize,
    ) -> Result<Self, String> {
        let mode = wg.solve(spec.omega, spec.mode).ok_or(format!(
            "mode_source: TE{} is not guided at omega = {}",
            spec.mode, spec.omega
        ))?;
        Ok(Self {
            i: spec.i,
            profile: mode.sample(ny, y_center),
            time: spec.time.clone(),
            amplitude: spec.amplitude,
        })
    }

    /// Add `amplitude·s(t)·E_mode(j)` onto column `i` (soft source).
    fn inject(&self, g: &mut Grid2d, t: f64) {
        let s = self.amplitude * self.time.eval(t);
        for j in 1..g.ny - 1 {
            let p = self.profile[j];
            if p != 0.0 {
                g.add_ez(self.i, j, s * p);
            }
        }
    }
}

/// Job spec accepted by the field executor (mirrors /api/execute).
///
/// F2 additions (all optional — F1 jobs deserialize unchanged):
/// `mode_source` (guided-mode launch), `ports` (mode-overlap port
/// monitors; non-empty triggers the two-run S-parameter extraction of
/// [`sparams`]), and `reference_eps` (explicit device-removed layout for
/// the reference run; defaults to [`EpsSpec::reference`]). `source` is now
/// optional so a job may be driven by the mode source alone.
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
    /// Point/line source (F1). Optional since F2; at least one of `source`
    /// and `mode_source` must be present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// Guided-mode line source (F2).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mode_source: Option<ModeSourceSpec>,
    /// Port monitors (F2). Non-empty ⇒ S-parameter extraction: requires
    /// `mode_source`, `frequencies`, and a `waveguide_x` layout.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub ports: Vec<PortSpec>,
    /// Device-removed layout for the extraction reference run; defaults to
    /// `eps.reference()`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reference_eps: Option<EpsSpec>,
    /// Angular frequencies ω for the DFT monitor at the probe cell and for
    /// the port monitors / S-parameters.
    #[serde(default)]
    pub frequencies: Vec<f64>,
    /// Probe cell [i, j]; defaults to the grid center.
    #[serde(default)]
    pub probe: Option<[usize; 2]>,
    /// Include a final flat Ez snapshot in the outputs (for rendering).
    #[serde(default)]
    pub snapshot: bool,
}

/// One completed FDTD run: final grid plus every monitor.
pub(crate) struct SimRun {
    pub dt: f64,
    pub grid: Grid2d,
    pub probe: Probe,
    pub dft: DftMonitor,
    pub ports: Vec<PortMonitor>,
}

/// The single shared leapfrog loop: step, inject (soft sources), record.
/// Progress is mapped linearly onto `(frac0, frac1)` so multi-run jobs
/// (S-parameter extraction) report a single 0→1 ramp.
pub(crate) fn simulate(
    job: &FieldJob,
    eps: Vec<f64>,
    injector: Option<&ModeInjector>,
    mut ports: Vec<PortMonitor>,
    (frac0, frac1): (f64, f64),
    on_progress: &mut dyn FnMut(Progress),
) -> SimRun {
    let spec = GridSpec {
        nx: job.nx,
        ny: job.ny,
        courant: job.courant,
        npml: job.npml,
        cpml: job.cpml.clone(),
    };
    let mut grid = Grid2d::new(&spec, eps);
    let dt = grid.dt;

    let (pi, pj) = job
        .probe
        .map(|p| (p[0], p[1]))
        .unwrap_or((job.nx / 2, job.ny / 2));
    let mut probe = Probe::new(pi, pj);
    let mut dft = DftMonitor::new(pi, pj, job.frequencies.clone());

    let tick = (job.steps / 100).max(1);
    for n in 0..job.steps {
        grid.step();
        let t = (n + 1) as f64 * dt;
        if let Some(src) = &job.source {
            src.inject(&mut grid, t);
        }
        if let Some(inj) = injector {
            inj.inject(&mut grid, t);
        }
        probe.record(&grid);
        dft.record(&grid, t);
        for p in &mut ports {
            p.record(&grid, t);
        }
        if (n + 1) % tick == 0 || n + 1 == job.steps {
            on_progress(Progress {
                fraction: frac0 + (frac1 - frac0) * (n + 1) as f64 / job.steps as f64,
                metrics: serde_json::json!({
                    "step": n + 1,
                    "t": t,
                    "ez_probe": probe.last(),
                }),
            });
        }
    }
    SimRun {
        dt,
        grid,
        probe,
        dft,
        ports,
    }
}

/// Assemble the executor outputs from a (device-)run: probe trace, DFT
/// spectra, optional snapshot, optional `sparams` block.
pub(crate) fn device_outputs(
    job: &FieldJob,
    run: &SimRun,
    sparams: Option<serde_json::Value>,
) -> serde_json::Value {
    let stride = run.probe.trace.len().div_ceil(MAX_TRACE_POINTS).max(1);
    let trace: Vec<f64> = run.probe.trace.iter().step_by(stride).copied().collect();
    let spectra = run.dft.spectra(run.dt);
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
            "ez": run.grid.snapshot_ez(),
        })
    });
    let mut outputs = serde_json::json!({
        "dt": run.dt,
        "steps": job.steps,
        "probe": { "i": run.probe.i, "j": run.probe.j, "stride": stride, "trace": trace },
        "dft": dft_out,
        "snapshot": snapshot,
    });
    if let Some(sp) = sparams {
        outputs["sparams"] = sp;
    }
    outputs
}

/// Run a [`FieldJob`] to completion, streaming ~1% progress ticks.
///
/// Jobs with `ports` run the two-pass S-parameter extraction of
/// [`sparams`]; plain jobs run once. Invalid jobs (missing sources, mode
/// not guided, …) return `outputs: {"error": …}` with an empty ledger,
/// matching the photonic executor's error convention.
///
/// Ledger accounting: `macs` ≈ 3 component updates × cells × steps × runs
/// (the Hx, Hy, Ez sweeps each touch every cell once per step; CPML strip
/// work is folded into the same estimate); `adc_samples` = one analog
/// field read per monitor per step (probe + DFT monitor + each port),
/// summed over runs.
pub fn run_job(job: &FieldJob, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
    match run_job_inner(job, on_progress) {
        Ok(result) => result,
        Err(msg) => ExecutionResult {
            ledger: EventLedger::default(),
            outputs: serde_json::json!({ "error": msg }),
        },
    }
}

fn run_job_inner(
    job: &FieldJob,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<ExecutionResult, String> {
    if job.source.is_none() && job.mode_source.is_none() {
        return Err("job needs a `source` or a `mode_source`".into());
    }
    if !job.ports.is_empty() {
        return sparams::run_sparams_job(job, on_progress);
    }

    let injector = match &job.mode_source {
        Some(ms) => {
            let (wg, y_center) = job
                .eps
                .slab_waveguide()
                .ok_or("mode_source requires eps.kind = \"waveguide_x\"")?;
            Some(ModeInjector::build(ms, &wg, y_center, job.ny)?)
        }
        None => None,
    };

    let t0 = std::time::Instant::now();
    let run = simulate(
        job,
        job.eps.build(job.nx, job.ny),
        injector.as_ref(),
        Vec::new(),
        (0.0, 1.0),
        on_progress,
    );

    let mut ledger = EventLedger::default();
    ledger.macs = 3 * (job.nx * job.ny) as u64 * job.steps as u64;
    ledger.adc_samples = run.probe.trace.len() as u64 + run.dft.samples();
    ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

    Ok(ExecutionResult {
        ledger,
        outputs: device_outputs(job, &run, None),
    })
}

/// Executor for the field column. With F2 it closes the device→circuit
/// loop: jobs with ports emit `outputs.sparams`, and
/// [`ExtractedSparams::two_port`] lifts the extraction into
/// `tei-sim-photonic` components.
pub struct FieldExecutor;

impl Executor for FieldExecutor {
    type Job = FieldJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        run_job(job, on_progress)
    }
}
