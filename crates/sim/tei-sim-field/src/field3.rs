//! F3 job layer — [`Field3Job`] / [`Field3Executor`]: 3D FDTD runs with
//! point-dipole sources, probes, per-probe DFT monitors, dispersive media
//! ([`crate::medium3`]) and a 2D rendering slice.
//!
//! `Field3Job` is a separate serde type from the 2D [`crate::FieldJob`]
//! (the roadmap's schema-stability rule: new capability, new job — F1/F2
//! jobs deserialize unchanged forever).
//!
//! ## Ledger convention
//!
//! `macs` = **6 × nx·ny·nz × steps** — six field-component updates per
//! Yee cell per leapfrog step. This is the 3D analogue of the 2D
//! convention (3 × cells × steps): the staggered component arrays are
//! each marginally smaller than nx·ny·nz, and the CPML ψ recursions plus
//! ADE auxiliary updates are folded into the same nominal count rather
//! than itemized. `adc_samples` = one analog field read per monitor per
//! step (each probe + its DFT monitor). `wall_seconds` is the measured
//! run time.

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::linalg::C64;

use crate::MAX_TRACE_POINTS;
use crate::grid::{self, CpmlParams};
use crate::grid3::{Axis, Comp, Grid3Spec, Grid3d, SliceField, default_courant3};
use crate::medium3::{DispMedia, MaterialRegion};
use crate::source::TimeProfile;

fn one() -> f64 {
    1.0
}

fn default_amplitude() -> f64 {
    1.0
}

/// Permittivity layout for the 3D grid: relative permittivity ε_r per
/// Yee cell, sampled at each E component's staggered position with the
/// integer-index ("inside its cell") convention of [`crate::medium3`].
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum EpsSpec3 {
    /// Uniform ε_r everywhere.
    Uniform {
        #[serde(default = "one")]
        eps_r: f64,
    },
    /// ε_r inside the half-open cell box [i0,i1)×[j0,j1)×[k0,k1),
    /// `background` elsewhere.
    Box {
        eps_r: f64,
        #[serde(default = "one")]
        background: f64,
        i0: usize,
        i1: usize,
        j0: usize,
        j1: usize,
        k0: usize,
        k1: usize,
    },
}

impl EpsSpec3 {
    fn build_comp(&self, di: usize, dj: usize, dk: usize) -> Vec<f64> {
        match *self {
            EpsSpec3::Uniform { eps_r } => vec![eps_r; di * dj * dk],
            EpsSpec3::Box {
                eps_r,
                background,
                i0,
                i1,
                j0,
                j1,
                k0,
                k1,
            } => {
                let mut v = vec![background; di * dj * dk];
                for i in i0..i1.min(di) {
                    for j in j0..j1.min(dj) {
                        for k in k0..k1.min(dk) {
                            v[(i * dj + j) * dk + k] = eps_r;
                        }
                    }
                }
                v
            }
        }
    }

    /// Materialize the per-component ε_r arrays: `[Ex-shaped, Ey-shaped,
    /// Ez-shaped]` as consumed by [`Grid3d::new`].
    pub fn build(&self, nx: usize, ny: usize, nz: usize) -> [Vec<f64>; 3] {
        [
            self.build_comp(nx - 1, ny, nz),
            self.build_comp(nx, ny - 1, nz),
            self.build_comp(nx, ny, nz - 1),
        ]
    }
}

/// A soft point dipole: adds `amplitude·s(t)` onto the E component along
/// `axis` at sample (i, j, k) every step (transparent to scattered waves,
/// like the 2D sources).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Dipole3 {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    /// Polarization axis (which E component is driven). Default z.
    #[serde(default = "default_axis_z")]
    pub axis: Axis,
    pub time: TimeProfile,
    #[serde(default = "default_amplitude")]
    pub amplitude: f64,
}

fn default_axis_z() -> Axis {
    Axis::Z
}

impl Dipole3 {
    /// Inject at time `t` (call after the field update, post-update time).
    pub fn inject(&self, g: &mut Grid3d, t: f64) {
        g.add_e(
            self.axis.e_comp(),
            self.i,
            self.j,
            self.k,
            self.amplitude * self.time.eval(t),
        );
    }
}

fn default_comp_ez() -> Comp {
    Comp::Ez
}

/// A time-series probe location: one E component at one sample.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Probe3Spec {
    pub i: usize,
    pub j: usize,
    pub k: usize,
    #[serde(default = "default_comp_ez")]
    pub component: Comp,
}

/// Materialized probe: records its component every step.
#[derive(Debug, Clone)]
pub struct Probe3 {
    pub spec: Probe3Spec,
    pub trace: Vec<f64>,
}

impl Probe3 {
    pub fn new(spec: Probe3Spec) -> Self {
        Self {
            spec,
            trace: Vec::new(),
        }
    }

    pub fn record(&mut self, g: &Grid3d) {
        self.trace
            .push(g.e_at(self.spec.component, self.spec.i, self.spec.j, self.spec.k));
    }

    pub fn last(&self) -> f64 {
        self.trace.last().copied().unwrap_or(0.0)
    }
}

/// On-the-fly DFT monitor at one E-component sample — the 3D sibling of
/// [`crate::DftMonitor`]: accumulates Σ E(tₙ)·e^{−iωtₙ} per registered ω.
#[derive(Debug, Clone)]
pub struct Dft3Monitor {
    comp: Comp,
    i: usize,
    j: usize,
    k: usize,
    omegas: Vec<f64>,
    accum: Vec<C64>,
    samples: u64,
}

impl Dft3Monitor {
    pub fn new(comp: Comp, i: usize, j: usize, k: usize, omegas: Vec<f64>) -> Self {
        let n = omegas.len();
        Self {
            comp,
            i,
            j,
            k,
            omegas,
            accum: vec![C64::ZERO; n],
            samples: 0,
        }
    }

    /// Accumulate E·e^{−iωt}; `t` is the physical time of the current E.
    pub fn record(&mut self, g: &Grid3d, t: f64) {
        let e = g.e_at(self.comp, self.i, self.j, self.k);
        for (a, &w) in self.accum.iter_mut().zip(&self.omegas) {
            *a = *a + C64::from_polar(e, -w * t);
        }
        self.samples += 1;
    }

    pub fn omegas(&self) -> &[f64] {
        &self.omegas
    }

    /// Raw accumulated sums Σ E·e^{−iωt} (one per ω).
    pub fn accum(&self) -> &[C64] {
        &self.accum
    }

    /// Riemann-sum DFT amplitudes: the raw sums scaled by Δt.
    pub fn spectra(&self, dt: f64) -> Vec<C64> {
        self.accum.iter().map(|&a| a * dt).collect()
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }
}

/// A 2D rendering slice through the final fields.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SnapshotSpec {
    /// Slice normal.
    pub axis: Axis,
    /// Sample index along the normal axis.
    pub index: usize,
    /// What to sample; default raw Ez.
    #[serde(default)]
    pub field: SliceField,
}

/// Job spec accepted by the 3D field executor (F3).
///
/// All lengths are in cells (Δ = 1), times in cell-traversal times
/// (c = 1), angular frequencies in rad per unit time. Δt = courant/√3.
///
/// ```json
/// {
///   "nx": 64, "ny": 64, "nz": 64, "steps": 800,
///   "courant": 0.5,
///   "npml": 10,
///   "cpml": { "m": 3.0, "sigma_scale": 1.0, "kappa_max": 3.0, "alpha_max": 0.05 },
///   "eps": { "kind": "box", "eps_r": 4.0, "background": 1.0,
///            "i0": 40, "i1": 50, "j0": 20, "j1": 44, "k0": 20, "k1": 44 },
///   "materials": [
///     { "model": { "kind": "drude", "omega_p": 0.5, "gamma": 0.01, "eps_inf": 1.0 },
///       "i0": 20, "i1": 30, "j0": 20, "j1": 44, "k0": 20, "k1": 44 }
///   ],
///   "source": { "i": 12, "j": 32, "k": 32, "axis": "z",
///               "time": { "type": "gaussian", "t0": 24.0, "tau": 8.0 },
///               "amplitude": 1.0 },
///   "probes": [ { "i": 55, "j": 32, "k": 32, "component": "ez" } ],
///   "frequencies": [0.25, 0.5],
///   "snapshot": { "axis": "z", "index": 32, "field": "e_mag" }
/// }
/// ```
///
/// Optional with defaults: `courant` (0.5), `npml` (10), `cpml`
/// (standard grading), `materials` (none), `probes` (grid-center Ez),
/// `frequencies` (none), `snapshot` (none).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Field3Job {
    /// Integer grid points along x.
    pub nx: usize,
    /// Integer grid points along y.
    pub ny: usize,
    /// Integer grid points along z.
    pub nz: usize,
    /// Number of leapfrog steps.
    pub steps: usize,
    /// Courant number S (Δt = S·Δ/√3); stable for S < 1.
    #[serde(default = "default_courant3")]
    pub courant: f64,
    /// CPML thickness in cells on each of the six faces (0 = PEC box).
    #[serde(default = "grid::default_npml")]
    pub npml: usize,
    #[serde(default)]
    pub cpml: CpmlParams,
    pub eps: EpsSpec3,
    /// Dispersive (Drude/Lorentz) box regions.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub materials: Vec<MaterialRegion>,
    /// Point dipole source.
    pub source: Dipole3,
    /// Probes; empty ⇒ one Ez probe at the grid center.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub probes: Vec<Probe3Spec>,
    /// Angular frequencies ω for the per-probe DFT monitors.
    #[serde(default)]
    pub frequencies: Vec<f64>,
    /// Optional final-state 2D slice for rendering.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub snapshot: Option<SnapshotSpec>,
}

fn check_sample(
    what: &str,
    comp: Comp,
    (i, j, k): (usize, usize, usize),
    (di, dj, dk): (usize, usize, usize),
) -> Result<(), String> {
    if i >= di || j >= dj || k >= dk {
        return Err(format!(
            "{what}: {comp:?} sample ({i}, {j}, {k}) outside dims ({di}, {dj}, {dk})"
        ));
    }
    Ok(())
}

/// Run a [`Field3Job`] to completion, streaming ~1% progress ticks
/// (`{"step", "t", "ez_probe"}` — the first probe's latest sample).
/// Invalid jobs return `outputs: {"error": …}` with an empty ledger,
/// matching the F1/F2 error convention. See the module docs for the
/// ledger convention.
pub fn run_job3(job: &Field3Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
    match run_job3_inner(job, on_progress) {
        Ok(result) => result,
        Err(msg) => ExecutionResult {
            ledger: EventLedger::default(),
            outputs: serde_json::json!({ "error": msg }),
        },
    }
}

fn run_job3_inner(
    job: &Field3Job,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<ExecutionResult, String> {
    let (nx, ny, nz) = (job.nx, job.ny, job.nz);
    if nx <= 2 * job.npml + 2 || ny <= 2 * job.npml + 2 || nz <= 2 * job.npml + 2 {
        return Err(format!(
            "grid ({nx}, {ny}, {nz}) must exceed 2·npml + 2 = {} per axis",
            2 * job.npml + 2
        ));
    }
    if job.courant <= 0.0 {
        return Err("courant must be positive".into());
    }
    let spec = Grid3Spec {
        nx,
        ny,
        nz,
        courant: job.courant,
        npml: job.npml,
        cpml: job.cpml.clone(),
    };
    let dt = job.courant / 3f64.sqrt();

    let mut eps = job.eps.build(nx, ny, nz);
    let mut media = DispMedia::build(&job.materials, (nx, ny, nz), dt, &mut eps)?;
    let mut grid = Grid3d::new(&spec, eps);

    let src_comp = job.source.axis.e_comp();
    check_sample(
        "source",
        src_comp,
        (job.source.i, job.source.j, job.source.k),
        grid.comp_dims(src_comp),
    )?;
    let probe_specs: Vec<Probe3Spec> = if job.probes.is_empty() {
        vec![Probe3Spec {
            i: nx / 2,
            j: ny / 2,
            k: nz / 2,
            component: Comp::Ez,
        }]
    } else {
        job.probes.clone()
    };
    for p in &probe_specs {
        check_sample(
            "probe",
            p.component,
            (p.i, p.j, p.k),
            grid.comp_dims(p.component),
        )?;
    }
    let mut probes: Vec<Probe3> = probe_specs.iter().cloned().map(Probe3::new).collect();
    let mut dfts: Vec<Dft3Monitor> = probe_specs
        .iter()
        .map(|p| Dft3Monitor::new(p.component, p.i, p.j, p.k, job.frequencies.clone()))
        .collect();

    let dispersive = !media.is_empty();
    let t0 = std::time::Instant::now();
    let tick = (job.steps / 100).max(1);
    for n in 0..job.steps {
        grid.update_h();
        if dispersive {
            media.pre(&grid.ex, &grid.ey, &grid.ez);
        }
        grid.update_e();
        if dispersive {
            media.post(&mut grid.ex, &mut grid.ey, &mut grid.ez);
        }
        let t = (n + 1) as f64 * dt;
        job.source.inject(&mut grid, t);
        for p in &mut probes {
            p.record(&grid);
        }
        for d in &mut dfts {
            d.record(&grid, t);
        }
        if (n + 1) % tick == 0 || n + 1 == job.steps {
            on_progress(Progress {
                fraction: (n + 1) as f64 / job.steps as f64,
                metrics: serde_json::json!({
                    "step": n + 1,
                    "t": t,
                    "ez_probe": probes[0].last(),
                }),
            });
        }
    }

    let mut ledger = EventLedger::default();
    ledger.macs = 6 * (nx * ny * nz) as u64 * job.steps as u64;
    ledger.adc_samples = probes.iter().map(|p| p.trace.len() as u64).sum::<u64>()
        + dfts.iter().map(|d| d.samples()).sum::<u64>();
    ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

    let probes_out: Vec<serde_json::Value> = probes
        .iter()
        .zip(&dfts)
        .map(|(p, d)| {
            let stride = p.trace.len().div_ceil(MAX_TRACE_POINTS).max(1);
            let trace: Vec<f64> = p.trace.iter().step_by(stride).copied().collect();
            let spectra = d.spectra(dt);
            let dft_out: Vec<serde_json::Value> = d
                .omegas()
                .iter()
                .zip(&spectra)
                .map(|(&w, a)| {
                    serde_json::json!({
                        "omega": w, "re": a.re, "im": a.im, "abs": a.abs(),
                    })
                })
                .collect();
            serde_json::json!({
                "i": p.spec.i, "j": p.spec.j, "k": p.spec.k,
                "component": p.spec.component,
                "stride": stride,
                "trace": trace,
                "dft": dft_out,
            })
        })
        .collect();
    let snapshot = job.snapshot.as_ref().map(|s| {
        let (n0, n1, data) = grid.slice(s.axis, s.index, s.field);
        serde_json::json!({
            "axis": s.axis, "index": s.index, "field": s.field,
            "n0": n0, "n1": n1, "data": data,
        })
    });

    Ok(ExecutionResult {
        ledger,
        outputs: serde_json::json!({
            "dt": dt,
            "steps": job.steps,
            "probes": probes_out,
            "snapshot": snapshot,
        }),
    })
}

/// Executor for 3D field jobs (F3). Registered by the serving layer
/// alongside [`crate::FieldExecutor`]; this crate only exports it.
pub struct Field3Executor;

impl Executor for Field3Executor {
    type Job = Field3Job;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        run_job3(job, on_progress)
    }
}
