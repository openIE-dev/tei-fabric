//! S-parameter extraction (F2) — the device→circuit closed loop of
//! docs/SIM-ROADMAP.md §3.7/§2: FDTD port amplitudes → complex S-matrix
//! column → a `tei_sim_photonic::Sparams` component the circuit layer can
//! star-compose.
//!
//! ## Method: reference-run subtraction
//!
//! The job is executed **twice** with identical source, grid, timestep and
//! step count:
//!
//! 1. **Reference run** — the device removed: `reference_eps` if given,
//!    otherwise [`crate::EpsSpec::reference`] (the uniform waveguide with
//!    the device section stripped). The mode amplitude recorded at the
//!    port-1 plane is the **incident** wave a₁(ω) — the launch is a
//!    bidirectional soft mode line, but only the forward lobe ever crosses
//!    port 1 (the backward lobe exits into the left CPML), so no
//!    directional launch is needed.
//! 2. **Device run** — the full `eps`. At port 1 the total field is
//!    incident + reflected; subtracting the reference run's amplitude
//!    isolates the backward wave, b₁(ω) = A₁ᵈᵉᵛ(ω) − a₁(ω). At every
//!    downstream port k ≥ 2 the field is purely outgoing, b_k(ω) = A_kᵈᵉᵛ(ω).
//!
//! With the source on port 1:
//!
//! ```text
//!   S₁₁(ω) = (A₁ᵈᵉᵛ − a₁)/a₁        S_k1(ω) = A_kᵈᵉᵛ/a₁   (k ≥ 2)
//! ```
//!
//! Port planes are the S-parameter reference planes: a straight guide of
//! length L between ports gives S₂₁ = e^{iβL} (magnitude *and* propagation
//! phase), which is how the validation suite recovers n_eff from FDTD.
//! Because the two runs share everything upstream of the device, the
//! subtraction is exact (bit-identical incident field); when the device
//! *is* the reference (a bare straight guide) S₁₁ ≡ 0 by construction —
//! reflection extraction is validated non-trivially by the etalon test.
//!
//! **Geometry contract** (checked at run time): `eps.kind = waveguide_x`,
//! the mode source column left of `ports[0].i`, ports ordered
//! source → port 1 → device → ports 2…; the reference must share the
//! waveguide cross-section (guaranteed when `reference_eps` is omitted).
//!
//! ## Output shape (`outputs.sparams`)
//!
//! ```json
//! {
//!   "frequencies": [0.22, 0.25, 0.28],
//!   "source_port": 1,
//!   "separation": "reference-run subtraction (bidirectional soft mode line source)",
//!   "entries": [
//!     {"from": 1, "to": 1, "omega": 0.22, "re": …, "im": …, "abs": …, "phase": …},
//!     {"from": 1, "to": 2, "omega": 0.22, "re": …, "im": …, "abs": …, "phase": …},
//!     …
//!   ]
//! }
//! ```
//!
//! `entries` is ordered ports-major, frequencies-minor; `to` is the 1-based
//! port index; `phase = atan2(im, re)` in (−π, π].
//!
//! ## Photonic handoff
//!
//! [`ExtractedSparams::two_port`] lifts a two-port extraction into the
//! photonic crate's [`tei_sim_photonic::Sparams`] (left = port 1, right =
//! port 2) at one frequency index, assuming a **reciprocal, mirror-symmetric
//! device** (S₁₂ = S₂₁, S₂₂ = S₁₁ — true for everything F2 can build:
//! straight guides and symmetric etalon sections). The dependency points
//! field → photonic; the photonic crate stays independent of FDTD.

use crate::mode::SlabMode;
use crate::port::PortMonitor;
use crate::{FieldJob, ModeInjector, simulate};
use tei_sim_core::exec::{ExecutionResult, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::linalg::{C64, CMat};

/// An extracted S-matrix column: S_k1(ω) for every port k and frequency ω.
#[derive(Debug, Clone)]
pub struct ExtractedSparams {
    /// Angular frequencies ω (normalized units), as requested by the job.
    pub frequencies: Vec<f64>,
    /// `s_col[k][f]` = S_{(k+1),1}(ω_f) — k is the 0-based port index.
    pub s_col: Vec<Vec<C64>>,
}

impl ExtractedSparams {
    /// The `outputs.sparams` JSON documented in the module header.
    pub fn to_json(&self) -> serde_json::Value {
        let entries: Vec<serde_json::Value> = self
            .s_col
            .iter()
            .enumerate()
            .flat_map(|(k, col)| {
                self.frequencies.iter().zip(col).map(move |(&w, s)| {
                    serde_json::json!({
                        "from": 1,
                        "to": k + 1,
                        "omega": w,
                        "re": s.re,
                        "im": s.im,
                        "abs": s.abs(),
                        "phase": s.im.atan2(s.re),
                    })
                })
            })
            .collect();
        serde_json::json!({
            "frequencies": self.frequencies,
            "source_port": 1,
            "separation":
                "reference-run subtraction (bidirectional soft mode line source)",
            "entries": entries,
        })
    }

    /// Lift a two-port extraction into the photonic circuit layer at
    /// frequency index `fi`: `Sparams` with left = port 1, right = port 2,
    /// under the reciprocal mirror-symmetric device assumption
    /// (S₁₂ = S₂₁, S₂₂ = S₁₁). Panics unless exactly two ports were
    /// extracted and `fi` is in range.
    pub fn two_port(&self, fi: usize) -> tei_sim_photonic::Sparams {
        assert_eq!(self.s_col.len(), 2, "two_port needs exactly 2 ports");
        let (s11, s21) = (self.s_col[0][fi], self.s_col[1][fi]);
        let mut full = CMat::zeros(2, 2);
        full[(0, 0)] = s11;
        full[(0, 1)] = s21; // reciprocity
        full[(1, 0)] = s21;
        full[(1, 1)] = s11; // mirror symmetry
        tei_sim_photonic::Sparams::from_full(&full, 1)
    }
}

/// Execute the two-run extraction described in the module docs. Called by
/// [`crate::run_job`] whenever the job declares ports.
pub(crate) fn run_sparams_job(
    job: &FieldJob,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<ExecutionResult, String> {
    let (extracted, result) = extract(job, on_progress)?;
    let _ = extracted;
    Ok(result)
}

/// The two-run extraction, returning both the typed [`ExtractedSparams`]
/// (for in-process handoff to `tei-sim-photonic`) and the executor-shaped
/// [`ExecutionResult`] whose `outputs.sparams` carries the same data as
/// JSON.
pub fn extract(
    job: &FieldJob,
    on_progress: &mut dyn FnMut(Progress),
) -> Result<(ExtractedSparams, ExecutionResult), String> {
    if job.frequencies.is_empty() {
        return Err("ports require at least one entry in `frequencies`".into());
    }
    let Some(ms) = &job.mode_source else {
        return Err("ports require a `mode_source` (the launch on port 1)".into());
    };
    let (wg, y_center) = job
        .eps
        .slab_waveguide()
        .ok_or("ports require eps.kind = \"waveguide_x\"")?;
    if job.ports.is_empty() {
        return Err("empty ports".into());
    }
    if ms.i >= job.ports[0].i {
        return Err(format!(
            "mode_source column ({}) must lie left of port 1 ({}) — the \
             extraction convention is source → port 1 → device → ports 2…",
            ms.i, job.ports[0].i
        ));
    }

    // Per-port, per-frequency mode profiles (frequency-dependent n_eff/γ).
    let mut monitors = Vec::with_capacity(job.ports.len());
    for (k, p) in job.ports.iter().enumerate() {
        if p.i + 1 >= job.nx {
            return Err(format!("port {} column {} outside the grid", k + 1, p.i));
        }
        let mut profiles = Vec::with_capacity(job.frequencies.len());
        for &w in &job.frequencies {
            let mode: SlabMode = wg.solve(w, p.mode).ok_or(format!(
                "port {}: TE{} is not guided at omega = {w}",
                k + 1,
                p.mode
            ))?;
            profiles.push(mode.sample(job.ny, y_center));
        }
        monitors.push(PortMonitor::new(p.i, job.frequencies.clone(), profiles));
    }
    let injector = ModeInjector::build(ms, &wg, y_center, job.ny)?;

    let t0 = std::time::Instant::now();
    let reference_eps = job
        .reference_eps
        .clone()
        .unwrap_or_else(|| job.eps.reference());
    let reference = simulate(
        job,
        reference_eps.build(job.nx, job.ny),
        Some(&injector),
        monitors.clone(),
        (0.0, 0.5),
        on_progress,
    );
    let device = simulate(
        job,
        job.eps.build(job.nx, job.ny),
        Some(&injector),
        monitors,
        (0.5, 1.0),
        on_progress,
    );

    let dt = device.dt;
    let a1 = reference.ports[0].amplitudes(dt);
    for (fi, a) in a1.iter().enumerate() {
        if a.abs() < 1e-30 {
            return Err(format!(
                "incident amplitude vanished at omega = {} — widen the source \
                 band or lengthen the run",
                job.frequencies[fi]
            ));
        }
    }
    let s_col: Vec<Vec<C64>> = device
        .ports
        .iter()
        .enumerate()
        .map(|(k, port)| {
            let dev = port.amplitudes(dt);
            if k == 0 {
                // Source port: subtract the incident wave, keep the reflection.
                dev.iter()
                    .zip(&reference.ports[0].amplitudes(dt))
                    .zip(&a1)
                    .map(|((&d, &r), &a)| (d - r) / a)
                    .collect()
            } else {
                dev.iter().zip(&a1).map(|(&d, &a)| d / a).collect()
            }
        })
        .collect();
    let extracted = ExtractedSparams {
        frequencies: job.frequencies.clone(),
        s_col,
    };

    let mut ledger = EventLedger::default();
    // Two full runs of the three component sweeps.
    ledger.macs = 2 * 3 * (job.nx * job.ny) as u64 * job.steps as u64;
    // One analog read per monitor per step, both runs.
    ledger.adc_samples = (reference.probe.trace.len() + device.probe.trace.len()) as u64
        + reference.dft.samples()
        + device.dft.samples()
        + reference.ports.iter().map(|p| p.samples()).sum::<u64>()
        + device.ports.iter().map(|p| p.samples()).sum::<u64>();
    ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

    let result = ExecutionResult {
        ledger,
        outputs: crate::device_outputs(job, &device, Some(extracted.to_json())),
    };
    Ok((extracted, result))
}
