//! tei-sim-photonic — SAX-class (Ghent/gdsfactory ecosystem) /
//! neuroptica-class photonic circuit simulator.
//!
//! Frequency-domain **S-matrix composition** for the photonic substrate
//! column: a component library of analytic transfer matrices (waveguide,
//! directional coupler, phase shifter, MZI, ring resonators), the
//! **Redheffer star product** for cascading multiport networks with all
//! internal reflections summed exactly, and **universal MZI meshes** —
//! Clements rectangular construction *and* decomposition — driving an
//! optical-power matrix-vector-multiply mode with event-ledger accounting
//! (modulator events, MACs, detector ADC samples) that feeds the
//! `tei-d-photonic` cost dialect.
//!
//! Lineage & citations:
//! - Clements, Humphreys, Metcalf, Kolthammer, Walmsley, *Optimal design
//!   for universal multiport interferometers*, Optica 3, 1460 (2016).
//! - Reck, Zeilinger, Bernstein, Bertani, *Experimental realization of any
//!   discrete unitary operator*, PRL 73, 58 (1994).
//! - Redheffer, *On a certain linear fractional transformation*,
//!   J. Math. Phys. 39, 269 (1960).
//! - Bogaerts et al., *Silicon microring resonators*, Laser Photon. Rev. 6,
//!   47 (2012) — ring transfer functions.
//! - Mezzadri, *How to generate random matrices from the classical compact
//!   groups*, Notices AMS 54, 592 (2007) — Haar sampling.
//!
//! Validation (tests/analytic.rs) is analytic/published only, per the
//! roadmap's binding policy: component unitarity, closed-form MZI and ring
//! transfer, FSR `λ²/(n_g L)`, critical-coupling extinction, star-product
//! identity/associativity, Clements round-trip to 1e-10, exact MVM ledger
//! counts, and bit-level determinism.

pub mod clements;
pub mod components;
pub mod haar;
pub mod mvm;
pub mod redheffer;

pub use clements::{ClementsMesh, MziSetting};
pub use components::{
    AllPassRing, Waveguide, add_drop_transfer, all_pass_transfer, coupler_50_50,
    directional_coupler, mzi_transfer, phase_shifter, single_port_phase,
};
pub use haar::haar_unitary;
pub use mvm::OpticalMvm;
pub use redheffer::{Sparams, cascade_2port};

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::linalg::{C64, CMat};
use tei_sim_core::rng::Rng;

/// How the target unitary of a [`PhotonicJob`] is specified.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum UnitarySpec {
    /// Haar-random `n×n` unitary drawn deterministically from `seed`.
    RandomHaar { seed: u64 },
    /// Explicit matrix, row-major real and imaginary parts (`n` rows of
    /// `n` entries each). Must be unitary to 1e-8.
    Given {
        re: Vec<Vec<f64>>,
        im: Vec<Vec<f64>>,
    },
}

/// Job spec accepted by the photonic executor (mirrors /api/execute).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct PhotonicJob {
    /// Mesh size N.
    pub n: usize,
    /// Target unitary to embed in the mesh.
    pub unitary: UnitarySpec,
    /// Number of optical MVM queries to run through the programmed mesh.
    #[serde(default = "default_queries")]
    pub n_queries: usize,
    /// Seed for the query input vectors.
    #[serde(default)]
    pub seed: u64,
}

fn default_queries() -> usize {
    16
}

/// Executor for the photonic column: embed the target unitary in a Clements
/// mesh, report reconstruction fidelity, then run optical MVM queries and
/// report RMS detected-power error against the ideal `|U·x|²`.
pub struct PhotonicExecutor;

impl PhotonicExecutor {
    fn build_target(job: &PhotonicJob) -> Result<CMat, String> {
        match &job.unitary {
            UnitarySpec::RandomHaar { seed } => Ok(haar_unitary(job.n, &mut Rng::new(*seed))),
            UnitarySpec::Given { re, im } => {
                if re.len() != job.n || im.len() != job.n {
                    return Err("unitary row count must equal n".into());
                }
                let mut u = CMat::zeros(job.n, job.n);
                for i in 0..job.n {
                    if re[i].len() != job.n || im[i].len() != job.n {
                        return Err("unitary column count must equal n".into());
                    }
                    for j in 0..job.n {
                        u[(i, j)] = C64::new(re[i][j], im[i][j]);
                    }
                }
                Ok(u)
            }
        }
    }
}

impl Executor for PhotonicExecutor {
    type Job = PhotonicJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = std::time::Instant::now();
        let mut ledger = EventLedger::default();

        let fail = |msg: String| ExecutionResult {
            ledger: EventLedger::default(),
            outputs: serde_json::json!({ "error": msg }),
        };

        let target = match Self::build_target(job) {
            Ok(u) => u,
            Err(e) => return fail(e),
        };
        let mvm = match OpticalMvm::from_unitary(&target) {
            Ok(m) => m,
            Err(e) => return fail(e),
        };
        mvm.program(&mut ledger);

        // Reconstruction fidelity: rebuild the mesh unitary and compare.
        let rebuilt = mvm.mesh.unitary();
        let mut max_err = 0.0f64;
        for (a, b) in rebuilt.data.iter().zip(&target.data) {
            max_err = max_err.max((*a - *b).abs());
        }
        let fro_err = rebuilt.frobenius_distance(&target);
        on_progress(Progress {
            fraction: 0.1,
            metrics: serde_json::json!({
                "stage": "decomposed",
                "n_mzis": mvm.mesh.n_mzis(),
                "reconstruction_max_error": max_err,
            }),
        });

        // Optical MVM queries: random unit-norm complex inputs, detected
        // powers vs ideal |U·x|².
        let mut rng = Rng::new(job.seed);
        let n = job.n;
        let mut sq_err_sum = 0.0;
        let report_every = (job.n_queries / 20).max(1);
        for q in 0..job.n_queries {
            let mut x: Vec<C64> = (0..n)
                .map(|_| C64::new(rng.normal(), rng.normal()))
                .collect();
            let norm = x.iter().map(|z| z.norm_sq()).sum::<f64>().sqrt();
            for z in x.iter_mut() {
                *z = *z * (1.0 / norm);
            }
            let detected = mvm.forward(&x, &mut ledger);
            // Ideal: direct U·x then |·|².
            for i in 0..n {
                let mut acc = C64::ZERO;
                for j in 0..n {
                    acc = acc + target[(i, j)] * x[j];
                }
                let d = detected[i] - acc.norm_sq();
                sq_err_sum += d * d;
            }
            if (q + 1) % report_every == 0 || q + 1 == job.n_queries {
                on_progress(Progress {
                    fraction: 0.1 + 0.9 * (q + 1) as f64 / job.n_queries as f64,
                    metrics: serde_json::json!({
                        "stage": "mvm",
                        "queries_done": q + 1,
                        "rms_error": (sq_err_sum / ((q + 1) * n) as f64).sqrt(),
                    }),
                });
            }
        }
        let mvm_rms = if job.n_queries > 0 {
            (sq_err_sum / (job.n_queries * n) as f64).sqrt()
        } else {
            0.0
        };

        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());
        ExecutionResult {
            ledger,
            outputs: serde_json::json!({
                "n": n,
                "n_mzis": mvm.mesh.n_mzis(),
                "n_phases": mvm.mesh.n_phases(),
                "reconstruction_max_error": max_err,
                "reconstruction_frobenius_error": fro_err,
                "mvm_rms_error": mvm_rms,
                "n_queries": job.n_queries,
            }),
        }
    }
}
