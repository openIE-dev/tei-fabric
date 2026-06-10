//! Optical-power matrix-vector multiplication on a Clements mesh.
//!
//! The photonic MVM pipeline: program the mesh phases to embed the target
//! matrix (one **modulator event per programmed phase**), modulate the input
//! amplitudes onto the modes, let light propagate (the mesh performs the
//! full `N×N` complex MVM in flight — ledgered as `N²` MACs), and
//! photodetect `|·|²` at the `N` outputs (`N` ADC samples per readout).
//!
//! ## Target matrices
//!
//! The mesh realizes any **unitary** `U ∈ U(N)` exactly — which includes
//! every real orthogonal matrix (suitably scaled real weights with
//! `WᵀW = I`). An arbitrary real matrix requires the standard SVD
//! factorization `W = U·Σ·V†` — two meshes plus a diagonal attenuator
//! column — which is out of scope for this milestone and documented as
//! future work (it composes directly from this module: two
//! [`OpticalMvm`]s and an amplitude screen).

use crate::clements::ClementsMesh;
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::linalg::{C64, CMat};

/// A mesh programmed as an optical matrix-vector multiplier.
#[derive(Debug, Clone)]
pub struct OpticalMvm {
    pub mesh: ClementsMesh,
}

impl OpticalMvm {
    /// Embed a target unitary via Clements decomposition.
    pub fn from_unitary(u: &CMat) -> Result<Self, String> {
        Ok(Self {
            mesh: ClementsMesh::decompose(u)?,
        })
    }

    /// Ledger the weight-load: one modulator event per programmed phase
    /// (2 per MZI + N output phases = N² total for an N×N mesh).
    pub fn program(&self, ledger: &mut EventLedger) {
        ledger.modulator_events += self.mesh.n_phases() as u64;
    }

    /// One optical MVM: propagate the input amplitude vector and
    /// photodetect. Returns the detected **powers** `|y_i|²`.
    /// Ledger: `macs += N²`, `adc_samples += N`.
    pub fn forward(&self, x: &[C64], ledger: &mut EventLedger) -> Vec<f64> {
        let n = self.mesh.n;
        let y = self.mesh.apply(x);
        ledger.macs += (n * n) as u64;
        ledger.adc_samples += n as u64;
        y.into_iter().map(|z| z.norm_sq()).collect()
    }
}
