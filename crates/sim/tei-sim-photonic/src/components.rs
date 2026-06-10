//! Component transfer matrices — the photonic circuit library.
//!
//! Everything here is a **frequency-domain field transfer**: a complex
//! amplitude (1-mode devices) or a `CMat` mapping input-mode amplitudes to
//! output-mode amplitudes (2-mode devices). Power is `|amplitude|²`.
//!
//! ## Conventions (binding for the whole crate)
//!
//! - **Propagation phase**: a waveguide of length `L` multiplies the field by
//!   `e^{iβL}` with `β = 2π·n_eff/λ` (forward-propagating `e^{i(βz−ωt)}`
//!   convention).
//! - **Dispersion-free (F1)**: `n_eff` is treated as wavelength-independent,
//!   so the group index `n_g = n_eff − λ·dn_eff/dλ = n_eff`. Dispersive
//!   models arrive when `tei-sim-field` F2 starts extracting S-parameters
//!   from port monitors (see docs/SIM-ROADMAP.md §3.7).
//! - **Directional coupler**: `[[t, i·r], [i·r, t]]` with `t² + r² = 1`,
//!   `t, r ≥ 0` real — straight-through amplitude `t`, cross-coupled
//!   amplitude `i·r` (the 90° cross-port phase of a symmetric, lossless,
//!   reciprocal coupler).
//! - **MZI internal phase θ**: bar power = `sin²(θ/2)`, so `θ = 0` is the
//!   full **cross** state and `θ = π` is the full **bar** state (see
//!   [`mzi_transfer`] for the derivation).

use tei_sim_core::linalg::{C64, CMat};

/// A straight waveguide segment, parameterized by wavelength at evaluation.
///
/// Field transfer `t(λ) = a · e^{iβL}` with `β = 2π n_eff / λ` and amplitude
/// factor `a = 10^{−(α_dB/cm · L_cm)/20}` from the propagation loss
/// `α` in dB/cm (power dB, hence the /20 for field amplitude).
#[derive(Debug, Clone, Copy)]
pub struct Waveguide {
    /// Physical length in µm.
    pub length_um: f64,
    /// Effective index (constant — dispersion-free F1 assumption).
    pub n_eff: f64,
    /// Propagation loss in dB/cm (power). 0 ⇒ lossless.
    pub loss_db_per_cm: f64,
}

impl Waveguide {
    /// Scalar field transfer at vacuum wavelength `lambda_um` (µm).
    pub fn transfer(&self, lambda_um: f64) -> C64 {
        let beta = std::f64::consts::TAU * self.n_eff / lambda_um;
        let length_cm = self.length_um * 1e-4;
        let amp = 10f64.powf(-self.loss_db_per_cm * length_cm / 20.0);
        C64::from_polar(amp, beta * self.length_um)
    }

    /// Two-port S-matrix `[[0, t], [t, 0]]` — matched (reflectionless) and
    /// reciprocal. Port 0 = left facet, port 1 = right facet.
    pub fn s_matrix(&self, lambda_um: f64) -> CMat {
        let t = self.transfer(lambda_um);
        let mut s = CMat::zeros(2, 2);
        s[(0, 1)] = t;
        s[(1, 0)] = t;
        s
    }
}

/// Directional coupler / beamsplitter transfer matrix.
///
/// `power_coupling` `K ∈ [0, 1]` is the fraction of power crossing over:
/// `r = √K`, `t = √(1−K)`, transfer `[[t, i·r], [i·r, t]]`.
///
/// Unitarity: rows have norm `t² + r² = 1` and inner product
/// `t·(−i r) + (i r)·t = 0`, so `S†S = I` exactly for any `K`.
pub fn directional_coupler(power_coupling: f64) -> CMat {
    assert!((0.0..=1.0).contains(&power_coupling), "K ∈ [0, 1]");
    let r = power_coupling.sqrt();
    let t = (1.0 - power_coupling).sqrt();
    let mut s = CMat::zeros(2, 2);
    s[(0, 0)] = C64::new(t, 0.0);
    s[(0, 1)] = C64::new(0.0, r);
    s[(1, 0)] = C64::new(0.0, r);
    s[(1, 1)] = C64::new(t, 0.0);
    s
}

/// The 50:50 coupler `(1/√2)·[[1, i], [i, 1]]`.
pub fn coupler_50_50() -> CMat {
    directional_coupler(0.5)
}

/// Two-mode phase shifter `diag(e^{iφ}, 1)` — phase on the **top** arm.
pub fn phase_shifter(phi: f64) -> CMat {
    let mut s = CMat::identity(2);
    s[(0, 0)] = C64::from_polar(1.0, phi);
    s
}

/// Single-port phase shifter: scalar `e^{iφ}`.
pub fn single_port_phase(phi: f64) -> C64 {
    C64::from_polar(1.0, phi)
}

/// 2×2 Mach-Zehnder interferometer: input phase `φ` (top arm), 50:50
/// coupler, internal phase `θ` (top arm), 50:50 coupler.
///
/// Built by composing the actual component matrices,
/// `M = C · P(θ) · C · P(φ)` with `C = (1/√2)[[1, i],[i, 1]]`,
/// `P(x) = diag(e^{ix}, 1)`. Carrying out the product:
///
/// ```text
/// C·P(θ)·C = ½ [[e^{iθ}−1,    i(e^{iθ}+1)],
///               [i(e^{iθ}+1), 1−e^{iθ}   ]]
///          = i·e^{iθ/2} [[sin(θ/2), cos(θ/2)],
///                        [cos(θ/2), −sin(θ/2)]]
/// ```
/// using `e^{iθ}−1 = 2i·e^{iθ/2}·sin(θ/2)` and
/// `e^{iθ}+1 = 2·e^{iθ/2}·cos(θ/2)`. Hence
///
/// ```text
/// M(θ, φ) = i·e^{iθ/2} [[e^{iφ} sin(θ/2),  cos(θ/2)],
///                       [e^{iφ} cos(θ/2), −sin(θ/2)]]
/// ```
///
/// **Stated convention**: bar transmission `|M₀₀|² = sin²(θ/2)`
/// (cross `|M₁₀|² = cos²(θ/2)`); `θ = 0` ⇒ full cross, `θ = π` ⇒ full bar.
/// This is the physical MZI of Clements et al., Optica 3, 1460 (2016); the
/// decomposition module uses the algebraically cleaner `T(θ, φ)` block,
/// which equals this `M` up to external phases (see `clements`).
pub fn mzi_transfer(theta: f64, phi: f64) -> CMat {
    let c = coupler_50_50();
    c.matmul(&phase_shifter(theta))
        .matmul(&c)
        .matmul(&phase_shifter(phi))
}

/// All-pass (single-bus) ring resonator field transfer at the through port.
///
/// `t` = real self-coupling of the bus coupler (cross-coupling `κ`,
/// `t² + κ² = 1`), `a` = single-round-trip amplitude factor, `theta` = βL =
/// round-trip phase. Summing the geometric series of round trips with the
/// `[[t, iκ],[iκ, t]]` coupler convention (Bogaerts et al., *Silicon
/// microring resonators*, Laser Photon. Rev. 6, 47 (2012)):
///
/// ```text
/// E_c = iκ·E_in + t·a·e^{iθ}·E_c          (circulating field)
/// E_t = t·E_in + iκ·a·e^{iθ}·E_c
///     ⇒ H(θ) = E_t/E_in = (t − a·e^{iθ}) / (1 − t·a·e^{iθ})
/// ```
///
/// Resonances at `θ = 2πm`; there `|H|² = (t−a)²/(1−ta)²`, which vanishes at
/// **critical coupling** `t = a`. Lossless (`a = 1`) ⇒ `|H| = 1` for all θ
/// (all-pass).
pub fn all_pass_transfer(t: f64, a: f64, theta: f64) -> C64 {
    let rt = C64::from_polar(a, theta); // a·e^{iθ}
    let num = C64::new(t, 0.0) - rt;
    let den = C64::ONE - rt * t;
    num / den
}

/// Add-drop (dual-bus) ring resonator: `(through, drop)` field transfers.
///
/// Couplers `t₁` (input bus) and `t₂` (drop bus), `κᵢ = √(1−tᵢ²)`; `a` and
/// `theta` are the full round-trip amplitude and phase. The drop path
/// crosses both couplers (factor `iκ₁·iκ₂ = −κ₁κ₂`) and propagates half the
/// ring (`√a·e^{iθ/2}`); the recirculation denominator is shared:
///
/// ```text
/// H_through = (t₁ − t₂·a·e^{iθ}) / (1 − t₁t₂·a·e^{iθ})
/// H_drop    = −κ₁κ₂·√a·e^{iθ/2} / (1 − t₁t₂·a·e^{iθ})
/// ```
///
/// Lossless power conservation `|H_t|² + |H_d|² = 1` at `a = 1` follows from
/// `(t₁² + t₂² − 2t₁t₂cosθ) + (1−t₁²)(1−t₂²) = |1 − t₁t₂e^{iθ}|²`.
pub fn add_drop_transfer(t1: f64, t2: f64, a: f64, theta: f64) -> (C64, C64) {
    let k1 = (1.0 - t1 * t1).sqrt();
    let k2 = (1.0 - t2 * t2).sqrt();
    let rt = C64::from_polar(a, theta);
    let half = C64::from_polar(a.sqrt(), theta / 2.0);
    let den = C64::ONE - rt * (t1 * t2);
    let through = (C64::new(t1, 0.0) - rt * t2) / den;
    let drop = (-half * (k1 * k2)) / den;
    (through, drop)
}

/// Wavelength-parameterized all-pass ring (geometry → `t`, `a`, `θ(λ)`).
#[derive(Debug, Clone, Copy)]
pub struct AllPassRing {
    /// Ring circumference in µm.
    pub circumference_um: f64,
    /// Effective index (dispersion-free ⇒ `n_g = n_eff`).
    pub n_eff: f64,
    /// Bus self-coupling `t` (real, `0 ≤ t ≤ 1`).
    pub t: f64,
    /// Round-trip loss in dB/cm of waveguide loss.
    pub loss_db_per_cm: f64,
}

impl AllPassRing {
    /// Round-trip phase `θ(λ) = 2π·n_eff·L/λ`.
    pub fn round_trip_phase(&self, lambda_um: f64) -> f64 {
        std::f64::consts::TAU * self.n_eff * self.circumference_um / lambda_um
    }

    /// Single-round-trip amplitude `a = 10^{−α·L_cm/20}`.
    pub fn round_trip_amplitude(&self) -> f64 {
        10f64.powf(-self.loss_db_per_cm * self.circumference_um * 1e-4 / 20.0)
    }

    /// Through-port field transfer at `lambda_um`.
    pub fn transfer(&self, lambda_um: f64) -> C64 {
        all_pass_transfer(
            self.t,
            self.round_trip_amplitude(),
            self.round_trip_phase(lambda_um),
        )
    }

    /// Closed-form free spectral range at `lambda_um`:
    /// `FSR_λ = λ²/(n_g·L)` with `n_g = n_eff` (dispersion-free).
    pub fn fsr_um(&self, lambda_um: f64) -> f64 {
        lambda_um * lambda_um / (self.n_eff * self.circumference_um)
    }
}
