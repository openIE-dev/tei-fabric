//! Analytic validation for tei-sim-gaussian (roadmap §3.8).
//!
//! Closed-form ground truth only, per the binding validation policy. All
//! formulas are in the crate's conventions: xpxp ordering, ħ = 2 (vacuum
//! σ = I), Ω = ⊕[[0,1],[−1,0]]. Reference: Weedbrook et al., "Gaussian
//! quantum information", Rev. Mod. Phys. 84, 621 (2012).

use tei_sim_core::exec::Executor;
use tei_sim_core::linalg::Mat;
use tei_sim_core::rng::Rng;
use tei_sim_gaussian::exec::{GaussianExecutor, GaussianJob};
use tei_sim_gaussian::{
    GaussianState, beamsplitter, embed_one_mode, embed_two_mode, is_physical, omega,
    physicality_margin, rotation, squeezer, two_mode_squeezer,
};

const TOL: f64 = 1e-12;

fn frob_diff(a: &Mat, b: &Mat) -> f64 {
    let mut s = 0.0;
    for i in 0..a.rows {
        for j in 0..a.cols {
            s += (a[(i, j)] - b[(i, j)]).powi(2);
        }
    }
    s.sqrt()
}

/// (a) Vacuum: σ = I exactly, ⟨n⟩ = 0.
#[test]
fn vacuum_is_identity() {
    let st = GaussianState::vacuum(3);
    assert_eq!(frob_diff(&st.cov, &Mat::identity(6)), 0.0);
    assert!(st.mean.iter().all(|&m| m == 0.0));
    for k in 0..3 {
        assert_eq!(st.mean_photon(k), 0.0);
    }
}

/// (b) Squeezed vacuum: Var(x̂) = e^{−2r}, Var(p̂) = e^{2r}; ⟨n⟩ = sinh²r.
#[test]
fn squeezed_vacuum_variances_and_photons() {
    for &r in &[0.1, 0.5, 1.3, 2.0] {
        let st = GaussianState::squeezed(&[(r, 0.0)]);
        let (vx, vp) = st.quadrature_variances(0);
        assert!((vx - (-2.0 * r).exp()).abs() < TOL, "r={r}: Var x = {vx}");
        assert!((vp - (2.0 * r).exp()).abs() < TOL, "r={r}: Var p = {vp}");
        let n = st.mean_photon(0);
        assert!((n - r.sinh().powi(2)).abs() < TOL, "r={r}: ⟨n⟩ = {n}");
    }
}

/// (c) Symplectic invariance: S Ω Sᵀ = Ω for squeezer, rotation,
/// beamsplitter, two-mode squeezer at random parameters (embedded in a
/// 3-mode register, including non-adjacent and swapped mode pairs).
#[test]
fn all_ops_preserve_symplectic_form() {
    let mut rng = Rng::new(20260610);
    let om = omega(3);
    for _ in 0..25 {
        let r = 3.0 * (rng.f64() - 0.5);
        let phi = std::f64::consts::TAU * rng.f64();
        let theta = std::f64::consts::TAU * rng.f64();
        let candidates = [
            embed_one_mode(3, rng.below(3), &squeezer(r, phi)),
            embed_one_mode(3, rng.below(3), &rotation(phi)),
            embed_two_mode(3, 0, 2, &beamsplitter(theta)),
            embed_two_mode(3, 2, 1, &beamsplitter(theta)),
            embed_two_mode(3, 1, 2, &two_mode_squeezer(r)),
            embed_two_mode(3, 2, 0, &two_mode_squeezer(r)),
        ];
        for s in &candidates {
            let sos = s.matmul(&om).matmul(&s.transpose());
            let d = frob_diff(&sos, &om);
            assert!(d < TOL, "SΩSᵀ − Ω = {d} (r={r}, φ={phi}, θ={theta})");
        }
    }
}

/// (d) Beamsplitter on |α, 0⟩: output amplitudes (α cos θ, −α sin θ) in this
/// crate's sign convention; covariance stays exactly I (coherent in →
/// coherent out).
#[test]
fn beamsplitter_on_coherent() {
    let alpha = (0.8, -0.3);
    let theta = 0.6;
    let mut st = GaussianState::coherent(&[alpha, (0.0, 0.0)]);
    st.beamsplit(0, 1, theta);
    let (c, s) = (theta.cos(), theta.sin());
    // Means in ħ=2: ⟨x̂⟩ = 2 Re α, ⟨p̂⟩ = 2 Im α.
    let want = [
        2.0 * alpha.0 * c,
        2.0 * alpha.1 * c,
        -2.0 * alpha.0 * s,
        -2.0 * alpha.1 * s,
    ];
    for (got, want) in st.mean.iter().zip(&want) {
        assert!((got - want).abs() < TOL, "mean {got} vs {want}");
    }
    assert!(frob_diff(&st.cov, &Mat::identity(4)) < TOL);
}

/// (e) EPR correlations of the two-mode squeezed vacuum.
///
/// Derivation (ħ = 2): TMS(r) on vacuum gives σ = [[cosh 2r·I₂, sinh 2r·Z],
/// [sinh 2r·Z, cosh 2r·I₂]], Z = diag(1, −1). Hence
///   Var(x̂₁ − x̂₂) = σx1x1 + σx2x2 − 2σx1x2 = 2cosh 2r − 2sinh 2r = 2e^{−2r},
///   Var(p̂₁ + p̂₂) = σp1p1 + σp2p2 + 2σp1p2 = 2cosh 2r − 2sinh 2r = 2e^{−2r}
/// (the p–p cross block carries Z's minus sign). Each reduced single mode is
/// thermal: σₖ = cosh 2r·I = (2n̄+1)·I with n̄ = sinh²r.
#[test]
fn epr_correlations_of_tms() {
    for &r in &[0.3, 0.9, 1.7] {
        let mut st = GaussianState::vacuum(2);
        st.two_mode_squeeze(0, 1, r);
        let c = &st.cov;
        let var_xminus = c[(0, 0)] + c[(2, 2)] - 2.0 * c[(0, 2)];
        let var_pplus = c[(1, 1)] + c[(3, 3)] + 2.0 * c[(1, 3)];
        let want = 2.0 * (-2.0 * r).exp();
        assert!((var_xminus - want).abs() < TOL, "Var(x₁−x₂) = {var_xminus}");
        assert!((var_pplus - want).abs() < TOL, "Var(p₁+p₂) = {var_pplus}");
        // Reduced single modes are thermal with n̄ = sinh²r.
        let nbar = r.sinh().powi(2);
        for k in 0..2 {
            let red = st.mode_cov(k);
            let mut th = Mat::identity(2);
            th[(0, 0)] = 2.0 * nbar + 1.0;
            th[(1, 1)] = 2.0 * nbar + 1.0;
            assert!(frob_diff(&red, &th) < TOL);
            assert!((st.mean_photon(k) - nbar).abs() < TOL);
        }
    }
}

/// (f) Homodyne conditioning on the TMS state.
///
/// Closed form: with σ as in test (e), measuring x̂₁ has marginal
/// N(0, cosh 2r). Conditioning x̂₂ via the Schur complement:
///   Var(x̂₂ | x̂₁ = m) = σx2x2 − σx2x1²/σx1x1
///                     = cosh 2r − sinh²2r / cosh 2r
///                     = (cosh²2r − sinh²2r)/cosh 2r = 1/cosh 2r,
/// which drops below the marginal cosh 2r (and below vacuum!) — the EPR
/// steering signature. The conditional mean is
///   ⟨x̂₂⟩ = σx2x1/σx1x1 · m = tanh 2r · m,
/// while p̂₂ is uncorrelated with x̂₁ and keeps Var = cosh 2r.
#[test]
fn homodyne_conditioning_closed_form() {
    let r = 0.8_f64;
    let c2 = (2.0 * r).cosh();
    let mut st = GaussianState::vacuum(2);
    st.two_mode_squeeze(0, 1, r);

    let (marg_mean, marg_var) = st.homodyne_marginal(0, 0.0);
    assert!(marg_mean.abs() < TOL);
    assert!((marg_var - c2).abs() < TOL);

    let m = 1.234;
    st.homodyne_project(0, 0.0, m);
    assert_eq!(st.n_modes(), 1);
    let (vx, vp) = st.quadrature_variances(0);
    assert!(
        (vx - 1.0 / c2).abs() < TOL,
        "Var(x₂|x₁) = {vx} vs {}",
        1.0 / c2
    );
    assert!((vp - c2).abs() < TOL, "Var(p₂|x₁) = {vp}");
    assert!(vx < marg_var, "conditional must beat marginal");
    let want_mean = (2.0 * r).tanh() * m;
    assert!((st.mean[0] - want_mean).abs() < TOL);
    assert!(st.mean[1].abs() < TOL);
}

/// (f, sampled) 50 000 homodyne shots on independently prepared TMS states:
/// empirical mean and variance match the marginal N(0, cosh 2r) within 3%.
#[test]
fn homodyne_sampling_matches_marginal() {
    let r = 0.8_f64;
    let c2 = (2.0 * r).cosh();
    let base = {
        let mut st = GaussianState::vacuum(2);
        st.two_mode_squeeze(0, 1, r);
        st
    };
    let mut rng = Rng::new(424242);
    let shots = 50_000;
    let (mut sum, mut sumsq) = (0.0, 0.0);
    for _ in 0..shots {
        let mut st = base.clone();
        let out = st.homodyne(0, 0.0, &mut rng);
        assert!((out.variance - c2).abs() < TOL);
        sum += out.sample;
        sumsq += out.sample * out.sample;
    }
    let n = shots as f64;
    let mean = sum / n;
    let var = sumsq / n - mean * mean;
    let sd = c2.sqrt();
    assert!(mean.abs() < 0.03 * sd, "empirical mean {mean}");
    assert!((var / c2 - 1.0).abs() < 0.03, "empirical var {var} vs {c2}");
}

/// (g) Thermal state: σ = (2n̄+1)·I, ⟨n⟩ = n̄.
#[test]
fn thermal_state_moments() {
    for &nbar in &[0.0, 0.4, 3.7] {
        let st = GaussianState::thermal(&[nbar]);
        let mut want = Mat::identity(2);
        want[(0, 0)] = 2.0 * nbar + 1.0;
        want[(1, 1)] = 2.0 * nbar + 1.0;
        assert_eq!(frob_diff(&st.cov, &want), 0.0);
        assert!((st.mean_photon(0) - nbar).abs() < TOL);
    }
}

/// (h) Physicality: every constructed state satisfies σ + iΩ ⪰ 0
/// (margin ≥ −1e-12); a matrix squeezed below vacuum in BOTH quadratures
/// fails.
#[test]
fn physicality_check() {
    let states = [
        GaussianState::vacuum(2),
        GaussianState::coherent(&[(1.1, -0.4), (0.0, 2.0)]),
        GaussianState::squeezed(&[(1.2, 0.7), (0.4, 0.0)]),
        GaussianState::thermal(&[0.3, 2.0]),
        {
            let mut st = GaussianState::vacuum(2);
            st.two_mode_squeeze(0, 1, 1.0);
            st.beamsplit(0, 1, 0.3);
            st
        },
    ];
    for st in &states {
        let margin = physicality_margin(&st.cov);
        assert!(margin >= -TOL, "physical state has margin {margin}");
        assert!(is_physical(&st.cov, TOL));
    }
    // Both quadratures "squeezed" below vacuum — violates Heisenberg:
    // ν = 0.5 < 1, margin ≈ −0.5.
    let mut bad = Mat::identity(2);
    bad[(0, 0)] = 0.5;
    bad[(1, 1)] = 0.5;
    let margin = physicality_margin(&bad);
    assert!((margin + 0.5).abs() < TOL, "margin {margin}");
    assert!(!is_physical(&bad, TOL));
}

/// (i) Determinism: seeded homodyne sampling is bit-reproducible, both at
/// the state level and through the executor.
#[test]
fn seeded_sampling_is_deterministic() {
    let base = {
        let mut st = GaussianState::vacuum(2);
        st.two_mode_squeeze(0, 1, 0.6);
        st
    };
    let run = |seed: u64| -> Vec<f64> {
        let mut rng = Rng::new(seed);
        (0..64)
            .map(|_| base.clone().homodyne(0, 0.4, &mut rng).sample)
            .collect()
    };
    assert_eq!(run(9), run(9));
    assert_ne!(run(9), run(10));

    let job: GaussianJob = serde_json::from_value(serde_json::json!({
        "n_modes": 2,
        "circuit": [{ "op": "two_mode_squeeze", "mode_a": 0, "mode_b": 1, "r": 0.6 }],
        "homodyne": { "mode": 0, "phi": 0.0, "shots": 5000 },
        "seed": 9,
    }))
    .unwrap();
    let a = GaussianExecutor.execute(&job, &mut |_| {});
    let b = GaussianExecutor.execute(&job, &mut |_| {});
    assert_eq!(
        a.outputs["homodyne"]["sample_mean"],
        b.outputs["homodyne"]["sample_mean"]
    );
    assert_eq!(
        a.outputs["homodyne"]["histogram"]["counts"],
        b.outputs["homodyne"]["histogram"]["counts"]
    );
}
