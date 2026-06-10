//! Ramp-time sweep harness — runs a [`CellSpec`] across T/RC ratios
//! (rayon-parallel, deterministic outputs) and produces the
//! **overhead curve** E(T)/E_abrupt that recalibrates the reversible cost
//! dialect's fixed `REVERSIBLE_OVERHEAD_L0 = 10³` into a measured,
//! frequency-dependent function.
//!
//! # The closed form this module is anchored to
//!
//! For the canonical R–C charged by a linear ramp 0 → V over T (then held),
//! the resistor current is i(t) = (CV/T)(1 − e^{−t/RC}) during the ramp and
//! an exponential settle after it; integrating i²R over both pieces gives
//! **exactly**
//!
//! ```text
//! E_R(T) = (RC/T)·CV²·[1 − (RC/T)·(1 − e^{−T/RC})]
//! ```
//!
//! with the two limits the validation suite pins down: T ≫ RC ⇒
//! E_R → (RC/T)·CV² (log-log slope −1, the adiabatic regime) and T ≪ RC ⇒
//! E_R → ½CV² (abrupt switching). Athas et al., "Low-power digital systems
//! based on adiabatic-switching principles", IEEE T-VLSI 2(4), 1994;
//! DeBenedictis arXiv:2009.00448 runs the same sweep over S2LAL cells.

use crate::cells::CellSpec;
use rayon::prelude::*;
use serde::Serialize;
use std::sync::mpsc;
use tei_sim_circuit::{CircuitError, TransientOpts, TransientResult, transient};

/// Transient steps per min(T, RC) — resolves both the ramp and the
/// exponential settle. Trapezoidal integration is second order, so 400
/// points per time constant puts the discretization error orders of
/// magnitude below the 1% validation tolerances.
pub const STEPS_PER_UNIT: f64 = 400.0;

/// The slope fit is restricted to T/RC ≥ this — the adiabatic regime where
/// the closed form is within 10% of its (RC/T)·CV² asymptote.
pub const ADIABATIC_REGIME_MIN_RATIO: f64 = 10.0;

/// Landauer limit kT·ln2 at 300 K [J] ≈ 2.871×10⁻²¹ — the optional
/// overhead-above-Landauer normalizer in the executor outputs.
pub const E_LANDAUER_300K: f64 = 1.380_649e-23 * 300.0 * std::f64::consts::LN_2;

/// Exact dissipation of [`crate::cells::ramp_charge_cell`] run to
/// completion: E = (RC/T)·CV²·[1 − (RC/T)(1 − e^{−T/RC})]. Degenerate
/// `t_ramp ≤ 0` returns the abrupt limit ½CV².
pub fn ramp_charge_exact(r: f64, c: f64, v: f64, t_ramp: f64) -> f64 {
    let cv2 = c * v * v;
    if t_ramp <= 0.0 {
        return 0.5 * cv2;
    }
    let x = r * c / t_ramp;
    x * cv2 * (1.0 - x * (1.0 - (-1.0 / x).exp()))
}

/// One completed sweep point.
#[derive(Debug, Clone, Serialize)]
pub struct SweepPoint {
    /// Ramp time in units of the switch time constant, T/RC.
    pub t_over_rc: f64,
    /// Total dissipated energy of the cell over the cycle [J].
    pub e_diss_j: f64,
    /// `e_diss_j` normalized by the cell's abrupt-switching limit
    /// ([`CellSpec::abrupt_limit_j`]).
    pub e_over_abrupt: f64,
    /// Transient steps the point cost (ledger bookkeeping).
    pub steps: u64,
}

/// Full result of a single-cell run at one ratio, with the underlying
/// transient (per-element energies, Tellegen residual) exposed for the
/// validation suite.
#[derive(Debug, Clone)]
pub struct CellRun {
    pub e_diss_j: f64,
    pub e_abrupt_j: f64,
    pub transient: TransientResult,
}

/// Run one cell at one T/RC ratio: builds the netlist via
/// [`CellSpec::build`], steps it with dt = min(T, RC)/[`STEPS_PER_UNIT`]
/// through the full clock cycle plus the settle tail, and returns the
/// dissipated energy. Practical ratio range ≈ 10⁻⁴ … 10⁵ (outside it the
/// fixed-step budget of the circuit solver trips, surfacing as
/// `CircuitError::Invalid`).
pub fn run_cell(cell: &CellSpec, t_over_rc: f64) -> Result<CellRun, CircuitError> {
    cell.validate()?;
    if !t_over_rc.is_finite() || t_over_rc <= 0.0 {
        return Err(CircuitError::Invalid(format!(
            "t_over_rc must be finite and positive, got {t_over_rc}"
        )));
    }
    let rc = cell.rc();
    let t_ramp = t_over_rc * rc;
    let (net, t_stop) = cell.build(t_ramp);
    let mut opts = TransientOpts::new(t_stop, t_ramp.min(rc) / STEPS_PER_UNIT);
    opts.store_stride = 1 << 20; // energies only; skip trace storage
    let res = transient(&net, &opts)?;
    Ok(CellRun {
        e_diss_j: res.dissipated_energy,
        e_abrupt_j: cell.abrupt_limit_j(),
        transient: res,
    })
}

/// Sweep a cell across T/RC ratios, rayon-parallel. `on_point(done, &p)` is
/// invoked **on the calling thread** once per completed point (in completion
/// order; `done` = points finished so far) — the executor turns these into
/// progress ticks. The returned vector is in the *input* ratio order
/// regardless of completion order, so outputs stay deterministic.
pub fn sweep_with_progress(
    cell: &CellSpec,
    ratios: &[f64],
    on_point: &mut dyn FnMut(usize, &SweepPoint),
) -> Result<Vec<SweepPoint>, CircuitError> {
    cell.validate()?;
    if ratios.is_empty() {
        return Err(CircuitError::Invalid(
            "ratios must contain at least one T/RC value".into(),
        ));
    }
    let n = ratios.len();
    let mut out: Vec<Option<SweepPoint>> = vec![None; n];
    let mut first_err: Option<CircuitError> = None;
    let (tx, rx) = mpsc::channel::<(usize, Result<SweepPoint, CircuitError>)>();

    std::thread::scope(|s| {
        s.spawn(move || {
            ratios
                .par_iter()
                .enumerate()
                .for_each_with(tx, |tx, (i, &ratio)| {
                    let point = run_cell(cell, ratio).map(|run| SweepPoint {
                        t_over_rc: ratio,
                        e_diss_j: run.e_diss_j,
                        e_over_abrupt: run.e_diss_j / run.e_abrupt_j,
                        steps: run.transient.steps,
                    });
                    let _ = tx.send((i, point));
                });
        });
        // Drain on the caller's thread so the (non-Send) progress callback
        // never crosses threads.
        for done in 1..=n {
            let (i, point) = rx.recv().expect("sweep worker disconnected");
            match point {
                Ok(p) => {
                    on_point(done, &p);
                    out[i] = Some(p);
                }
                Err(e) => {
                    if first_err.is_none() {
                        first_err = Some(e);
                    }
                }
            }
        }
    });

    if let Some(e) = first_err {
        return Err(e);
    }
    Ok(out.into_iter().map(|p| p.expect("point filled")).collect())
}

/// **The recalibration product**: (T/RC, E/E_abrupt) pairs for a cell across
/// the given ratios — the measured, frequency-dependent overhead function
/// that replaces the cost dialect's fixed `REVERSIBLE_OVERHEAD_L0 = 10³`
/// constant. Pair with [`fitted_slope`] for the scaling exponent.
pub fn overhead_curve(cell: &CellSpec, ratios: &[f64]) -> Result<Vec<(f64, f64)>, CircuitError> {
    let points = sweep_with_progress(cell, ratios, &mut |_, _| {})?;
    Ok(points
        .into_iter()
        .map(|p| (p.t_over_rc, p.e_over_abrupt))
        .collect())
}

/// Least-squares log-log slope of an overhead curve, restricted to the
/// adiabatic regime T/RC ≥ [`ADIABATIC_REGIME_MIN_RATIO`]. The ideal
/// R–C adiabatic law gives −1. Returns NaN when fewer than two points fall
/// in the regime (serializes as JSON `null` in the executor outputs).
pub fn fitted_slope(curve: &[(f64, f64)]) -> f64 {
    let pts: Vec<(f64, f64)> = curve
        .iter()
        .filter(|(t, e)| *t >= ADIABATIC_REGIME_MIN_RATIO && *e > 0.0)
        .map(|(t, e)| (t.ln(), e.ln()))
        .collect();
    if pts.len() < 2 {
        return f64::NAN;
    }
    let n = pts.len() as f64;
    let xb = pts.iter().map(|p| p.0).sum::<f64>() / n;
    let yb = pts.iter().map(|p| p.1).sum::<f64>() / n;
    let sxy: f64 = pts.iter().map(|(x, y)| (x - xb) * (y - yb)).sum();
    let sxx: f64 = pts.iter().map(|(x, _)| (x - xb) * (x - xb)).sum();
    sxy / sxx
}

#[cfg(test)]
mod tests {
    use super::*;

    /// fitted_slope recovers an exact power law e = 7·t⁻¹ to 1e-12, and
    /// ignores the sub-regime points that would bias it.
    #[test]
    fn fitted_slope_exact_power_law() {
        let curve: Vec<(f64, f64)> = [0.5, 2.0, 10.0, 100.0, 1000.0]
            .iter()
            .map(|&t| (t, 7.0 / t))
            .collect();
        let s = fitted_slope(&curve);
        assert!((s + 1.0).abs() < 1e-12, "slope = {s}");
        // Fewer than two regime points (only t = 10 qualifies here) → NaN.
        assert!(fitted_slope(&curve[..3]).is_nan());
    }

    /// The closed form's two limits: x→∞ gives ½CV², large T gives (RC/T)CV².
    #[test]
    fn ramp_charge_exact_limits() {
        let (r, c, v) = (1e3, 1e-9, 1.0);
        let cv2 = c * v * v;
        let rc = r * c;
        let e_fast = ramp_charge_exact(r, c, v, 1e-4 * rc);
        assert!((e_fast - 0.5 * cv2).abs() / (0.5 * cv2) < 1e-3);
        let e_slow = ramp_charge_exact(r, c, v, 1e4 * rc);
        assert!((e_slow - 1e-4 * cv2).abs() / (1e-4 * cv2) < 1e-3);
    }

    /// Empty / invalid sweep inputs surface as Invalid, not panics.
    #[test]
    fn sweep_input_validation() {
        let cell = CellSpec::RampCharge {
            r_ohm: 1e3,
            c_f: 1e-9,
            v: 1.0,
        };
        assert!(matches!(
            sweep_with_progress(&cell, &[], &mut |_, _| {}),
            Err(CircuitError::Invalid(_))
        ));
        assert!(matches!(
            run_cell(&cell, -1.0),
            Err(CircuitError::Invalid(_))
        ));
    }
}
