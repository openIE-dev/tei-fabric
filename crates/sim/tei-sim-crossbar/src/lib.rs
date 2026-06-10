//! tei-sim-crossbar — CrossSim-class (Sandia) analog in-memory MVM simulator.
//!
//! Functional simulator for the in-memory-compute substrate column: a weight
//! matrix `W` (rows = inputs k, cols = outputs n) is mapped onto device
//! conductances and matrix-vector multiplies `y_j = Σᵢ xᵢ·W[i][j]` are
//! executed through a stack of device and peripheral non-idealities:
//!
//! - **programming variability** — per-device multiplicative lognormal error
//!   `G_prog = G_target · exp(σ_prog·Z)`, `Z ~ N(0,1)`. The lognormal form is
//!   the standard fit to ReRAM/PCM write distributions (see e.g. the CrossSim
//!   device models, Sandia, <https://cross-sim.sandia.gov>, and Wan et al.,
//!   "A compute-in-memory chip based on resistive random-access memory",
//!   Nature 608, 2022).
//! - **read noise** — per-read additive Gaussian on each device current with
//!   standard deviation proportional to the device conductance
//!   (`σ_i = σ_read·|G_i|`), i.e. multiplicative shot/thermal read noise.
//! - **conductance drift** — PCM power law `G(t) = G0·(t/t0)^(−ν)`
//!   (Ielmini et al., IEEE TED 2007; Le Gallo & Sebastian, "An overview of
//!   phase-change memory device physics", J. Phys. D 53, 213002, 2020),
//!   applied deterministically through an `age = t/t0` parameter.
//! - **DAC quantization** — `b_in`-bit uniform mid-rise quantizer over the
//!   input full scale ±`input_range`, with clipping.
//! - **ADC transfer** — `b_out`-bit uniform quantizer over a configurable
//!   range with clipping and optional bow-shaped INL. Ideal quantization SNR
//!   for a full-scale sinusoid is the classic `6.02·b + 1.76 dB`
//!   (Bennett, "Spectra of quantized signals", Bell Syst. Tech. J. 27, 1948).
//! - **IR drop** — three fidelity modes; see [`IrDropMode`]. The exact
//!   resistive-mesh solve is deferred to `tei-sim-circuit` (roadmap §3.5 M1).
//!
//! Matrices larger than the physical array are tiled `⌈k/size⌉ × ⌈n/size⌉`
//! with partial sums accumulated in the digital domain; the ADC fires once
//! per (row-tile, output column), so a full MVM costs `n·⌈k/size⌉` ADC
//! samples and `k·n` MACs — both counted in the [`EventLedger`].
//!
//! **Signed-weight convention.** Hardware realizes signed weights as a
//! balanced differential conductance pair `(G⁺, G⁻)` read on complementary
//! columns. This crate collapses the pair into one *effective signed
//! conductance* `G = G⁺ − G⁻` per cell, with all multiplicative device
//! effects (lognormal programming, drift, read noise scaled by |G|) applied
//! to the effective value. The differential periphery is a circuit-level
//! detail owned by `tei-sim-circuit`.
//!
//! Validation (tests/analytic.rs): analytic + published only — independent
//! read-noise variance propagation σ_y² = Σ xᵢ²σᵢ², quantization SNR
//! 6.02·b + 1.76 dB, drift-exponent recovery, tiling exactness, lognormal
//! mean exp(μ + σ²/2), seed determinism.

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

// ───────────────────────────── device model ─────────────────────────────

/// ADC transfer parameters: uniform `bits`-bit quantizer over ±`range`
/// (output-domain units — internally the column current is normalized by the
/// weight→conductance scale before digitization, which is mathematically
/// identical to an ADC range of `range·g_scale` amperes) with clipping, plus
/// an optional bow-shaped integral non-linearity.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdcParams {
    pub bits: u32,
    /// Full-scale: codes span [−range, +range]; inputs outside clip.
    pub range: f64,
    /// Peak INL in LSB. Modeled as a half-sine bow over the code axis,
    /// `e(c) = inl_lsb·Δ·sin(π·c/(2^b − 1))` — the standard low-order INL
    /// signature of flash/SAR converters. 0 disables it.
    #[serde(default)]
    pub inl_lsb: f64,
}

/// IR-drop fidelity modes for the parasitic wire resistance of the array.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IrDropMode {
    /// Zero wire resistance — currents sum losslessly.
    #[default]
    Ideal,
    /// First-order closed form. Each cell (i, j) of a physical tile of
    /// `r` rows sees, to first order, its own series wire resistance
    ///
    /// ```text
    /// R_path(i, j) = R_wire · ((j + 1) + (r − i))
    /// ```
    ///
    /// — `j + 1` row-wire segments from the input driver at the left edge
    /// to the cell, plus `r − i` column-wire segments from the cell down to
    /// the sense amplifier (virtual ground) at the bottom edge. The device
    /// branch then behaves as `G` in series with `R_path`, i.e. an effective
    /// conductance
    ///
    /// ```text
    /// G_eff = G / (1 + |G|·R_path)
    /// ```
    ///
    /// **Approximation:** this treats every cell's current path as
    /// independent, ignoring the voltage drops caused by *other* cells'
    /// currents sharing the same wire segments. It is exact for a single
    /// active device and first-order accurate when the aggregate drop is
    /// small (`N·Ḡ·R_wire ≪ 1`); it underestimates the degradation of a
    /// fully-active array. The coupled solve is [`IrDropMode::ExactMesh`].
    FirstOrder { r_wire: f64 },
    /// Exact resistive-mesh solve (every wire segment a resistor, full MNA).
    /// Deferred to `tei-sim-circuit` M1 per docs/SIM-ROADMAP.md §3.3/§3.5;
    /// selecting it currently panics via `unimplemented!()`.
    ExactMesh,
}

/// Full device + periphery parameter set for a crossbar.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DeviceParams {
    /// Maximum device conductance magnitude, siemens. The largest |weight|
    /// maps to this; everything else scales linearly.
    pub g_max: f64,
    /// Lognormal programming-error σ (0 = perfect write).
    pub sigma_prog: f64,
    /// Relative per-read noise: each read draws `N(0, (σ_read·|G|)²)` on the
    /// device conductance (0 = noiseless read).
    pub sigma_read: f64,
    /// PCM drift exponent ν in `G(t) = G0·(t/t0)^(−ν)` (0 = no drift).
    pub drift_nu: f64,
    /// Normalized age `t/t0` at read time (1 = freshly programmed).
    pub age: f64,
    /// Input DAC resolution; `None` = ideal analog input.
    pub dac_bits: Option<u32>,
    /// Input full scale ±`input_range` for the DAC.
    pub input_range: f64,
    /// Output ADC; `None` = ideal analog readout.
    pub adc: Option<AdcParams>,
    /// Parasitic wire-resistance fidelity.
    pub ir_drop: IrDropMode,
}

impl Default for DeviceParams {
    fn default() -> Self {
        Self {
            g_max: 100e-6, // 100 µS — typical ReRAM LRS scale.
            sigma_prog: 0.0,
            sigma_read: 0.0,
            drift_nu: 0.0,
            age: 1.0,
            dac_bits: None,
            input_range: 1.0,
            adc: None,
            ir_drop: IrDropMode::Ideal,
        }
    }
}

/// Uniform mid-rise quantizer over ±range with clipping: step Δ = 2R/2^b,
/// reconstruction levels at (c + ½)Δ − R for codes c ∈ [0, 2^b).
fn quantize_uniform(v: f64, range: f64, bits: u32) -> f64 {
    let levels = (1u64 << bits) as f64;
    let step = 2.0 * range / levels;
    let code = ((v + range) / step).floor().clamp(0.0, levels - 1.0);
    (code + 0.5) * step - range
}

/// ADC transfer: uniform quantization + clipping + optional INL bow.
fn adc_transfer(v: f64, p: &AdcParams) -> f64 {
    let levels = (1u64 << p.bits) as f64;
    let step = 2.0 * p.range / levels;
    let code = ((v + p.range) / step).floor().clamp(0.0, levels - 1.0);
    let mut out = (code + 0.5) * step - p.range;
    if p.inl_lsb != 0.0 {
        out += p.inl_lsb * step * (std::f64::consts::PI * code / (levels - 1.0)).sin();
    }
    out
}

// ───────────────────────────── crossbar array ─────────────────────────────

/// One physical tile: a contiguous block of the weight matrix programmed
/// onto a ≤ `array_size`² device array.
#[derive(Debug, Clone)]
struct Tile {
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
    /// Programmed effective signed conductance G0 (post-lognormal), row-major.
    g: Vec<f64>,
}

/// A weight matrix mapped onto tiled crossbar arrays with a programmable
/// device model. Construction *programs* the devices (drawing lognormal
/// write errors from `rng`); [`CrossbarArray::mvm`] then executes noisy
/// matrix-vector products and [`CrossbarArray::ideal_mvm`] the exact
/// digital reference.
#[derive(Debug, Clone)]
pub struct CrossbarArray {
    rows: usize,
    cols: usize,
    array_size: usize,
    params: DeviceParams,
    /// Weight → conductance scale, S per weight unit: g_max / max|w|.
    g_scale: f64,
    /// Ideal weights, row-major (the digital reference).
    w_ideal: Vec<f64>,
    tiles: Vec<Tile>,
}

impl CrossbarArray {
    /// Program `weights` (row-major, `rows × cols`) onto tiled physical
    /// arrays of side `array_size`. Lognormal write errors are drawn from
    /// `rng` in a fixed tile-major, row-major device order, so identical
    /// seeds program identical arrays.
    pub fn program(
        weights: &[f64],
        rows: usize,
        cols: usize,
        array_size: usize,
        params: DeviceParams,
        rng: &mut Rng,
    ) -> Self {
        assert_eq!(weights.len(), rows * cols, "weights must be rows×cols");
        assert!(array_size > 0, "array_size must be positive");
        let w_max = weights.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
        let g_scale = if w_max > 0.0 {
            params.g_max / w_max
        } else {
            params.g_max
        };

        let row_tiles = rows.div_ceil(array_size);
        let col_tiles = cols.div_ceil(array_size);
        let mut tiles = Vec::with_capacity(row_tiles * col_tiles);
        for tr in 0..row_tiles {
            for tc in 0..col_tiles {
                let row0 = tr * array_size;
                let col0 = tc * array_size;
                let trows = array_size.min(rows - row0);
                let tcols = array_size.min(cols - col0);
                let mut g = Vec::with_capacity(trows * tcols);
                for i in 0..trows {
                    for j in 0..tcols {
                        let target = weights[(row0 + i) * cols + (col0 + j)] * g_scale;
                        // G_prog = G_target · exp(σ_prog·Z): multiplicative
                        // lognormal on the magnitude, sign preserved.
                        let prog = if params.sigma_prog > 0.0 {
                            target * (params.sigma_prog * rng.normal()).exp()
                        } else {
                            target
                        };
                        g.push(prog);
                    }
                }
                tiles.push(Tile {
                    row0,
                    col0,
                    rows: trows,
                    cols: tcols,
                    g,
                });
            }
        }

        Self {
            rows,
            cols,
            array_size,
            params,
            g_scale,
            w_ideal: weights.to_vec(),
            tiles,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Weight → conductance scale (siemens per weight unit).
    pub fn g_scale(&self) -> f64 {
        self.g_scale
    }

    /// PCM drift factor `(t/t0)^(−ν)` at the configured age.
    fn drift_factor(&self) -> f64 {
        if self.params.drift_nu == 0.0 {
            1.0
        } else {
            self.params.age.powf(-self.params.drift_nu)
        }
    }

    /// Effective signed conductance of device (row, col) at read time:
    /// programmed value (lognormal included) aged by the drift power law.
    pub fn conductance(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols);
        let tr = row / self.array_size;
        let tc = col / self.array_size;
        let col_tiles = self.cols.div_ceil(self.array_size);
        let tile = &self.tiles[tr * col_tiles + tc];
        let (i, j) = (row - tile.row0, col - tile.col0);
        tile.g[i * tile.cols + j] * self.drift_factor()
    }

    /// Exact digital reference `y_j = Σᵢ xᵢ·W[i][j]` — no device model.
    pub fn ideal_mvm(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.rows);
        let mut y = vec![0.0; self.cols];
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.w_ideal[i * self.cols..(i + 1) * self.cols];
            for (yj, &w) in y.iter_mut().zip(row) {
                *yj += xi * w;
            }
        }
        y
    }

    /// Noisy MVM through the full non-ideality stack. Per row-tile partial
    /// sums are digitized (one ADC sample per tile × output column) and
    /// accumulated in the digital domain. Ledger: `macs += rows·cols`,
    /// `adc_samples += cols·⌈rows/array_size⌉`.
    pub fn mvm(&self, x: &[f64], rng: &mut Rng, ledger: &mut EventLedger) -> Vec<f64> {
        assert_eq!(x.len(), self.rows);

        // DAC: quantize the input vector once; every tile in a row-block is
        // driven by the same quantized line voltages.
        let xq: Vec<f64> = match self.params.dac_bits {
            Some(b) => x
                .iter()
                .map(|&v| quantize_uniform(v, self.params.input_range, b))
                .collect(),
            None => x.to_vec(),
        };

        let drift = self.drift_factor();
        let mut y = vec![0.0; self.cols];
        for tile in &self.tiles {
            let xs = &xq[tile.row0..tile.row0 + tile.rows];
            for j in 0..tile.cols {
                // Analog column current, normalized by g_scale into output
                // (weight·input) units — identical math, friendlier numbers.
                let mut acc = 0.0;
                for (i, &xi) in xs.iter().enumerate() {
                    let g0 = tile.g[i * tile.cols + j] * drift;
                    let g_eff = match self.params.ir_drop {
                        IrDropMode::Ideal => g0,
                        IrDropMode::FirstOrder { r_wire } => {
                            // See IrDropMode::FirstOrder for the derivation.
                            let r_path = r_wire * ((j + 1) + (tile.rows - i)) as f64;
                            g0 / (1.0 + g0.abs() * r_path)
                        }
                        IrDropMode::ExactMesh => unimplemented!(
                            "exact resistive-mesh IR drop lands with tei-sim-circuit M1 \
                             (docs/SIM-ROADMAP.md §3.5)"
                        ),
                    };
                    // Per-read conductance noise σ = σ_read·|G| (a device
                    // property — scaled by the aged programmed conductance).
                    let g_read = if self.params.sigma_read > 0.0 {
                        g_eff + self.params.sigma_read * g0.abs() * rng.normal()
                    } else {
                        g_eff
                    };
                    acc += xi * g_read;
                }
                ledger.macs += tile.rows as u64;
                ledger.adc_samples += 1;
                let mut partial = acc / self.g_scale;
                if let Some(adc) = &self.params.adc {
                    partial = adc_transfer(partial, adc);
                }
                y[tile.col0 + j] += partial;
            }
        }
        y
    }
}

// ───────────────────────────── executor ─────────────────────────────

/// Job spec accepted by the crossbar executor (mirrors /api/execute):
/// a `rows × cols` random weight matrix is programmed onto `array_size`
/// tiles with the given device model, then `n_queries` random-input MVMs
/// run and the noisy outputs are scored against the digital reference.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrossbarJob {
    pub rows: usize,
    pub cols: usize,
    #[serde(default = "default_array_size")]
    pub array_size: usize,
    #[serde(default)]
    pub device: DeviceParams,
    pub n_queries: u64,
    #[serde(default)]
    pub seed: u64,
}

fn default_array_size() -> usize {
    256
}

pub struct CrossbarExecutor;

impl Executor for CrossbarExecutor {
    type Job = CrossbarJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = std::time::Instant::now();
        let mut rng = Rng::new(job.seed);

        // Random weights in [−1, 1], programmed once.
        let weights: Vec<f64> = (0..job.rows * job.cols)
            .map(|_| 2.0 * rng.f64() - 1.0)
            .collect();
        let array = CrossbarArray::program(
            &weights,
            job.rows,
            job.cols,
            job.array_size,
            job.device.clone(),
            &mut rng,
        );

        let mut ledger = EventLedger::default();
        let mut sum_err2 = 0.0;
        let mut sum_sig2 = 0.0;
        let mut elements = 0u64;
        let report_every = (job.n_queries / 100).max(1);

        for q in 0..job.n_queries {
            let x: Vec<f64> = (0..job.rows).map(|_| 2.0 * rng.f64() - 1.0).collect();
            let y_ideal = array.ideal_mvm(&x);
            let y_noisy = array.mvm(&x, &mut rng, &mut ledger);
            for (yn, yi) in y_noisy.iter().zip(&y_ideal) {
                let e = yn - yi;
                sum_err2 += e * e;
                sum_sig2 += yi * yi;
            }
            elements += job.cols as u64;

            if (q + 1) % report_every == 0 || q + 1 == job.n_queries {
                let rms = (sum_err2 / elements as f64).sqrt();
                on_progress(Progress {
                    fraction: (q + 1) as f64 / job.n_queries as f64,
                    metrics: serde_json::json!({
                        "query": q + 1,
                        "rms_error": rms,
                        "snr_db": snr_db(sum_sig2, sum_err2),
                    }),
                });
            }
        }
        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

        let rms_error = (sum_err2 / elements.max(1) as f64).sqrt();
        ExecutionResult {
            ledger: ledger.clone(),
            outputs: serde_json::json!({
                "rows": job.rows,
                "cols": job.cols,
                "array_size": job.array_size,
                "n_queries": job.n_queries,
                "rms_error": rms_error,
                "snr_db": snr_db(sum_sig2, sum_err2),
                "macs": ledger.macs,
                "adc_samples": ledger.adc_samples,
            }),
        }
    }
}

/// SNR in dB; `None` (JSON null) when the error is exactly zero.
fn snr_db(sig2: f64, err2: f64) -> Option<f64> {
    (err2 > 0.0).then(|| 10.0 * (sig2 / err2).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mid-rise quantizer: full-scale endpoints clip to the outermost
    /// reconstruction levels and 0 maps within Δ/2.
    #[test]
    fn quantizer_basics() {
        let step = 2.0 / 16.0; // b=4, range 1
        assert!((quantize_uniform(1.0, 1.0, 4) - (1.0 - step / 2.0)).abs() < 1e-15);
        assert!((quantize_uniform(-1.0, 1.0, 4) - (-1.0 + step / 2.0)).abs() < 1e-15);
        assert!(quantize_uniform(0.0, 1.0, 4).abs() <= step / 2.0 + 1e-15);
        // Clipping beyond full scale.
        assert!((quantize_uniform(5.0, 1.0, 4) - (1.0 - step / 2.0)).abs() < 1e-15);
    }

    /// First-order IR drop strictly reduces output magnitude and converges
    /// to ideal as R_wire → 0.
    #[test]
    fn ir_drop_first_order_sanity() {
        let w = vec![1.0; 64];
        let x = vec![1.0; 8];
        let mk = |r_wire: f64| {
            let params = DeviceParams {
                ir_drop: if r_wire == 0.0 {
                    IrDropMode::Ideal
                } else {
                    IrDropMode::FirstOrder { r_wire }
                },
                ..Default::default()
            };
            let arr = CrossbarArray::program(&w, 8, 8, 8, params, &mut Rng::new(1));
            arr.mvm(&x, &mut Rng::new(2), &mut EventLedger::default())[0]
        };
        let ideal = mk(0.0);
        let small = mk(1e-3); // |G|·R_path ~ 1e-9 — negligible
        let large = mk(200.0); // |G|·R_path ~ 2e-4·per-segment — visible
        assert!((small - ideal).abs() / ideal < 1e-6, "{small} vs {ideal}");
        assert!(large < ideal, "{large} !< {ideal}");
        assert!(large > 0.9 * ideal, "first-order should stay small here");
    }

    /// CrossbarJob round-trips through serde with defaults filled in.
    #[test]
    fn job_serde_defaults() {
        let job: CrossbarJob =
            serde_json::from_str(r#"{"rows": 64, "cols": 32, "n_queries": 10}"#).unwrap();
        assert_eq!(job.array_size, 256);
        assert_eq!(job.seed, 0);
        assert!(matches!(job.device.ir_drop, IrDropMode::Ideal));
        let s = serde_json::to_string(&job).unwrap();
        let back: CrossbarJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.rows, 64);
    }
}
