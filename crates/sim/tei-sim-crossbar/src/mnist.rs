//! MNIST accuracy-vs-noise demo (docs/SIM-ROADMAP.md §3.3 stretch goal):
//! train a 784→128→10 MLP in pure Rust, map both weight matrices onto the
//! noisy [`CrossbarArray`] machinery, and measure classification accuracy as
//! a function of device noise.
//!
//! Published anchor for direction/magnitude: NeuRRAM (Wan et al., "A
//! compute-in-memory chip based on resistive random-access memory", Nature
//! 608, 2022) reports MNIST at ~99% in software dropping to ~97–98% measured
//! on-chip — i.e. a sub-2% loss at realistic device noise, degrading toward
//! chance as noise grows. The ignored tests in `tests/mnist.rs` check the
//! same direction and magnitude.

use rayon::prelude::*;
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

use crate::idx::{self, Dataset};
use crate::mlp::{Mlp, TrainConfig, argmax};
use crate::{AdcParams, CrossbarArray, DeviceParams};

/// Per-image RNG stream-splitting constant (golden-ratio increment, same as
/// splitmix64's), so rayon-parallel evaluation is order-independent and
/// bit-deterministic.
pub const PER_IMAGE_STREAM: u64 = 0x9E37_79B9_7F4A_7C15;

/// Number of training images used to calibrate DAC/ADC ranges.
pub const CALIB_IMAGES: usize = 256;

/// Trained model + dataset, ready for digital or crossbar evaluation.
pub struct Pipeline {
    pub mlp: Mlp,
    pub data: Dataset,
    pub config: TrainConfig,
    /// Wall-clock training seconds (None when weights came from cache).
    pub train_seconds: Option<f64>,
}

impl Pipeline {
    /// Load MNIST, then load cached TEIMLP01 weights if present (keyed on
    /// the training config) or train from scratch and cache them under the
    /// MNIST cache dir.
    pub fn train_or_load(config: TrainConfig) -> Result<Self, String> {
        let data = Dataset::load()?;
        let cache = idx::mnist_dir().join(format!(
            "mlp-784x128x10-seed{}-e{}-b{}.teimlp",
            config.seed, config.epochs, config.batch
        ));
        if let Ok(mlp) = Mlp::load(&cache) {
            if mlp.n_in == data.pixels {
                return Ok(Self {
                    mlp,
                    data,
                    config,
                    train_seconds: None,
                });
            }
        }
        let images = idx::to_f32(&data.train_images);
        let mut mlp = Mlp::new(data.pixels, 128, 10, &mut Rng::new(config.seed));
        let t0 = std::time::Instant::now();
        mlp.train(&images, &data.train_labels, &config);
        let train_seconds = t0.elapsed().as_secs_f64();
        mlp.save(&cache)?;
        Ok(Self {
            mlp,
            data,
            config,
            train_seconds: Some(train_seconds),
        })
    }

    /// Digital f32 test-set accuracy. Rayon-parallel; the reduction is an
    /// order-independent integer count, so the result is deterministic.
    pub fn digital_accuracy(&self) -> f64 {
        let n = self.data.n_test();
        let correct: usize = (0..n)
            .into_par_iter()
            .filter(|&i| {
                let x = idx::to_f32(
                    &self.data.test_images[i * self.data.pixels..(i + 1) * self.data.pixels],
                );
                self.mlp.predict(&x) == self.data.test_labels[i] as usize
            })
            .count();
        correct as f64 / n as f64
    }
}

/// The MLP with both matmuls programmed onto noisy crossbar arrays.
/// Biases and the ReLU stay digital (they live in the periphery on real
/// CIM parts — NeuRRAM keeps neuron nonlinearities in mixed-signal
/// peripheral circuits too).
pub struct CrossbarMlp {
    l1: CrossbarArray,
    l2: CrossbarArray,
    b1: Vec<f64>,
    b2: Vec<f64>,
    seed: u64,
}

impl CrossbarMlp {
    /// Program both weight matrices onto crossbar tiles with the given
    /// noise levels, 8-bit DAC/ADC, and per-layer ranges calibrated from
    /// the tile partial sums of [`CALIB_IMAGES`] training images.
    ///
    /// Programming noise is drawn from `Rng::new(seed)`; read noise during
    /// [`CrossbarMlp::predict`] is drawn from a per-image stream
    /// `Rng::new(seed ^ i·PER_IMAGE_STREAM)`, so evaluation is deterministic
    /// under rayon.
    pub fn program(
        mlp: &Mlp,
        data: &Dataset,
        sigma_prog: f64,
        sigma_read: f64,
        array_size: usize,
        seed: u64,
    ) -> Self {
        let w1: Vec<f64> = mlp.w1.iter().map(|&w| w as f64).collect();
        let w2: Vec<f64> = mlp.w2.iter().map(|&w| w as f64).collect();

        // ── Calibration: max |tile partial sum| and max input per layer over
        // the first CALIB_IMAGES training images (clean digital math, same
        // row tiling the crossbar uses). The ADC fires once per row-tile ×
        // output column, so its range must cover tile partials, not totals.
        let n_calib = CALIB_IMAGES.min(data.n_train());
        let mut max_p1 = 0.0f64; // layer-1 tile partials
        let mut max_p2 = 0.0f64; // layer-2 tile partials
        let mut max_h = 0.0f64; // layer-2 inputs (hidden activations)
        for i in 0..n_calib {
            let px = &data.train_images[i * data.pixels..(i + 1) * data.pixels];
            let x: Vec<f64> = px.iter().map(|&p| p as f64 / 255.0).collect();
            let (h, p1) = tile_forward(&x, &w1, mlp.n_hidden, array_size);
            max_p1 = max_p1.max(p1);
            let h: Vec<f64> = h
                .iter()
                .zip(&mlp.b1)
                .map(|(&v, &b)| (v + b as f64).max(0.0))
                .collect();
            max_h = max_h.max(h.iter().fold(0.0f64, |m, &v| m.max(v)));
            let (_, p2) = tile_forward(&h, &w2, mlp.n_out, array_size);
            max_p2 = max_p2.max(p2);
        }
        // Headroom over the calibration max — the test set has unseen
        // extremes; mild clipping of rare outliers is part of the model.
        let headroom = 1.1;

        let params = |input_range: f64, adc_range: f64| DeviceParams {
            sigma_prog,
            sigma_read,
            dac_bits: Some(8),
            input_range,
            adc: Some(AdcParams {
                bits: 8,
                range: adc_range * headroom,
                inl_lsb: 0.0,
            }),
            ..Default::default()
        };

        let mut prog_rng = Rng::new(seed);
        let l1 = CrossbarArray::program(
            &w1,
            mlp.n_in,
            mlp.n_hidden,
            array_size,
            params(1.0, max_p1),
            &mut prog_rng,
        );
        let l2 = CrossbarArray::program(
            &w2,
            mlp.n_hidden,
            mlp.n_out,
            array_size,
            params(max_h * headroom, max_p2),
            &mut prog_rng,
        );

        Self {
            l1,
            l2,
            b1: mlp.b1.iter().map(|&b| b as f64).collect(),
            b2: mlp.b2.iter().map(|&b| b as f64).collect(),
            seed,
        }
    }

    /// Classify one image (f64 pixels in [0,1]) through both noisy crossbar
    /// layers. `image_index` selects the deterministic per-image read-noise
    /// stream. Returns (predicted class, per-image event ledger).
    pub fn predict(&self, x: &[f64], image_index: u64) -> (usize, EventLedger) {
        let mut rng = Rng::new(self.seed ^ image_index.wrapping_mul(PER_IMAGE_STREAM));
        let mut ledger = EventLedger::default();
        let y1 = self.l1.mvm(x, &mut rng, &mut ledger);
        let h: Vec<f64> = y1
            .iter()
            .zip(&self.b1)
            .map(|(&v, &b)| (v + b).max(0.0))
            .collect();
        let y2 = self.l2.mvm(&h, &mut rng, &mut ledger);
        let z: Vec<f64> = y2.iter().zip(&self.b2).map(|(&v, &b)| v + b).collect();
        (argmax(&z), ledger)
    }

    /// Accuracy over the first `n` test images (rayon-parallel,
    /// deterministic — per-image RNG streams + order-independent count).
    pub fn accuracy(&self, data: &Dataset, n: usize) -> f64 {
        let n = n.min(data.n_test());
        let correct: usize = (0..n)
            .into_par_iter()
            .filter(|&i| {
                let px = &data.test_images[i * data.pixels..(i + 1) * data.pixels];
                let x: Vec<f64> = px.iter().map(|&p| p as f64 / 255.0).collect();
                self.predict(&x, i as u64).0 == data.test_labels[i] as usize
            })
            .count();
        correct as f64 / n as f64
    }

    /// Predictions over the first `n` test images, in image order.
    pub fn predictions(&self, data: &Dataset, n: usize) -> Vec<usize> {
        let n = n.min(data.n_test());
        (0..n)
            .into_par_iter()
            .map(|i| {
                let px = &data.test_images[i * data.pixels..(i + 1) * data.pixels];
                let x: Vec<f64> = px.iter().map(|&p| p as f64 / 255.0).collect();
                self.predict(&x, i as u64).0
            })
            .collect()
    }
}

/// Digital tile-partial forward used for ADC calibration: returns the column
/// totals and the max |partial sum| over row tiles of `array_size` rows —
/// the exact quantity the crossbar ADC digitizes.
fn tile_forward(x: &[f64], w: &[f64], cols: usize, array_size: usize) -> (Vec<f64>, f64) {
    let rows = x.len();
    let mut y = vec![0.0f64; cols];
    let mut max_partial = 0.0f64;
    let mut row0 = 0;
    while row0 < rows {
        let trows = array_size.min(rows - row0);
        let mut partial = vec![0.0f64; cols];
        for i in 0..trows {
            let xi = x[row0 + i];
            if xi == 0.0 {
                continue;
            }
            let wrow = &w[(row0 + i) * cols..(row0 + i + 1) * cols];
            for (pj, &wij) in partial.iter_mut().zip(wrow) {
                *pj += xi * wij;
            }
        }
        for (yj, &pj) in y.iter_mut().zip(&partial) {
            max_partial = max_partial.max(pj.abs());
            *yj += pj;
        }
        row0 += trows;
    }
    (y, max_partial)
}

/// Sweep accuracy vs device noise σ (applied to both σ_prog and σ_read) over
/// the first `n_images` test images. Returns (σ, accuracy) pairs.
pub fn noise_sweep(pipeline: &Pipeline, sigmas: &[f64], n_images: usize) -> Vec<(f64, f64)> {
    sigmas
        .iter()
        .map(|&s| {
            let xb = CrossbarMlp::program(&pipeline.mlp, &pipeline.data, s, s, 256, 7);
            (s, xb.accuracy(&pipeline.data, n_images))
        })
        .collect()
}

/// The canonical sweep points for the demo.
pub const SWEEP_SIGMAS: [f64; 6] = [0.0, 0.01, 0.03, 0.05, 0.1, 0.2];

// ─── executor: the accuracy axis of the calibration loop ────────────────────

/// Job for the MNIST accuracy column: sweep test-set accuracy across device
/// noise σ (applied to both σ_prog and σ_read). `operating_sigma` is the
/// point whose measured loss-vs-digital feeds the in-memory dialect's
/// `accuracy_loss` via the calibration loop.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct MnistJob {
    #[serde(default = "default_sweep_sigmas")]
    pub sigmas: Vec<f64>,
    /// Test images evaluated per sweep point.
    #[serde(default = "default_subset")]
    pub subset: usize,
    #[serde(default = "default_operating_sigma")]
    pub operating_sigma: f64,
    /// Training seed (weights are cached per config under the MNIST dir).
    #[serde(default = "default_train_seed")]
    pub seed: u64,
}

fn default_sweep_sigmas() -> Vec<f64> {
    SWEEP_SIGMAS.to_vec()
}
fn default_subset() -> usize {
    2000
}
fn default_operating_sigma() -> f64 {
    0.03
}
fn default_train_seed() -> u64 {
    42
}

pub struct MnistExecutor;

impl tei_sim_core::exec::Executor for MnistExecutor {
    type Job = MnistJob;

    fn execute(
        &self,
        job: &MnistJob,
        on_progress: &mut dyn FnMut(tei_sim_core::exec::Progress),
    ) -> tei_sim_core::exec::ExecutionResult {
        use tei_sim_core::exec::{ExecutionResult, Progress};
        let t0 = std::time::Instant::now();
        let mut ledger = EventLedger::default();

        let pipeline = match Pipeline::train_or_load(TrainConfig {
            seed: job.seed,
            ..TrainConfig::default()
        }) {
            Ok(p) => p,
            Err(e) => {
                return ExecutionResult {
                    ledger,
                    outputs: serde_json::json!({ "error": e }),
                };
            }
        };
        let digital = pipeline.digital_accuracy();
        on_progress(Progress {
            fraction: 0.1,
            metrics: serde_json::json!({
                "stage": "trained",
                "digital_accuracy": digital,
                "train_seconds": pipeline.train_seconds,
            }),
        });

        let mut sigmas = job.sigmas.clone();
        if !sigmas
            .iter()
            .any(|s| (s - job.operating_sigma).abs() < 1e-12)
        {
            sigmas.push(job.operating_sigma);
        }
        sigmas.sort_by(f64::total_cmp);
        sigmas.dedup();

        let n = job.subset.min(pipeline.data.n_test()).max(1);
        // Per-image MAC/ADC counts are data-independent (fixed network), so
        // one instrumented predict prices the whole evaluation exactly.
        let px = &pipeline.data.test_images[0..pipeline.data.pixels];
        let x0: Vec<f64> = px.iter().map(|&p| p as f64 / 255.0).collect();

        let mut curve = Vec::with_capacity(sigmas.len());
        let mut measured_loss = 0.0;
        for (i, &s) in sigmas.iter().enumerate() {
            let xb = CrossbarMlp::program(&pipeline.mlp, &pipeline.data, s, s, 256, 7);
            let (_, l1) = xb.predict(&x0, 0);
            ledger.macs += l1.macs * n as u64;
            ledger.adc_samples += l1.adc_samples * n as u64;
            let acc = xb.accuracy(&pipeline.data, n);
            let loss = (digital - acc).max(0.0);
            if (s - job.operating_sigma).abs() < 1e-12 {
                measured_loss = loss;
            }
            curve.push(serde_json::json!({
                "sigma": s, "accuracy": acc, "loss_vs_digital": loss,
            }));
            on_progress(Progress {
                fraction: 0.1 + 0.9 * (i + 1) as f64 / sigmas.len() as f64,
                metrics: serde_json::json!({ "sigma": s, "accuracy": acc }),
            });
        }

        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());
        ExecutionResult {
            ledger,
            outputs: serde_json::json!({
                "digital_accuracy": digital,
                "curve": curve,
                "operating_sigma": job.operating_sigma,
                "measured_accuracy_loss": measured_loss,
                "n_images": n,
                "train_seconds": pipeline.train_seconds,
            }),
        }
    }
}
