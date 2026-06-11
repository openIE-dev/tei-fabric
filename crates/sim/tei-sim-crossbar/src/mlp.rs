//! Minimal f32 MLP (784→128→10, ReLU, softmax cross-entropy) trained with
//! SGD + momentum — the pure-Rust trainer behind the MNIST accuracy demo
//! (docs/SIM-ROADMAP.md §3.3 stretch goal).
//!
//! Everything is deterministic: He initialization, Fisher–Yates epoch
//! shuffles, and gradient accumulation all draw from
//! [`tei_sim_core::rng::Rng`] in a fixed order, so a given seed reproduces
//! bit-identical weights on every platform.

use tei_sim_core::rng::Rng;

/// Binary weight-file magic (version-stamped).
pub const MAGIC: &[u8; 8] = b"TEIMLP01";

/// Fully-connected 2-layer perceptron. Weight layout matches the crossbar
/// convention: row-major with rows = inputs, cols = outputs, i.e.
/// `w1[i * n_hidden + j]` connects input `i` to hidden unit `j`.
#[derive(Debug, Clone, PartialEq)]
pub struct Mlp {
    pub n_in: usize,
    pub n_hidden: usize,
    pub n_out: usize,
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

/// Gradient buffers mirroring [`Mlp`].
#[derive(Debug, Clone)]
pub struct Grads {
    pub w1: Vec<f32>,
    pub b1: Vec<f32>,
    pub w2: Vec<f32>,
    pub b2: Vec<f32>,
}

impl Grads {
    pub fn zeros(m: &Mlp) -> Self {
        Self {
            w1: vec![0.0; m.w1.len()],
            b1: vec![0.0; m.b1.len()],
            w2: vec![0.0; m.w2.len()],
            b2: vec![0.0; m.b2.len()],
        }
    }

    pub fn clear(&mut self) {
        self.w1.fill(0.0);
        self.b1.fill(0.0);
        self.w2.fill(0.0);
        self.b2.fill(0.0);
    }
}

/// Training hyperparameters.
#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub epochs: usize,
    pub batch: usize,
    /// Learning rate (applied to batch-averaged gradients).
    pub lr: f32,
    /// Classical momentum coefficient.
    pub momentum: f32,
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 5,
            batch: 32,
            lr: 0.05,
            momentum: 0.9,
            seed: 42,
        }
    }
}

impl Mlp {
    /// He-normal initialization (Var = 2/fan_in — the ReLU-correct scaling,
    /// He et al. 2015), biases zero, weights drawn from `rng` in a fixed
    /// order (w1 row-major, then w2).
    pub fn new(n_in: usize, n_hidden: usize, n_out: usize, rng: &mut Rng) -> Self {
        let mut draw = |fan_in: usize, n: usize| -> Vec<f32> {
            let s = (2.0 / fan_in as f64).sqrt();
            (0..n).map(|_| (s * rng.normal()) as f32).collect()
        };
        Self {
            n_in,
            n_hidden,
            n_out,
            w1: draw(n_in, n_in * n_hidden),
            b1: vec![0.0; n_hidden],
            w2: draw(n_hidden, n_hidden * n_out),
            b2: vec![0.0; n_out],
        }
    }

    /// Hidden layer: ReLU(x·W1 + b1).
    pub fn hidden(&self, x: &[f32]) -> Vec<f32> {
        debug_assert_eq!(x.len(), self.n_in);
        let mut h = self.b1.clone();
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue; // MNIST is ~80% zero pixels.
            }
            let row = &self.w1[i * self.n_hidden..(i + 1) * self.n_hidden];
            for (hj, &w) in h.iter_mut().zip(row) {
                *hj += xi * w;
            }
        }
        for hj in &mut h {
            if *hj < 0.0 {
                *hj = 0.0;
            }
        }
        h
    }

    /// Output logits from a hidden activation: h·W2 + b2.
    pub fn logits_from_hidden(&self, h: &[f32]) -> Vec<f32> {
        let mut z = self.b2.clone();
        for (j, &hj) in h.iter().enumerate() {
            if hj == 0.0 {
                continue;
            }
            let row = &self.w2[j * self.n_out..(j + 1) * self.n_out];
            for (zk, &w) in z.iter_mut().zip(row) {
                *zk += hj * w;
            }
        }
        z
    }

    pub fn logits(&self, x: &[f32]) -> Vec<f32> {
        self.logits_from_hidden(&self.hidden(x))
    }

    pub fn predict(&self, x: &[f32]) -> usize {
        argmax(&self.logits(x))
    }

    /// Forward + backward for one sample; accumulates gradients into `g`
    /// and returns the cross-entropy loss −log p(label).
    pub fn backprop(&self, x: &[f32], label: usize, g: &mut Grads) -> f32 {
        let h = self.hidden(x);
        let z = self.logits_from_hidden(&h);
        let p = softmax(&z);
        let loss = -(p[label].max(f32::MIN_POSITIVE)).ln();

        // dL/dz = p − onehot(label).
        let mut dz = p;
        dz[label] -= 1.0;

        // Layer 2: dW2[j,k] = h_j·dz_k, db2 = dz, dh = W2·dz.
        let mut dh = vec![0.0f32; self.n_hidden];
        for (j, &hj) in h.iter().enumerate() {
            let wrow = &self.w2[j * self.n_out..(j + 1) * self.n_out];
            let grow = &mut g.w2[j * self.n_out..(j + 1) * self.n_out];
            let mut acc = 0.0;
            for k in 0..self.n_out {
                grow[k] += hj * dz[k];
                acc += wrow[k] * dz[k];
            }
            dh[j] = acc;
        }
        for (gb, &d) in g.b2.iter_mut().zip(&dz) {
            *gb += d;
        }

        // ReLU': zero where the pre-activation was clipped (h == 0).
        for (dhj, &hj) in dh.iter_mut().zip(&h) {
            if hj == 0.0 {
                *dhj = 0.0;
            }
        }

        // Layer 1: dW1[i,j] = x_i·dh_j, db1 = dh.
        for (i, &xi) in x.iter().enumerate() {
            if xi == 0.0 {
                continue;
            }
            let grow = &mut g.w1[i * self.n_hidden..(i + 1) * self.n_hidden];
            for (gw, &d) in grow.iter_mut().zip(&dh) {
                *gw += xi * d;
            }
        }
        for (gb, &d) in g.b1.iter_mut().zip(&dh) {
            *gb += d;
        }
        loss
    }

    /// Mini-batch SGD + momentum over `images` (flattened n×n_in, f32 in
    /// [0,1]) / `labels`. Sequential and fully deterministic (epoch order is
    /// a Fisher–Yates shuffle drawn from `Rng::new(cfg.seed)`). Returns the
    /// mean training loss of the final epoch.
    pub fn train(&mut self, images: &[f32], labels: &[u8], cfg: &TrainConfig) -> f32 {
        let n = labels.len();
        assert_eq!(images.len(), n * self.n_in);
        let mut rng = Rng::new(cfg.seed);
        let mut order: Vec<usize> = (0..n).collect();
        let mut g = Grads::zeros(self);
        let mut v = Grads::zeros(self); // momentum velocity
        let mut last_epoch_loss = 0.0;

        for _epoch in 0..cfg.epochs {
            // Fisher–Yates, deterministic.
            for i in (1..n).rev() {
                let j = rng.below(i + 1);
                order.swap(i, j);
            }
            let mut epoch_loss = 0.0f64;
            for batch in order.chunks(cfg.batch) {
                g.clear();
                for &s in batch {
                    let x = &images[s * self.n_in..(s + 1) * self.n_in];
                    epoch_loss += self.backprop(x, labels[s] as usize, &mut g) as f64;
                }
                let scale = cfg.lr / batch.len() as f32;
                let step = |w: &mut [f32], vel: &mut [f32], grad: &[f32]| {
                    for ((wi, vi), &gi) in w.iter_mut().zip(vel.iter_mut()).zip(grad) {
                        *vi = cfg.momentum * *vi - scale * gi;
                        *wi += *vi;
                    }
                };
                step(&mut self.w1, &mut v.w1, &g.w1);
                step(&mut self.b1, &mut v.b1, &g.b1);
                step(&mut self.w2, &mut v.w2, &g.w2);
                step(&mut self.b2, &mut v.b2, &g.b2);
            }
            last_epoch_loss = (epoch_loss / n as f64) as f32;
        }
        last_epoch_loss
    }

    // ───────────────────────── persistence (TEIMLP01) ─────────────────────────

    /// Serialize: magic, u32-LE dims, then w1/b1/w2/b2 as f32 LE.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(
            8 + 12 + 4 * (self.w1.len() + self.b1.len() + self.w2.len() + self.b2.len()),
        );
        out.extend_from_slice(MAGIC);
        for d in [self.n_in, self.n_hidden, self.n_out] {
            out.extend_from_slice(&(d as u32).to_le_bytes());
        }
        for v in [&self.w1, &self.b1, &self.w2, &self.b2] {
            for &x in v.iter() {
                out.extend_from_slice(&x.to_le_bytes());
            }
        }
        out
    }

    pub fn from_bytes(bytes: &[u8]) -> Result<Self, String> {
        if bytes.len() < 20 || &bytes[..8] != MAGIC {
            return Err("not a TEIMLP01 weight file".to_string());
        }
        let dim = |o: usize| {
            u32::from_le_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize
        };
        let (n_in, n_hidden, n_out) = (dim(8), dim(12), dim(16));
        let counts = [n_in * n_hidden, n_hidden, n_hidden * n_out, n_out];
        let total: usize = counts.iter().sum();
        if bytes.len() != 20 + 4 * total {
            return Err(format!(
                "TEIMLP01 payload is {} bytes, dims {n_in}×{n_hidden}×{n_out} require {}",
                bytes.len() - 20,
                4 * total
            ));
        }
        let mut off = 20;
        let mut take = |count: usize| -> Vec<f32> {
            let v = bytes[off..off + 4 * count]
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            off += 4 * count;
            v
        };
        Ok(Self {
            n_in,
            n_hidden,
            n_out,
            w1: take(counts[0]),
            b1: take(counts[1]),
            w2: take(counts[2]),
            b2: take(counts[3]),
        })
    }

    pub fn save(&self, path: &std::path::Path) -> Result<(), String> {
        std::fs::write(path, self.to_bytes()).map_err(|e| format!("{}: {e}", path.display()))
    }

    pub fn load(path: &std::path::Path) -> Result<Self, String> {
        let bytes = std::fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
        Self::from_bytes(&bytes)
    }
}

/// Numerically-stable softmax.
pub fn softmax(z: &[f32]) -> Vec<f32> {
    let m = z.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let e: Vec<f32> = z.iter().map(|&v| (v - m).exp()).collect();
    let s: f32 = e.iter().sum();
    e.iter().map(|&v| v / s).collect()
}

/// Index of the maximum element (first on ties).
pub fn argmax<T: PartialOrd + Copy>(v: &[T]) -> usize {
    let mut best = 0;
    for (i, &x) in v.iter().enumerate().skip(1) {
        if x > v[best] {
            best = i;
        }
    }
    best
}
