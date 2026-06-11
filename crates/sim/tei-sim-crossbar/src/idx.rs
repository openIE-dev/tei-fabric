//! Hand-rolled IDX (LeCun MNIST) parser + dataset loader.
//!
//! The IDX format (<https://yann.lecun.com/exdb/mnist/>) is a 4-byte magic
//! `[0x00, 0x00, dtype, ndims]` followed by `ndims` big-endian u32 dimension
//! sizes, then the raw data. We only need `dtype = 0x08` (unsigned byte),
//! which covers all four classic MNIST files.
//!
//! Files are expected unzipped in `$MNIST_DIR` (default
//! `~/.cache/tei-fabric/mnist/`) — run `scripts/fetch-mnist.sh` to populate.

use std::path::PathBuf;

/// Unsigned-byte dtype code in the IDX magic.
const DTYPE_U8: u8 = 0x08;

/// A parsed IDX file: dimension sizes plus the raw byte payload.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Idx {
    pub dims: Vec<usize>,
    pub data: Vec<u8>,
}

impl Idx {
    /// Serialize back to IDX bytes (test round-trips, tooling).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(4 + 4 * self.dims.len() + self.data.len());
        out.extend_from_slice(&[0, 0, DTYPE_U8, self.dims.len() as u8]);
        for &d in &self.dims {
            out.extend_from_slice(&(d as u32).to_be_bytes());
        }
        out.extend_from_slice(&self.data);
        out
    }
}

/// Parse an IDX byte buffer (unsigned-byte dtype only).
pub fn parse_idx(bytes: &[u8]) -> Result<Idx, String> {
    if bytes.len() >= 2 && bytes[0] == 0x1f && bytes[1] == 0x8b {
        return Err(
            "file is gzipped — run scripts/fetch-mnist.sh to download and gunzip MNIST".to_string(),
        );
    }
    if bytes.len() < 4 {
        return Err(format!("IDX header truncated: {} bytes", bytes.len()));
    }
    if bytes[0] != 0 || bytes[1] != 0 {
        return Err(format!(
            "bad IDX magic {:02x}{:02x} (expected 0000)",
            bytes[0], bytes[1]
        ));
    }
    if bytes[2] != DTYPE_U8 {
        return Err(format!(
            "unsupported IDX dtype 0x{:02x} (only 0x08 unsigned byte)",
            bytes[2]
        ));
    }
    let ndims = bytes[3] as usize;
    let header = 4 + 4 * ndims;
    if bytes.len() < header {
        return Err(format!(
            "IDX header truncated: {} bytes, need {header} for {ndims} dims",
            bytes.len()
        ));
    }
    let mut dims = Vec::with_capacity(ndims);
    for d in 0..ndims {
        let o = 4 + 4 * d;
        dims.push(
            u32::from_be_bytes([bytes[o], bytes[o + 1], bytes[o + 2], bytes[o + 3]]) as usize,
        );
    }
    let expected: usize = dims.iter().product();
    let data = &bytes[header..];
    if data.len() != expected {
        return Err(format!(
            "IDX payload is {} bytes, dims {:?} require {expected}",
            data.len(),
            dims
        ));
    }
    Ok(Idx {
        dims,
        data: data.to_vec(),
    })
}

/// Where MNIST lives: `$MNIST_DIR`, else `~/.cache/tei-fabric/mnist`.
pub fn mnist_dir() -> PathBuf {
    if let Ok(d) = std::env::var("MNIST_DIR") {
        return PathBuf::from(d);
    }
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/tei-fabric/mnist")
}

/// The four classic MNIST splits, images flattened row-major (n × 784 bytes).
#[derive(Debug, Clone)]
pub struct Dataset {
    pub train_images: Vec<u8>,
    pub train_labels: Vec<u8>,
    pub test_images: Vec<u8>,
    pub test_labels: Vec<u8>,
    /// Pixels per image (28 × 28 = 784).
    pub pixels: usize,
}

impl Dataset {
    pub fn n_train(&self) -> usize {
        self.train_labels.len()
    }

    pub fn n_test(&self) -> usize {
        self.test_labels.len()
    }

    /// Load all four files from [`mnist_dir`].
    pub fn load() -> Result<Self, String> {
        let dir = mnist_dir();
        let read = |name: &str| -> Result<Idx, String> {
            let path = dir.join(name);
            let bytes = std::fs::read(&path).map_err(|e| {
                format!(
                    "cannot read {} ({e}) — run scripts/fetch-mnist.sh",
                    path.display()
                )
            })?;
            parse_idx(&bytes).map_err(|e| format!("{}: {e}", path.display()))
        };

        let train_images = read("train-images-idx3-ubyte")?;
        let train_labels = read("train-labels-idx1-ubyte")?;
        let test_images = read("t10k-images-idx3-ubyte")?;
        let test_labels = read("t10k-labels-idx1-ubyte")?;

        for (idx, want_dims, name) in [
            (&train_images, 3, "train images"),
            (&test_images, 3, "test images"),
            (&train_labels, 1, "train labels"),
            (&test_labels, 1, "test labels"),
        ] {
            if idx.dims.len() != want_dims {
                return Err(format!(
                    "{name}: expected {want_dims}-d IDX, got dims {:?}",
                    idx.dims
                ));
            }
        }
        let pixels = train_images.dims[1] * train_images.dims[2];
        if train_images.dims[0] != train_labels.dims[0]
            || test_images.dims[0] != test_labels.dims[0]
        {
            return Err("image/label counts disagree".to_string());
        }

        Ok(Self {
            train_images: train_images.data,
            train_labels: train_labels.data,
            test_images: test_images.data,
            test_labels: test_labels.data,
            pixels,
        })
    }
}

/// Normalize u8 pixels to f32 in [0, 1].
pub fn to_f32(pixels: &[u8]) -> Vec<f32> {
    pixels.iter().map(|&p| p as f32 / 255.0).collect()
}
