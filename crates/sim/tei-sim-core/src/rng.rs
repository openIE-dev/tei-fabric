//! Deterministic PRNG — splitmix64 seeding + xoshiro256++ generation.
//!
//! Hand-rolled so every platform (including wasm32) produces bit-identical
//! streams for the same seed. xoshiro256++ passes BigCrush; splitmix64 is
//! the recommended seeder (Blackman & Vigna, <https://prng.di.unimi.it>).

/// splitmix64 — used to expand a single u64 seed into xoshiro state.
fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// xoshiro256++ generator.
#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    /// Cached second normal deviate from Box-Muller.
    spare_normal: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self {
            s,
            spare_normal: None,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[0].wrapping_add(s[3]).rotate_left(23).wrapping_add(s[0]);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1).
    #[inline]
    pub fn f64(&mut self) -> f64 {
        // 53 high bits → [0,1) with full double precision.
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        // Lemire-style rejection-free for our purposes (bias negligible at
        // sim scales, but do the widening-multiply trick anyway).
        ((self.next_u64() as u128 * n as u128) >> 64) as usize
    }

    /// Standard normal via Box-Muller (cached pair).
    pub fn normal(&mut self) -> f64 {
        if let Some(z) = self.spare_normal.take() {
            return z;
        }
        // Avoid u1 == 0.
        let u1 = loop {
            let u = self.f64();
            if u > 0.0 {
                break u;
            }
        };
        let u2 = self.f64();
        let r = (-2.0 * u1.ln()).sqrt();
        let (s, c) = (std::f64::consts::TAU * u2).sin_cos();
        self.spare_normal = Some(r * s);
        r * c
    }

    /// Exponential with rate λ.
    pub fn exponential(&mut self, lambda: f64) -> f64 {
        let u = loop {
            let u = self.f64();
            if u > 0.0 {
                break u;
            }
        };
        -u.ln() / lambda
    }

    /// Bernoulli with probability p.
    #[inline]
    pub fn bernoulli(&mut self, p: f64) -> bool {
        self.f64() < p
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mean and variance of the uniform stream within statistical bounds.
    #[test]
    fn uniform_moments() {
        let mut rng = Rng::new(42);
        let n = 200_000;
        let (mut sum, mut sumsq) = (0.0, 0.0);
        for _ in 0..n {
            let x = rng.f64();
            sum += x;
            sumsq += x * x;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        // mean → 1/2 (σ_mean ≈ 0.000645); var → 1/12.
        assert!((mean - 0.5).abs() < 0.005, "mean {mean}");
        assert!((var - 1.0 / 12.0).abs() < 0.005, "var {var}");
    }

    /// Box-Muller normal: mean 0, var 1, symmetric tails.
    #[test]
    fn normal_moments() {
        let mut rng = Rng::new(7);
        let n = 200_000;
        let (mut sum, mut sumsq) = (0.0, 0.0);
        for _ in 0..n {
            let z = rng.normal();
            sum += z;
            sumsq += z * z;
        }
        let mean = sum / n as f64;
        let var = sumsq / n as f64 - mean * mean;
        assert!(mean.abs() < 0.02, "mean {mean}");
        assert!((var - 1.0).abs() < 0.03, "var {var}");
    }

    /// Lag-1 autocorrelation of the uniform stream is ~0.
    #[test]
    fn low_autocorrelation() {
        let mut rng = Rng::new(123);
        let n = 100_000;
        let xs: Vec<f64> = (0..n).map(|_| rng.f64() - 0.5).collect();
        let num: f64 = xs.windows(2).map(|w| w[0] * w[1]).sum();
        let den: f64 = xs.iter().map(|x| x * x).sum();
        let rho = num / den;
        assert!(rho.abs() < 0.02, "lag-1 autocorrelation {rho}");
    }

    /// Determinism: same seed → identical stream.
    #[test]
    fn deterministic() {
        let mut a = Rng::new(99);
        let mut b = Rng::new(99);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }
}
