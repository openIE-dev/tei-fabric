//! Linux / GPU measured meters: RAPL (Intel/AMD package energy counter) and
//! NVML (NVIDIA GPU energy). Both are `Measured` tier — real energy counters,
//! the digital-baseline analog of macmon on the Mac. They activate only on the
//! hardware they describe; everywhere else they report unavailable and
//! [`crate::detect_meter`] falls through.
//!
//! Ported from `invisible-infrastructure`'s `inv-energy` (`rapl.rs`,
//! `nvml.rs`), kept dependency-light: RAPL is pure `std` sysfs reads; NVML is
//! behind the optional `nvml` feature so the default build never pulls
//! `nvml-wrapper`.

use crate::{Acquisition, HostMeter};
use std::time::Instant;

// ── RAPL (Intel/AMD, Linux) ─────────────────────────────────────────────

const RAPL_ENERGY_PATH: &str = "/sys/class/powercap/intel-rapl:0/energy_uj";
const RAPL_RANGE_PATH: &str = "/sys/class/powercap/intel-rapl:0/max_energy_range_uj";

/// Intel/AMD RAPL meter: reads the package `energy_uj` counter before and
/// after the run and takes the delta — a direct energy measurement, not a
/// power estimate. Linux-only; the counter is often root-readable only, so
/// availability probes a real read (a non-root agent falls back gracefully).
pub struct RaplMeter {
    available: bool,
}

impl Default for RaplMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl RaplMeter {
    pub fn new() -> Self {
        Self {
            available: cfg!(target_os = "linux") && Self::read_uj().is_some(),
        }
    }

    /// Current package energy counter (microjoules), if readable + parseable.
    fn read_uj() -> Option<u64> {
        std::fs::read_to_string(RAPL_ENERGY_PATH)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    }

    /// Counter wrap range (microjoules), for delta correction.
    fn range_uj() -> Option<u64> {
        std::fs::read_to_string(RAPL_RANGE_PATH)
            .ok()?
            .trim()
            .parse::<u64>()
            .ok()
    }
}

impl HostMeter for RaplMeter {
    fn name(&self) -> &str {
        "rapl"
    }
    fn available(&self) -> bool {
        self.available
    }
    fn acquisition(&self) -> Acquisition {
        Acquisition::Measured
    }

    fn measure(&self, f: Box<dyn FnOnce() + '_>) -> (f64, f64, u32) {
        let e0 = Self::read_uj();
        let t = Instant::now();
        f();
        let dur = t.elapsed().as_secs_f64();
        let e1 = Self::read_uj();
        match (e0, e1) {
            (Some(a), Some(b)) => {
                // The counter wraps at max_energy_range_uj.
                let delta_uj = if b >= a {
                    b - a
                } else {
                    Self::range_uj().map(|r| r.saturating_sub(a) + b).unwrap_or(0)
                };
                (delta_uj as f64 / 1_000_000.0, dur, 0)
            }
            _ => (0.0, dur, 0),
        }
    }
}

// ── NVML (NVIDIA GPU) — optional `nvml` feature ─────────────────────────

/// NVIDIA GPU meter (NVML). With the `nvml` feature and a GPU present, reads
/// the driver's cumulative energy counter (delta over the run) summed across
/// devices, falling back to a 2-point power×time integration. Without the
/// feature it is always unavailable.
pub struct NvmlMeter {
    #[cfg(feature = "nvml")]
    nvml: Option<nvml_wrapper::Nvml>,
    available: bool,
}

impl Default for NvmlMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl NvmlMeter {
    pub fn new() -> Self {
        #[cfg(feature = "nvml")]
        {
            match nvml_wrapper::Nvml::init() {
                Ok(nvml) => {
                    let has_gpu = nvml.device_count().unwrap_or(0) > 0;
                    NvmlMeter { nvml: Some(nvml), available: has_gpu }
                }
                Err(_) => NvmlMeter { nvml: None, available: false },
            }
        }
        #[cfg(not(feature = "nvml"))]
        {
            NvmlMeter { available: false }
        }
    }

    /// Cumulative GPU energy across all devices (joules), if the driver
    /// exposes the counter.
    #[cfg(feature = "nvml")]
    fn total_energy_j(&self) -> Option<f64> {
        let nvml = self.nvml.as_ref()?;
        let n = nvml.device_count().ok()?;
        let mut mj: u128 = 0;
        for i in 0..n {
            let dev = nvml.device_by_index(i).ok()?;
            mj += dev.total_energy_consumption().ok()? as u128;
        }
        Some(mj as f64 / 1000.0)
    }

    /// Instantaneous GPU power across all devices (watts).
    #[cfg(feature = "nvml")]
    fn total_power_w(&self) -> Option<f64> {
        let nvml = self.nvml.as_ref()?;
        let n = nvml.device_count().ok()?;
        let mut mw: u64 = 0;
        for i in 0..n {
            let dev = nvml.device_by_index(i).ok()?;
            mw += dev.power_usage().ok()? as u64;
        }
        Some(mw as f64 / 1000.0)
    }
}

impl HostMeter for NvmlMeter {
    fn name(&self) -> &str {
        "nvml"
    }
    fn available(&self) -> bool {
        self.available
    }
    fn acquisition(&self) -> Acquisition {
        Acquisition::Measured
    }

    #[cfg(feature = "nvml")]
    fn watts(&self) -> Option<f64> {
        self.total_power_w()
    }

    #[cfg(feature = "nvml")]
    fn measure(&self, f: Box<dyn FnOnce() + '_>) -> (f64, f64, u32) {
        // Prefer the energy counter (a true delta); fall back to power×time.
        let e0 = self.total_energy_j();
        let p0 = self.total_power_w().unwrap_or(0.0);
        let t = Instant::now();
        f();
        let dur = t.elapsed().as_secs_f64();
        if let (Some(a), Some(b)) = (e0, self.total_energy_j()) {
            if b >= a {
                return (b - a, dur, 0);
            }
        }
        let p1 = self.total_power_w().unwrap_or(p0);
        ((p0 + p1) / 2.0 * dur, dur, 0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rapl_reports_measured_tier_and_safe_off_linux() {
        let m = RaplMeter::new();
        assert_eq!(m.name(), "rapl");
        assert_eq!(m.acquisition(), Acquisition::Measured);
        // Off Linux it must be unavailable (no panic, no fake reading).
        if !cfg!(target_os = "linux") {
            assert!(!m.available());
        }
    }

    #[test]
    fn nvml_unavailable_without_feature_or_gpu() {
        let m = NvmlMeter::new();
        assert_eq!(m.name(), "nvml");
        assert_eq!(m.acquisition(), Acquisition::Measured);
        #[cfg(not(feature = "nvml"))]
        assert!(!m.available());
    }
}
