//! Host-side energy meter for the cloud runtime.
//!
//! `tei-serve` runs on a real machine (today: an Apple-Silicon Mac Studio in
//! prod). When the cloud runtime executes a workload, the *simulator* reports
//! what the work would cost on the **target substrate** (a cost-surface model,
//! `Table`-tier). This crate reports the *other* number: the **real wall-power
//! the host machine actually burned** running that work.
//!
//! Those two numbers must never be confused, so every reading carries
//! **two-axis provenance**:
//!
//! - [`Acquisition`] — *how* the joules were obtained: `Modeled` (cost
//!   surface), `Estimated` (load→watts envelope, what Apple Silicon gives us
//!   without PMC access), or `Measured` (direct power counters — IOReport/PMC,
//!   the documented upgrade path).
//! - [`Identity`] — *what* the joules describe: a [`Identity::Target`] TEI
//!   substrate, or a [`Identity::StandIn`] (the Mac Studio is a stand-in for
//!   future TEI silicon).
//!
//! A figure is only honest "this is what it costs on TEI hardware" when it is
//! `Measured` × `Target`. The Mac Studio gives `Estimated` × `StandIn`: a real
//! machine, an honest tier, not a target prediction.
//!
//! ## Credit
//! The Apple-Silicon load→watts method is ported from
//! `invisible-infrastructure`'s `inv-energy` crate (`apple.rs`,
//! `AppleSiliconMeter`). It is reproduced here — rather than depended on across
//! workspaces — so the prod fabric server stays self-contained and
//! independently deployable.

use serde::Serialize;

/// How a joule figure was obtained — the fidelity of acquisition, weakest to
/// strongest is `Modeled` < `Estimated` < `Measured`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Acquisition {
    /// Predicted by a cost model (the cost surface / dialect tables). No
    /// hardware ran. Mirrors the ledger's `JoulesSource::Table`.
    Modeled,
    /// Derived from a coarse activity→power envelope (e.g. Apple-Silicon
    /// load average → watts). A real run, but power is inferred, not counted.
    Estimated,
    /// Read from hardware power/energy counters (RAPL, NVML, IOReport/PMC,
    /// an INA228 shunt). The strongest tier.
    Measured,
}

/// What a joule figure *describes* — the substrate it is attributed to.
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Identity {
    /// The figure describes the actual TEI target substrate.
    Target { name: String },
    /// The figure describes a stand-in running the work in the target's place
    /// (the Mac Studio, a rented FPGA/GPU, a specialized-fabric cloud). It is
    /// not the target; a correction is required before it estimates the target.
    StandIn { name: String },
}

impl Identity {
    pub fn stand_in(name: impl Into<String>) -> Self {
        Identity::StandIn { name: name.into() }
    }
    pub fn target(name: impl Into<String>) -> Self {
        Identity::Target { name: name.into() }
    }
}

/// An attested host-energy reading for one measured run.
#[derive(Debug, Clone, Serialize)]
pub struct RunReceipt {
    /// Real energy the host burned during the run (joules).
    pub host_joules: f64,
    /// Mean power over the run (watts).
    pub avg_watts: f64,
    /// Wall-clock duration of the measured region (seconds).
    pub duration_s: f64,
    /// How `host_joules` was obtained.
    pub acquisition: Acquisition,
    /// What `host_joules` describes (the stand-in machine, here).
    pub identity: Identity,
    /// The meter that produced the reading (e.g. `"apple-silicon"`).
    pub meter: String,
}

/// A host energy meter: read cumulative joules + instantaneous watts now.
pub trait HostMeter {
    /// Stable meter name (for the receipt + audit trail).
    fn name(&self) -> &str;
    /// True when this meter can produce real readings on this machine.
    fn available(&self) -> bool;
    /// The acquisition tier this meter achieves.
    fn acquisition(&self) -> Acquisition;
    /// Current instantaneous power draw, watts. `None` if unavailable.
    fn watts(&self) -> Option<f64>;
}

/// Apple-Silicon meter: estimates package power from the 1-minute load average
/// scaled by core count, across an idle→peak envelope. This is the
/// `Estimated` tier — a real machine, inferred power. Upgrading to `Measured`
/// means reading IOReport/PMC energy counters (no extra privileges) in
/// [`AppleSiliconMeter::watts`].
///
/// Ported from `inv-energy::apple::AppleSiliconMeter`.
pub struct AppleSiliconMeter {
    available: bool,
}

impl Default for AppleSiliconMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl AppleSiliconMeter {
    /// Idle package power, watts. Conservative Apple-Silicon desktop floor.
    const IDLE_WATTS: f64 = 8.0;
    /// Peak package power, watts. Conservative ceiling for the load-scaled
    /// envelope (not a measured TDP).
    const PEAK_WATTS: f64 = 40.0;

    pub fn new() -> Self {
        Self {
            available: cfg!(target_os = "macos") && cfg!(target_arch = "aarch64"),
        }
    }

    #[cfg(target_os = "macos")]
    fn read_watts(&self) -> Option<f64> {
        let mut load: [f64; 3] = [0.0; 3];
        // SAFETY: getloadavg writes up to 3 f64s into the provided buffer.
        let n = unsafe { libc::getloadavg(load.as_mut_ptr(), 3) };
        if n < 1 {
            return None;
        }
        let cores = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1) as f64;
        let utilization = (load[0] / cores).clamp(0.0, 1.0);
        Some(Self::IDLE_WATTS + (Self::PEAK_WATTS - Self::IDLE_WATTS) * utilization)
    }
}

impl HostMeter for AppleSiliconMeter {
    fn name(&self) -> &str {
        "apple-silicon"
    }
    fn available(&self) -> bool {
        self.available
    }
    fn acquisition(&self) -> Acquisition {
        Acquisition::Estimated
    }

    #[cfg(target_os = "macos")]
    fn watts(&self) -> Option<f64> {
        if self.available { self.read_watts() } else { None }
    }
    #[cfg(not(target_os = "macos"))]
    fn watts(&self) -> Option<f64> {
        None
    }
}

/// A meter that reports a fixed watt figure regardless of activity. Used as the
/// portable fallback on hosts where no real meter is available, so the cloud
/// runtime still produces a (clearly `Estimated`) receipt off-Mac.
pub struct FixedWattsMeter {
    watts: f64,
    name: &'static str,
}

impl FixedWattsMeter {
    pub fn new(watts: f64) -> Self {
        Self { watts, name: "fixed-tdp" }
    }
}

impl HostMeter for FixedWattsMeter {
    fn name(&self) -> &str {
        self.name
    }
    fn available(&self) -> bool {
        true
    }
    fn acquisition(&self) -> Acquisition {
        Acquisition::Estimated
    }
    fn watts(&self) -> Option<f64> {
        Some(self.watts)
    }
}

/// Pick the best available host meter for this machine. Apple Silicon when we
/// can, a conservative fixed-TDP estimate otherwise — never nothing, so the
/// runtime always emits a labeled receipt.
pub fn detect_meter() -> Box<dyn HostMeter + Send + Sync> {
    let apple = AppleSiliconMeter::new();
    if apple.available() {
        Box::new(apple)
    } else {
        // Generic server fallback; conservative and clearly Estimated.
        Box::new(FixedWattsMeter::new(25.0))
    }
}

/// Measure the real host energy consumed while running `f`.
///
/// Samples power before and after and integrates over the wall-clock duration
/// (mean of the two readings × elapsed time). Good for runs from ~10ms up; for
/// very long runs the caller should sample periodically and sum. The returned
/// [`RunReceipt`] is stamped `StandIn` because the host is standing in for the
/// target substrate.
pub fn measure_run<M, F, R>(meter: &M, host_name: impl Into<String>, f: F) -> (R, RunReceipt)
where
    M: HostMeter + ?Sized,
    F: FnOnce() -> R,
{
    let w0 = meter.watts().unwrap_or(0.0);
    let start = std::time::Instant::now();
    let out = f();
    let duration_s = start.elapsed().as_secs_f64();
    let w1 = meter.watts().unwrap_or(w0);
    let avg_watts = (w0 + w1) / 2.0;
    let host_joules = avg_watts * duration_s;
    let receipt = RunReceipt {
        host_joules,
        avg_watts,
        duration_s,
        acquisition: meter.acquisition(),
        identity: Identity::stand_in(host_name),
        meter: meter.name().to_string(),
    };
    (out, receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn apple_meter_names_itself() {
        assert_eq!(AppleSiliconMeter::new().name(), "apple-silicon");
    }

    #[test]
    fn fixed_meter_always_available_and_estimated() {
        let m = FixedWattsMeter::new(30.0);
        assert!(m.available());
        assert_eq!(m.acquisition(), Acquisition::Estimated);
        assert_eq!(m.watts(), Some(30.0));
    }

    #[test]
    fn detect_meter_never_returns_unavailable() {
        assert!(detect_meter().available());
    }

    #[test]
    fn measure_run_stamps_standin_and_positive_energy() {
        let m = FixedWattsMeter::new(20.0);
        let (val, receipt) = measure_run(&m, "test-host", || {
            // a little real work so duration > 0
            (0u64..50_000).fold(0u64, |a, b| a.wrapping_add(b))
        });
        assert!(val > 0);
        assert!(receipt.host_joules >= 0.0);
        assert_eq!(receipt.avg_watts, 20.0);
        assert_eq!(receipt.acquisition, Acquisition::Estimated);
        match receipt.identity {
            Identity::StandIn { name } => assert_eq!(name, "test-host"),
            _ => panic!("host energy must be stamped StandIn, never Target"),
        }
    }

    #[test]
    fn identity_serializes_with_kind_tag() {
        let j = serde_json::to_value(Identity::stand_in("Apple Mac Studio")).unwrap();
        assert_eq!(j["kind"], "stand_in");
        assert_eq!(j["name"], "Apple Mac Studio");
    }
}
