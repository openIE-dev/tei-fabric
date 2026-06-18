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
//!   surface), `Estimated` (load→watts envelope), or `Measured` (power
//!   counters — on Apple Silicon, macmon's IOReport energy readings).
//! - [`Identity`] — *what* the joules describe: a [`Identity::Target`] TEI
//!   substrate, or a [`Identity::StandIn`] (the Mac Studio is a stand-in for
//!   future TEI silicon).
//!
//! A figure is only honest "this is what it costs on TEI hardware" when it is
//! `Measured` × `Target`. With macmon warm the Mac Studio gives
//! `Measured` × `StandIn`: a real machine, real counters, but not the target.
//!
//! ## Why a background sampler
//! macmon needs ~3 s of warmup before its first sample, so spawning it
//! per-run would stall every request. Instead one long-lived `macmon pipe`
//! process runs in a background thread ([`prewarm`] starts it at boot),
//! caching the latest `all_power` for instant reads. Until it is warm,
//! [`detect_meter`] returns the `Estimated` Apple-Silicon meter — the system
//! reports `Estimated` during the brief warmup window, `Measured` after.
//!
//! ## Credit
//! The Apple-Silicon load→watts fallback is ported from
//! `invisible-infrastructure`'s `inv-energy` crate (`apple.rs`). The measured
//! tier reads [macmon](https://github.com/vladkens/macmon)'s `pipe` JSON
//! (`all_power`, the IOReport SoC power) — sudoless, no extra privileges.

pub mod correction;
pub mod linux;
pub use correction::{Correction, CorrectionMethod, TargetEstimate, fpga_to_asic_default};
pub use linux::{NvmlMeter, RaplMeter};

use serde::Serialize;
use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

/// How a joule figure was obtained — weakest to strongest is
/// `Modeled` < `Estimated` < `Measured`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Acquisition {
    /// Predicted by a cost model (the cost surface / dialect tables). No
    /// hardware ran. Mirrors the ledger's `JoulesSource::Table`.
    Modeled,
    /// Derived from a coarse activity→power envelope (e.g. Apple-Silicon
    /// load average → watts). A real run, but power is inferred, not counted.
    Estimated,
    /// Read from hardware power/energy counters (macmon/IOReport on Apple
    /// Silicon, RAPL, NVML, an INA228 shunt). The strongest tier.
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
    /// Number of power samples integrated (0 ⇒ the meter used a point estimate).
    pub samples: u32,
    /// How `host_joules` was obtained.
    pub acquisition: Acquisition,
    /// What `host_joules` describes (the stand-in machine, here).
    pub identity: Identity,
    /// The meter that produced the reading (e.g. `"macmon"`).
    pub meter: String,
}

/// A host energy meter.
pub trait HostMeter: Send + Sync {
    /// Stable meter name (for the receipt + audit trail).
    fn name(&self) -> &str;
    /// True when this meter can produce real readings on this machine.
    fn available(&self) -> bool;
    /// The acquisition tier this meter achieves.
    fn acquisition(&self) -> Acquisition;
    /// Current instantaneous power draw, watts. `None` if unavailable.
    fn watts(&self) -> Option<f64> {
        None
    }

    /// Run `f`, returning `(joules, duration_s, samples)` measured across its
    /// execution. The default integrates the mean of the start/end [`watts`]
    /// readings over the wall duration (a 2-point estimate, `samples = 0`).
    /// Sampling meters (macmon) override this to integrate real power.
    ///
    /// `f` is boxed so the trait stays object-safe; callers normally use the
    /// [`measure_run`] helper, which threads the workload's return value out.
    fn measure(&self, f: Box<dyn FnOnce() + '_>) -> (f64, f64, u32) {
        let w0 = self.watts().unwrap_or(0.0);
        let t = Instant::now();
        f();
        let dur = t.elapsed().as_secs_f64();
        let w1 = self.watts().unwrap_or(w0);
        ((w0 + w1) / 2.0 * dur, dur, 0)
    }
}

// ── macmon background sampler ───────────────────────────────────────────

/// Latest `all_power` reading from the long-lived macmon process.
struct SamplerState {
    /// f64 bits of the latest `all_power` (watts).
    watts_bits: AtomicU64,
    /// Wall-clock ms of the last update (0 ⇒ never sampled).
    updated_ms: AtomicU64,
}

impl SamplerState {
    fn latest(&self) -> Option<(f64, u64)> {
        let ms = self.updated_ms.load(Ordering::Relaxed);
        if ms == 0 {
            return None;
        }
        Some((
            f64::from_bits(self.watts_bits.load(Ordering::Relaxed)),
            ms,
        ))
    }
}

fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

static SAMPLER: OnceLock<Arc<SamplerState>> = OnceLock::new();

/// Start (once) the long-lived `macmon pipe` reader thread and return the
/// shared latest-power cell. Idempotent.
fn sampler() -> &'static Arc<SamplerState> {
    SAMPLER.get_or_init(|| {
        let state = Arc::new(SamplerState {
            watts_bits: AtomicU64::new(0),
            updated_ms: AtomicU64::new(0),
        });
        let st = state.clone();
        let _ = std::thread::Builder::new()
            .name("macmon-sampler".into())
            .spawn(move || {
                // Respawn loop: if macmon exits, pause and restart.
                loop {
                    if let Ok(mut child) = Command::new("macmon")
                        .args(["pipe", "-i", "100"])
                        .stdout(Stdio::piped())
                        .stderr(Stdio::null())
                        .spawn()
                    {
                        if let Some(out) = child.stdout.take() {
                            for line in BufReader::new(out).lines() {
                                match line {
                                    Ok(l) => {
                                        if let Some(p) = parse_all_power(&l) {
                                            st.watts_bits.store(p.to_bits(), Ordering::Relaxed);
                                            st.updated_ms.store(now_ms(), Ordering::Relaxed);
                                        }
                                    }
                                    Err(_) => break,
                                }
                            }
                        }
                        let _ = child.kill();
                        let _ = child.wait();
                    }
                    std::thread::sleep(Duration::from_secs(2));
                }
            });
        state
    })
}

/// Staleness window: macmon samples every 100 ms, so anything within 2 s is
/// current. Beyond that the sampler is warming up or has stalled.
const MAX_AGE_MS: u64 = 2000;

/// Latest macmon power if it is fresh (sampler warm and live).
fn fresh_watts() -> Option<f64> {
    let (w, ms) = sampler().latest()?;
    if now_ms().saturating_sub(ms) <= MAX_AGE_MS {
        Some(w)
    } else {
        None
    }
}

/// Start the macmon sampler now so it is warm before the first request. Call
/// once at process startup. No-op where macmon is absent.
pub fn prewarm() {
    if cfg!(target_os = "macos") && MacmonMeter::probe() {
        let _ = sampler();
    }
}

/// True when the macmon sampler has produced a fresh reading.
pub fn macmon_warm() -> bool {
    fresh_watts().is_some()
}

/// macmon-backed meter: integrates `all_power` (IOReport SoC power: CPU + GPU
/// + ANE) sampled across the run from the background sampler. The
/// **`Measured`** tier on Apple Silicon — real energy counters, sudoless.
pub struct MacmonMeter;

impl Default for MacmonMeter {
    fn default() -> Self {
        Self::new()
    }
}

impl MacmonMeter {
    pub fn new() -> Self {
        let _ = sampler(); // ensure the sampler thread is running
        MacmonMeter
    }

    /// `macmon --version` succeeds ⇒ the binary is usable on this host.
    pub fn probe() -> bool {
        Command::new("macmon")
            .arg("--version")
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
}

impl HostMeter for MacmonMeter {
    fn name(&self) -> &str {
        "macmon"
    }
    fn available(&self) -> bool {
        true
    }
    fn acquisition(&self) -> Acquisition {
        Acquisition::Measured
    }
    fn watts(&self) -> Option<f64> {
        fresh_watts()
    }

    fn measure(&self, f: Box<dyn FnOnce() + '_>) -> (f64, f64, u32) {
        let collected = Arc::new(Mutex::new(Vec::<f64>::new()));
        if let Some(w) = fresh_watts() {
            collected.lock().unwrap().push(w);
        }

        // Poll the cached reading during the run (reads are instant). macmon
        // updates ~every 100 ms; polling at 40 ms time-weights the mean.
        let stop = Arc::new(AtomicBool::new(false));
        let (s2, c2) = (stop.clone(), collected.clone());
        let poller = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                if let Some(w) = fresh_watts() {
                    c2.lock().unwrap().push(w);
                }
                std::thread::sleep(Duration::from_millis(40));
            }
        });

        let t = Instant::now();
        f();
        let dur = t.elapsed().as_secs_f64();

        stop.store(true, Ordering::Relaxed);
        let _ = poller.join();
        if let Some(w) = fresh_watts() {
            collected.lock().unwrap().push(w);
        }

        let samples = collected.lock().unwrap();
        if samples.is_empty() {
            // Sampler not warm — honest zero-sample point estimate.
            return (0.0, dur, 0);
        }
        let mean = samples.iter().sum::<f64>() / samples.len() as f64;
        (mean * dur, dur, samples.len() as u32)
    }
}

/// Extract `all_power` (watts) from one macmon pipe JSON line.
fn parse_all_power(line: &str) -> Option<f64> {
    serde_json::from_str::<serde_json::Value>(line)
        .ok()?
        .get("all_power")?
        .as_f64()
}

/// Apple-Silicon load→watts meter: estimates package power from the 1-minute
/// load average across an idle→peak envelope. The **`Estimated`** tier — a
/// real machine, inferred power. Used when macmon is unavailable or warming.
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

/// A meter that reports a fixed watt figure regardless of activity. Portable
/// fallback for hosts with no real meter, so the cloud runtime still produces
/// a (clearly `Estimated`) receipt off-Mac.
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

/// Pick the best available host meter: macmon **once warm** (Measured) →
/// Apple-Silicon load model (Estimated) → fixed-TDP (Estimated). Never
/// nothing, so the runtime always emits a labeled receipt. Call [`prewarm`]
/// at startup so macmon is warm by the first request.
pub fn detect_meter() -> Box<dyn HostMeter + Send + Sync> {
    // macOS: macmon (once warm) → Apple-Silicon load model.
    if cfg!(target_os = "macos") && MacmonMeter::probe() {
        let _ = sampler();
        if macmon_warm() {
            return Box::new(MacmonMeter::new());
        }
        // macmon present but still warming — report Estimated for now.
    }
    let apple = AppleSiliconMeter::new();
    if apple.available() {
        return Box::new(apple);
    }
    // GPU / Linux hosts: NVIDIA GPU energy (compute-dominant) → CPU RAPL.
    let nvml = NvmlMeter::new();
    if nvml.available() {
        return Box::new(nvml);
    }
    let rapl = RaplMeter::new();
    if rapl.available() {
        return Box::new(rapl);
    }
    Box::new(FixedWattsMeter::new(25.0))
}

/// Measure the real host energy consumed while running `f`.
///
/// Delegates integration to the meter (macmon polls + integrates real power;
/// simpler meters use a 2-point watts estimate) and stamps the result
/// `StandIn` because the host is standing in for the target substrate.
pub fn measure_run<F, R>(
    meter: &(dyn HostMeter + Send + Sync),
    host_name: impl Into<String>,
    f: F,
) -> (R, RunReceipt)
where
    F: FnOnce() -> R,
{
    let mut out: Option<R> = None;
    let (host_joules, duration_s, samples) = {
        let slot = &mut out;
        meter.measure(Box::new(move || {
            *slot = Some(f());
        }))
    };
    let avg_watts = if duration_s > 0.0 { host_joules / duration_s } else { 0.0 };
    let receipt = RunReceipt {
        host_joules,
        avg_watts,
        duration_s,
        samples,
        acquisition: meter.acquisition(),
        identity: Identity::stand_in(host_name),
        meter: meter.name().to_string(),
    };
    (out.expect("measure ran the closure"), receipt)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_meter_two_point_measures_energy() {
        let m = FixedWattsMeter::new(20.0);
        let (val, receipt) = measure_run(&m, "test-host", || {
            (0u64..50_000).fold(0u64, |a, b| a.wrapping_add(b))
        });
        assert!(val > 0);
        assert!(receipt.host_joules >= 0.0);
        assert_eq!(receipt.avg_watts.round(), 20.0); // joules/dur == fixed watts
        assert_eq!(receipt.acquisition, Acquisition::Estimated);
        assert_eq!(receipt.samples, 0);
        match receipt.identity {
            Identity::StandIn { name } => assert_eq!(name, "test-host"),
            _ => panic!("host energy must be stamped StandIn, never Target"),
        }
    }

    #[test]
    fn macmon_meter_reports_measured_tier() {
        assert_eq!(MacmonMeter::new().acquisition(), Acquisition::Measured);
        assert_eq!(MacmonMeter::new().name(), "macmon");
    }

    #[test]
    fn detect_meter_never_returns_unavailable() {
        assert!(detect_meter().available());
    }

    #[test]
    fn identity_serializes_with_kind_tag() {
        let j = serde_json::to_value(Identity::stand_in("Apple Mac Studio")).unwrap();
        assert_eq!(j["kind"], "stand_in");
        assert_eq!(j["name"], "Apple Mac Studio");
    }

    #[test]
    #[cfg(target_os = "macos")]
    fn macmon_integrates_real_power_when_warm() {
        if !MacmonMeter::probe() {
            return; // macmon not installed here
        }
        prewarm();
        // Wait out macmon's ~3 s warmup (cap ~8 s).
        let t = Instant::now();
        while !macmon_warm() && t.elapsed() < Duration::from_secs(8) {
            std::thread::sleep(Duration::from_millis(100));
        }
        if !macmon_warm() {
            return; // environment didn't warm in time; don't flake
        }
        let m = MacmonMeter::new();
        let (_v, receipt) = measure_run(&m, "test-mac", || {
            let t = Instant::now();
            let mut acc = 0u64;
            while t.elapsed() < Duration::from_millis(400) {
                acc = acc.wrapping_add((0u64..20_000).fold(0, |a, b| a.wrapping_add(b * b)));
            }
            acc
        });
        assert_eq!(receipt.acquisition, Acquisition::Measured);
        assert!(receipt.host_joules > 0.0);
        assert!(receipt.samples >= 1, "expected integrated samples, got 0");
    }
}
