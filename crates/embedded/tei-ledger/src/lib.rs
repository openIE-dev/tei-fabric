//! tei-ledger — the TEI Embedded device contract (E0).
//!
//! The fabric prices computational primitives in joules, executes them,
//! counts what physically happened in an event ledger, and feeds measured
//! energy back to replace assumed constants. This crate is that contract
//! for current-generation hardware: the embedded [`EventLedger`], joule
//! [provenance](JoulesSource), per-substrate [`CycleSource`]s, fixed-
//! capacity [`CostTable`]s, the lowest-joule [dispatch](CostTable::cheapest)
//! rule, and the [`CalibrationReport`] that POSTs to the fabric's
//! `/api/calibration` family.
//!
//! Everything here is `no_std`, allocation-free, and deterministic. Every
//! ecosystem binding (Embassy, Arduino, MicroPython, Zephyr, the Pi-class
//! `teid`) links this one crate — the LVGL/TinyUSB one-core shape with the
//! language inverted (see EMBEDDED-ROADMAP.md §4).
//!
//! ## Counter-set provenance
//!
//! The ledger fields and the [`CycleSource`] abstraction follow the
//! 2026-06 verification research (EMBEDDED-ROADMAP.md §8):
//! - ARM cores: DWT CYCCNT (architectural M3/M4/M7; *optional* on M33 —
//!   check `DWT_CTRL.NOCYCCNT`; absent on M0+/M23 → SysTick fallback) or
//!   the Armv8.1-M PMU on M55/M85.
//! - RP2350 Hazard3: 64-bit `mcycle` — but `mcountinhibit` resets to 0x5,
//!   counters are OFF until firmware clears it.
//! - ESP32-C6 HP core: Espressif's custom PCCR CSR (0x7e2), not `mcycle`;
//!   ESP32-P4 HP and the C6/P4 LP cores use standard `mcycle`.
//! - nRF54 FLPR: rv32emc, **no cycle counter at all** — time via GRTC.
//!
//! Joule provenance is the tier model: `Measured` requires hardware the
//! target itself can read (INA228-class 40-bit hardware joule accumulation,
//! or SiLabs `BSP_CurrentGet()` over the board controller). EnergyTrace and
//! PPK2-class bench gear calibrate `CyclesProxy` tables; shipped datasheet
//! defaults are `Table` — and the ledger always says which it was.

#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

/// Where a joules figure came from. Honesty is the contract: a consumer
/// can always tell measurement from estimate from folklore.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
#[cfg_attr(feature = "serde", serde(rename_all = "snake_case"))]
pub enum JoulesSource {
    /// Read by the target itself from energy-accumulating hardware
    /// (INA228-class monitor, board-controller AEM query).
    Measured,
    /// Cycles/time × a per-power-state table calibrated on a bench
    /// (PPK2/Joulescope/EnergyTrace) or crowd-sourced via the fabric.
    CyclesProxy,
    /// Shipped datasheet defaults; no calibration has touched this board.
    /// Dispatch *ratios* between substrates remain useful even here.
    #[default]
    Table,
}

/// The embedded event ledger — counters current hardware counts cheaply.
///
/// The embedded analogue of the sim layer's `EventLedger` (sweeps, spikes,
/// MACs become cycles, DMA transfers, accelerator invocations), sharing
/// its design rule: count what physically happened, then price it.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EventLedger {
    /// Substrate cycles consumed (source: the substrate's [`CycleSource`]).
    pub cycles: u64,
    /// Retired instructions, where a counter exists (RISC-V `minstret`,
    /// ARM PMU). `None` ≠ zero — it means "this substrate cannot count".
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub instructions: Option<u64>,
    /// DMA transfer completions attributed to this work.
    pub dma_transfers: u64,
    /// ADC samples digitized.
    pub adc_samples: u64,
    /// Accelerator invocations (PIO program runs, FMAC/CORDIC jobs,
    /// NPU inferences, LP-core wakes…).
    pub accel_invocations: u64,
    /// Microseconds spent in sleep states while this work was pending.
    pub sleep_us: u64,
    /// Microseconds awake on this work.
    pub active_us: u64,
    /// Energy, if a figure exists. `None` means "not priced yet", never 0 J.
    #[cfg_attr(feature = "serde", serde(skip_serializing_if = "Option::is_none"))]
    pub joules: Option<f64>,
    /// Provenance of `joules` (meaningful when `joules.is_some()`; for
    /// `None` it records what the board *would* report).
    pub joules_source: JoulesSource,
}

impl EventLedger {
    /// An empty ledger with the given provenance tier.
    pub const fn new(source: JoulesSource) -> Self {
        Self {
            cycles: 0,
            instructions: None,
            dma_transfers: 0,
            adc_samples: 0,
            accel_invocations: 0,
            sleep_us: 0,
            active_us: 0,
            joules: None,
            joules_source: source,
        }
    }

    /// Fold another ledger in. Counters add; joules add when both sides
    /// have a figure (one-sided joules stay, preserving partial pricing);
    /// provenance degrades to the weaker tier (Measured < CyclesProxy <
    /// Table) so a merged figure never overclaims.
    pub fn merge(&mut self, other: &EventLedger) {
        self.cycles += other.cycles;
        self.instructions = match (self.instructions, other.instructions) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        };
        self.dma_transfers += other.dma_transfers;
        self.adc_samples += other.adc_samples;
        self.accel_invocations += other.accel_invocations;
        self.sleep_us += other.sleep_us;
        self.active_us += other.active_us;
        self.joules = match (self.joules, other.joules) {
            (Some(a), Some(b)) => Some(a + b),
            (a, b) => a.or(b),
        };
        self.joules_source = weaker(self.joules_source, other.joules_source);
    }
}

/// The weaker of two provenance tiers (Measured strongest, Table weakest).
const fn weaker(a: JoulesSource, b: JoulesSource) -> JoulesSource {
    match (a, b) {
        (JoulesSource::Table, _) | (_, JoulesSource::Table) => JoulesSource::Table,
        (JoulesSource::CyclesProxy, _) | (_, JoulesSource::CyclesProxy) => {
            JoulesSource::CyclesProxy
        }
        _ => JoulesSource::Measured,
    }
}

/// A cycle counter for one substrate. There is deliberately no blanket
/// implementation: which register (or peripheral timer) counts is a
/// per-substrate fact — DWT CYCCNT, PMU CCNTR, `mcycle`, Espressif PCCR,
/// or a GRTC/SysTick time fallback where no counter exists.
pub trait CycleSource {
    /// Current free-running count. Wrapping is the caller's problem;
    /// [`CycleSource::delta`] handles single-wrap spans for `u32` counters.
    fn now(&self) -> u64;

    /// Cycles elapsed since `start` (same source). Default assumes a
    /// non-wrapping 64-bit count; 32-bit hardware sources should override
    /// with wrapping subtraction in their native width.
    fn delta(&self, start: u64) -> u64 {
        self.now().wrapping_sub(start)
    }
}

/// An energy meter the target itself can read (the T0 tier): an
/// INA228-class accumulator, a board-controller AEM query, a PMIC ADC.
pub trait EnergyMeter {
    /// Joules accumulated since the last [`reset`](EnergyMeter::reset),
    /// or `None` if the meter is unreachable right now.
    fn joules(&mut self) -> Option<f64>;
    /// Zero the accumulation (INA228 RSTACC-style).
    fn reset(&mut self);
}

/// One calibrated price: this primitive on this substrate costs this much.
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CostEntry {
    /// Periodic Stack primitive id (the same id space the fabric uses).
    pub primitive_id: u32,
    /// Substrate name, fabric conventions: "cpu@150mhz", "pio", "lp-core",
    /// "npu", "dma", "flpr"…
    pub substrate: &'static str,
    /// Joules per operation on this substrate.
    pub j_per_op: f64,
    /// Provenance of this entry.
    pub source: JoulesSource,
}

/// A fixed-capacity, allocation-free cost table. `N` is the board's whole
/// price list; tables are small (primitives × substrates), so linear scans
/// beat any clever structure at MCU scale.
#[derive(Debug, Clone)]
pub struct CostTable<const N: usize> {
    entries: [Option<CostEntry>; N],
    len: usize,
}

impl<const N: usize> Default for CostTable<N> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const N: usize> CostTable<N> {
    /// An empty table.
    pub const fn new() -> Self {
        Self {
            entries: [None; N],
            len: 0,
        }
    }

    /// Number of entries.
    pub const fn len(&self) -> usize {
        self.len
    }

    /// True when no entries are present.
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Insert or replace the entry for (primitive, substrate). Returns
    /// `Err(entry)` when the table is full and the pair is new.
    pub fn upsert(&mut self, entry: CostEntry) -> Result<(), CostEntry> {
        for slot in self.entries.iter_mut().take(self.len) {
            if let Some(e) = slot {
                if e.primitive_id == entry.primitive_id && e.substrate == entry.substrate {
                    *slot = Some(entry);
                    return Ok(());
                }
            }
        }
        if self.len == N {
            return Err(entry);
        }
        self.entries[self.len] = Some(entry);
        self.len += 1;
        Ok(())
    }

    /// All entries for one primitive.
    pub fn for_primitive(&self, primitive_id: u32) -> impl Iterator<Item = &CostEntry> {
        self.entries
            .iter()
            .take(self.len)
            .filter_map(move |e| match e {
                Some(e) if e.primitive_id == primitive_id => Some(e),
                _ => None,
            })
    }

    /// THE dispatch rule, identical to the fabric's: the lowest-joule
    /// substrate that prices this primitive. Ties break toward the
    /// stronger provenance, then first-inserted (deterministic).
    pub fn cheapest(&self, primitive_id: u32) -> Option<&CostEntry> {
        let mut best: Option<&CostEntry> = None;
        for e in self.for_primitive(primitive_id) {
            best = match best {
                None => Some(e),
                Some(b) => {
                    if e.j_per_op < b.j_per_op
                        || (e.j_per_op == b.j_per_op && stronger_than(e.source, b.source))
                    {
                        Some(e)
                    } else {
                        Some(b)
                    }
                }
            };
        }
        best
    }
}

/// True when `a` is strictly stronger provenance than `b`.
const fn stronger_than(a: JoulesSource, b: JoulesSource) -> bool {
    rank(a) < rank(b)
}

const fn rank(s: JoulesSource) -> u8 {
    match s {
        JoulesSource::Measured => 0,
        JoulesSource::CyclesProxy => 1,
        JoulesSource::Table => 2,
    }
}

/// What a board POSTs to the fabric after measuring a primitive: the
/// community calibration store's row format (`/api/calibration` family).
#[derive(Debug, Clone, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CalibrationReport<'a> {
    /// Board identity, forge conventions: "pico2", "esp32c6-devkitc"…
    pub board_id: &'a str,
    /// Substrate measured (see [`CostEntry::substrate`]).
    pub substrate: &'a str,
    /// Periodic Stack primitive id.
    pub primitive_id: u32,
    /// Operations executed during the measurement window.
    pub n_ops: u64,
    /// What physically happened.
    pub ledger: EventLedger,
    /// The headline: joules per op, `ledger.joules / n_ops`.
    pub j_per_op: f64,
}

impl<'a> CalibrationReport<'a> {
    /// Build a report from a priced ledger. Returns `None` when the
    /// ledger has no joules or no ops — an unpriced report is not a
    /// report, and the contract refuses to fabricate one.
    pub fn from_ledger(
        board_id: &'a str,
        substrate: &'a str,
        primitive_id: u32,
        n_ops: u64,
        ledger: EventLedger,
    ) -> Option<Self> {
        let joules = ledger.joules?;
        if n_ops == 0 {
            return None;
        }
        Some(Self {
            board_id,
            substrate,
            primitive_id,
            n_ops,
            ledger,
            j_per_op: joules / n_ops as f64,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Fake(u64);
    impl CycleSource for Fake {
        fn now(&self) -> u64 {
            self.0
        }
    }

    #[test]
    fn merge_sums_counters_and_degrades_provenance() {
        let mut a = EventLedger::new(JoulesSource::Measured);
        a.cycles = 100;
        a.joules = Some(1.0e-6);
        let mut b = EventLedger::new(JoulesSource::CyclesProxy);
        b.cycles = 50;
        b.joules = Some(2.0e-6);
        b.instructions = Some(10);
        a.merge(&b);
        assert_eq!(a.cycles, 150);
        assert_eq!(a.instructions, Some(10));
        assert_eq!(a.joules, Some(3.0e-6));
        assert_eq!(a.joules_source, JoulesSource::CyclesProxy);
    }

    #[test]
    fn merge_keeps_one_sided_joules() {
        let mut a = EventLedger::new(JoulesSource::Measured);
        a.joules = Some(5.0e-7);
        let b = EventLedger::new(JoulesSource::Measured);
        a.merge(&b);
        assert_eq!(a.joules, Some(5.0e-7));
        assert_eq!(a.joules_source, JoulesSource::Measured);
    }

    #[test]
    fn cheapest_picks_lowest_joule_then_strongest_provenance() {
        let mut t: CostTable<8> = CostTable::new();
        t.upsert(CostEntry {
            primitive_id: 79,
            substrate: "cpu@150mhz",
            j_per_op: 96e-6,
            source: JoulesSource::Measured,
        })
        .unwrap();
        t.upsert(CostEntry {
            primitive_id: 79,
            substrate: "pio",
            j_per_op: 41e-6,
            source: JoulesSource::CyclesProxy,
        })
        .unwrap();
        t.upsert(CostEntry {
            primitive_id: 79,
            substrate: "dma",
            j_per_op: 41e-6,
            source: JoulesSource::Table,
        })
        .unwrap();
        let best = t.cheapest(79).unwrap();
        assert_eq!(best.substrate, "pio"); // ties break to stronger provenance
        assert!(t.cheapest(18).is_none());
    }

    #[test]
    fn upsert_replaces_and_reports_full() {
        let mut t: CostTable<1> = CostTable::new();
        let e = CostEntry {
            primitive_id: 1,
            substrate: "cpu",
            j_per_op: 1.0,
            source: JoulesSource::Table,
        };
        t.upsert(e).unwrap();
        t.upsert(CostEntry { j_per_op: 2.0, ..e }).unwrap(); // replace, not full
        assert_eq!(t.cheapest(1).unwrap().j_per_op, 2.0);
        let other = CostEntry {
            primitive_id: 2,
            ..e
        };
        assert!(t.upsert(other).is_err());
    }

    #[test]
    fn report_refuses_unpriced_ledgers() {
        let l = EventLedger::new(JoulesSource::Table);
        assert!(CalibrationReport::from_ledger("pico2", "pio", 79, 100, l).is_none());
        let mut priced = l;
        priced.joules = Some(4.1e-3);
        let r = CalibrationReport::from_ledger("pico2", "pio", 79, 100, priced).unwrap();
        assert_eq!(r.j_per_op, 4.1e-5);
        assert!(CalibrationReport::from_ledger("pico2", "pio", 79, 0, priced).is_none());
    }

    #[test]
    fn cycle_source_delta() {
        let c = Fake(1000);
        assert_eq!(c.delta(400), 600);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn report_serializes_to_fabric_shape() {
        extern crate std;
        let mut l = EventLedger::new(JoulesSource::Measured);
        l.cycles = 1_500_000;
        l.accel_invocations = 1;
        l.active_us = 980;
        l.joules = Some(4.1e-5);
        let r = CalibrationReport::from_ledger("pico2", "pio", 79, 1, l).unwrap();
        let v: serde_json::Value = serde_json::to_value(&r).unwrap();
        assert_eq!(v["board_id"], "pico2");
        assert_eq!(v["substrate"], "pio");
        assert_eq!(v["primitive_id"], 79);
        assert_eq!(v["ledger"]["joules_source"], "measured");
        assert_eq!(v["ledger"]["cycles"], 1_500_000);
        // None fields are omitted, not null — keeps frames small.
        assert!(v["ledger"].get("instructions").is_none());
    }
}
