//! teiOS E1a — host-testable core of the RP2040 firmware.
//!
//! The firmware binary ([`main.rs`](../src/main.rs)) owns the hardware
//! (USB CDC, the 1 MHz TIMER, the DMA sniffer); the board-independent
//! logic — the software CRC32 substrate, the deterministic workload,
//! and the JSON-lines writers — lives in the shared `teios-core` crate
//! (also used by teios-rp2350). This crate pins the RP2040 board
//! identity: board strings, substrate names, and the shipped cost
//! table; the tests at the bottom lock the board's wire format to
//! `tei-ledger`'s serde shape.
//!
//! # Cycles on the Cortex-M0+ (the CycleSource doctrine)
//!
//! The M0+ has **no DWT CYCCNT** — no CPU cycle counter exists on this
//! chip, full stop. Per the roadmap's per-substrate `CycleSource`
//! doctrine, the firmware derives cycles from the RP2040 TIMER (1 MHz,
//! 64-bit): `cycles = elapsed_us × clk_sys_MHz` (125 at the embassy-rp
//! default clock). The boot line therefore reports `cyccnt:false` —
//! there is no cycle counter — while cpu ledgers carry the honest
//! timer-derived proxy in `cycles` (quantization ±1 µs = ±125 cycles
//! per measured span; see `fw::TimerCycleSource`). Energy provenance
//! is unchanged: it lives in `joules_source`, and this image ships at
//! the Table tier.
//!
//! # The JSON-lines protocol (what TEI Studio's web console parses)
//!
//! One JSON object per `\n`-terminated line on the USB CDC serial port,
//! discriminated by `"type"`. Fields follow `tei-ledger`'s serde shape:
//! snake_case names, `None` fields **omitted** (never `null`). Numbers
//! may use exponent notation (`4e-6`) — any JSON parser accepts both.
//!
//! - `{"type":"boot","board_id":"feather-rp2040",
//!    "firmware":"teios-rp2040/0.1.0","primitive_id":36,
//!    "buf_len":65536,"cyccnt":false}`
//!   Once after USB connects. `cyccnt` is always `false` on RP2040
//!   (see above); `cycles` in cpu ledgers is the timer proxy.
//! - `{"type":"ledger","board_id":"feather-rp2040",
//!    "substrate":"cpu@125mhz","primitive_id":36,"n_ops":1,
//!    "ledger":{"cycles":1250000,"dma_transfers":0,"adc_samples":0,
//!    "accel_invocations":0,"sleep_us":0,"active_us":10000,
//!    "joules_source":"table"}}`
//!   One per primitive run per substrate. The `ledger` object is exactly
//!   `tei_ledger::EventLedger` in serde form; `instructions`/`joules`
//!   appear only when present.
//! - `{"type":"check","ok":true,"crc_cpu":2378668581,"crc_dma":2378668581}`
//!   Both substrates ran the same primitive over the same buffer; `ok`
//!   is their agreement. CRCs are the u32 values as JSON numbers.
//! - `{"type":"dispatch","primitive_id":36,"chosen":"dma-sniffer",
//!    "j_per_op":4e-6,"joules_source":"table","alternatives":[
//!    {"substrate":"cpu@125mhz","j_per_op":1.8e-4,"joules_source":"table"}]}`
//!   The lowest-joule verdict from [`CostTable::cheapest`], plus every
//!   priced alternative for the primitive.

#![cfg_attr(not(test), no_std)]

use core::fmt::{self, Write};

use tei_ledger::{CostEntry, CostTable, EventLedger, JoulesSource};
use teios_core::BoardConfig;
pub use teios_core::{BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, crc32_software, fill_pattern};

#[cfg(all(feature = "board-feather-rp2040", feature = "board-pico"))]
compile_error!("enable exactly one board feature: board-feather-rp2040 OR board-pico");
#[cfg(not(any(feature = "board-feather-rp2040", feature = "board-pico")))]
compile_error!(
    "enable a board feature: board-feather-rp2040 (default) or \
     --no-default-features --features board-pico"
);

/// Board identity, forge conventions. Selected by the board feature,
/// which also selects the matching boot2 flash bootloader.
#[cfg(feature = "board-feather-rp2040")]
pub const BOARD_ID: &str = "feather-rp2040";
/// Board identity, forge conventions. Selected by the board feature,
/// which also selects the matching boot2 flash bootloader.
#[cfg(all(feature = "board-pico", not(feature = "board-feather-rp2040")))]
pub const BOARD_ID: &str = "pico";

/// Firmware identity reported in the boot line.
pub const FIRMWARE: &str = concat!("teios-rp2040/", env!("CARGO_PKG_VERSION"));

/// This board, as stamped on every JSON line.
pub const BOARD: BoardConfig = BoardConfig {
    board_id: BOARD_ID,
    firmware: FIRMWARE,
};

/// Substrate name for the software CRC32 on the Cortex-M0+ at the
/// embassy-rp default 125 MHz system clock.
pub const SUBSTRATE_CPU: &str = "cpu@125mhz";

/// Substrate name for the RP2040 DMA sniffer (hardware CRC32 computed
/// by the DMA engine while the CPU is free to sleep — the same
/// SNIFF_DATA/SNIFF_CTRL block the RP2350 kept).
pub const SUBSTRATE_DMA: &str = "dma-sniffer";

// ---------------------------------------------------------------------------
// Shipped cost table — Table tier.
// ---------------------------------------------------------------------------

/// ILLUSTRATIVE J/op for one CRC32 pass over [`BUF_LEN`] on the M0+ at
/// 125 MHz (software, table-driven). Placeholder default pending the
/// E1 bench measurement (PPK2) — no published figure exists; the
/// ledger says `joules_source: table` so nothing overclaims. Kept
/// above the RP2350's M33 placeholder (the two-cycle-per-bit M0+
/// works harder per byte), but it is a placeholder, not a measurement.
pub const CPU_J_PER_OP: f64 = 1.8e-4;

/// ILLUSTRATIVE J/op for the same pass via the DMA sniffer. Same
/// caveat as [`CPU_J_PER_OP`]: a placeholder Table-tier default, kept
/// plausibly proportioned (the sniffer moves one word per DMA transfer
/// with the CPU idle) until the E1 bench replaces it.
pub const DMA_J_PER_OP: f64 = 4.0e-6;

/// The cost table this image ships with: both substrates priced at the
/// Table tier (shipped defaults, no calibration has touched this board).
pub fn shipped_cost_table() -> CostTable<COST_CAPACITY> {
    let mut t = CostTable::new();
    let _ = t.upsert(CostEntry {
        primitive_id: PRIMITIVE_HASH,
        substrate: SUBSTRATE_CPU,
        j_per_op: CPU_J_PER_OP,
        source: JoulesSource::Table,
    });
    let _ = t.upsert(CostEntry {
        primitive_id: PRIMITIVE_HASH,
        substrate: SUBSTRATE_DMA,
        j_per_op: DMA_J_PER_OP,
        source: JoulesSource::Table,
    });
    t
}

// ---------------------------------------------------------------------------
// JSON-lines writers — teios-core's, with this board's identity bound.
// ---------------------------------------------------------------------------

/// `{"type":"boot",...}` — once after USB connects. On RP2040 the
/// firmware always passes `cyccnt: false` (no cycle counter on M0+).
pub fn write_boot_line<W: Write>(w: &mut W, cyccnt: bool) -> fmt::Result {
    teios_core::write_boot_line(w, &BOARD, cyccnt)
}

/// `{"type":"ledger",...}` — one primitive run on one substrate.
pub fn write_ledger_line<W: Write>(
    w: &mut W,
    substrate: &str,
    n_ops: u64,
    ledger: &EventLedger,
) -> fmt::Result {
    teios_core::write_ledger_line(w, &BOARD, substrate, n_ops, ledger)
}

/// `{"type":"report",...}` — a calibration report for one priced substrate,
/// stamped with this board's id. Studio relays it to the fabric.
pub fn write_report_line<W: Write>(
    w: &mut W,
    entry: &tei_ledger::CostEntry,
    n_ops: u64,
) -> fmt::Result {
    teios_core::write_report_line(w, &BOARD, entry, n_ops)
}

pub use teios_core::{write_check_line, write_dispatch_line};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// The wire-format lock for this board: the emitted ledger line must
    /// equal serde_json's rendering of the same `EventLedger` and carry
    /// the board identity. (The generic shape tests live in teios-core.)
    #[test]
    fn ledger_line_matches_tei_ledger_serde_shape() {
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = 1_250_000; // timer proxy: 10 000 µs × 125 MHz
        l.active_us = 10_000;
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_CPU, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "ledger");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["substrate"], "cpu@125mhz");
        assert_eq!(v["primitive_id"], 36);
        assert_eq!(v["n_ops"], 1);
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        // None fields are omitted, never null.
        assert!(v["ledger"].get("instructions").is_none());
        assert!(v["ledger"].get("joules").is_none());
    }

    #[test]
    fn boot_line_carries_board_identity_and_no_cyccnt() {
        let mut b = String::new();
        write_boot_line(&mut b, false).unwrap();
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["type"], "boot");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["firmware"], FIRMWARE);
        assert_eq!(v["primitive_id"], 36);
        assert_eq!(v["buf_len"], 65_536);
        // M0+ has no DWT CYCCNT; the firmware reports the truth.
        assert_eq!(v["cyccnt"], false);
    }

    /// Each board feature pins its literal id (the forge convention).
    #[cfg(feature = "board-feather-rp2040")]
    #[test]
    fn board_id_is_feather_rp2040() {
        assert_eq!(BOARD_ID, "feather-rp2040");
    }

    /// Each board feature pins its literal id (the forge convention).
    #[cfg(all(feature = "board-pico", not(feature = "board-feather-rp2040")))]
    #[test]
    fn board_id_is_pico() {
        assert_eq!(BOARD_ID, "pico");
    }

    #[test]
    fn shipped_table_dispatches_to_dma_sniffer() {
        let t = shipped_cost_table();
        assert_eq!(t.len(), 2);
        let best = t.cheapest(PRIMITIVE_HASH).unwrap();
        assert_eq!(best.substrate, SUBSTRATE_DMA);
        assert_eq!(best.source, JoulesSource::Table);
        assert!(t.cheapest(9999).is_none());
    }

    #[test]
    fn dispatch_line_names_cheapest_and_lists_alternatives() {
        let t = shipped_cost_table();
        let mut s = String::new();
        let chosen = write_dispatch_line(&mut s, &t, PRIMITIVE_HASH).unwrap();
        assert_eq!(chosen, Some(SUBSTRATE_DMA));
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "dispatch");
        assert_eq!(v["chosen"], "dma-sniffer");
        assert!((v["j_per_op"].as_f64().unwrap() - DMA_J_PER_OP).abs() < 1e-18);
        let alts = v["alternatives"].as_array().unwrap();
        assert_eq!(
            alts,
            &vec![
                json!({"substrate": "cpu@125mhz", "j_per_op": CPU_J_PER_OP, "joules_source": "table"})
            ]
        );
    }

    /// Every line the firmware can emit fits the firmware's line buffer.
    #[test]
    fn lines_fit_firmware_buffer() {
        const LINE_CAP: usize = teios_core::LINE_CAP; // keep in sync with fw.rs
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = u64::MAX;
        l.instructions = Some(u64::MAX);
        l.dma_transfers = u64::MAX;
        l.adc_samples = u64::MAX;
        l.accel_invocations = u64::MAX;
        l.sleep_us = u64::MAX;
        l.active_us = u64::MAX;
        l.joules = Some(1.234_567_890_123e-6);
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_CPU, u64::MAX, &l).unwrap();
        assert!(s.len() <= LINE_CAP, "ledger line {} > {LINE_CAP}", s.len());
        let mut d = String::new();
        write_dispatch_line(&mut d, &shipped_cost_table(), PRIMITIVE_HASH).unwrap();
        assert!(d.len() <= LINE_CAP);
    }
}
