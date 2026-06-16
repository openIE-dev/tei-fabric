//! teiOS E1b — host-testable core of the Portenta H7 firmware.
//!
//! The firmware ([`main.rs`](../src/main.rs) / `fw.rs`) owns the
//! hardware (USB CDC, the M7 DWT cycle counter, the STM32 CRC
//! peripheral, and — the headline — the second Cortex-M4 core); the
//! board-independent protocol lives in the shared `teios-core` crate,
//! same as the RP2040/RP2350 images. This crate pins the Portenta H7
//! identity and its substrate names + shipped cost table.
//!
//! # The heterogeneous-dispatch story (why this board)
//!
//! The STM32H747XI is two CPUs on one die — a Cortex-M7 at 480 MHz and
//! a Cortex-M4 at 240 MHz — plus the STM32 hardware CRC peripheral.
//! That makes it the first *true heterogeneous-core* teiOS target: the
//! same Hash primitive (CRC32 over [`BUF_LEN`]) raced on the M7
//! software path, the M4 software path, and the CRC peripheral, each
//! priced into a ledger, the cheapest dispatched. The M7 *has* a real
//! DWT CYCCNT (unlike the RP2040's M0+), so the boot line reports
//! `cyccnt:true` and cpu ledgers carry true cycle counts.
//!
//! # Cycles on the M7
//!
//! Architectural DWT CYCCNT — enable `DEMCR.TRCENA` + `DWT_CTRL.CYCCNTENA`
//! and read `DWT.CYCCNT`. True per-cycle granularity (no timer proxy).
//!
//! # The JSON-lines protocol
//!
//! Identical to the RP2040 image (see teios-core / teios-rp2040 docs):
//! one `\n`-terminated JSON object per line over USB CDC, discriminated
//! by `"type"` (boot / ledger / check / dispatch), `tei-ledger` serde
//! shape, `None` fields omitted. TEI Studio's console parses it the
//! same way regardless of board.

#![cfg_attr(not(test), no_std)]

use core::fmt::{self, Write};

use tei_ledger::{CostEntry, CostTable, EventLedger, JoulesSource};
use teios_core::BoardConfig;
pub use teios_core::{BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, crc32_software, fill_pattern};

#[cfg(all(feature = "board-portenta-h7", feature = "board-portenta-h7-lite"))]
compile_error!("enable exactly one board feature: board-portenta-h7 OR board-portenta-h7-lite");

/// Board identity (forge conventions). The H7 and H7 Lite share silicon
/// and firmware; only the id string differs.
#[cfg(not(feature = "board-portenta-h7-lite"))]
pub const BOARD_ID: &str = "portenta-h7";
/// Board identity (forge conventions).
#[cfg(feature = "board-portenta-h7-lite")]
pub const BOARD_ID: &str = "portenta-h7-lite";

/// Firmware identity reported in the boot line.
pub const FIRMWARE: &str = concat!("teios-h747/", env!("CARGO_PKG_VERSION"));

/// This board, as stamped on every JSON line.
pub const BOARD: BoardConfig = BoardConfig {
    board_id: BOARD_ID,
    firmware: FIRMWARE,
};

/// Software CRC32 on the Cortex-M7 at 480 MHz.
pub const SUBSTRATE_M7: &str = "cpu-m7@480mhz";
/// Software CRC32 on the Cortex-M4 at 240 MHz (the second core).
pub const SUBSTRATE_M4: &str = "cpu-m4@240mhz";
/// CRC32 by the STM32 hardware CRC peripheral (CPU free to sleep).
pub const SUBSTRATE_CRC_HW: &str = "crc-hw";

// ---------------------------------------------------------------------------
// Shipped cost table — Table tier (ILLUSTRATIVE; pending the bench pass).
// ---------------------------------------------------------------------------

/// ILLUSTRATIVE J/op, M7 software CRC32 over [`BUF_LEN`] at 480 MHz.
/// Placeholder Table-tier default pending the bench (no published
/// figure); `joules_source: table` keeps it honest.
pub const M7_J_PER_OP: f64 = 9.0e-5;
/// ILLUSTRATIVE J/op, M4 software CRC32 at 240 MHz — slower clock, fewer
/// joules/s but more seconds; placeholder until measured.
pub const M4_J_PER_OP: f64 = 1.1e-4;
/// ILLUSTRATIVE J/op, STM32 CRC peripheral (the cheap path — hardware
/// computes while the core idles); placeholder until measured.
pub const CRC_HW_J_PER_OP: f64 = 6.0e-6;

/// The cost table this image ships with — all three substrates at the
/// Table tier (no calibration has touched this board yet).
pub fn shipped_cost_table() -> CostTable<COST_CAPACITY> {
    let mut t = CostTable::new();
    for (sub, j) in [
        (SUBSTRATE_M7, M7_J_PER_OP),
        (SUBSTRATE_M4, M4_J_PER_OP),
        (SUBSTRATE_CRC_HW, CRC_HW_J_PER_OP),
    ] {
        let _ = t.upsert(CostEntry {
            primitive_id: PRIMITIVE_HASH,
            substrate: sub,
            j_per_op: j,
            source: JoulesSource::Table,
        });
    }
    t
}

// ---------------------------------------------------------------------------
// JSON-lines writers — teios-core's, with this board's identity bound.
// ---------------------------------------------------------------------------

/// `{"type":"boot",...}` — once after USB connects. The M7 has a real
/// DWT CYCCNT, so the firmware passes `cyccnt: true`.
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
    use serde_json::Value;

    #[test]
    fn cost_table_prices_three_substrates_crc_hw_cheapest() {
        let t = shipped_cost_table();
        let cheapest = t.cheapest(PRIMITIVE_HASH).unwrap();
        assert_eq!(cheapest.substrate, SUBSTRATE_CRC_HW);
        assert_eq!(t.for_primitive(PRIMITIVE_HASH).count(), 3);
    }

    #[test]
    fn ledger_line_matches_tei_ledger_serde_shape() {
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = 650_000; // real M7 DWT cycles
        l.active_us = 1354;
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_M7, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "ledger");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["substrate"], "cpu-m7@480mhz");
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        assert!(v["ledger"].get("joules").is_none());
    }

    #[test]
    fn boot_line_reports_cyccnt_true_on_m7() {
        let mut b = String::new();
        write_boot_line(&mut b, true).unwrap();
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["type"], "boot");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["cyccnt"], true);
    }
}
