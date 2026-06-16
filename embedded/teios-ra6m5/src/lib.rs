//! teiOS E1d — host-testable core of the Portenta C33 firmware
//! (Renesas RA6M5, R7FA6M5, Cortex-M33 @ 200 MHz).
//!
//! The firmware (`fw.rs`) owns the hardware (the RA6M5 CRCA hardware-CRC
//! peripheral, the M33 DWT cycle counter, the semihosting transport);
//! the board-independent protocol lives in the shared `teios-core`
//! crate, same as the RP2040/H7/nRF images. This crate pins the Portenta
//! C33 identity, its substrate names, and the shipped cost table.
//!
//! # Why this board: a genuine HW-vs-SW race on a third silicon family
//!
//! The RA6M5 has a **CRCA hardware-CRC peripheral** that computes CRC-32
//! in dedicated logic. So unlike the Nicla (single M33 path), the C33
//! gets a real two-substrate race for the Hash primitive: software
//! CRC32 on the M33 vs the CRCA peripheral, both priced, cross-checked,
//! the cheaper dispatched — the fabric model on the Renesas RA family,
//! the third MCU vendor in the matrix after RP and ST/Nordic.
//!
//! # No mature async HAL → bare-metal
//!
//! There is no embassy time-driver for the RA family, so the firmware is
//! plain `cortex-m-rt` + the `ra6m5-pac` register crate, with a
//! cycle-delay cadence (synchronous `app`, no async runtime).
//!
//! # Cycles on the M33
//!
//! The Cortex-M33 has an architectural DWT CYCCNT — enable
//! `DEMCR.TRCENA` + `DWT_CTRL.CYCCNTENA` and read `DWT.CYCCNT`. True
//! per-cycle granularity, so the boot line reports `cyccnt:true`.

#![cfg_attr(not(test), no_std)]

use core::fmt::{self, Write};

use tei_ledger::{CostEntry, CostTable, EventLedger, JoulesSource};
use teios_core::BoardConfig;
pub use teios_core::{BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, crc32_software, fill_pattern};

/// Board identity (forge conventions).
pub const BOARD_ID: &str = "portenta-c33";

/// Firmware identity reported in the boot line.
pub const FIRMWARE: &str = concat!("teios-ra6m5/", env!("CARGO_PKG_VERSION"));

/// This board, as stamped on every JSON line.
pub const BOARD: BoardConfig = BoardConfig {
    board_id: BOARD_ID,
    firmware: FIRMWARE,
};

/// Software CRC32 on the Cortex-M33 at 200 MHz.
pub const SUBSTRATE_M33: &str = "cpu-m33@200mhz";
/// CRC32 by the RA6M5 CRCA hardware-CRC peripheral (CPU free to sleep).
pub const SUBSTRATE_CRC_HW: &str = "crca";

// ---------------------------------------------------------------------------
// Shipped cost table — Table tier (ILLUSTRATIVE; pending the bench pass).
// ---------------------------------------------------------------------------

/// ILLUSTRATIVE J/op, M33 software CRC32 over [`BUF_LEN`] at 200 MHz.
/// Placeholder Table-tier default pending the bench; `joules_source:
/// table` keeps it honest.
pub const M33_J_PER_OP: f64 = 8.0e-5;
/// ILLUSTRATIVE J/op, CRCA peripheral (the cheap path — hardware
/// computes while the core idles); placeholder until measured.
pub const CRC_HW_J_PER_OP: f64 = 5.0e-6;

/// The cost table this image ships with — both substrates at the Table
/// tier (no calibration has touched this board yet).
pub fn shipped_cost_table() -> CostTable<COST_CAPACITY> {
    let mut t = CostTable::new();
    for (sub, j) in [(SUBSTRATE_M33, M33_J_PER_OP), (SUBSTRATE_CRC_HW, CRC_HW_J_PER_OP)] {
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

/// `{"type":"boot",...}` — once at startup. The M33 has a real DWT
/// CYCCNT, so the firmware passes `cyccnt: true`.
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
/// stamped with this board's id. The relay carries it to the fabric.
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
    fn cost_table_prices_two_substrates_crc_hw_cheapest() {
        let t = shipped_cost_table();
        let cheapest = t.cheapest(PRIMITIVE_HASH).unwrap();
        assert_eq!(cheapest.substrate, SUBSTRATE_CRC_HW);
        assert_eq!(t.for_primitive(PRIMITIVE_HASH).count(), 2);
    }

    #[test]
    fn ledger_line_matches_tei_ledger_serde_shape() {
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = 800_000; // real M33 DWT cycles
        l.active_us = 4000;
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_M33, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "ledger");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["substrate"], "cpu-m33@200mhz");
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        assert!(v["ledger"].get("joules").is_none());
    }

    #[test]
    fn boot_line_reports_cyccnt_true_on_m33() {
        let mut b = String::new();
        write_boot_line(&mut b, true).unwrap();
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["type"], "boot");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["cyccnt"], true);
    }
}
