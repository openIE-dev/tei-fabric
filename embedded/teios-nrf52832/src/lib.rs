//! teiOS E1c — host-testable core of the Nicla Voice / Nicla Sense ME
//! firmware (Nordic nRF52832, Cortex-M4F @ 64 MHz).
//!
//! The firmware (`fw.rs`) owns the hardware (the UART transport, the M4
//! DWT cycle counter); the board-independent protocol lives in the
//! shared `teios-core` crate, same as the RP2040/H7 images. This crate
//! pins the Nicla board identity, its substrate names, and the shipped
//! cost table.
//!
//! # The TEI story for these boards: "sleep is a substrate"
//!
//! The nRF52832 is a single general-compute core (the M4F), so for the
//! Hash primitive (CRC32 over [`BUF_LEN`]) there is exactly **one**
//! runnable substrate — the M4 software path, DWT-counted. What makes
//! these boards interesting to the fabric is the *other* die on the
//! module: an always-on AI accelerator that runs at sub-milliwatt while
//! the host M4 sleeps —
//!
//! - **Nicla Voice**: Syntiant **NDP120** always-on audio NPU.
//! - **Nicla Sense ME**: Bosch **BHI260AP** self-learning sensor-fusion hub.
//!
//! Those are *fixed-function* accelerators — they don't run our CRC32 —
//! so they appear in the shipped cost table as a **priced menu entry**
//! (`SUBSTRATE_ACCEL`), and actually offloading a primitive to them is
//! the documented bring-up stretch (mirroring how the H7's second M4
//! core is handled). The default app runs the M4 substrate, prices it,
//! and dispatches; when an accelerator kernel lands, it becomes a real
//! second row and the dispatch verdict can flip to "let the host sleep".
//!
//! # Cycles on the M4
//!
//! The Cortex-M4 has an architectural DWT CYCCNT — enable
//! `DEMCR.TRCENA` + `DWT_CTRL.CYCCNTENA` and read `DWT.CYCCNT`. True
//! per-cycle granularity, so the boot line reports `cyccnt:true`.
//!
//! # The JSON-lines protocol
//!
//! Identical to the other images, but the transport is **UART** (the
//! nRF52832 has no USB peripheral): one `\n`-terminated JSON object per
//! line, `tei-ledger` serde shape, `None` fields omitted.

#![cfg_attr(not(test), no_std)]

use core::fmt::{self, Write};

use tei_ledger::{CostEntry, CostTable, EventLedger, JoulesSource};
use teios_core::BoardConfig;
pub use teios_core::{BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, crc32_software, fill_pattern};

#[cfg(all(feature = "board-nicla-voice", feature = "board-nicla-sense"))]
compile_error!("enable exactly one board feature: board-nicla-voice OR board-nicla-sense");

/// Board identity (forge conventions).
#[cfg(not(feature = "board-nicla-sense"))]
pub const BOARD_ID: &str = "nicla-voice";
/// Board identity (forge conventions).
#[cfg(feature = "board-nicla-sense")]
pub const BOARD_ID: &str = "nicla-sense";

/// Firmware identity reported in the boot line.
pub const FIRMWARE: &str = concat!("teios-nrf52832/", env!("CARGO_PKG_VERSION"));

/// This board, as stamped on every JSON line.
pub const BOARD: BoardConfig = BoardConfig {
    board_id: BOARD_ID,
    firmware: FIRMWARE,
};

/// Software CRC32 on the Cortex-M4F at 64 MHz — the one general-compute
/// substrate for the Hash primitive on this silicon.
pub const SUBSTRATE_M4: &str = "cpu-m4@64mhz";

/// The always-on AI accelerator on the module — a priced menu entry; an
/// actual offload kernel is the documented bring-up stretch.
#[cfg(not(feature = "board-nicla-sense"))]
pub const SUBSTRATE_ACCEL: &str = "ndp120"; // Syntiant audio NPU (Nicla Voice)
#[cfg(feature = "board-nicla-sense")]
pub const SUBSTRATE_ACCEL: &str = "bhi260"; // Bosch sensor-fusion hub (Nicla Sense ME)

// ---------------------------------------------------------------------------
// Shipped cost table — Table tier (ILLUSTRATIVE; pending the bench pass).
// ---------------------------------------------------------------------------

/// ILLUSTRATIVE J/op, M4 software CRC32 over [`BUF_LEN`] at 64 MHz.
/// Placeholder Table-tier default pending the bench; `joules_source:
/// table` keeps it honest.
pub const M4_J_PER_OP: f64 = 1.4e-4;
/// ILLUSTRATIVE J/op for the always-on accelerator — sub-mW class, so
/// far cheaper *if* a CRC-equivalent kernel is ever mapped to it.
/// Placeholder until measured; present so the cost table shows the menu.
pub const ACCEL_J_PER_OP: f64 = 3.0e-6;

/// The cost table this image ships with: the M4 software substrate plus
/// the accelerator menu entry, both at the Table tier (no calibration
/// has touched this board yet).
pub fn shipped_cost_table() -> CostTable<COST_CAPACITY> {
    let mut t = CostTable::new();
    for (sub, j) in [(SUBSTRATE_M4, M4_J_PER_OP), (SUBSTRATE_ACCEL, ACCEL_J_PER_OP)] {
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

/// `{"type":"boot",...}` — once after the UART comes up. The M4 has a
/// real DWT CYCCNT, so the firmware passes `cyccnt: true`.
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

pub use teios_core::{write_check_line, write_dispatch_line};

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    #[test]
    fn cost_table_prices_m4_and_accel_accel_cheapest() {
        let t = shipped_cost_table();
        // The accelerator is the cheap menu entry; the M4 is the only
        // currently-runnable substrate. Both are present.
        assert_eq!(t.for_primitive(PRIMITIVE_HASH).count(), 2);
        let cheapest = t.cheapest(PRIMITIVE_HASH).unwrap();
        assert_eq!(cheapest.substrate, SUBSTRATE_ACCEL);
    }

    #[test]
    fn ledger_line_matches_tei_ledger_serde_shape() {
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = 600_000; // real M4 DWT cycles
        l.active_us = 9375;
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_M4, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "ledger");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["substrate"], "cpu-m4@64mhz");
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        assert!(v["ledger"].get("joules").is_none());
    }

    #[test]
    fn boot_line_reports_cyccnt_true_on_m4() {
        let mut b = String::new();
        write_boot_line(&mut b, true).unwrap();
        let v: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v["type"], "boot");
        assert_eq!(v["board_id"], BOARD_ID);
        assert_eq!(v["cyccnt"], true);
    }
}
