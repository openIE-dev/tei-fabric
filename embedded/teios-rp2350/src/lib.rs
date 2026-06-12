//! teiOS E1 — host-testable core of the RP2350 firmware.
//!
//! The firmware binary ([`main.rs`](../src/main.rs)) owns the hardware
//! (USB CDC, DWT CYCCNT, the DMA sniffer); everything that can be proven
//! on a host lives here: the software CRC32 substrate, the shipped cost
//! table, and the JSON-lines writer whose wire format is locked to
//! `tei-ledger`'s serde shape by the tests at the bottom of this file.
//!
//! # The JSON-lines protocol (what TEI Studio's web console parses)
//!
//! One JSON object per `\n`-terminated line on the USB CDC serial port,
//! discriminated by `"type"`. Fields follow `tei-ledger`'s serde shape:
//! snake_case names, `None` fields **omitted** (never `null`). Numbers
//! may use exponent notation (`4e-6`) — any JSON parser accepts both.
//!
//! - `{"type":"boot","board_id":"pico2","firmware":"teios-rp2350/0.1.0",
//!    "primitive_id":36,"buf_len":65536,"cyccnt":true}`
//!   Once after USB connects. `cyccnt:false` means the M33 reported
//!   `DWT_CTRL.NOCYCCNT` (no cycle counter) and cpu ledgers carry
//!   `cycles:0`.
//! - `{"type":"ledger","board_id":"pico2","substrate":"cpu@150mhz",
//!    "primitive_id":36,"n_ops":1,"ledger":{"cycles":524288,
//!    "dma_transfers":0,"adc_samples":0,"accel_invocations":0,
//!    "sleep_us":0,"active_us":3495,"joules_source":"table"}}`
//!   One per primitive run per substrate. The `ledger` object is exactly
//!   `tei_ledger::EventLedger` in serde form; `instructions`/`joules`
//!   appear only when present.
//! - `{"type":"check","ok":true,"crc_cpu":2378668581,"crc_dma":2378668581}`
//!   Both substrates ran the same primitive over the same buffer; `ok`
//!   is their agreement. CRCs are the u32 values as JSON numbers.
//! - `{"type":"dispatch","primitive_id":36,"chosen":"dma-sniffer",
//!    "j_per_op":4e-6,"joules_source":"table","alternatives":[
//!    {"substrate":"cpu@150mhz","j_per_op":9e-5,"joules_source":"table"}]}`
//!   The lowest-joule verdict from [`CostTable::cheapest`], plus every
//!   priced alternative for the primitive.

#![cfg_attr(not(test), no_std)]

use core::fmt::{self, Write};

use tei_ledger::{CostEntry, CostTable, EventLedger, JoulesSource};

/// Periodic Stack primitive id for Hash (embedded profile).
pub const PRIMITIVE_HASH: u32 = 36;

/// Board identity, forge conventions.
pub const BOARD_ID: &str = "pico2";

/// Firmware identity reported in the boot line.
pub const FIRMWARE: &str = concat!("teios-rp2350/", env!("CARGO_PKG_VERSION"));

/// Substrate name for the software CRC32 on the Cortex-M33 at the
/// embassy-rp default 150 MHz system clock.
pub const SUBSTRATE_CPU: &str = "cpu@150mhz";

/// Substrate name for the RP2350 DMA sniffer (hardware CRC32 computed
/// by the DMA engine while the CPU is free to sleep).
pub const SUBSTRATE_DMA: &str = "dma-sniffer";

/// Size of the workload buffer: 64 KiB.
pub const BUF_LEN: usize = 64 * 1024;

// ---------------------------------------------------------------------------
// Shipped cost table — Table tier.
// ---------------------------------------------------------------------------

/// ILLUSTRATIVE J/op for one CRC32 pass over [`BUF_LEN`] on the M33 at
/// 150 MHz (software, table-driven). Placeholder default pending the
/// novel E1 bench measurement (PPK2) — no published figure exists; the
/// ledger says `joules_source: table` so nothing overclaims.
pub const CPU_J_PER_OP: f64 = 9.0e-5;

/// ILLUSTRATIVE J/op for the same pass via the DMA sniffer. Same
/// caveat as [`CPU_J_PER_OP`]: a placeholder Table-tier default, kept
/// plausibly proportioned (the sniffer moves one word per DMA transfer
/// with the CPU idle) until the E1 bench replaces it.
pub const DMA_J_PER_OP: f64 = 4.0e-6;

/// Capacity of the board's price list.
pub const COST_CAPACITY: usize = 8;

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
// The primitive: CRC32 (IEEE 802.3, reflected — zlib/`crc32` compatible).
// ---------------------------------------------------------------------------

/// Reflected CRC-32 polynomial (0x04C11DB7 bit-reversed).
const CRC32_POLY_REFLECTED: u32 = 0xEDB8_8320;

const fn crc32_table() -> [u32; 256] {
    let mut table = [0u32; 256];
    let mut i = 0;
    while i < 256 {
        let mut c = i as u32;
        let mut k = 0;
        while k < 8 {
            c = if c & 1 != 0 {
                CRC32_POLY_REFLECTED ^ (c >> 1)
            } else {
                c >> 1
            };
            k += 1;
        }
        table[i] = c;
        i += 1;
    }
    table
}

static CRC32_TABLE: [u32; 256] = crc32_table();

/// Software CRC32 — the `cpu@150mhz` substrate. Deterministic,
/// table-driven, byte at a time. Matches zlib's `crc32()`, and matches
/// the DMA sniffer configured as CRC32R with `OUT_REV | OUT_INV` and
/// seed `0xFFFF_FFFF`.
pub fn crc32_software(data: &[u8]) -> u32 {
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = CRC32_TABLE[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

/// Fill the workload buffer with a deterministic xorshift32 pattern so
/// every board hashes identical bytes (and a host test can pin the CRC).
pub fn fill_pattern(buf: &mut [u8]) {
    let mut state = 0x9E37_79B9u32;
    for b in buf.iter_mut() {
        // xorshift32 (Marsaglia)
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        *b = (state >> 24) as u8;
    }
}

// ---------------------------------------------------------------------------
// JSON-lines writer — hand-rolled, no_std, allocation-free.
// Wire format locked to tei-ledger's serde shape by the host tests.
// ---------------------------------------------------------------------------

/// serde's `rename_all = "snake_case"` name for a provenance tier.
pub const fn joules_source_str(s: JoulesSource) -> &'static str {
    match s {
        JoulesSource::Measured => "measured",
        JoulesSource::CyclesProxy => "cycles_proxy",
        JoulesSource::Table => "table",
    }
}

/// The `ledger` object body, exactly as `serde_json` would emit
/// `EventLedger`: declaration order, `None` fields omitted.
fn write_ledger_object<W: Write>(w: &mut W, l: &EventLedger) -> fmt::Result {
    write!(w, "{{\"cycles\":{}", l.cycles)?;
    if let Some(i) = l.instructions {
        write!(w, ",\"instructions\":{i}")?;
    }
    write!(
        w,
        ",\"dma_transfers\":{},\"adc_samples\":{},\"accel_invocations\":{},\"sleep_us\":{},\"active_us\":{}",
        l.dma_transfers, l.adc_samples, l.accel_invocations, l.sleep_us, l.active_us
    )?;
    if let Some(j) = l.joules {
        write!(w, ",\"joules\":{j:e}")?;
    }
    write!(
        w,
        ",\"joules_source\":\"{}\"}}",
        joules_source_str(l.joules_source)
    )
}

/// `{"type":"boot",...}` — once after USB connects.
pub fn write_boot_line<W: Write>(w: &mut W, cyccnt: bool) -> fmt::Result {
    write!(
        w,
        "{{\"type\":\"boot\",\"board_id\":\"{BOARD_ID}\",\"firmware\":\"{FIRMWARE}\",\"primitive_id\":{PRIMITIVE_HASH},\"buf_len\":{BUF_LEN},\"cyccnt\":{cyccnt}}}"
    )
}

/// `{"type":"ledger",...}` — one primitive run on one substrate.
pub fn write_ledger_line<W: Write>(
    w: &mut W,
    substrate: &str,
    n_ops: u64,
    ledger: &EventLedger,
) -> fmt::Result {
    write!(
        w,
        "{{\"type\":\"ledger\",\"board_id\":\"{BOARD_ID}\",\"substrate\":\"{substrate}\",\"primitive_id\":{PRIMITIVE_HASH},\"n_ops\":{n_ops},\"ledger\":"
    )?;
    write_ledger_object(w, ledger)?;
    w.write_char('}')
}

/// `{"type":"check",...}` — cross-substrate agreement on the result.
pub fn write_check_line<W: Write>(w: &mut W, crc_cpu: u32, crc_dma: u32) -> fmt::Result {
    write!(
        w,
        "{{\"type\":\"check\",\"ok\":{},\"crc_cpu\":{crc_cpu},\"crc_dma\":{crc_dma}}}",
        crc_cpu == crc_dma
    )
}

/// `{"type":"dispatch",...}` — the lowest-joule verdict plus every
/// priced alternative. Returns the chosen substrate name (`None` when
/// the table holds no price for the primitive — nothing is written).
pub fn write_dispatch_line<W: Write, const N: usize>(
    w: &mut W,
    table: &CostTable<N>,
    primitive_id: u32,
) -> Result<Option<&'static str>, fmt::Error> {
    let Some(chosen) = table.cheapest(primitive_id) else {
        return Ok(None);
    };
    write!(
        w,
        "{{\"type\":\"dispatch\",\"primitive_id\":{primitive_id},\"chosen\":\"{}\",\"j_per_op\":{:e},\"joules_source\":\"{}\",\"alternatives\":[",
        chosen.substrate,
        chosen.j_per_op,
        joules_source_str(chosen.source)
    )?;
    let mut first = true;
    for e in table.for_primitive(primitive_id) {
        if e.substrate == chosen.substrate {
            continue;
        }
        if !first {
            w.write_char(',')?;
        }
        first = false;
        write!(
            w,
            "{{\"substrate\":\"{}\",\"j_per_op\":{:e},\"joules_source\":\"{}\"}}",
            e.substrate,
            e.j_per_op,
            joules_source_str(e.source)
        )?;
    }
    w.write_str("]}")?;
    Ok(Some(chosen.substrate))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    /// Independent bit-at-a-time reference (no table) for cross-checking
    /// the table-driven implementation.
    fn crc32_bitwise(data: &[u8]) -> u32 {
        let mut c = 0xFFFF_FFFFu32;
        for &b in data {
            c ^= b as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 {
                    CRC32_POLY_REFLECTED ^ (c >> 1)
                } else {
                    c >> 1
                };
            }
        }
        c ^ 0xFFFF_FFFF
    }

    #[test]
    fn crc32_known_vectors() {
        // The canonical CRC-32/ISO-HDLC check value.
        assert_eq!(crc32_software(b"123456789"), 0xCBF4_3926);
        assert_eq!(crc32_software(b""), 0);
        assert_eq!(
            crc32_software(b"The quick brown fox jumps over the lazy dog"),
            0x414F_A339
        );
    }

    #[test]
    fn crc32_table_matches_bitwise_over_pattern_buffer() {
        let mut buf = vec![0u8; BUF_LEN];
        fill_pattern(&mut buf);
        assert_eq!(crc32_software(&buf), crc32_bitwise(&buf));
        // Determinism: refilling produces the identical buffer.
        let mut again = vec![0u8; BUF_LEN];
        fill_pattern(&mut again);
        assert_eq!(buf, again);
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

    /// The wire-format lock: the hand-rolled ledger object must equal
    /// serde_json's rendering of the same `EventLedger`, including the
    /// omission (not nulling) of `None` fields.
    #[test]
    fn ledger_line_matches_tei_ledger_serde_shape() {
        let mut l = EventLedger::new(JoulesSource::Table);
        l.cycles = 524_288;
        l.active_us = 3_495;
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_CPU, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "ledger");
        assert_eq!(v["board_id"], "pico2");
        assert_eq!(v["substrate"], "cpu@150mhz");
        assert_eq!(v["primitive_id"], 36);
        assert_eq!(v["n_ops"], 1);
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        // None fields are omitted, never null.
        assert!(v["ledger"].get("instructions").is_none());
        assert!(v["ledger"].get("joules").is_none());
    }

    #[test]
    fn ledger_line_carries_some_fields_when_present() {
        let mut l = EventLedger::new(JoulesSource::CyclesProxy);
        l.cycles = 100;
        l.instructions = Some(80);
        l.dma_transfers = 16_384;
        l.accel_invocations = 1;
        l.joules = Some(4.0e-6);
        let mut s = String::new();
        write_ledger_line(&mut s, SUBSTRATE_DMA, 1, &l).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["ledger"], serde_json::to_value(l).unwrap());
        assert_eq!(v["ledger"]["instructions"], 80);
        assert_eq!(v["ledger"]["joules_source"], "cycles_proxy");
        assert!((v["ledger"]["joules"].as_f64().unwrap() - 4.0e-6).abs() < 1e-12);
    }

    #[test]
    fn dispatch_line_names_cheapest_and_lists_alternatives() {
        let t = shipped_cost_table();
        let mut s = String::new();
        let chosen = write_dispatch_line(&mut s, &t, PRIMITIVE_HASH).unwrap();
        assert_eq!(chosen, Some(SUBSTRATE_DMA));
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "dispatch");
        assert_eq!(v["primitive_id"], 36);
        assert_eq!(v["chosen"], "dma-sniffer");
        assert_eq!(v["joules_source"], "table");
        assert!((v["j_per_op"].as_f64().unwrap() - DMA_J_PER_OP).abs() < 1e-18);
        let alts = v["alternatives"].as_array().unwrap();
        assert_eq!(alts.len(), 1);
        assert_eq!(
            alts[0],
            json!({"substrate": "cpu@150mhz", "j_per_op": CPU_J_PER_OP, "joules_source": "table"})
        );
        // Unpriced primitive: nothing written, None returned.
        let mut empty = String::new();
        assert_eq!(write_dispatch_line(&mut empty, &t, 9999).unwrap(), None);
        assert!(empty.is_empty());
    }

    #[test]
    fn check_and_boot_lines_parse() {
        let mut s = String::new();
        write_check_line(&mut s, 0xDEAD_BEEF, 0xDEAD_BEEF).unwrap();
        let v: Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["type"], "check");
        assert_eq!(v["ok"], true);
        assert_eq!(v["crc_cpu"], 0xDEAD_BEEFu32);

        let mut s2 = String::new();
        write_check_line(&mut s2, 1, 2).unwrap();
        let v2: Value = serde_json::from_str(&s2).unwrap();
        assert_eq!(v2["ok"], false);

        let mut b = String::new();
        write_boot_line(&mut b, true).unwrap();
        let v3: Value = serde_json::from_str(&b).unwrap();
        assert_eq!(v3["type"], "boot");
        assert_eq!(v3["board_id"], "pico2");
        assert_eq!(v3["primitive_id"], 36);
        assert_eq!(v3["buf_len"], 65_536);
        assert_eq!(v3["cyccnt"], true);
    }

    /// Every line the firmware can emit fits the firmware's line buffer.
    #[test]
    fn lines_fit_firmware_buffer() {
        const LINE_CAP: usize = 512; // keep in sync with main.rs
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
