//! teiOS E1d firmware — Arduino Portenta C33 (Renesas RA6M5, M33).
//!
//! The FIXED harness: the M33 DWT cycle counter, the RA6M5 CRCA
//! hardware-CRC peripheral, the semihosting transport, and the
//! synchronous driver loop that builds a [`Tei`] and calls the user's
//! [`crate::app::app`]. Only `src/app.rs` is user-editable.
//!
//! ## Hardware-verification status (read this)
//!
//! Compile-verified for `thumbv8m.main-none-eabihf`; **not yet
//! hardware-verified.** The bench items (silent at compile time):
//!
//! 1. **Transport.** Output goes over **semihosting** (`hstdout`), which
//!    streams the ledger to whatever debug probe is attached — no
//!    clock/baud/pin-mux bring-up needed, which is exactly how you bring
//!    a board up before its UART is wired. Production moves to an SCI
//!    UART (the RA6M5 SCI clock-tree + PFS pin-mux is the bench step).
//! 2. **CRCA correctness + cadence.** The CRCA register sequence below
//!    targets zlib/IEEE CRC-32 (GPS=CRC-32, LSB-first/reflected, seed
//!    0xFFFF_FFFF, software final XOR). On-die agreement with the M33
//!    software path is the `check` line every pass. The 1 Hz cadence is
//!    a fixed cycle-delay against the *reset* clock (no PLL bring-up),
//!    so it is approximate until the clock tree is configured.
//!
//! ## The two substrates
//!
//! - `cpu-m33@200mhz` — software CRC32, DWT-counted.
//! - `crca` — the RA6M5 hardware CRC peripheral. The CPU is free while
//!   it computes; priced as the cheap path in the cost table.

use core::fmt::Write as _;

use cortex_m_rt::entry;
use cortex_m_semihosting::hio::{self, HostStream};
use heapless::String;
use panic_halt as _;
use ra6m5_pac::crc::{Crcdor, CrcdirBy, crccr0::{Gps, Lms}};
use ra6m5_pac::mstp::mstpcrc::Mstpc1;
use ra6m5_pac::{self as pac, NoBitfieldReg};
use tei_ledger::{CostTable, EventLedger, JoulesSource};
use teios_ra6m5::{
    BUF_LEN, COST_CAPACITY, SUBSTRATE_CRC_HW, crc32_software, shipped_cost_table, write_boot_line,
    write_check_line, write_dispatch_line, write_ledger_line,
};

const LINE_CAP: usize = teios_core::LINE_CAP;

/// The 64 KiB workload, in flash. `const fn` mirrors
/// `teios_core::fill_pattern` (xorshift32) byte-for-byte so the CRC
/// matches every other board.
const fn build_workload() -> [u8; BUF_LEN] {
    let mut buf = [0u8; BUF_LEN];
    let mut state = 0x9E37_79B9u32;
    let mut i = 0;
    while i < BUF_LEN {
        state ^= state << 13;
        state ^= state >> 17;
        state ^= state << 5;
        buf[i] = (state >> 24) as u8;
        i += 1;
    }
    buf
}
static WORKLOAD: [u8; BUF_LEN] = build_workload();

/// The M33 cycle source: the real DWT CYCCNT.
struct DwtCycleSource;
impl tei_ledger::CycleSource for DwtCycleSource {
    fn now(&self) -> u64 {
        cortex_m::peripheral::DWT::cycle_count() as u64
    }
    fn delta(&self, start: u64) -> u64 {
        (self.now() as u32).wrapping_sub(start as u32) as u64
    }
}

/// Release the CRCA module stop and configure it for zlib/IEEE CRC-32.
/// MSTPCRC.MSTPC1 is the CRC module-stop bit (1 = stopped at reset).
fn crca_init() {
    unsafe {
        // MSTPC1 = 0 cancels the CRC module stop (1 = stopped at reset).
        pac::MSTP.mstpcrc().modify(|r| r.mstpc1().set(Mstpc1::new(0)));
        // GPS = 0b100 (CRC-32), LMS = 0 (LSB-first → reflected, zlib).
        pac::CRC
            .crccr0()
            .modify(|r| r.gps().set(Gps::new(0b100)).lms().set(Lms::new(0)));
    }
}

/// Compute zlib CRC-32 with the CRCA peripheral: seed 0xFFFF_FFFF into
/// CRCDOR, feed bytes through CRCDIR_BY, read CRCDOR, apply final XOR.
fn crca_crc32(buf: &[u8]) -> u32 {
    unsafe {
        pac::CRC
            .crcdor()
            .write(Crcdor::default().set(0xFFFF_FFFF));
        for &b in buf {
            pac::CRC.crcdir_by().write(CrcdirBy::default().set(b));
        }
        pac::CRC.crcdor().read().get() ^ 0xFFFF_FFFF
    }
}

/// The user-app harness — defined in `tei`, exported for `app.rs`.
pub mod tei {
    use super::*;

    pub type TeiError = ();

    #[allow(dead_code)]
    pub struct Run {
        /// CRC result — apps read it to cross-check substrates.
        pub result: u32,
        /// The priced ledger for this run — apps may inspect it.
        pub ledger: EventLedger,
    }

    /// The safe surface an app may touch on the Portenta C33.
    pub struct Tei<'a> {
        pub(super) out: &'a mut HostStream,
        pub(super) cycles: &'a DwtCycleSource,
        pub(super) buf: &'a [u8; BUF_LEN],
        pub(super) table: &'a CostTable<COST_CAPACITY>,
        pub(super) line: String<LINE_CAP>,
    }

    impl<'a> Tei<'a> {
        /// Run `primitive` on the named substrate; price it; stream the
        /// ledger line. `crca` uses the hardware peripheral; everything
        /// else the M33 software path (DWT-counted).
        pub fn run_on(&mut self, substrate: &'static str, _primitive: u32) -> Result<Run, TeiError> {
            use tei_ledger::CycleSource;
            let c0 = self.cycles.now();
            let (result, accel) = if substrate == SUBSTRATE_CRC_HW {
                (crca_crc32(self.buf), true)
            } else {
                (crc32_software(self.buf), false)
            };
            let mut l = EventLedger::new(JoulesSource::Table);
            l.cycles = self.cycles.delta(c0);
            if accel {
                l.accel_invocations = 1;
            }
            self.line.clear();
            write_ledger_line(&mut self.line, substrate, 1, &l).ok();
            self.send()?;
            Ok(Run { result, ledger: l })
        }

        pub fn check(&mut self, a: u32, b: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_check_line(&mut self.line, a, b).ok();
            self.send()
        }

        pub fn dispatch(&mut self, primitive: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_dispatch_line(&mut self.line, self.table, primitive).ok();
            self.send()
        }

        #[allow(dead_code)]
        pub fn buf(&self) -> &[u8] {
            self.buf
        }

        /// Approximate delay — fixed cycle count against the reset clock
        /// (no PLL bring-up). Bench-tunable once the clock tree is set.
        pub fn sleep_ms(&mut self, ms: u64) {
            // ~reset-clock cycles; correct order of magnitude, not exact.
            cortex_m::asm::delay((ms as u32).saturating_mul(8_000));
        }

        fn send(&mut self) -> Result<(), TeiError> {
            self.out.write_str(self.line.as_str()).map_err(|_| ())?;
            self.out.write_str("\n").map_err(|_| ())
        }
    }
}
use tei::Tei;

#[entry]
fn main() -> ! {
    // M33 DWT cycle counter.
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();
    let cycles = DwtCycleSource;

    crca_init();

    let table = shipped_cost_table();
    let mut out = hio::hstdout().unwrap();

    let mut boot: String<LINE_CAP> = String::new();
    write_boot_line(&mut boot, true).ok(); // M33 has DWT CYCCNT
    let _ = out.write_str(boot.as_str());
    let _ = out.write_str("\n");

    loop {
        let mut tei = Tei {
            out: &mut out,
            cycles: &cycles,
            buf: &WORKLOAD,
            table: &table,
            line: String::new(),
        };
        let _ = crate::app::app(&mut tei);
    }
}
