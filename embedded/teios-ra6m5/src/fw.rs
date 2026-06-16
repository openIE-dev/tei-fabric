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
use tei_ledger::{EnergyMeter, EventLedger};
use tei_rt::{Runtime, Substrate};
use teios_ra6m5::{
    BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, SUBSTRATE_CRC_HW, SUBSTRATE_M33, crc32_software,
    shipped_cost_table, write_boot_line, write_check_line, write_dispatch_line, write_ledger_line,
    write_report_line,
};

/// The Portenta C33's substrates: M33 software CRC32 and the RA6M5 CRCA
/// hardware-CRC peripheral. Both are pure `fn(&[u8]) -> u32` (the CRCA one
/// pokes the PAC directly, after `crca_init`), so the runtime context is `()`.
fn m33_substrate(_: &mut (), d: &[u8]) -> u32 {
    crc32_software(d)
}
fn crca_substrate(_: &mut (), d: &[u8]) -> u32 {
    crca_crc32(d)
}
const SUBSTRATES: &[Substrate<()>] = &[
    Substrate { id: SUBSTRATE_M33, primitive_id: PRIMITIVE_HASH, run: m33_substrate },
    Substrate { id: SUBSTRATE_CRC_HW, primitive_id: PRIMITIVE_HASH, run: crca_substrate },
];

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
        /// The teiOS runtime kernel (cost table + substrate registry).
        pub(super) rt: &'a mut Runtime<'static, (), COST_CAPACITY>,
        pub(super) line: String<LINE_CAP>,
        /// Optional INA228 (`--features measured-ina228`, over `crate::riic`).
        /// When present, ledgers carry Measured joules instead of Table-tier
        /// constants. `'static` dyn — the meter lives for the whole program.
        pub(super) meter: Option<&'a mut (dyn EnergyMeter + 'static)>,
    }

    impl<'a> Tei<'a> {
        /// Run `primitive` on the named substrate; price it; stream the
        /// ledger line. `crca` uses the hardware peripheral; everything
        /// else the M33 software path (DWT-counted).
        pub fn run_on(&mut self, substrate: &'static str, primitive: u32) -> Result<Run, TeiError> {
            // The runtime does the work: run the named substrate, time it on
            // the DWT cycle source, read the meter, re-price the table.
            let run = match self.rt.run_on(
                substrate,
                &mut (),
                self.buf,
                1,
                self.cycles,
                self.meter.as_deref_mut(),
            ) {
                Some(r) => r,
                None => self
                    .rt
                    .run(primitive, &mut (), self.buf, 1, self.cycles, self.meter.as_deref_mut())
                    .ok_or(())?,
            };
            self.emit(run)
        }

        /// The teiOS call: run `primitive` on the runtime's dispatched
        /// substrate — the kernel chooses (M33 vs CRCA by measured joules).
        pub fn run(&mut self, primitive: u32) -> Result<Run, TeiError> {
            let run = self
                .rt
                .run(primitive, &mut (), self.buf, 1, self.cycles, self.meter.as_deref_mut())
                .ok_or(())?;
            self.emit(run)
        }

        /// Decorate a runtime [`tei_rt::Run`] with the board's extra counters,
        /// stream it, return the result + ledger.
        fn emit(&mut self, run: tei_rt::Run) -> Result<Run, TeiError> {
            let mut ledger = run.ledger;
            if run.substrate == SUBSTRATE_CRC_HW {
                ledger.accel_invocations = 1;
            }
            self.line.clear();
            write_ledger_line(&mut self.line, run.substrate, 1, &ledger).ok();
            self.send()?;
            Ok(Run { result: run.result, ledger })
        }

        /// Publish the runtime's calibrated prices home — one `report` line
        /// per priced substrate; the relay carries them to the fabric.
        pub fn publish(&mut self, primitive: u32) -> Result<(), TeiError> {
            let mut snap: [Option<tei_ledger::CostEntry>; COST_CAPACITY] = [None; COST_CAPACITY];
            let mut k = 0;
            for e in self.rt.costs().for_primitive(primitive) {
                snap[k] = Some(*e);
                k += 1;
            }
            for e in snap.iter().take(k).flatten() {
                self.line.clear();
                write_report_line(&mut self.line, e, 1).ok();
                self.send()?;
            }
            Ok(())
        }

        pub fn check(&mut self, a: u32, b: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_check_line(&mut self.line, a, b).ok();
            self.send()
        }

        pub fn dispatch(&mut self, primitive: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_dispatch_line(&mut self.line, self.rt.costs(), primitive).ok();
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

    // The teiOS runtime: the M33 + CRCA substrates and the shipped cost table.
    let mut rt = Runtime::new(SUBSTRATES, shipped_cost_table());
    let mut out = hio::hstdout().unwrap();

    // Optional INA228 on IIC0 via our RIIC master. BENCH-PENDING: the SCL/SDA
    // pins + shunt (0.015 Ω) / max-current (5 A) must match the part wired
    // in-line on the C33 carrier's supply rail. Without the feature the
    // ledger stays Table-tier.
    #[cfg(feature = "measured-ina228")]
    let mut ina = {
        let i2c = unsafe { crate::riic::Riic0::new() };
        tei_ina228::Ina228::new(i2c, tei_ina228::DEFAULT_ADDR, 0.015, 5.0, true).ok()
    };

    let mut boot: String<LINE_CAP> = String::new();
    write_boot_line(&mut boot, true).ok(); // M33 has DWT CYCCNT
    let _ = out.write_str(boot.as_str());
    let _ = out.write_str("\n");

    loop {
        #[cfg(feature = "measured-ina228")]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> =
            ina.as_mut().map(|m| m as &mut (dyn EnergyMeter + 'static));
        #[cfg(not(feature = "measured-ina228"))]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> = None;
        let mut tei = Tei {
            out: &mut out,
            cycles: &cycles,
            buf: &WORKLOAD,
            rt: &mut rt,
            line: String::new(),
            meter,
        };
        let _ = crate::app::app(&mut tei);
    }
}
