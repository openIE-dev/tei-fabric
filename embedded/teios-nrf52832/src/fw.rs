//! teiOS E1c firmware — Arduino Nicla Voice / Nicla Sense ME
//! (Nordic nRF52832, Cortex-M4F).
//!
//! The FIXED harness: the M4 DWT cycle counter, the UART transport, the
//! software-CRC32 substrate, and the once-per-second driver loop that
//! builds a [`Tei`] and calls the user's [`crate::app::app`]. Only
//! `src/app.rs` is user-editable through the forge.
//!
//! ## Hardware-verification status (read this)
//!
//! Compile-verified for `thumbv7em-none-eabihf`; **not yet
//! hardware-verified.** Two board-specific items are the on-bench step:
//!
//! 1. **UART pins + transport.** The nRF52832 has **no USB**, so the
//!    ledger stream goes out a UARTE at 115200. The TX/RX pins below
//!    (P0.06/P0.08, the nRF52 default) must be mapped to whatever the
//!    Nicla carrier exposes (ESLOV / debug header / a USB-UART bridge);
//!    that pin map is the bench step.
//! 2. **Flash base.** `memory.x` is the bare-metal layout (app at 0x0);
//!    a Nicla with its Arduino bootloader places the app above it. Set
//!    `FLASH ORIGIN` to the bootloader's app offset when flashing
//!    through it.
//!
//! ## RAM-constrained: the workload lives in flash
//!
//! The nRF52832 has only **64 KB of RAM** — exactly [`BUF_LEN`] — so the
//! 64 KiB workload cannot sit in RAM like it does on the RP2040/H7. It
//! is a `const`-built xorshift32 array in flash (`.rodata`), byte-for-byte
//! identical to `teios_core::fill_pattern`, so the CRC matches every
//! other board. The M4 reads it straight from flash (XIP).

use embassy_executor::Spawner;
use embassy_nrf::uarte::{self, Uarte};
use embassy_nrf::{bind_interrupts, peripherals};
use embassy_time::{Duration, Instant, Ticker, Timer};
use heapless::String;
use panic_halt as _;
use tei_ledger::{EnergyMeter, EventLedger};
use tei_rt::{Runtime, Substrate};
use teios_nrf52832::{
    BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, SUBSTRATE_M4, crc32_software, shipped_cost_table,
    write_boot_line, write_check_line, write_dispatch_line, write_ledger_line, write_report_line,
};

/// The Nicla nRF52832's only runnable substrate: software CRC32 on the M4
/// (pure compute → context `()`). The always-on accelerator (NDP120 / BHI260)
/// is a priced cost-table menu entry, not a runtime substrate.
fn m4_substrate(_: &mut (), d: &[u8]) -> u32 {
    crc32_software(d)
}
const SUBSTRATES: &[Substrate<()>] = &[Substrate {
    id: SUBSTRATE_M4,
    primitive_id: PRIMITIVE_HASH,
    run: m4_substrate,
}];

bind_interrupts!(struct Irqs {
    UARTE0 => uarte::InterruptHandler<peripherals::UARTE0>;
});

// TWIM (I2C) interrupt — only when the INA228 meter is built in.
#[cfg(feature = "measured-ina228")]
bind_interrupts!(struct IrqsI2c {
    TWISPI0 => embassy_nrf::twim::InterruptHandler<peripherals::TWISPI0>;
});

const LINE_CAP: usize = teios_core::LINE_CAP;

/// The 64 KiB workload, in flash. `const fn` mirrors
/// `teios_core::fill_pattern` (xorshift32) byte-for-byte.
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

/// The M4 cycle source: the real DWT CYCCNT (architectural on Cortex-M4).
struct DwtCycleSource;
impl tei_ledger::CycleSource for DwtCycleSource {
    fn now(&self) -> u64 {
        cortex_m::peripheral::DWT::cycle_count() as u64
    }
    fn delta(&self, start: u64) -> u64 {
        (self.now() as u32).wrapping_sub(start as u32) as u64
    }
}

/// The user-app harness — defined in `tei`, exported for `app.rs`.
pub mod tei {
    use super::*;

    pub type TeiError = uarte::Error;

    #[allow(dead_code)]
    pub struct Run {
        /// CRC result — apps read it to cross-check substrates.
        pub result: u32,
        /// The priced ledger for this run — apps may inspect it.
        pub ledger: EventLedger,
    }

    /// The safe surface an app may touch on the Nicla nRF52832.
    pub struct Tei<'a> {
        pub(super) uart: &'a mut Uarte<'static, peripherals::UARTE0>,
        pub(super) cycles: &'a DwtCycleSource,
        pub(super) buf: &'a [u8; BUF_LEN],
        /// The teiOS runtime kernel (cost table + substrate registry).
        pub(super) rt: &'a mut Runtime<'static, (), COST_CAPACITY>,
        pub(super) line: String<LINE_CAP>,
        /// Optional INA228 (`--features measured-ina228`). When present,
        /// ledgers carry Measured joules. `'static` dyn — lives for the program.
        pub(super) meter: Option<&'a mut (dyn EnergyMeter + 'static)>,
    }

    impl<'a> Tei<'a> {
        /// Run `primitive` on the named substrate; price it; stream the
        /// ledger line. Only the M4 software path is runnable on this
        /// silicon; the accelerator is a priced cost-table menu entry.
        pub async fn run_on(
            &mut self,
            substrate: &'static str,
            primitive: u32,
        ) -> Result<Run, TeiError> {
            let t0 = Instant::now();
            let run = match self
                .rt
                .run_on(substrate, &mut (), self.buf, 1, self.cycles, self.meter.as_deref_mut())
            {
                Some(r) => r,
                None => self
                    .rt
                    .run(primitive, &mut (), self.buf, 1, self.cycles, self.meter.as_deref_mut())
                    .ok_or(uarte::Error::BufferNotInRAM)?,
            };
            self.emit(run, t0).await
        }

        /// The teiOS call: run `primitive` on the runtime's dispatched
        /// substrate (here always the M4 — the only runnable one).
        pub async fn run(&mut self, primitive: u32) -> Result<Run, TeiError> {
            let t0 = Instant::now();
            let run = self
                .rt
                .run(primitive, &mut (), self.buf, 1, self.cycles, self.meter.as_deref_mut())
                .ok_or(uarte::Error::BufferNotInRAM)?;
            self.emit(run, t0).await
        }

        async fn emit(&mut self, run: tei_rt::Run, t0: Instant) -> Result<Run, TeiError> {
            let mut ledger = run.ledger;
            ledger.active_us = t0.elapsed().as_micros();
            self.line.clear();
            write_ledger_line(&mut self.line, run.substrate, 1, &ledger).ok();
            send_line(self.uart, &self.line).await?;
            Ok(Run { result: run.result, ledger })
        }

        /// Publish the runtime's calibrated prices home (Studio → HUB/FLEET).
        pub async fn publish(&mut self, primitive: u32) -> Result<(), TeiError> {
            let mut snap: [Option<tei_ledger::CostEntry>; COST_CAPACITY] = [None; COST_CAPACITY];
            let mut k = 0;
            for e in self.rt.costs().for_primitive(primitive) {
                snap[k] = Some(*e);
                k += 1;
            }
            for e in snap.iter().take(k).flatten() {
                self.line.clear();
                write_report_line(&mut self.line, e, 1).ok();
                send_line(self.uart, &self.line).await?;
            }
            Ok(())
        }

        /// Cross-check two substrate results — available for apps that
        /// race a second substrate (e.g. an accelerator offload).
        #[allow(dead_code)]
        pub async fn check(&mut self, a: u32, b: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_check_line(&mut self.line, a, b).ok();
            send_line(self.uart, &self.line).await
        }

        pub async fn dispatch(&mut self, primitive: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_dispatch_line(&mut self.line, self.rt.costs(), primitive).ok();
            send_line(self.uart, &self.line).await
        }

        #[allow(dead_code)]
        pub fn buf(&self) -> &[u8] {
            self.buf
        }

        pub async fn sleep_ms(&mut self, ms: u64) {
            Timer::after(Duration::from_millis(ms)).await;
        }
    }
}
use tei::Tei;

#[embassy_executor::main]
async fn main(_spawner: Spawner) {
    let p = embassy_nrf::init(Default::default());

    // M4 DWT cycle counter.
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();
    let cycles = DwtCycleSource;

    // UART transport (no USB on nRF52832). BENCH-BRING-UP: P0.06/P0.08
    // are the nRF52 default UART pins; remap to the Nicla carrier.
    let mut cfg = uarte::Config::default();
    cfg.baudrate = uarte::Baudrate::BAUD115200;
    // new(uarte, rxd, txd, irq, config)
    let mut uart = Uarte::new(p.UARTE0, p.P0_08, p.P0_06, Irqs, cfg);

    // The teiOS runtime: the M4 substrate + the shipped cost table.
    let mut rt = Runtime::new(SUBSTRATES, shipped_cost_table());

    // Optional INA228 on TWIM0 (P0.14 SDA / P0.15 SCL). BENCH-PENDING: the
    // pins + shunt (0.015 Ω) / max-current (5 A) must match the part wired
    // in-line on the supply rail. Without the feature the ledger stays Table.
    #[cfg(feature = "measured-ina228")]
    let mut ina = {
        use embassy_nrf::twim::{Config as TwimConfig, Twim};
        // nRF EasyDMA needs the TX buffer in RAM; INA228 writes are ≤8 bytes.
        static mut TWIM_BUF: [u8; 16] = [0; 16];
        let tx = unsafe { &mut *core::ptr::addr_of_mut!(TWIM_BUF) };
        let i2c = Twim::new(p.TWISPI0, IrqsI2c, p.P0_14, p.P0_15, TwimConfig::default(), tx);
        tei_ina228::Ina228::new(i2c, tei_ina228::DEFAULT_ADDR, 0.015, 5.0, true).ok()
    };

    let mut boot: String<LINE_CAP> = String::new();
    write_boot_line(&mut boot, true).ok(); // M4 has DWT CYCCNT
    let _ = send_line(&mut uart, &boot).await;

    let mut ticker = Ticker::every(Duration::from_secs(1));
    loop {
        #[cfg(feature = "measured-ina228")]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> =
            ina.as_mut().map(|m| m as &mut (dyn EnergyMeter + 'static));
        #[cfg(not(feature = "measured-ina228"))]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> = None;
        let mut tei = Tei {
            uart: &mut uart,
            cycles: &cycles,
            buf: &WORKLOAD,
            rt: &mut rt,
            line: String::new(),
            meter,
        };
        let _ = crate::app::app(&mut tei).await;
        ticker.next().await;
    }
}

/// Send one `\n`-terminated line over the UART. The buffer is a RAM
/// `String` (EasyDMA requires the TX buffer in RAM, never flash).
async fn send_line(
    uart: &mut Uarte<'static, peripherals::UARTE0>,
    s: &str,
) -> Result<(), uarte::Error> {
    uart.write(s.as_bytes()).await?;
    uart.write(b"\n").await
}
