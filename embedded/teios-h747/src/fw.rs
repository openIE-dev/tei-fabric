//! teiOS E1b firmware — Arduino Portenta H7 (STM32H747XI, M7 core).
//!
//! The FIXED harness: RCC/clock bring-up, USB CDC, the M7 DWT cycle
//! counter, and the substrate implementations — software CRC32 on the
//! M7 and the STM32 hardware CRC peripheral — plus the once-per-second
//! driver loop that builds a [`Tei`] and calls the user's
//! [`crate::app::app`] each pass. Only `src/app.rs` is user-editable
//! through the forge.
//!
//! ## Hardware-verification status (read this)
//!
//! This firmware is **compile-verified for thumbv7em-none-eabihf, not
//! yet hardware-verified.** The Portenta's USB-C is USB **high-speed
//! over an external ULPI PHY** (USB3300) and the 480 MHz clock tree from
//! the 25 MHz HSE both need on-bench bring-up with a debugger — a wrong
//! RCC/PHY guess bricks enumeration silently. The substrate logic, the
//! CRC peripheral config, the ledger protocol, and the link layout
//! (app at 0x08040000, above Arduino's bootloader) are sound; the
//! USB-HS/ULPI + RCC specifics are the documented bench step. Flash via
//! double-tap reset → DFU:
//!   `dfu-util -d 2341:035b -a 0 -s 0x08040000:leave -D teios-h747.bin`
//!
//! ## The CRC-peripheral substrate
//!
//! teios-core's `crc32_software` is zlib/IEEE CRC32 (poly 0x04C11DB7,
//! init 0xFFFF_FFFF, reflect in+out, final xor 0xFFFF_FFFF). The STM32
//! CRC unit reproduces it with `reverse_input: Byte`, `reverse_output:
//! true`, default poly; the final XOR is applied in software (the unit
//! has no xorout). A host test in `teios-core` pins the equivalence
//! conceptually; on-die agreement is the `check` line every pass.

use core::mem::MaybeUninit;

use embassy_executor::Spawner;
use embassy_stm32::crc::{Config as CrcConfig, Crc, InputReverseConfig, PolySize};
use embassy_stm32::peripherals::USB_OTG_HS;
use embassy_stm32::time::Hertz;
use embassy_stm32::usb::{Driver, InterruptHandler};
use embassy_stm32::{Config, SharedData, bind_interrupts};
use embassy_time::{Duration, Instant, Ticker, Timer};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use heapless::String;
use panic_halt as _;
use static_cell::StaticCell;
use tei_ledger::{CostTable, EnergyMeter, EventLedger, JoulesSource};
use teios_h747::{
    BOARD_ID, BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, SUBSTRATE_CRC_HW, crc32_software,
    fill_pattern, shipped_cost_table, write_boot_line, write_check_line, write_dispatch_line,
    write_ledger_line,
};

bind_interrupts!(struct Irqs {
    OTG_HS => InterruptHandler<USB_OTG_HS>;
});

const LINE_CAP: usize = teios_core::LINE_CAP;

/// Workload buffer in AXI SRAM (.bss — no flash cost). Word-aligned for
/// the CRC unit's 32-bit feed.
#[repr(align(4))]
struct Aligned([u8; BUF_LEN]);
static mut WORKLOAD: Aligned = Aligned([0; BUF_LEN]);
static mut EP_OUT: [u8; 256] = [0; 256];

/// Endpoint/descriptor scratch for embassy-usb.
static CONFIG_DESC: StaticCell<[u8; 256]> = StaticCell::new();
static BOS_DESC: StaticCell<[u8; 256]> = StaticCell::new();
static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
static STATE: StaticCell<State> = StaticCell::new();

/// Dual-core init handshake area (the M7 is the primary core).
static SHARED_DATA: MaybeUninit<SharedData> = MaybeUninit::uninit();

/// The M7 cycle source: the real DWT CYCCNT (architectural on Cortex-M7).
struct DwtCycleSource;
impl tei_ledger::CycleSource for DwtCycleSource {
    fn now(&self) -> u64 {
        // SAFETY: read-only access to the DWT cycle counter.
        cortex_m::peripheral::DWT::cycle_count() as u64
    }
    fn delta(&self, start: u64) -> u64 {
        // 32-bit counter; wrapping subtraction in u32 space.
        (self.now() as u32).wrapping_sub(start as u32) as u64
    }
}

/// The user-app harness — defined in `tei`, exported for `app.rs`.
pub mod tei {
    use super::*;

    pub type TeiError = EndpointError;

    pub struct Run {
        pub result: u32,
        /// The priced ledger for this run — apps may inspect it.
        #[allow(dead_code)]
        pub ledger: EventLedger,
    }

    /// The safe surface an app may touch on the Portenta H7.
    pub struct Tei<'a> {
        pub(super) class: &'a mut CdcAcmClass<'static, Driver<'static, USB_OTG_HS>>,
        pub(super) cycles: &'a DwtCycleSource,
        pub(super) crc: &'a mut Crc<'static>,
        pub(super) buf: &'a [u8; BUF_LEN],
        pub(super) table: &'a CostTable<COST_CAPACITY>,
        pub(super) line: String<LINE_CAP>,
        /// Optional INA228 on the supply rail (`--features measured-ina228`).
        /// When present, ledgers carry Measured joules. `'static` dyn — the
        /// meter lives for the program (embassy main never returns).
        pub(super) meter: Option<&'a mut (dyn EnergyMeter + 'static)>,
    }

    impl<'a> Tei<'a> {
        /// Run `primitive` on the named substrate; price it; stream the
        /// ledger line. crc-hw uses the STM32 CRC peripheral, everything
        /// else the M7 software path (DWT-counted).
        pub async fn run_on(
            &mut self,
            substrate: &'static str,
            _primitive: u32,
        ) -> Result<Run, TeiError> {
            if let Some(m) = self.meter.as_deref_mut() {
                m.reset();
            }
            let (result, mut ledger) = if substrate == SUBSTRATE_CRC_HW {
                let t0 = Instant::now();
                self.crc.reset();
                // CRC unit, then software final-XOR for zlib CRC32.
                let raw = self.crc.feed_bytes(self.buf);
                let crc = raw ^ 0xFFFF_FFFF;
                let mut l = EventLedger::new(JoulesSource::Table);
                l.accel_invocations = 1;
                l.active_us = t0.elapsed().as_micros();
                (crc, l)
            } else {
                let t0 = Instant::now();
                use tei_ledger::CycleSource;
                let c0 = self.cycles.now();
                let crc = crc32_software(self.buf);
                let mut l = EventLedger::new(JoulesSource::Table);
                l.cycles = self.cycles.delta(c0);
                l.active_us = t0.elapsed().as_micros();
                (crc, l)
            };
            if let Some(m) = self.meter.as_deref_mut() {
                if let Some(j) = m.joules() {
                    ledger.joules = Some(j);
                    ledger.joules_source = JoulesSource::Measured;
                }
            }
            self.line.clear();
            write_ledger_line(&mut self.line, substrate, 1, &ledger).ok();
            send_line(self.class, &self.line).await?;
            Ok(Run { result, ledger })
        }

        pub async fn check(&mut self, a: u32, b: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_check_line(&mut self.line, a, b).ok();
            send_line(self.class, &self.line).await
        }

        pub async fn dispatch(&mut self, primitive: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_dispatch_line(&mut self.line, self.table, primitive).ok();
            send_line(self.class, &self.line).await
        }

        /// The workload buffer the substrates hash — apps may read it.
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

#[embassy_executor::task]
async fn usb_task(mut usb: UsbDevice<'static, Driver<'static, USB_OTG_HS>>) -> ! {
    usb.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    // RCC: 25 MHz HSE (Portenta crystal) → PLL → 480 MHz sys, 48 MHz for
    // USB. BENCH-BRING-UP: these dividers are the documented step; a
    // wrong value here is silent on the host (compile is fine) and only
    // shows on the scope. Conservative skeleton config below.
    let mut config = Config::default();
    {
        use embassy_stm32::rcc::*;
        config.rcc.hse = Some(Hse {
            freq: Hertz(25_000_000),
            mode: HseMode::Oscillator,
        });
        config.rcc.pll1 = Some(Pll {
            source: PllSource::HSE,
            prediv: PllPreDiv::DIV5,
            mul: PllMul::MUL192,
            divp: Some(PllDiv::DIV2),
            divq: Some(PllDiv::DIV20),
            divr: None,
        });
        config.rcc.sys = Sysclk::PLL1_P;
        config.rcc.voltage_scale = VoltageScale::Scale0;
    }
    // Dual-core part: the M7 is the primary core and brings up the clock
    // tree; `init_primary` publishes the clock config to the M4 via
    // SHARED_DATA. (Booting the M4 to actually run a substrate is the
    // documented inter-core stretch — see SUBSTRATE_M4.)
    let p = embassy_stm32::init_primary(config, &SHARED_DATA);

    // M7 DWT cycle counter.
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    core.DWT.enable_cycle_counter();
    let cycles = DwtCycleSource;

    // STM32 hardware CRC, configured for zlib/IEEE CRC32. The H7 CRC unit
    // resets to polynomial 0x04C11DB7 / init 0xFFFF_FFFF (the CRC32 poly);
    // we set byte-reversed input + reversed output and apply the final
    // XOR in software, reproducing teios-core's `crc32_software`.
    let crc_cfg = CrcConfig::new(
        InputReverseConfig::Byte, // reflect input bytes (zlib convention)
        true,                     // reflect output
        PolySize::Width32,
        0xFFFF_FFFF, // init value
        0x04C1_1DB7, // IEEE/zlib CRC32 polynomial
    )
    .unwrap();
    let mut crc = Crc::new(p.CRC, crc_cfg);

    // USB CDC over OTG_HS through the Portenta's external ULPI PHY
    // (USB3300). BENCH-BRING-UP: the ULPI pin map below is the standard
    // STM32H7 OTG_HS_ULPI alternate-function set; PHY reset/power and the
    // 60 MHz ULPI clock are the on-bench step.
    let mut usb_cfg = embassy_stm32::usb::Config::default();
    usb_cfg.vbus_detection = false;
    let driver = Driver::new_hs_ulpi(
        p.USB_OTG_HS,
        Irqs,
        p.PA5,  // ULPI_CK
        p.PC2,  // ULPI_DIR
        p.PC3,  // ULPI_NXT
        p.PC0,  // ULPI_STP
        p.PA3,  // ULPI_D0
        p.PB0,  // ULPI_D1
        p.PB1,  // ULPI_D2
        p.PB10, // ULPI_D3
        p.PB11, // ULPI_D4
        p.PB12, // ULPI_D5
        p.PB13, // ULPI_D6
        p.PB5,  // ULPI_D7
        unsafe { &mut *core::ptr::addr_of_mut!(EP_OUT) },
        usb_cfg,
    );

    let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
    config.manufacturer = Some("OpenIE");
    config.product = Some("teiOS E1b (Portenta H7)");
    config.serial_number = Some(BOARD_ID);
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        CONFIG_DESC.init([0; 256]),
        BOS_DESC.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );
    let mut class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    spawner.spawn(usb_task(builder.build()).unwrap());

    let buf: &'static mut [u8; BUF_LEN] = unsafe { &mut (*core::ptr::addr_of_mut!(WORKLOAD)).0 };
    fill_pattern(buf);
    let table = shipped_cost_table();

    // Optional INA228 on I2C1 (PB6 SCL / PB7 SDA). BENCH-PENDING: the I2C pins
    // + the shunt (0.015 Ω) / max-current (5 A) must match the part wired
    // in-line on the supply rail. Without the feature the ledger stays Table.
    #[cfg(feature = "measured-ina228")]
    let mut ina = {
        let i2c = embassy_stm32::i2c::I2c::new_blocking(p.I2C1, p.PB6, p.PB7, Default::default());
        tei_ina228::Ina228::new(i2c, tei_ina228::DEFAULT_ADDR, 0.015, 5.0, true).ok()
    };

    loop {
        class.wait_connection().await;
        #[cfg(feature = "measured-ina228")]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> =
            ina.as_mut().map(|m| m as &mut (dyn EnergyMeter + 'static));
        #[cfg(not(feature = "measured-ina228"))]
        let meter: Option<&mut (dyn EnergyMeter + 'static)> = None;
        let _ = stream(&mut class, &cycles, &mut crc, buf, &table, meter).await;
    }
}

async fn stream(
    class: &mut CdcAcmClass<'static, Driver<'static, USB_OTG_HS>>,
    cycles: &DwtCycleSource,
    crc: &mut Crc<'static>,
    buf: &[u8; BUF_LEN],
    table: &CostTable<COST_CAPACITY>,
    mut meter: Option<&mut (dyn EnergyMeter + 'static)>,
) -> Result<(), EndpointError> {
    let mut boot: String<LINE_CAP> = String::new();
    write_boot_line(&mut boot, true).ok(); // M7 has DWT CYCCNT
    send_line(class, &boot).await?;

    let mut ticker = Ticker::every(Duration::from_secs(1));
    loop {
        let mut tei = Tei {
            class,
            cycles,
            crc,
            buf,
            table,
            line: String::new(),
            meter: meter.as_deref_mut(),
        };
        crate::app::app(&mut tei).await?;
        ticker.next().await;
    }
}

async fn send_line(
    class: &mut CdcAcmClass<'static, Driver<'static, USB_OTG_HS>>,
    s: &str,
) -> Result<(), EndpointError> {
    let max = class.max_packet_size() as usize;
    for chunk in s.as_bytes().chunks(max) {
        class.write_packet(chunk).await?;
    }
    class.write_packet(b"\n").await
}

// Suppress unused warnings for the bench-stretch surface.
#[allow(dead_code)]
fn _unused() {
    let _ = PRIMITIVE_HASH;
}
