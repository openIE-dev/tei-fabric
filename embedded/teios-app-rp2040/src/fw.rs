//! teiOS app-skeleton firmware — RP2040. The FIXED harness: USB CDC,
//! the substrate implementations (software CRC32 on the M0+ with a
//! timer-proxy cycle source; the DMA-sniffer hardware CRC32), and the
//! once-per-second driver loop that builds a [`Tei`] and calls the
//! user's [`crate::app::app`] each pass. Only `src/app.rs` is editable
//! through the forge; this file is the vetted surface.
//!
//! Substrate + protocol details are identical to the shipped
//! teios-rp2040 image (see that crate's fw.rs docs).

use core::sync::atomic::{Ordering, compiler_fence};

use embassy_executor::Spawner;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_rp::{bind_interrupts, pac};
use embassy_time::{Duration, Instant, Ticker, Timer};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use heapless::String;
use panic_halt as _;
use static_cell::StaticCell;
use tei_ledger::{CostTable, CycleSource, EventLedger, JoulesSource};
use teios_app_rp2040::{
    BOARD_ID, BUF_LEN, COST_CAPACITY, SUBSTRATE_DMA, crc32_software,
    fill_pattern, shipped_cost_table, write_boot_line, write_check_line, write_dispatch_line,
    write_ledger_line,
};

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [embassy_rp::binary_info::EntryAddr; 4] = [
    embassy_rp::binary_info::rp_program_name!(c"teiOS app"),
    embassy_rp::binary_info::rp_program_description!(
        c"TEI priced-primitive dispatch — user app built by the forge"
    ),
    embassy_rp::binary_info::rp_cargo_version!(),
    embassy_rp::binary_info::rp_program_build_attribute!(),
];

const DMA_CH: usize = 0;
const LINE_CAP: usize = teios_core::LINE_CAP;

static mut WORKLOAD: [u8; BUF_LEN] = [0; BUF_LEN];
static mut DMA_SCRATCH: u32 = 0;

struct TimerCycleSource {
    clk_sys_mhz: u64,
}
impl CycleSource for TimerCycleSource {
    fn now(&self) -> u64 {
        Instant::now().as_micros() * self.clk_sys_mhz
    }
}

/// The user-app harness — defined in `tei`, exported for `app.rs`.
pub mod tei {
    use super::*;

    /// The only error an app sees: the USB host went away. teiOS catches
    /// it and re-waits for a connection.
    pub type TeiError = EndpointError;

    /// One substrate run: the result value and what it cost.
    pub struct Run {
        pub result: u32,
        pub ledger: EventLedger,
    }

    /// The safe surface an app may touch. Holds the USB class, the
    /// cycle source, the workload buffer, and the cost table; every
    /// emitting method streams a JSON line Studio parses.
    pub struct Tei<'a> {
        pub(super) class: &'a mut CdcAcmClass<'static, Driver<'static, USB>>,
        pub(super) timer: &'a TimerCycleSource,
        pub(super) buf: &'a [u8; BUF_LEN],
        pub(super) table: &'a CostTable<COST_CAPACITY>,
        pub(super) line: String<LINE_CAP>,
    }

    impl<'a> Tei<'a> {
        /// Run `primitive` on the named substrate. Prices the run into a
        /// ledger, streams the ledger line, returns the result+ledger.
        /// Unknown substrate names run on the CPU (safe default).
        pub async fn run_on(
            &mut self,
            substrate: &'static str,
            _primitive: u32,
        ) -> Result<Run, TeiError> {
            let (result, ledger) = if substrate == SUBSTRATE_DMA {
                let t0 = Instant::now();
                let crc = dma_sniffer_crc32(self.buf);
                let mut l = EventLedger::new(JoulesSource::Table);
                l.dma_transfers = (BUF_LEN / 4) as u64;
                l.accel_invocations = 1;
                l.active_us = t0.elapsed().as_micros();
                (crc, l)
            } else {
                let t0 = Instant::now();
                let c0 = self.timer.now();
                let crc = crc32_software(self.buf);
                let mut l = EventLedger::new(JoulesSource::Table);
                l.cycles = self.timer.delta(c0);
                l.active_us = t0.elapsed().as_micros();
                (crc, l)
            };
            self.line.clear();
            write_ledger_line(&mut self.line, substrate, 1, &ledger).ok();
            send_line(self.class, &self.line).await?;
            Ok(Run { result, ledger })
        }

        /// Emit a cross-substrate result check.
        pub async fn check(&mut self, a: u32, b: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_check_line(&mut self.line, a, b).ok();
            send_line(self.class, &self.line).await
        }

        /// Emit the dispatch verdict for `primitive` (lowest joules wins,
        /// straight from the cost table).
        pub async fn dispatch(&mut self, primitive: u32) -> Result<(), TeiError> {
            self.line.clear();
            write_dispatch_line(&mut self.line, self.table, primitive).ok();
            send_line(self.class, &self.line).await
        }

        /// The deterministic workload buffer (identical bytes everywhere).
        pub fn buf(&self) -> &[u8] {
            self.buf
        }

        /// Sleep this pass.
        pub async fn sleep_ms(&mut self, ms: u64) {
            Timer::after(Duration::from_millis(ms)).await;
        }
    }
}
use tei::Tei;

#[embassy_executor::task]
async fn usb_task(mut usb: UsbDevice<'static, Driver<'static, USB>>) -> ! {
    usb.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());
    let _dma_ch0 = p.DMA_CH0;

    let cyccnt = false;
    let timer = TimerCycleSource {
        clk_sys_mhz: (embassy_rp::clocks::clk_sys_freq() / 1_000_000) as u64,
    };

    let driver = Driver::new(p.USB, Irqs);
    let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
    config.manufacturer = Some("OpenIE");
    config.product = Some("teiOS app (RP2040)");
    config.serial_number = Some(BOARD_ID);
    config.max_power = 100;
    config.max_packet_size_0 = 64;

    static CONFIG_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static BOS_DESCRIPTOR: StaticCell<[u8; 256]> = StaticCell::new();
    static CONTROL_BUF: StaticCell<[u8; 64]> = StaticCell::new();
    let mut builder = embassy_usb::Builder::new(
        driver,
        config,
        CONFIG_DESCRIPTOR.init([0; 256]),
        BOS_DESCRIPTOR.init([0; 256]),
        &mut [],
        CONTROL_BUF.init([0; 64]),
    );
    static STATE: StaticCell<State> = StaticCell::new();
    let mut class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    spawner.spawn(usb_task(builder.build()).unwrap());

    let buf: &'static mut [u8; BUF_LEN] = unsafe { &mut *core::ptr::addr_of_mut!(WORKLOAD) };
    fill_pattern(buf);
    let table = shipped_cost_table();

    loop {
        class.wait_connection().await;
        let _ = stream(&mut class, &timer, cyccnt, buf, &table).await;
    }
}

/// Build the [`Tei`] harness and run the user app once per second.
async fn stream(
    class: &mut CdcAcmClass<'static, Driver<'static, USB>>,
    timer: &TimerCycleSource,
    cyccnt: bool,
    buf: &[u8; BUF_LEN],
    table: &CostTable<COST_CAPACITY>,
) -> Result<(), EndpointError> {
    let mut boot: String<LINE_CAP> = String::new();
    write_boot_line(&mut boot, cyccnt).ok();
    send_line(class, &boot).await?;

    let mut ticker = Ticker::every(Duration::from_secs(1));
    loop {
        let mut tei = Tei {
            class,
            timer,
            buf,
            table,
            line: String::new(),
        };
        crate::app::app(&mut tei).await?;
        ticker.next().await;
    }
}

async fn send_line(
    class: &mut CdcAcmClass<'static, Driver<'static, USB>>,
    s: &str,
) -> Result<(), EndpointError> {
    let max = class.max_packet_size() as usize;
    for chunk in s.as_bytes().chunks(max) {
        class.write_packet(chunk).await?;
    }
    class.write_packet(b"\n").await
}

fn dma_sniffer_crc32(buf: &[u8]) -> u32 {
    debug_assert_eq!(buf.len() % 4, 0);
    let dma = pac::DMA;
    let ch = dma.ch(DMA_CH);

    dma.sniff_data().write_value(0xFFFF_FFFF);
    dma.sniff_ctrl().write(|w| {
        w.set_en(true);
        w.set_dmach(DMA_CH as u8);
        w.set_calc(pac::dma::vals::Calc::CRC32R);
        w.set_out_rev(true);
        w.set_out_inv(true);
    });

    ch.read_addr().write_value(buf.as_ptr() as u32);
    ch.write_addr()
        .write_value(core::ptr::addr_of_mut!(DMA_SCRATCH) as u32);
    ch.trans_count().write_value((buf.len() / 4) as u32);

    compiler_fence(Ordering::SeqCst);
    cortex_m::asm::dsb();

    ch.ctrl_trig().write(|w| {
        w.set_incr_read(true);
        w.set_incr_write(false);
        w.set_data_size(pac::dma::vals::DataSize::SIZE_WORD);
        w.set_treq_sel(pac::dma::vals::TreqSel::PERMANENT);
        w.set_chain_to(DMA_CH as u8);
        w.set_sniff_en(true);
        w.set_en(true);
    });

    while ch.ctrl_trig().read().busy() {}
    compiler_fence(Ordering::SeqCst);

    dma.sniff_ctrl().write(|w| w.set_en(false));
    dma.sniff_data().read()
}
