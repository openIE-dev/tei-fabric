//! teiOS E1 firmware — RP2350 (Raspberry Pi Pico 2).
//!
//! Boot → USB CDC serial → every second, run the Hash primitive
//! (CRC32 over a 64 KiB buffer) on TWO on-die substrates:
//!
//! - `cpu@150mhz` — software, table-driven CRC32 on the Cortex-M33,
//!   cycles counted by DWT CYCCNT (`DWT_CTRL.NOCYCCNT` checked at boot;
//!   the boot line reports whether the counter exists).
//! - `dma-sniffer` — the RP2350 DMA sniffer computes the CRC32 in
//!   hardware as the DMA channel streams the buffer; the CPU does no
//!   per-byte work. embassy-rp does not expose the sniffer, so it is
//!   driven through the rp-pac registers directly (channel reserved by
//!   holding `DMA_CH0`).
//!
//! Each run becomes a `tei_ledger::EventLedger`; both are priced from
//! the shipped Table-tier cost table; `CostTable::cheapest` issues the
//! dispatch verdict; everything streams as JSON lines (see the crate
//! root docs in `lib.rs` for the protocol TEI Studio parses).

use core::sync::atomic::{Ordering, compiler_fence};

use cortex_m::peripheral::DWT;
use embassy_executor::Spawner;
use embassy_rp::peripherals::USB;
use embassy_rp::usb::{Driver, InterruptHandler};
use embassy_rp::{bind_interrupts, pac};
use embassy_time::{Duration, Instant, Ticker};
use embassy_usb::UsbDevice;
use embassy_usb::class::cdc_acm::{CdcAcmClass, State};
use embassy_usb::driver::EndpointError;
use heapless::String;
use panic_halt as _;
use static_cell::StaticCell;
use tei_ledger::{CostTable, CycleSource, EventLedger, JoulesSource};
use teios_rp2350::{
    BUF_LEN, COST_CAPACITY, PRIMITIVE_HASH, SUBSTRATE_CPU, SUBSTRATE_DMA, crc32_software,
    fill_pattern, shipped_cost_table, write_boot_line, write_check_line, write_dispatch_line,
    write_ledger_line,
};

bind_interrupts!(struct Irqs {
    USBCTRL_IRQ => InterruptHandler<USB>;
});

// Program metadata for `picotool info`.
#[unsafe(link_section = ".bi_entries")]
#[used]
pub static PICOTOOL_ENTRIES: [embassy_rp::binary_info::EntryAddr; 4] = [
    embassy_rp::binary_info::rp_program_name!(c"teiOS E1"),
    embassy_rp::binary_info::rp_program_description!(
        c"TEI priced-primitive dispatch: CRC32 on cpu vs DMA sniffer, ledger as JSON lines on USB CDC"
    ),
    embassy_rp::binary_info::rp_cargo_version!(),
    embassy_rp::binary_info::rp_program_build_attribute!(),
];

/// DMA channel driven via pac (reserved by holding `p.DMA_CH0`).
const DMA_CH: usize = 0;

/// Line buffer capacity — host test `lines_fit_firmware_buffer`
/// proves every emittable line fits.
const LINE_CAP: usize = 512;

/// The 64 KiB workload buffer, in striped SRAM (.bss — no flash cost).
static mut WORKLOAD: [u8; BUF_LEN] = [0; BUF_LEN];

/// Dummy DMA write target (write address not incremented).
static mut DMA_SCRATCH: u32 = 0;

/// DWT CYCCNT as a [`CycleSource`] — 32-bit hardware counter, so
/// `delta` overrides the default with wrapping subtraction in the
/// counter's native width (good for spans < 2^32 cycles ≈ 28 s @150 MHz).
struct DwtCycles;

impl CycleSource for DwtCycles {
    fn now(&self) -> u64 {
        DWT::cycle_count() as u64
    }

    fn delta(&self, start: u64) -> u64 {
        DWT::cycle_count().wrapping_sub(start as u32) as u64
    }
}

#[embassy_executor::task]
async fn usb_task(mut usb: UsbDevice<'static, Driver<'static, USB>>) -> ! {
    usb.run().await
}

#[embassy_executor::main]
async fn main(spawner: Spawner) {
    let p = embassy_rp::init(Default::default());

    // Reserve DMA channel 0 for the sniffer run: embassy-rp never touches
    // a channel whose Peripheral singleton we hold.
    let _dma_ch0 = p.DMA_CH0;

    // DWT CYCCNT: enable DEMCR.TRCENA, then the counter — but the M33's
    // cycle counter is architecturally optional, so honor NOCYCCNT and
    // report the truth in the boot line instead of lying with zeros.
    let mut core = cortex_m::Peripherals::take().unwrap();
    core.DCB.enable_trace();
    let cyccnt = DWT::has_cycle_counter();
    if cyccnt {
        core.DWT.enable_cycle_counter();
    }

    // USB CDC-ACM serial.
    let driver = Driver::new(p.USB, Irqs);
    let mut config = embassy_usb::Config::new(0xc0de, 0xcafe);
    config.manufacturer = Some("OpenIE");
    config.product = Some("teiOS E1 (Pico 2)");
    config.serial_number = Some("pico2");
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
        &mut [], // no msos descriptors
        CONTROL_BUF.init([0; 64]),
    );
    static STATE: StaticCell<State> = StaticCell::new();
    let mut class = CdcAcmClass::new(&mut builder, STATE.init(State::new()), 64);
    spawner.spawn(usb_task(builder.build()).unwrap());

    // Deterministic workload: every board hashes identical bytes.
    let buf: &'static mut [u8; BUF_LEN] = unsafe { &mut *core::ptr::addr_of_mut!(WORKLOAD) };
    fill_pattern(buf);

    let table = shipped_cost_table();
    let dwt = DwtCycles;

    loop {
        class.wait_connection().await;
        // On disconnect (EndpointError::Disabled) fall out and re-wait.
        let _ = stream(&mut class, &dwt, cyccnt, buf, &table).await;
    }
}

/// The minutes-to-first-ledger loop: one pass per second on each
/// substrate, four JSON lines per pass.
async fn stream(
    class: &mut CdcAcmClass<'static, Driver<'static, USB>>,
    dwt: &DwtCycles,
    cyccnt: bool,
    buf: &[u8; BUF_LEN],
    table: &CostTable<COST_CAPACITY>,
) -> Result<(), EndpointError> {
    let mut line: String<LINE_CAP> = String::new();

    line.clear();
    write_boot_line(&mut line, cyccnt).ok();
    send_line(class, &line).await?;

    let mut ticker = Ticker::every(Duration::from_secs(1));
    loop {
        // --- substrate 1: cpu@150mhz (software CRC32, DWT cycles) ---
        let t0 = Instant::now();
        let c0 = dwt.now();
        let crc_cpu = crc32_software(buf);
        let cpu_cycles = if cyccnt { dwt.delta(c0) } else { 0 };
        let cpu_us = t0.elapsed().as_micros();

        let mut cpu_ledger = EventLedger::new(JoulesSource::Table);
        cpu_ledger.cycles = cpu_cycles;
        cpu_ledger.active_us = cpu_us;

        line.clear();
        write_ledger_line(&mut line, SUBSTRATE_CPU, 1, &cpu_ledger).ok();
        send_line(class, &line).await?;

        // --- substrate 2: dma-sniffer (hardware CRC32 in the DMA engine) ---
        let t0 = Instant::now();
        let crc_dma = dma_sniffer_crc32(buf);
        let dma_us = t0.elapsed().as_micros();

        let mut dma_ledger = EventLedger::new(JoulesSource::Table);
        dma_ledger.dma_transfers = (BUF_LEN / 4) as u64; // one per 32-bit word
        dma_ledger.accel_invocations = 1;
        dma_ledger.active_us = dma_us;

        line.clear();
        write_ledger_line(&mut line, SUBSTRATE_DMA, 1, &dma_ledger).ok();
        send_line(class, &line).await?;

        // --- cross-substrate result check ---
        line.clear();
        write_check_line(&mut line, crc_cpu, crc_dma).ok();
        send_line(class, &line).await?;

        // --- the dispatch verdict: lowest joules wins ---
        line.clear();
        write_dispatch_line(&mut line, table, PRIMITIVE_HASH).ok();
        send_line(class, &line).await?;

        ticker.next().await;
    }
}

/// One JSON line over CDC, chunked to the endpoint packet size. The
/// trailing `\n` goes as its own (always short) packet, which also
/// flushes the transfer — no ZLP bookkeeping needed.
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

/// CRC32 over `buf` computed by the RP2350 DMA sniffer while the DMA
/// channel streams the buffer word-by-word into a dummy word.
///
/// Sniffer recipe for the zlib/IEEE (reflected) CRC32 that
/// [`crc32_software`] computes: `CALC = CRC32R` (bit-reversed data in),
/// seed `SNIFF_DATA = 0xFFFF_FFFF`, and read the result through
/// `OUT_REV | OUT_INV` (bit-reverse + invert applied on read).
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
    ch.trans_count().write(|w| {
        w.set_count((buf.len() / 4) as u32);
        w.set_mode(pac::dma::vals::TransCountMode::NORMAL);
    });

    // Buffer writes must be visible to the DMA master before triggering.
    compiler_fence(Ordering::SeqCst);
    cortex_m::asm::dsb();

    ch.ctrl_trig().write(|w| {
        w.set_incr_read(true);
        w.set_incr_write(false);
        w.set_data_size(pac::dma::vals::DataSize::SIZE_WORD);
        w.set_treq_sel(pac::dma::vals::TreqSel::PERMANENT);
        w.set_chain_to(DMA_CH as u8); // chain-to-self == no chain
        w.set_sniff_en(true);
        w.set_en(true);
    });

    // 16384 word transfers complete in ~hundreds of microseconds; a
    // busy-wait keeps the timing attribution simple for E1. (Next:
    // IRQ + WFE so the M33 truly sleeps while the sniffer works —
    // that is the measurement the bench pass wants.)
    while ch.ctrl_trig().read().busy() {}
    compiler_fence(Ordering::SeqCst);

    dma.sniff_ctrl().write(|w| w.set_en(false));
    dma.sniff_data().read()
}
