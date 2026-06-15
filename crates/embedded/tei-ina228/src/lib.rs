//! TI **INA228** driver — Measured-tier joules for teiOS boards.
//!
//! The INA228 is a high-precision (20-bit ΔΣ) I²C power/energy monitor with a
//! **hardware energy accumulator**: it integrates power × time on-chip into the
//! 40-bit `ENERGY` register, so the target reads accumulated joules directly —
//! no host-side integration, no timing jitter. That makes it the canonical
//! `tei_ledger::EnergyMeter` (the trait's `reset()` is documented "RSTACC-style"
//! precisely for this part).
//!
//! Board-agnostic over `embedded-hal` 1.0 `I2c`, so the same driver works on the
//! RP2040/STM32/nRF/RA forge skeletons. Put the INA228's shunt **in-line on the
//! board's supply rail** (the path whose energy you want), wire its I²C to the
//! board, and the ledger's joules become `JoulesSource::Measured`.
//!
//! # Energy math (TI INA228 datasheet, SBOS725)
//!
//! - `CURRENT_LSB = max_expected_current / 2^19`  (the CURRENT register is 20-bit
//!   signed → 2^19 positive codes).
//! - `SHUNT_CAL  = 13107.2e6 × CURRENT_LSB × R_shunt`  (× 4 when `ADCRANGE = 1`,
//!   the ±40.96 mV high-resolution range).
//! - `Power[W]   = 3.2 × CURRENT_LSB × POWER_register`.
//! - `Energy[J]  = 16 × Power_LSB × ENERGY_register = 51.2 × CURRENT_LSB × ENERGY`.
//!
//! The three pure functions below encode exactly that and are unit-tested; the
//! I²C plumbing is generic + compile-checked.

#![no_std]

use embedded_hal::i2c::I2c;
use tei_ledger::EnergyMeter;

/// Default 7-bit I²C address (A0=A1=GND). Adafruit's INA228 STEMMA QT ships here.
pub const DEFAULT_ADDR: u8 = 0x40;

// Register map (subset we use).
const REG_CONFIG: u8 = 0x00;
const REG_ADC_CONFIG: u8 = 0x01;
const REG_SHUNT_CAL: u8 = 0x02;
const REG_ENERGY: u8 = 0x09; // 40-bit, unsigned
const REG_DEVICE_ID: u8 = 0x3F;

// CONFIG bits.
const CONFIG_RST: u16 = 0x8000; // full reset
const CONFIG_RSTACC: u16 = 0x4000; // clear ENERGY/CHARGE accumulators
const CONFIG_ADCRANGE: u16 = 0x0010; // 1 = ±40.96 mV (4× resolution)

// ADC_CONFIG: continuous bus+shunt+temp, 1052 µs conversions, avg 1
// (datasheet reset value).
const ADC_CONFIG_CONTINUOUS: u16 = 0xFB68;

// ---------------------------------------------------------------------------
// Pure energy math (host-tested — this is the part that can be wrong).
// ---------------------------------------------------------------------------

/// `CURRENT_LSB = max_expected_current / 2^19` (amps per code).
pub fn current_lsb(max_expected_current_a: f64) -> f64 {
    max_expected_current_a / 524_288.0 // 2^19
}

/// `SHUNT_CAL` register value for the given current LSB + shunt resistance.
/// `low_range` selects ADCRANGE=1 (±40.96 mV), which multiplies SHUNT_CAL by 4.
pub fn shunt_cal(current_lsb: f64, shunt_ohms: f64, low_range: bool) -> u16 {
    let mut cal = 13_107_200_000.0 * current_lsb * shunt_ohms; // 13107.2e6
    if low_range {
        cal *= 4.0;
    }
    if cal < 0.0 {
        0
    } else if cal > u16::MAX as f64 {
        u16::MAX
    } else {
        cal as u16
    }
}

/// `Energy[J] = 51.2 × CURRENT_LSB × ENERGY_register`.
pub fn energy_joules(current_lsb: f64, energy_register: u64) -> f64 {
    51.2 * current_lsb * energy_register as f64
}

// ---------------------------------------------------------------------------
// Driver
// ---------------------------------------------------------------------------

/// An INA228 on an I²C bus, configured for a known shunt + max current.
pub struct Ina228<I2C> {
    i2c: I2C,
    addr: u8,
    current_lsb: f64,
    config_word: u16, // ADCRANGE bit retained for RSTACC writes
}

impl<I2C: I2c> Ina228<I2C> {
    /// Configure an INA228: compute `CURRENT_LSB`/`SHUNT_CAL` from the shunt and
    /// the maximum current you expect to measure, set continuous conversion, and
    /// zero the energy accumulator. `low_range` = use the ±40.96 mV ADC range
    /// (4× shunt resolution; correct for small shunts / low currents).
    pub fn new(
        mut i2c: I2C,
        addr: u8,
        shunt_ohms: f64,
        max_current_a: f64,
        low_range: bool,
    ) -> Result<Self, I2C::Error> {
        let lsb = current_lsb(max_current_a);
        let cal = shunt_cal(lsb, shunt_ohms, low_range);
        let config_word = if low_range { CONFIG_ADCRANGE } else { 0 };

        write_u16(&mut i2c, addr, REG_CONFIG, config_word | CONFIG_RSTACC)?;
        write_u16(&mut i2c, addr, REG_ADC_CONFIG, ADC_CONFIG_CONTINUOUS)?;
        write_u16(&mut i2c, addr, REG_SHUNT_CAL, cal)?;

        Ok(Self { i2c, addr, current_lsb: lsb, config_word })
    }

    /// `0x2280` (device 0x228) on a real INA228 — a presence check.
    pub fn device_id(&mut self) -> Result<u16, I2C::Error> {
        read_u16(&mut self.i2c, self.addr, REG_DEVICE_ID)
    }

    /// Release the I²C bus.
    pub fn release(self) -> I2C {
        self.i2c
    }
}

impl<I2C: I2c> EnergyMeter for Ina228<I2C> {
    /// Joules accumulated since the last [`reset`](EnergyMeter::reset). Reads the
    /// 40-bit hardware `ENERGY` register; `None` if the bus is unreachable.
    fn joules(&mut self) -> Option<f64> {
        let count = read_u40(&mut self.i2c, self.addr, REG_ENERGY).ok()?;
        Some(energy_joules(self.current_lsb, count))
    }

    /// Zero the on-chip energy accumulator (CONFIG.RSTACC), preserving ADCRANGE.
    fn reset(&mut self) {
        let _ = write_u16(&mut self.i2c, self.addr, REG_CONFIG, self.config_word | CONFIG_RSTACC);
    }
}

fn write_u16<I2C: I2c>(i2c: &mut I2C, addr: u8, reg: u8, val: u16) -> Result<(), I2C::Error> {
    i2c.write(addr, &[reg, (val >> 8) as u8, val as u8])
}

fn read_u16<I2C: I2c>(i2c: &mut I2C, addr: u8, reg: u8) -> Result<u16, I2C::Error> {
    let mut b = [0u8; 2];
    i2c.write_read(addr, &[reg], &mut b)?;
    Ok(((b[0] as u16) << 8) | b[1] as u16)
}

fn read_u40<I2C: I2c>(i2c: &mut I2C, addr: u8, reg: u8) -> Result<u64, I2C::Error> {
    let mut b = [0u8; 5];
    i2c.write_read(addr, &[reg], &mut b)?;
    Ok(b.iter().fold(0u64, |acc, &x| (acc << 8) | x as u64))
}

// Suppress an unused-import lint when only the constants matter on some targets.
#[allow(dead_code)]
const _RST: u16 = CONFIG_RST;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn current_lsb_is_max_over_2pow19() {
        // 10 A full scale → ~19.07 µA per code.
        let lsb = current_lsb(10.0);
        assert!((lsb - 1.907_348_6e-5).abs() < 1e-10, "got {lsb:e}");
    }

    #[test]
    fn energy_matches_datasheet_coefficient() {
        // 51.2 × CURRENT_LSB × ENERGY. With 10 A LSB and 1e6 counts → ~976.6 J.
        let lsb = current_lsb(10.0);
        let j = energy_joules(lsb, 1_000_000);
        assert!((j - 976.562_5).abs() < 0.01, "got {j}");
        // zero counts → zero joules
        assert_eq!(energy_joules(lsb, 0), 0.0);
    }

    #[test]
    fn shunt_cal_low_range_is_4x() {
        let lsb = current_lsb(10.0);
        let hi = shunt_cal(lsb, 0.015, false);
        let lo = shunt_cal(lsb, 0.015, true);
        // 13107.2e6 × 1.907e-5 × 0.015 ≈ 3751
        assert!((3700..3800).contains(&(hi as u32)), "hi={hi}");
        assert_eq!(lo as u32, hi as u32 * 4); // low range is exactly 4×
    }

    #[test]
    fn shunt_cal_saturates_to_u16() {
        // absurd config (100 A LSB × 1 Ω → cal ≈ 2.5e6) → clamp, never overflow
        assert_eq!(shunt_cal(current_lsb(100.0), 1.0, false), u16::MAX);
    }
}
