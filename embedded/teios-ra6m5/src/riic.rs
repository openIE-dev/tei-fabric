//! Minimal blocking **RIIC master** on IIC0, exposing the
//! `embedded_hal::i2c::I2c` trait — the driver the Portenta C33 was missing.
//!
//! The board-agnostic [`tei_ina228`] EnergyMeter is generic over
//! `embedded_hal::i2c::I2c`. Every other teiOS board gets that trait from an
//! embassy HAL; the RA6M5 image is bare-metal `cortex-m-rt` + `ra6m5-pac`
//! with no HAL, so nothing implemented the trait and the C33 stayed
//! Table-tier. This module provides it: a small register-level RIIC master,
//! so the INA228 binds and the C33 joins the Measured-joules path.
//!
//! ## Honesty boundary (read this)
//!
//! Compile-verified against `ra6m5-pac`. The exact RIIC register *sequence*
//! and the bit-rate timing are the **on-bench step** — the same boundary as
//! the CRCA path in `fw.rs` (there is no RA6M5 on the bench yet). In
//! particular the bit-rate divisors (`ICMR1.CKS`, `ICBRH`, `ICBRL`) are
//! nominal for ~100 kHz and must be tuned once the PCLKB clock tree is
//! configured (the firmware currently runs at the reset clock), and the
//! receive sequence (WAIT / ACKBT last-byte NACK) is the classic RIIC
//! fiddly bit to confirm on a scope.

use embedded_hal::i2c::{
    self, ErrorKind, ErrorType, I2c, NoAcknowledgeSource, Operation, SevenBitAddress,
};
use ra6m5_pac::iic0::iccr1::{Ice, Iicrst};
use ra6m5_pac::iic0::iccr2::{Bbsy, Rs, Sp, St};
use ra6m5_pac::iic0::icfer::{Nacke, Scle};
use ra6m5_pac::iic0::icmr3::{Ackbt, Ackwp, Wait};
use ra6m5_pac::iic0::icsr2::{Nackf, Rdrf, Stop, Tdre, Tend};
use ra6m5_pac::mstp::mstpcrb::Mstpb9;
use ra6m5_pac::{self as pac, NoBitfieldReg};

/// Bounded spin so a stuck bus (no pull-ups, no device) returns an error
/// instead of hanging the firmware forever.
const SPIN_LIMIT: u32 = 1_000_000;

/// I²C errors mapped onto the `embedded-hal` taxonomy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Error {
    /// A device did not ACK its address or a data byte.
    Nack,
    /// Lost master arbitration on a multi-master bus.
    Arbitration,
    /// Bus stuck (never went free / no response within the spin budget).
    Bus,
    /// A status flag never asserted within the spin budget.
    Timeout,
}

impl i2c::Error for Error {
    fn kind(&self) -> ErrorKind {
        match self {
            Error::Nack => ErrorKind::NoAcknowledge(NoAcknowledgeSource::Unknown),
            Error::Arbitration => ErrorKind::ArbitrationLoss,
            Error::Bus => ErrorKind::Bus,
            Error::Timeout => ErrorKind::Other,
        }
    }
}

/// The IIC0 channel as a blocking master. Construct once, then hand it to
/// `tei_ina228::Ina228::new`.
pub struct Riic0 {
    _private: (),
}

impl Riic0 {
    /// Bring IIC0 out of module-stop and into master mode at ~100 kHz.
    ///
    /// # Safety
    /// Assumes exclusive ownership of the IIC0 peripheral and its pins; only
    /// construct once.
    pub unsafe fn new() -> Self {
        // Release IIC0 from module-stop (MSTPCRB.MSTPB9 = 0 ⇒ running).
        pac::MSTP
            .mstpcrb()
            .modify(|r| r.mstpb9().set(Mstpb9::new(0)));

        // Enter internal reset for configuration (ICE=0, IICRST=1), then
        // enable the peripheral while still in reset (Renesas init order).
        pac::IIC0
            .iccr1()
            .modify(|r| r.ice().set(Ice::_0).iicrst().set(Iicrst::_1));
        pac::IIC0.iccr1().modify(|r| r.ice().set(Ice::_1));

        // Bit rate ≈ 100 kHz. CKS divider + BRH/BRL counts (each a 5-bit u8
        // field) are PCLKB-derived; NOMINAL here, tuned on bench once the
        // clock tree is set.
        pac::IIC0.icmr1().modify(|r| r.cks().set(0b011));
        pac::IIC0.icbrh().modify(|r| r.brh().set(0b11000));
        pac::IIC0.icbrl().modify(|r| r.brl().set(0b11100));

        // Function enable: NACK arbitration-detect + SCL synchronous circuit.
        pac::IIC0
            .icfer()
            .modify(|r| r.nacke().set(Nacke::_1).scle().set(Scle::_1));

        // Release internal reset → bus idle, ready as master.
        pac::IIC0.iccr1().modify(|r| r.iicrst().set(Iicrst::_0));

        Self { _private: () }
    }

    /// Spin until `cond()` or the budget runs out (→ `err`).
    fn spin_until<F: Fn() -> bool>(cond: F, err: Error) -> Result<(), Error> {
        let mut n = 0u32;
        while !cond() {
            n += 1;
            if n >= SPIN_LIMIT {
                return Err(err);
            }
        }
        Ok(())
    }

    fn nack_seen() -> bool {
        unsafe { pac::IIC0.icsr2().read().nackf().get() == Nackf::_1 }
    }

    /// Issue a START (from bus-free) or a repeated START.
    fn start(&mut self, restart: bool) -> Result<(), Error> {
        if restart {
            unsafe { pac::IIC0.iccr2().modify(|r| r.rs().set(Rs::_1)) };
        } else {
            Self::spin_until(
                || unsafe { pac::IIC0.iccr2().read().bbsy().get() == Bbsy::_0 },
                Error::Bus,
            )?;
            unsafe { pac::IIC0.iccr2().modify(|r| r.st().set(St::_1)) };
        }
        Ok(())
    }

    /// Send the addressing byte: 7-bit address + R/W̅.
    fn send_addr(&mut self, addr: u8, read: bool) -> Result<(), Error> {
        Self::spin_until(
            || unsafe { pac::IIC0.icsr2().read().tdre().get() == Tdre::_1 },
            Error::Timeout,
        )?;
        let byte = (addr << 1) | (read as u8);
        unsafe { pac::IIC0.icdrt().write(ra6m5_pac::iic0::Icdrt::default().set(byte)) };
        Ok(())
    }

    fn write_bytes(&mut self, bytes: &[u8]) -> Result<(), Error> {
        for &b in bytes {
            Self::spin_until(
                || unsafe { pac::IIC0.icsr2().read().tdre().get() == Tdre::_1 },
                Error::Timeout,
            )?;
            if Self::nack_seen() {
                return Err(Error::Nack);
            }
            unsafe { pac::IIC0.icdrt().write(ra6m5_pac::iic0::Icdrt::default().set(b)) };
        }
        Self::spin_until(
            || unsafe { pac::IIC0.icsr2().read().tend().get() == Tend::_1 },
            Error::Timeout,
        )?;
        if Self::nack_seen() {
            return Err(Error::Nack);
        }
        Ok(())
    }

    fn read_bytes(&mut self, buf: &mut [u8]) -> Result<(), Error> {
        if buf.is_empty() {
            return Ok(());
        }
        let n = buf.len();
        // WAIT mode lets us NACK the final byte cleanly.
        unsafe { pac::IIC0.icmr3().modify(|r| r.wait().set(Wait::_1)) };
        // Dummy read of ICDRR starts the receive clocking.
        let _ = unsafe { pac::IIC0.icdrr().read().get() };
        for i in 0..n {
            Self::spin_until(
                || unsafe { pac::IIC0.icsr2().read().rdrf().get() == Rdrf::_1 },
                Error::Timeout,
            )?;
            if i == n - 1 {
                // NACK the last byte: unlock ACKBT (ACKWP) then set it.
                unsafe {
                    pac::IIC0.icmr3().modify(|r| r.ackwp().set(Ackwp::_1));
                    pac::IIC0.icmr3().modify(|r| r.ackbt().set(Ackbt::_1));
                }
            }
            buf[i] = unsafe { pac::IIC0.icdrr().read().get() };
        }
        unsafe { pac::IIC0.icmr3().modify(|r| r.wait().set(Wait::_0)) };
        Ok(())
    }

    /// Issue STOP and clear the latched STOP/NACK flags.
    fn stop(&mut self) -> Result<(), Error> {
        unsafe {
            pac::IIC0.icsr2().modify(|r| r.stop().set(Stop::_0));
            pac::IIC0.iccr2().modify(|r| r.sp().set(Sp::_1));
        }
        Self::spin_until(
            || unsafe { pac::IIC0.icsr2().read().stop().get() == Stop::_1 },
            Error::Timeout,
        )?;
        unsafe {
            pac::IIC0
                .icsr2()
                .modify(|r| r.nackf().set(Nackf::_0).stop().set(Stop::_0))
        };
        Ok(())
    }
}

impl ErrorType for Riic0 {
    type Error = Error;
}

impl I2c<SevenBitAddress> for Riic0 {
    fn transaction(
        &mut self,
        address: SevenBitAddress,
        operations: &mut [Operation<'_>],
    ) -> Result<(), Self::Error> {
        let mut first = true;
        for op in operations.iter_mut() {
            let read = matches!(op, Operation::Read(_));
            self.start(!first)?;
            self.send_addr(address, read)?;
            match op {
                Operation::Write(bytes) => self.write_bytes(bytes)?,
                Operation::Read(buf) => self.read_bytes(buf)?,
            }
            first = false;
        }
        if !first {
            self.stop()?;
        }
        Ok(())
    }
}
