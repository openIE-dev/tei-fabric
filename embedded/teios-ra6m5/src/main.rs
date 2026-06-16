//! teiOS E1d binary. The firmware lives in [`fw`] and builds only for the
//! embedded target; on the host this is a stub so `cargo test` (host
//! triple) can build the package and exercise the lib without the
//! RA6M5 PAC.

#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod app;
#[cfg(all(target_arch = "arm", target_os = "none"))]
mod fw;
// The RIIC master (embedded-hal I2c) backing the INA228 EnergyMeter — only
// when the Measured-joules variant is built.
#[cfg(all(target_arch = "arm", target_os = "none", feature = "measured-ina228"))]
mod riic;

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {
    eprintln!("teios-ra6m5 is firmware: build with --target thumbv8m.main-none-eabihf");
}
