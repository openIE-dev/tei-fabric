//! teiOS E1c binary. The firmware lives in [`fw`] and builds only for the
//! embedded target; on the host this is a stub so `cargo test` (host
//! triple) can build the package and exercise the lib without the embassy
//! stack.

#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod app;
#[cfg(all(target_arch = "arm", target_os = "none"))]
mod fw;

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {
    eprintln!("teios-nrf52832 is firmware: build with --target thumbv7em-none-eabihf");
}
