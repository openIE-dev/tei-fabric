//! teiOS E1 binary. The real firmware lives in [`fw`] and only builds
//! for the embedded target; on the host this is a stub so `cargo test`
//! (host triple) can build the package without the embassy stack.

#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

#[cfg(all(target_arch = "arm", target_os = "none"))]
mod fw;

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {
    eprintln!("teios-rp2350 is firmware: build with --target thumbv8m.main-none-eabihf");
}
