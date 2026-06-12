// Placeholder — replaced after dependency API verification.
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_std)]
#![cfg_attr(all(target_arch = "arm", target_os = "none"), no_main)]

#[cfg(not(all(target_arch = "arm", target_os = "none")))]
fn main() {}
