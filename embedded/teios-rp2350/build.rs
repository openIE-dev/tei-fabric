//! Copies `memory.x` into the linker search path and applies the link
//! arguments for the embedded binary. Host test builds (the `lib` target
//! on the host triple) get the search path too, which is harmless — the
//! `-T` link args are bin-only and the bin is a stub off-target.

use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let out = &PathBuf::from(env::var_os("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(include_bytes!("memory.x"))
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");

    // Only the embedded target links against link.x.
    let target = env::var("TARGET").unwrap_or_default();
    if target.starts_with("thumbv8m.main") {
        println!("cargo:rustc-link-arg-bins=--nmagic");
        println!("cargo:rustc-link-arg-bins=-Tlink.x");
    }
}
