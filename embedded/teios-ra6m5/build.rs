//! Place memory.x on the linker search path and add the embedded link
//! args — but ONLY for the bare-metal target, so host `cargo test` of
//! the lib links normally.
use std::env;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;

fn main() {
    let target = env::var("TARGET").unwrap_or_default();
    let embedded = target.ends_with("-none-eabi") || target.ends_with("-none-eabihf");
    if !embedded {
        return; // host build (tests) — no memory.x, no link script
    }
    let out = PathBuf::from(env::var("OUT_DIR").unwrap());
    File::create(out.join("memory.x"))
        .unwrap()
        .write_all(include_bytes!("memory.x"))
        .unwrap();
    println!("cargo:rustc-link-search={}", out.display());
    println!("cargo:rerun-if-changed=memory.x");
    println!("cargo:rustc-link-arg=-Tlink.x");
}
