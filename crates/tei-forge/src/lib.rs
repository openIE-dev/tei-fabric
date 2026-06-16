//! tei-forge — the teiOS app build service.
//!
//! Studio's CODE→BUILD is real because of this crate: a user edits an
//! `app()` body in the browser, and the forge validates it, splices it
//! into a *vetted skeleton* (whose `Cargo.toml` the user can never
//! touch), compiles it for a real board under a macOS seatbelt sandbox
//! with no network and a wall-timeout, and packages a flashable UF2.
//!
//! ## The security model (read this before changing anything)
//!
//! Compiling untrusted Rust on a host is the dangerous part — not the
//! *target* code (it's cross-compiled to `thumbv6m` and never runs on
//! the host) but **build-time execution**: `build.rs`, proc-macros, and
//! `const`-eval all run on the build host during compilation.
//!
//! Three layers, defense in depth:
//! 1. **Fixed dependencies (the core invariant).** The forge copies the
//!    skeleton and writes *only* `src/app.rs`. The skeleton's
//!    `Cargo.toml`/`Cargo.lock` are authoritative and never touched by
//!    user input ⇒ no user deps ⇒ no new `build.rs` / proc-macros enter
//!    the tree. The only build-time code is `const`-eval (no syscalls)
//!    and the vetted embassy macros already in the skeleton.
//! 2. **A token denylist** ([`validate`]) rejecting the constructs that
//!    reach host I/O at expand time (`include!`, `env!`, `unsafe`,
//!    `asm!`, `extern`, `std`, …) — see each entry's rationale.
//! 3. **A seatbelt sandbox** (`sandbox-exec`): the `cargo build` runs
//!    with all network denied and writes confined to the build + scratch
//!    dirs, killed on a wall-timeout. On hosts without `sandbox-exec`
//!    (non-mac CI) the forge logs a warning and degrades — but the
//!    primary deployment is the Mac build host where it is present.
//!
//! Residual risk the sandbox does NOT stop: CPU/RAM exhaustion within
//! the timeout, and reads of any file the build user can read (no exfil
//! path, since network is denied). The denylist is token-level, not a
//! full parser — defense rests on the no-user-deps invariant + seatbelt,
//! not the denylist alone.

use std::path::{Path, PathBuf};
use std::process::Command;

use serde::{Deserialize, Serialize};

mod build;
mod sandbox;
mod validate;

pub use build::{build, BuildOpts};
pub use validate::{validate, DENYLIST};

/// Re-export the board registry so the HTTP layer (which deps only this
/// crate) can read identity + pinouts without a separate chipdb dep.
pub use ofpga_chipdb;

/// What Studio's BUILD tab POSTs: a target board id and the contents of
/// the user's `src/app.rs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeRequest {
    /// Board id matching a skeleton target (e.g. "feather-rp2040",
    /// "pico").
    pub target: String,
    /// The user-editable `app.rs` source.
    pub app_source: String,
    /// Build the **Measured** variant — wire the board's EnergyMeter into
    /// the firmware so the ledger reports `JoulesSource::Measured` instead
    /// of Table-tier constants. Honored only when the target's energy
    /// source has a shipped firmware driver (see [`Target::measured_feature`]);
    /// a no-op otherwise.
    #[serde(default)]
    pub measured: bool,
}

/// The build outcome — serde-stable for the HTTP layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeResult {
    pub ok: bool,
    /// Path to the produced `.uf2` on the build host (the HTTP layer
    /// serves its bytes; not a URL).
    pub artifact_path: Option<PathBuf>,
    /// Flash/family tag, e.g. "rp2040" or "stm32h7".
    pub uf2_family: String,
    /// Artifact file extension (no dot): "uf2" or "bin".
    #[serde(default)]
    pub artifact_ext: String,
    /// Image bytes (payload, not the artifact file size).
    pub bytes: usize,
    /// Hex SHA-256 of the produced UF2 file.
    pub sha256: String,
    /// Combined stdout+stderr from cargo, truncated to ~64 KB.
    pub logs: String,
    /// Set when `ok == false`: validation message or "build failed".
    pub error: Option<String>,
}

impl ForgeResult {
    fn failed(error: impl Into<String>, logs: impl Into<String>) -> Self {
        Self {
            ok: false,
            artifact_path: None,
            uf2_family: String::new(),
            artifact_ext: String::new(),
            bytes: 0,
            sha256: String::new(),
            logs: logs.into(),
            error: Some(error.into()),
        }
    }
}

/// How a built ELF becomes a flashable artifact.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Packaging {
    /// elf2uf2 via the skeleton's `scripts/elf2uf2.py` — mass-storage
    /// drag-drop boards (RP2040/RP2350). Artifact is `.uf2`.
    Uf2,
    /// `llvm-objcopy -O binary` — DFU boards flashed with dfu-util
    /// (Portenta H7). Artifact is `.bin`; the flash base lives in the
    /// skeleton's `memory.x`, and the dfu-util `-s <base>` is documented
    /// in the skeleton README (not stamped into the image).
    Bin,
}

impl Packaging {
    /// Artifact file extension (no dot).
    pub fn ext(self) -> &'static str {
        match self {
            Packaging::Uf2 => "uf2",
            Packaging::Bin => "bin",
        }
    }
}

/// One buildable skeleton: where it lives, its cross-target, and how its
/// ELF is packaged into a flashable artifact.
#[derive(Debug, Clone)]
pub struct Target {
    pub id: &'static str,
    /// Skeleton project dir, relative to the workspace root.
    pub skeleton: &'static str,
    /// rustc target triple.
    pub triple: &'static str,
    /// How the ELF is packaged.
    pub packaging: Packaging,
    /// Flash/family tag surfaced to the UI (rp2040 / stm32h7).
    pub family: &'static str,
    /// Build features to pass (board id selection, boot2 choice).
    pub features: &'static [&'static str],
    /// `--no-default-features` when selecting a non-default board.
    pub no_default_features: bool,
}

impl Target {
    /// The board's energy-measurement path, from the single registry
    /// (`ofpga-chipdb`). The energy story is NOT anchored to one part —
    /// this is the abstraction; the INA228 is just one driver.
    pub fn energy_source(&self) -> ofpga_chipdb::boards::EnergySource {
        board_info(self.id)
            .map(ofpga_chipdb::boards::energy_source)
            .unwrap_or(ofpga_chipdb::boards::EnergySource::None)
    }

    /// The cargo feature that wires the EnergyMeter into the firmware for
    /// the **Measured** variant, derived from [`Self::energy_source`].
    /// Today only the external-shunt (INA228) path has a shipped firmware
    /// driver → `"measured-ina228"`; every other source (on-board PMIC,
    /// Linux software telemetry, accelerator, or none) returns `None`
    /// here — those boards measure outside the cross-compiled image.
    /// Added to the build only when [`ForgeRequest::measured`] is set.
    pub fn measured_feature(&self) -> Option<&'static str> {
        use ofpga_chipdb::boards::{EnergySource, ShuntDriver};
        match self.energy_source() {
            EnergySource::ExternalShunt(ShuntDriver::Ina228) => Some("measured-ina228"),
            _ => None,
        }
    }
}

/// The targets the forge can build today. Bench-first: the RP2040
/// Feather and Pico, which David owns and which need no bench energy
/// gear to demo the dispatch loop.
pub const TARGETS: &[Target] = &[
    Target {
        id: "feather-rp2040",
        skeleton: "embedded/teios-app-rp2040",
        triple: "thumbv6m-none-eabi",
        packaging: Packaging::Uf2,
        family: "rp2040",
        features: &["board-feather-rp2040"],
        no_default_features: false,
    },
    Target {
        id: "pico",
        skeleton: "embedded/teios-app-rp2040",
        triple: "thumbv6m-none-eabi",
        packaging: Packaging::Uf2,
        family: "rp2040",
        features: &["board-pico"],
        no_default_features: true,
    },
    // Portenta H7 / H7 Lite — Cortex-M7, DFU-flashed .bin (E1b). The
    // skeleton is the teios-h747 crate; its target/ default is gated so
    // the forge cross-builds with --target thumbv7em-none-eabihf. The
    // image is hardware-pending (USB-HS/RCC bench bring-up) but the
    // build is real and the artifact valid — see the crate README.
    Target {
        id: "portenta-h7",
        skeleton: "embedded/teios-h747",
        triple: "thumbv7em-none-eabihf",
        packaging: Packaging::Bin,
        family: "stm32h7",
        features: &["board-portenta-h7"],
        no_default_features: false,
    },
    Target {
        id: "portenta-h7-lite",
        skeleton: "embedded/teios-h747",
        triple: "thumbv7em-none-eabihf",
        packaging: Packaging::Bin,
        family: "stm32h7",
        features: &["board-portenta-h7-lite"],
        no_default_features: true,
    },
    // Nicla Voice / Nicla Sense ME — nRF52832 (Cortex-M4F), UART
    // transport (no USB on this part), DFU/probe-flashed .bin (E1c).
    // Skeleton is the teios-nrf52832 crate.
    Target {
        id: "nicla-voice",
        skeleton: "embedded/teios-nrf52832",
        triple: "thumbv7em-none-eabihf",
        packaging: Packaging::Bin,
        family: "nrf52832",
        features: &["board-nicla-voice"],
        no_default_features: false,
    },
    Target {
        id: "nicla-sense",
        skeleton: "embedded/teios-nrf52832",
        triple: "thumbv7em-none-eabihf",
        packaging: Packaging::Bin,
        family: "nrf52832",
        features: &["board-nicla-sense"],
        no_default_features: true,
    },
    // Portenta C33 — Renesas RA6M5 (Cortex-M33), bare-metal (no embassy
    // HAL), CRCA hardware-CRC substrate, semihosting transport,
    // probe-flashed .bin (E1d). Skeleton is the teios-ra6m5 crate.
    Target {
        id: "portenta-c33",
        skeleton: "embedded/teios-ra6m5",
        triple: "thumbv8m.main-none-eabihf",
        packaging: Packaging::Bin,
        family: "ra6m5",
        features: &["board-portenta-c33"],
        no_default_features: false,
    },
];

/// Look up a target by id.
pub fn target(id: &str) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.id == id)
}

/// Bench boards that aren't forge build targets (Tier-0 / their own boot
/// flow) but ARE in the chipdb registry, so Studio's BOARD view can show
/// their identity / 3D / pinout. View-only — not buildable by the forge.
/// Ids are chipdb aliases (resolve via [`board_info`]).
pub const BENCH_BOARDS: &[&str] = &[
    "nano-matter",
    "openmv-ae3",
    "tachyon",
    "coral-dev-mini",
    // Linux SBC + Hailo NPU — software telemetry + accelerator energy.
    "pi5",
    // Alchitry FPGA family — fabric is the substrate, bitstream is the
    // "program", external shunt for joules (Tier-0, not a forge target).
    "alchitry-au-v2",
    "alchitry-au-plus",
    "alchitry-pt-v2",
];

/// The chipdb board backing a forge target — the canonical identity
/// (name, vendor, chip, family, price, url). Board identity is owned by
/// `ofpga-chipdb` (the single board registry); the forge `Target` carries
/// only build-specific fields. The target id is registered as a chipdb
/// alias, so this resolves for every target. Returns `None` only if a
/// target was added without a matching chipdb entry (caught by tests).
pub fn board_info(id: &str) -> Option<&'static ofpga_chipdb::boards::Board> {
    ofpga_chipdb::boards::find_board(id)
}

#[cfg(test)]
mod chipdb_tests {
    use super::*;

    #[test]
    fn every_target_resolves_in_chipdb() {
        for t in TARGETS {
            let b = board_info(t.id)
                .unwrap_or_else(|| panic!("forge target {} not in ofpga-chipdb", t.id));
            assert!(!b.name.is_empty(), "{} has no chipdb name", t.id);
        }
    }
}

/// Locate the workspace root by walking up from `start` for the marker
/// `crates/tei-forge` (so the forge finds its skeletons regardless of
/// the server's CWD).
pub fn workspace_root(start: &Path) -> Option<PathBuf> {
    let mut dir = Some(start);
    while let Some(d) = dir {
        if d.join("crates/tei-forge/Cargo.toml").is_file() {
            return Some(d.to_path_buf());
        }
        dir = d.parent();
    }
    None
}

/// Hex SHA-256 of bytes — so the HTTP layer can match a requested
/// artifact hash without re-implementing hashing.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::{Digest, Sha256};
    let d = Sha256::new().chain_update(bytes).finalize();
    let mut s = String::with_capacity(64);
    for b in d {
        s.push_str(&format!("{b:02x}"));
    }
    s
}

/// Is a cargo toolchain reachable? Cheap preflight for the HTTP layer to
/// report a clean "build host has no toolchain" instead of a build error.
pub fn toolchain_available() -> bool {
    Command::new("cargo")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
