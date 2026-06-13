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

/// What Studio's BUILD tab POSTs: a target board id and the contents of
/// the user's `src/app.rs`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeRequest {
    /// Board id matching a skeleton target (e.g. "feather-rp2040",
    /// "pico").
    pub target: String,
    /// The user-editable `app.rs` source.
    pub app_source: String,
}

/// The build outcome — serde-stable for the HTTP layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForgeResult {
    pub ok: bool,
    /// Path to the produced `.uf2` on the build host (the HTTP layer
    /// serves its bytes; not a URL).
    pub artifact_path: Option<PathBuf>,
    /// UF2 family tag, e.g. "rp2040".
    pub uf2_family: String,
    /// Image bytes (payload, not the UF2 file size).
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
            bytes: 0,
            sha256: String::new(),
            logs: logs.into(),
            error: Some(error.into()),
        }
    }
}

/// One buildable skeleton: where it lives, its cross-target, and the UF2
/// family/base the packager stamps.
#[derive(Debug, Clone)]
pub struct Target {
    pub id: &'static str,
    /// Skeleton project dir, relative to the workspace root.
    pub skeleton: &'static str,
    /// rustc target triple.
    pub triple: &'static str,
    /// elf2uf2 family name (rp2040 → 0xe48bff56).
    pub uf2_family: &'static str,
    /// Build features to pass (board id selection, boot2 choice).
    pub features: &'static [&'static str],
    /// `--no-default-features` when selecting a non-default board.
    pub no_default_features: bool,
}

/// The targets the forge can build today. Bench-first: the RP2040
/// Feather and Pico, which David owns and which need no bench energy
/// gear to demo the dispatch loop.
pub const TARGETS: &[Target] = &[
    Target {
        id: "feather-rp2040",
        skeleton: "embedded/teios-app-rp2040",
        triple: "thumbv6m-none-eabi",
        uf2_family: "rp2040",
        features: &["board-feather-rp2040"],
        no_default_features: false,
    },
    Target {
        id: "pico",
        skeleton: "embedded/teios-app-rp2040",
        triple: "thumbv6m-none-eabi",
        uf2_family: "rp2040",
        features: &["board-pico"],
        no_default_features: true,
    },
];

/// Look up a target by id.
pub fn target(id: &str) -> Option<&'static Target> {
    TARGETS.iter().find(|t| t.id == id)
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
