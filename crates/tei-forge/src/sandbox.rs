//! The seatbelt sandbox — layer 3 of the security model (see crate docs).
//!
//! Wraps the `cargo build` in macOS `sandbox-exec` with a profile that
//! denies all network and confines writes to the build + scratch dirs.
//! Reads stay broad (the toolchain and registry must be readable) but
//! with no network there is no exfiltration path. On hosts lacking
//! `sandbox-exec` the caller degrades to an unsandboxed build with a
//! logged warning.

use std::path::Path;

/// Absolute path to the macOS sandbox binary.
pub const SANDBOX_EXEC: &str = "/usr/bin/sandbox-exec";

/// Is the seatbelt sandbox available on this host?
pub fn available() -> bool {
    Path::new(SANDBOX_EXEC).exists()
}

/// Build the seatbelt profile string. `write_dirs` are the absolute
/// subpaths the build is allowed to write (the temp project + its target
/// dir + the OS temp root for cargo's own scratch).
pub fn profile(write_dirs: &[&Path]) -> String {
    let mut p = String::from(
        "(version 1)\n\
         (deny default)\n\
         (allow process-fork)\n\
         (allow process-exec)\n\
         (allow sysctl-read)\n\
         (allow mach-lookup)\n\
         (allow signal (target self))\n\
         (allow file-read*)\n\
         (deny network*)\n",
    );
    for d in write_dirs {
        // subpath must be absolute; lossy is fine for the profile text.
        p.push_str(&format!(
            "(allow file-write* (subpath \"{}\"))\n",
            d.display()
        ));
    }
    p
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn profile_denies_network_and_allows_listed_writes() {
        let d = PathBuf::from("/tmp/forge-xyz");
        let prof = profile(&[&d]);
        assert!(prof.contains("(deny network*)"));
        assert!(prof.contains("(deny default)"));
        assert!(prof.contains("/tmp/forge-xyz"));
        assert!(prof.contains("(allow file-read*)"));
    }
}
