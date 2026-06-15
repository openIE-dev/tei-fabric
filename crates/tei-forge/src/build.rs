//! The build pipeline: validate → splice into a temp copy of the
//! skeleton → cargo build (offline, sandboxed, timed) → elf2uf2 →
//! packaged artifact + logs. Layer 1 (fixed deps) is enforced here by
//! writing *only* `src/app.rs` into the copied skeleton.

use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use sha2::{Digest, Sha256};

use crate::{sandbox, target, validate, ForgeRequest, ForgeResult, Packaging, Target};

/// Build knobs.
#[derive(Debug, Clone)]
pub struct BuildOpts {
    /// Workspace root (contains `embedded/…` skeletons).
    pub workspace_root: PathBuf,
    /// Where produced UF2s are kept (created if absent).
    pub results_dir: PathBuf,
    /// Persistent CARGO_TARGET_DIR shared across builds: dep artifacts
    /// cache here, so only the changed user crate recompiles (cold ~40s,
    /// warm ~seconds). Forge-controlled, sandbox-allowed.
    pub shared_target: PathBuf,
    /// Wall-clock kill time for the cargo build.
    pub timeout: Duration,
}

impl BuildOpts {
    pub fn new(workspace_root: PathBuf) -> Self {
        let results_dir = workspace_root.join("target/forge-artifacts");
        let shared_target = workspace_root.join("target/forge-target");
        Self {
            workspace_root,
            results_dir,
            shared_target,
            timeout: Duration::from_secs(120),
        }
    }
}

const LOG_CAP: usize = 64 * 1024;

fn truncate_logs(mut s: String) -> String {
    if s.len() > LOG_CAP {
        let keep = &s[s.len() - LOG_CAP..];
        s = format!("…(truncated)…\n{keep}");
    }
    s
}

/// Run the full pipeline for `req`.
pub fn build(req: &ForgeRequest, opts: &BuildOpts) -> ForgeResult {
    // 1. validate (layer 2)
    if let Err(e) = validate(&req.app_source) {
        return ForgeResult::failed(e, String::new());
    }
    let Some(tgt) = target(&req.target) else {
        return ForgeResult::failed(format!("unknown target: {}", req.target), String::new());
    };
    let skeleton = opts.workspace_root.join(tgt.skeleton);
    if !skeleton.join("Cargo.toml").is_file() {
        return ForgeResult::failed(
            format!("skeleton missing: {}", skeleton.display()),
            String::new(),
        );
    }

    // 2. fresh temp copy of the skeleton; write ONLY app.rs (layer 1)
    let tmp = match tempfile::Builder::new().prefix("tei-forge-").tempdir() {
        Ok(t) => t,
        Err(e) => return ForgeResult::failed(format!("tempdir: {e}"), String::new()),
    };
    let proj = tmp.path().join("proj");
    if let Err(e) = copy_skeleton(&skeleton, &proj) {
        return ForgeResult::failed(format!("copy skeleton: {e}"), String::new());
    }
    // The skeleton's relative `path = "../.."` deps break once copied out
    // of the workspace. Canonicalize them to absolute paths pointing back
    // at the real crates. This is forge-controlled, NOT user input — the
    // dependency set is unchanged, only its location is made absolute, so
    // the fixed-deps invariant holds.
    if let Err(e) = absolutize_path_deps(&proj.join("Cargo.toml"), &skeleton) {
        return ForgeResult::failed(format!("rewrite deps: {e}"), String::new());
    }
    if let Err(e) = fs::write(proj.join("src/app.rs"), &req.app_source) {
        return ForgeResult::failed(format!("write app.rs: {e}"), String::new());
    }

    // 3. cargo build — offline, sandboxed, timed. Use the shared target
    // dir so deps cache across builds.
    let target_dir = opts.shared_target.clone();
    let (status_ok, logs, mut sandboxed) = run_cargo(req, tgt, &proj, &target_dir, opts);
    let logs = truncate_logs(logs);
    if !status_ok {
        let mut out = ForgeResult::failed("build failed", logs);
        if !sandboxed {
            out.logs
                .push_str("\n[forge] note: sandbox-exec absent — built unsandboxed.");
        }
        return out;
    }

    // 4. package the ELF → flashable artifact (UF2 or raw .bin)
    let elf = target_dir
        .join(tgt.triple)
        .join("release")
        .join(skeleton_bin_name(&skeleton));
    if !elf.is_file() {
        return ForgeResult::failed(format!("expected ELF not found: {}", elf.display()), logs);
    }
    if let Err(e) = fs::create_dir_all(&opts.results_dir) {
        return ForgeResult::failed(format!("results dir: {e}"), logs);
    }
    let ext = tgt.packaging.ext();
    let artifact = opts.results_dir.join(format!(
        "teios-{}-{}.{ext}",
        tgt.id,
        short_hash(&req.app_source)
    ));
    let pkg_result = match tgt.packaging {
        Packaging::Uf2 => package_uf2(&skeleton, &elf, &artifact, &proj),
        Packaging::Bin => package_bin(&elf, &artifact),
    };
    if let Err(e) = pkg_result {
        let mut l = logs;
        l.push_str(&format!("\n[package {ext}]\n{e}"));
        return ForgeResult::failed(format!("{ext} packaging failed"), truncate_logs(l));
    }

    // 5. measure + hash
    let bytes = fs::metadata(&elf).map(|m| m.len() as usize).unwrap_or(0);
    let (sha, _len) = sha256_file(&artifact).unwrap_or_default();
    let _ = &mut sandboxed;
    ForgeResult {
        ok: true,
        artifact_path: Some(artifact),
        uf2_family: tgt.family.to_string(),
        artifact_ext: ext.to_string(),
        bytes,
        sha256: sha,
        logs,
        error: None,
    }
}

/// Package via the skeleton's `scripts/elf2uf2.py` (RP-class boards).
fn package_uf2(skeleton: &Path, elf: &Path, out: &Path, proj: &Path) -> Result<(), String> {
    let script = skeleton.join("scripts/elf2uf2.py");
    let o = Command::new("python3")
        .arg(&script)
        .arg(elf)
        .arg(out)
        .current_dir(proj)
        .output()
        .map_err(|e| format!("spawn elf2uf2: {e}"))?;
    if o.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&o.stderr).into_owned())
    }
}

/// Package via `llvm-objcopy -O binary` (DFU boards). objcopy ships in
/// the rust toolchain's sysroot, so no extra install is needed on the
/// build host.
fn package_bin(elf: &Path, out: &Path) -> Result<(), String> {
    let objcopy = llvm_objcopy().ok_or("llvm-objcopy not found in rust sysroot")?;
    let o = Command::new(&objcopy)
        .arg("-O")
        .arg("binary")
        .arg(elf)
        .arg(out)
        .output()
        .map_err(|e| format!("spawn llvm-objcopy: {e}"))?;
    if o.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&o.stderr).into_owned())
    }
}

/// Find `llvm-objcopy` under the active rust toolchain's sysroot.
fn llvm_objcopy() -> Option<PathBuf> {
    let out = Command::new("rustc").arg("--print").arg("sysroot").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let sysroot = String::from_utf8(out.stdout).ok()?;
    let sysroot = Path::new(sysroot.trim());
    // bin/ layout: <sysroot>/lib/rustlib/<host>/bin/llvm-objcopy
    for entry in walk_for(sysroot, "llvm-objcopy", 5) {
        return Some(entry);
    }
    None
}

/// Shallow bounded search for a filename under `root` (depth-limited so a
/// pathological sysroot can't hang the build).
fn walk_for(root: &Path, name: &str, depth: usize) -> Vec<PathBuf> {
    let mut hits = Vec::new();
    if depth == 0 {
        return hits;
    }
    let Ok(rd) = fs::read_dir(root) else {
        return hits;
    };
    for e in rd.flatten() {
        let p = e.path();
        if p.is_dir() {
            hits.extend(walk_for(&p, name, depth - 1));
        } else if p.file_name().and_then(|n| n.to_str()) == Some(name) {
            hits.push(p);
        }
        if !hits.is_empty() {
            break;
        }
    }
    hits
}

/// Spawn cargo (optionally under seatbelt), enforce the wall-timeout by
/// killing the process group, return (ok, combined logs, sandboxed?).
fn run_cargo(
    req: &ForgeRequest,
    tgt: &Target,
    proj: &Path,
    target_dir: &Path,
    opts: &BuildOpts,
) -> (bool, String, bool) {
    let mut args: Vec<String> = Vec::new();
    let sandboxed = sandbox::available();
    // cargo writes target/ (and Cargo.lock into proj). Create target_dir
    // so it can be canonicalized for the seatbelt subpath.
    let _ = fs::create_dir_all(target_dir);
    if sandboxed {
        // Seatbelt matches the RESOLVED path: temp dirs are symlinks
        // (/var/folders → /private/var/folders), so canonicalize every
        // write-subpath or the writes are denied.
        let canon = |p: &Path| p.canonicalize().unwrap_or_else(|_| p.to_path_buf());
        let proj_c = canon(proj);
        let target_c = canon(target_dir);
        let tmp_c = canon(&std::env::temp_dir());
        let prof = sandbox::profile(&[&proj_c, &target_c, &tmp_c]);
        args.push("-p".into());
        args.push(prof);
        args.push("cargo".into());
    }
    let mut cargo_args = vec![
        "build".to_string(),
        "--release".to_string(),
        "--offline".to_string(),
        "--target".to_string(),
        tgt.triple.to_string(),
    ];
    if tgt.no_default_features {
        cargo_args.push("--no-default-features".into());
    }
    for f in tgt.features {
        cargo_args.push("--features".into());
        cargo_args.push((*f).into());
    }
    // Measured variant: wire in the INA228 EnergyMeter feature when the
    // request asks for it AND the target supports it (no-op on bare-metal
    // RA6M5, which declares no measured_feature).
    if req.measured {
        if let Some(mf) = tgt.measured_feature {
            cargo_args.push("--features".into());
            cargo_args.push(mf.into());
        }
    }

    let program = if sandboxed {
        sandbox::SANDBOX_EXEC
    } else {
        "cargo"
    };
    let mut cmd = Command::new(program);
    if sandboxed {
        cmd.args(&args);
    }
    cmd.args(&cargo_args)
        .current_dir(proj)
        .env("CARGO_NET_OFFLINE", "1")
        .env("CARGO_TARGET_DIR", target_dir)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

    let started = Instant::now();
    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => return (false, format!("spawn cargo: {e}"), sandboxed),
    };

    // Poll for completion or timeout. (No process-group plumbing in std;
    // kill the child — its build subprocesses are short-lived rustc
    // invocations that exit when the parent's pipes close.)
    loop {
        match child.try_wait() {
            Ok(Some(status)) => {
                let mut logs = String::new();
                if let Some(mut o) = child.stdout.take() {
                    let _ = o.read_to_string(&mut logs);
                }
                if let Some(mut e) = child.stderr.take() {
                    let _ = e.read_to_string(&mut logs);
                }
                return (status.success(), logs, sandboxed);
            }
            Ok(None) => {
                if started.elapsed() > opts.timeout {
                    let _ = child.kill();
                    let _ = child.wait();
                    return (
                        false,
                        format!(
                            "build exceeded {}s wall-timeout — killed",
                            opts.timeout.as_secs()
                        ),
                        sandboxed,
                    );
                }
                std::thread::sleep(Duration::from_millis(100));
            }
            Err(e) => return (false, format!("wait cargo: {e}"), sandboxed),
        }
    }
}

/// Copy the skeleton tree, excluding `target/` (huge — the build makes
/// its own) and `Cargo.lock` (it pins the workspace; `--offline` resolves
/// from the shared cache instead). `.cargo/config.toml`, memory.x,
/// build.rs, scripts/ all come along — the authoritative build inputs.
fn copy_skeleton(src: &Path, dst: &Path) -> std::io::Result<()> {
    fs::create_dir_all(dst)?;
    for entry in fs::read_dir(src)? {
        let entry = entry?;
        let name = entry.file_name();
        if name == "target" || name == "Cargo.lock" {
            continue;
        }
        let from = entry.path();
        let to = dst.join(&name);
        if from.is_dir() {
            copy_skeleton(&from, &to)?;
        } else {
            fs::copy(&from, &to)?;
        }
    }
    Ok(())
}

/// Rewrite every `path = "<relative>"` in the copied Cargo.toml to the
/// absolute path it resolves to from the ORIGINAL skeleton dir. Only the
/// dep location changes; the dep set is identical.
fn absolutize_path_deps(cargo_toml: &Path, orig_skeleton: &Path) -> std::io::Result<()> {
    let text = fs::read_to_string(cargo_toml)?;
    let mut out = String::with_capacity(text.len());
    for line in text.lines() {
        if let Some(rewritten) = rewrite_path_line(line, orig_skeleton) {
            out.push_str(&rewritten);
        } else {
            out.push_str(line);
        }
        out.push('\n');
    }
    fs::write(cargo_toml, out)
}

/// If `line` contains `path = "<rel>"`, replace `<rel>` with its absolute
/// form resolved against `base`. Returns None if there's no path dep.
fn rewrite_path_line(line: &str, base: &Path) -> Option<String> {
    let key = "path = \"";
    let start = line.find(key)? + key.len();
    let rest = &line[start..];
    let end = rest.find('"')?;
    let rel = &rest[..end];
    if rel.starts_with('/') {
        return None; // already absolute
    }
    let abs = base.join(rel);
    let abs = abs.canonicalize().unwrap_or(abs);
    Some(format!(
        "{}{}{}",
        &line[..start],
        abs.display(),
        &line[start + end..]
    ))
}

/// The bin name = the skeleton's package name (its dir name here).
fn skeleton_bin_name(skeleton: &Path) -> String {
    skeleton
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("app")
        .to_string()
}

fn short_hash(s: &str) -> String {
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let d = h.finalize();
    hex(&d[..6])
}

fn sha256_file(p: &Path) -> Option<(String, u64)> {
    let mut f = fs::File::open(p).ok()?;
    let mut h = Sha256::new();
    let mut buf = [0u8; 8192];
    let mut total = 0u64;
    loop {
        let n = f.read(&mut buf).ok()?;
        if n == 0 {
            break;
        }
        total += n as u64;
        h.update(&buf[..n]);
    }
    Some((hex(&h.finalize()), total))
}

fn hex(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push_str(&format!("{b:02x}"));
    }
    s
}
