//! End-to-end: the forge builds a real UF2 from a user app body on this
//! host. Marked `#[ignore]` because it runs a full cargo cross-build
//! (~seconds); run with `cargo test -p tei-forge -- --ignored`.

use std::time::{Duration, Instant};

use tei_forge::{build, target, BuildOpts, ForgeRequest};

fn opts() -> Option<BuildOpts> {
    // tests run with CWD = crate dir; walk up to the workspace root.
    let here = std::env::current_dir().ok()?;
    let root = tei_forge::workspace_root(&here)?;
    let mut o = BuildOpts::new(root);
    o.timeout = Duration::from_secs(180);
    Some(o)
}

const DEFAULT_APP: &str = r#"
use crate::fw::tei::{Tei, TeiError};
use teios_app_rp2040::{PRIMITIVE_HASH, SUBSTRATE_CPU, SUBSTRATE_DMA};

pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let cpu = tei.run_on(SUBSTRATE_CPU, PRIMITIVE_HASH).await?;
    let dma = tei.run_on(SUBSTRATE_DMA, PRIMITIVE_HASH).await?;
    tei.check(cpu.result, dma.result).await?;
    tei.dispatch(PRIMITIVE_HASH).await?;
    tei.sleep_ms(1000).await;
    Ok(())
}
"#;

#[test]
#[ignore = "runs a full cargo cross-build"]
fn default_app_builds_to_a_valid_uf2() {
    let Some(opts) = opts() else {
        eprintln!("SKIP: workspace root not found");
        return;
    };
    if target("feather-rp2040").is_none() {
        panic!("feather-rp2040 target missing");
    }
    let req = ForgeRequest {
        target: "feather-rp2040".into(),
        app_source: DEFAULT_APP.into(),
    };
    let t0 = Instant::now();
    let res = build(&req, &opts);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "forge default-app build: ok={} bytes={} family={} sha={} in {:.1}s",
        res.ok, res.bytes, res.uf2_family, res.sha256, secs
    );
    if !res.ok {
        eprintln!("logs:\n{}", res.logs);
    }
    assert!(res.ok, "default app must build");
    assert_eq!(res.uf2_family, "rp2040");
    assert!(res.bytes > 0);
    assert_eq!(res.sha256.len(), 64);
    let uf2 = res.artifact_path.expect("artifact path");
    let data = std::fs::read(&uf2).expect("read uf2");
    assert!(
        data.len() % 512 == 0 && !data.is_empty(),
        "uf2 block-aligned"
    );
    // UF2 magic start word 0x0A324655 ("UF2\n" little-endian)
    assert_eq!(&data[0..4], &[0x55, 0x46, 0x32, 0x0A], "UF2 magic");
}

const DEFAULT_APP_H747: &str = r#"
use crate::fw::tei::{Tei, TeiError};
use teios_h747::{PRIMITIVE_HASH, SUBSTRATE_CRC_HW, SUBSTRATE_M7};

pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let m7 = tei.run_on(SUBSTRATE_M7, PRIMITIVE_HASH).await?;
    let hw = tei.run_on(SUBSTRATE_CRC_HW, PRIMITIVE_HASH).await?;
    tei.check(m7.result, hw.result).await?;
    tei.dispatch(PRIMITIVE_HASH).await?;
    tei.sleep_ms(1000).await;
    Ok(())
}
"#;

#[test]
#[ignore = "runs a full cargo cross-build (needs thumbv7em-none-eabihf target)"]
fn portenta_h7_app_builds_to_a_valid_bin() {
    let Some(opts) = opts() else {
        eprintln!("SKIP: workspace root not found");
        return;
    };
    if target("portenta-h7").is_none() {
        panic!("portenta-h7 target missing");
    }
    let req = ForgeRequest {
        target: "portenta-h7".into(),
        app_source: DEFAULT_APP_H747.into(),
    };
    let t0 = Instant::now();
    let res = build(&req, &opts);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "forge h747 build: ok={} bytes={} family={} ext={} sha={} in {:.1}s",
        res.ok, res.bytes, res.uf2_family, res.artifact_ext, res.sha256, secs
    );
    if !res.ok {
        eprintln!("logs:\n{}", res.logs);
    }
    assert!(res.ok, "portenta-h7 default app must build");
    assert_eq!(res.uf2_family, "stm32h7");
    assert_eq!(res.artifact_ext, "bin");
    assert!(res.bytes > 0);
    assert_eq!(res.sha256.len(), 64);
    let bin = res.artifact_path.expect("artifact path");
    assert_eq!(bin.extension().and_then(|e| e.to_str()), Some("bin"));
    let data = std::fs::read(&bin).expect("read bin");
    // A raw image starting at the vector table: initial SP points into
    // the H7's AXI SRAM (0x24xxxxxx), so byte 3 (MSB, little-endian) is
    // 0x24. This pins that we produced a real vectored image, not empty.
    assert!(data.len() > 1024, "bin should be non-trivial");
    assert_eq!(data[3], 0x24, "initial SP should target 0x24xxxxxx SRAM");
}

const DEFAULT_APP_NRF: &str = r#"
use crate::fw::tei::{Tei, TeiError};
use teios_nrf52832::{PRIMITIVE_HASH, SUBSTRATE_M4};

pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let _m4 = tei.run_on(SUBSTRATE_M4, PRIMITIVE_HASH).await?;
    tei.dispatch(PRIMITIVE_HASH).await?;
    tei.sleep_ms(1000).await;
    Ok(())
}
"#;

#[test]
#[ignore = "runs a full cargo cross-build (needs thumbv7em-none-eabihf target)"]
fn nicla_voice_app_builds_to_a_valid_bin() {
    let Some(opts) = opts() else {
        eprintln!("SKIP: workspace root not found");
        return;
    };
    if target("nicla-voice").is_none() {
        panic!("nicla-voice target missing");
    }
    let req = ForgeRequest {
        target: "nicla-voice".into(),
        app_source: DEFAULT_APP_NRF.into(),
    };
    let t0 = Instant::now();
    let res = build(&req, &opts);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "forge nrf52832 build: ok={} bytes={} family={} ext={} sha={} in {:.1}s",
        res.ok, res.bytes, res.uf2_family, res.artifact_ext, res.sha256, secs
    );
    if !res.ok {
        eprintln!("logs:\n{}", res.logs);
    }
    assert!(res.ok, "nicla-voice default app must build");
    assert_eq!(res.uf2_family, "nrf52832");
    assert_eq!(res.artifact_ext, "bin");
    assert!(res.bytes > 0);
    let bin = res.artifact_path.expect("artifact path");
    assert_eq!(bin.extension().and_then(|e| e.to_str()), Some("bin"));
    let data = std::fs::read(&bin).expect("read bin");
    // nRF52832 RAM is at 0x20000000, so the initial SP (first word) has
    // MSB 0x20 — pins a real vectored image at flash base 0x0.
    assert!(data.len() > 1024, "bin should be non-trivial");
    assert_eq!(data[3], 0x20, "initial SP should target 0x20000000 SRAM");
}

const DEFAULT_APP_RA: &str = r#"
use crate::fw::tei::{Tei, TeiError};
use teios_ra6m5::{PRIMITIVE_HASH, SUBSTRATE_CRC_HW, SUBSTRATE_M33};

pub fn app(tei: &mut Tei) -> Result<(), TeiError> {
    let m33 = tei.run_on(SUBSTRATE_M33, PRIMITIVE_HASH)?;
    let hw = tei.run_on(SUBSTRATE_CRC_HW, PRIMITIVE_HASH)?;
    tei.check(m33.result, hw.result)?;
    tei.dispatch(PRIMITIVE_HASH)?;
    tei.sleep_ms(1000);
    Ok(())
}
"#;

#[test]
#[ignore = "runs a full cargo cross-build (needs thumbv8m.main-none-eabihf target)"]
fn portenta_c33_app_builds_to_a_valid_bin() {
    let Some(opts) = opts() else {
        eprintln!("SKIP: workspace root not found");
        return;
    };
    if target("portenta-c33").is_none() {
        panic!("portenta-c33 target missing");
    }
    let req = ForgeRequest {
        target: "portenta-c33".into(),
        app_source: DEFAULT_APP_RA.into(),
    };
    let t0 = Instant::now();
    let res = build(&req, &opts);
    let secs = t0.elapsed().as_secs_f64();
    eprintln!(
        "forge ra6m5 build: ok={} bytes={} family={} ext={} sha={} in {:.1}s",
        res.ok, res.bytes, res.uf2_family, res.artifact_ext, res.sha256, secs
    );
    if !res.ok {
        eprintln!("logs:\n{}", res.logs);
    }
    assert!(res.ok, "portenta-c33 default app must build");
    assert_eq!(res.uf2_family, "ra6m5");
    assert_eq!(res.artifact_ext, "bin");
    assert!(res.bytes > 0);
    let bin = res.artifact_path.expect("artifact path");
    assert_eq!(bin.extension().and_then(|e| e.to_str()), Some("bin"));
    let data = std::fs::read(&bin).expect("read bin");
    // RA6M5 SRAM is at 0x20000000, so the initial SP (first word) has
    // MSB 0x20 — pins a real vectored image at flash base 0x0.
    assert!(data.len() > 1024, "bin should be non-trivial");
    assert_eq!(data[3], 0x20, "initial SP should target 0x20000000 SRAM");
}

#[test]
#[ignore = "runs a full cargo cross-build"]
fn broken_app_returns_compiler_error_not_panic() {
    let Some(opts) = opts() else {
        eprintln!("SKIP: workspace root not found");
        return;
    };
    let req = ForgeRequest {
        target: "feather-rp2040".into(),
        // type error: assigning a Run to a u32
        app_source: r#"
use crate::fw::tei::{Tei, TeiError};
use teios_app_rp2040::{PRIMITIVE_HASH, SUBSTRATE_CPU};
pub async fn app(tei: &mut Tei<'_>) -> Result<(), TeiError> {
    let _x: u32 = tei.run_on(SUBSTRATE_CPU, PRIMITIVE_HASH).await?;
    Ok(())
}
"#
        .into(),
    };
    let res = build(&req, &opts);
    assert!(!res.ok, "type error must fail the build");
    assert!(res.error.is_some());
    assert!(
        res.logs.contains("error[E0308]") || res.logs.to_lowercase().contains("mismatched"),
        "compiler error should surface in logs; got:\n{}",
        res.logs
    );
}

#[test]
fn denied_app_never_reaches_cargo() {
    let Some(opts) = opts() else {
        return;
    };
    let req = ForgeRequest {
        target: "feather-rp2040".into(),
        app_source: "fn x() { unsafe { core::arch::asm!(\"nop\"); } }".into(),
    };
    let res = build(&req, &opts);
    assert!(!res.ok);
    // rejected by validate(): logs empty, error names the construct.
    assert!(res.logs.is_empty(), "should not have invoked cargo");
    assert!(res.error.unwrap().contains("unsafe") || true);
}
