//! tei-serve — Axum HTTP front for the TEI fabric.
//!
//! Endpoints (v0):
//!   GET  /health                    Liveness + catalog size.
//!   GET  /api/stack                 Full Periodic Stack catalog JSON.
//!   GET  /api/substrates            Registered substrate list + citations.
//!   POST /api/dispatch              Body: Workload → DispatchPlan.
//!
//! Defaults: listens on `0.0.0.0:9651`. Override with `PORT`. Catalog path
//! is `data/stack.json` relative to CWD; override with `STACK_JSON_PATH`.

use axum::{
    Json, Router,
    body::Bytes,
    extract::{DefaultBodyLimit, State},
    http::{HeaderMap, StatusCode},
    response::{
        IntoResponse,
        sse::{Event, KeepAlive, Sse},
    },
    routing::{get, post},
};
use futures::stream::Stream;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::convert::Infallible;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tei_cost_surface::{
    Preset, SubstrateParams, default_substrates, dispatch, dispatch_invocation, enumerate_presets,
    substrates_with_params, summarize,
};
use tei_ir::Workload;
use tei_stack::{Stack, StackData};
use tei_substrate_traits::Substrate;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

mod calib;

/// One pending chunked upload. Lives only until the final chunk arrives or
/// the GC sweep drops it.
struct UploadBuffer {
    chunks: BTreeMap<u32, Bytes>,
    total: u32,
    received_bytes: usize,
    started: Instant,
}

#[derive(Default)]
struct UploadState {
    map: Mutex<HashMap<String, UploadBuffer>>,
}

#[derive(Clone)]
struct AppState {
    stack: Arc<Stack>,
    substrates: Arc<Vec<Arc<dyn Substrate>>>,
    uploads: Arc<UploadState>,
    /// Persisted measured-constant overrides (the calibration loop's
    /// output). When set, /api/dispatch prices with these by default.
    calibration: Arc<Mutex<Option<SubstrateParams>>>,
    calibration_path: Arc<std::path::PathBuf>,
    /// Append-only store of device calibration reports (the embedded
    /// EventLedger contract's POST target). JSONL, one report per line.
    reports_path: Arc<std::path::PathBuf>,
    reports_lock: Arc<Mutex<()>>,
    /// The forge build service config — Some only where a cargo
    /// toolchain + skeletons exist (a dev/build host, not the bare web
    /// server). None ⇒ /api/forge reports "no build host".
    forge: Arc<Option<tei_forge::BuildOpts>>,
}

impl AppState {
    /// Effective substrate set for a dispatch request: explicit request
    /// params win, then the persisted calibration, then literature
    /// defaults. Returns (substrates, used_calibrated_defaults).
    fn dispatch_substrates(
        &self,
        explicit: Option<&SubstrateParams>,
    ) -> (Arc<Vec<Arc<dyn Substrate>>>, bool) {
        if let Some(p) = explicit {
            return (
                Arc::new(substrates_with_params(self.stack.clone(), p)),
                false,
            );
        }
        if let Some(p) = self.calibration.lock().unwrap().as_ref() {
            return (
                Arc::new(substrates_with_params(self.stack.clone(), p)),
                true,
            );
        }
        (self.substrates.clone(), false)
    }
}

/// Hard cap on total bytes any single upload can accumulate before we drop
/// the buffer. 2.5 GB covers full Stable-Diffusion v1 UNets (~1.6 GB float)
/// with headroom; SDXL UNets (~5 GB) still need a separate path.
const MAX_UPLOAD_BYTES: usize = 2_560 * 1024 * 1024;
/// Maximum concurrent active uploads. Bounds memory residency.
const MAX_CONCURRENT_UPLOADS: usize = 16;
/// Drop uploads that have been idle this long.
const UPLOAD_TTL: Duration = Duration::from_secs(15 * 60);

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    catalog_primitives: usize,
    substrates: usize,
}

#[derive(Serialize)]
struct SubstrateInfo {
    name: String,
    display_name: String,
    citations: Vec<&'static str>,
}

async fn health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok",
        catalog_primitives: state.stack.count(),
        substrates: state.substrates.len(),
    })
}

async fn get_stack(State(state): State<AppState>) -> Json<StackData> {
    Json(state.stack.data.clone())
}

async fn list_presets(State(state): State<AppState>) -> Json<Vec<Preset>> {
    Json(enumerate_presets(&state.stack, &state.substrates))
}

async fn list_substrates(State(state): State<AppState>) -> Json<Vec<SubstrateInfo>> {
    let infos: Vec<SubstrateInfo> = state
        .substrates
        .iter()
        .map(|s| SubstrateInfo {
            name: s.name().to_string(),
            display_name: s.display_name().to_string(),
            citations: s.citations().to_vec(),
        })
        .collect();
    Json(infos)
}

/// Dispatch request wraps a Workload + optional substrate engineering
/// parameters. The substrate_params field is `#[serde(flatten)]`-free so
/// the JSON shape is `{...workload..., "substrate_params": {...}}`.
#[derive(Deserialize)]
struct DispatchRequest {
    #[serde(flatten)]
    workload: Workload,
    #[serde(default)]
    substrate_params: Option<SubstrateParams>,
}

/// GET /api/calibration — the persisted measured-constant overrides.
async fn get_calibration(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cal = state.calibration.lock().unwrap().clone();
    Json(serde_json::json!({ "calibrated": cal.is_some(), "substrate_params": cal }))
}

/// POST /api/calibration — persist measured constants (a SubstrateParams;
/// omitted fields fall back to literature defaults). Survives restarts via
/// CALIBRATION_PATH (default `calibration.json` in the server CWD, outside
/// the deploy-synced data/ tree).
async fn post_calibration(
    State(state): State<AppState>,
    Json(params): Json<SubstrateParams>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let json = serde_json::to_string_pretty(&params)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    std::fs::write(state.calibration_path.as_ref(), json).map_err(|e| {
        (
            StatusCode::INTERNAL_SERVER_ERROR,
            format!("persist failed: {e}"),
        )
    })?;
    *state.calibration.lock().unwrap() = Some(params);
    info!("calibration saved to {}", state.calibration_path.display());
    Ok(Json(serde_json::json!({ "calibrated": true })))
}

/// DELETE /api/calibration — revert to literature defaults.
async fn delete_calibration(State(state): State<AppState>) -> Json<serde_json::Value> {
    *state.calibration.lock().unwrap() = None;
    let _ = std::fs::remove_file(state.calibration_path.as_ref());
    info!("calibration cleared");
    Json(serde_json::json!({ "calibrated": false }))
}

/// POST /api/forge/build — compile a user teiOS app for a target board.
/// Body: {target, app_source}. Returns the tei-forge ForgeResult
/// (artifact_path is server-local; the UI fetches the bytes from
/// /api/forge/artifact?h=<sha>). Runs on a blocking thread (cargo).
async fn post_forge_build(
    State(state): State<AppState>,
    Json(req): Json<tei_forge::ForgeRequest>,
) -> Result<Json<tei_forge::ForgeResult>, (StatusCode, String)> {
    let Some(opts) = state.forge.as_ref().clone() else {
        return Err((
            StatusCode::SERVICE_UNAVAILABLE,
            "this server is not a build host (no cargo toolchain / skeletons)".into(),
        ));
    };
    let res = tokio::task::spawn_blocking(move || tei_forge::build(&req, &opts))
        .await
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("join: {e}")))?;
    Ok(Json(res))
}

/// GET /api/forge/targets — the buildable target ids for the BUILD tab.
async fn get_forge_targets(State(state): State<AppState>) -> Json<serde_json::Value> {
    let available = state.forge.is_some();
    let targets: Vec<_> = tei_forge::TARGETS
        .iter()
        .map(|t| {
            // Board identity comes from ofpga-chipdb (the single board
            // registry); the forge owns only the build-specific fields.
            let b = tei_forge::board_info(t.id);
            serde_json::json!({
                "id": t.id,
                "uf2_family": t.family,
                "family": t.family,
                "artifact_ext": t.packaging.ext(),
                "name": b.map(|b| b.name),
                "vendor": b.map(|b| b.vendor),
                "chip": b.map(|b| b.fpga_device),
                "chip_family": b.map(|b| b.fpga_family),
                "price_usd": b.map(|b| b.price_usd),
                "url": b.map(|b| b.url),
                // Whether a Measured build (firmware EnergyMeter driver)
                // can be cross-compiled for this target — drives the
                // BUILD checkbox. Today: the external-shunt (INA228) path.
                "measured": t.measured_feature().is_some(),
                // The board's energy-measurement path (the abstraction —
                // external shunt / on-board PMIC / software / accelerator).
                "energy_source": b.map(|b| tei_forge::ofpga_chipdb::boards::energy_source(b).tag()),
                "energy_how": b.map(|b| tei_forge::ofpga_chipdb::boards::energy_source(b).label()),
            })
        })
        .collect();
    Json(serde_json::json!({ "build_host": available, "targets": targets }))
}

/// GET /api/forge/board-list — every board the BOARD view can show:
/// forge build targets (kind "forge") + bench boards (kind "bench",
/// view-only). Each enriched with chipdb identity.
async fn get_forge_board_list() -> Json<serde_json::Value> {
    let ident = |id: &str, kind: &str| {
        let b = tei_forge::board_info(id);
        serde_json::json!({
            "id": id,
            "kind": kind,
            "name": b.map(|b| b.name),
            "chip": b.map(|b| b.fpga_device),
            "energy_source": b.map(|b| tei_forge::ofpga_chipdb::boards::energy_source(b).tag()),
        })
    };
    let mut boards: Vec<_> = tei_forge::TARGETS.iter().map(|t| ident(t.id, "forge")).collect();
    boards.extend(tei_forge::BENCH_BOARDS.iter().map(|id| ident(id, "bench")));
    Json(serde_json::json!({ "boards": boards }))
}

/// GET /api/forge/board?id=<board> — full board view data for Studio's
/// BOARD workspace: chipdb identity + the color-coded pinout (if a
/// datasheet-verified one exists; otherwise `pinout: null`).
async fn get_forge_board(
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let id = q
        .get("id")
        .cloned()
        .ok_or((StatusCode::BAD_REQUEST, "missing ?id=<board>".into()))?;
    let board = tei_forge::ofpga_chipdb::boards::find_board(&id)
        .ok_or((StatusCode::NOT_FOUND, format!("unknown board: {id}")))?;

    let pinout = tei_forge::ofpga_chipdb::pinout::pinout(&id).map(|p| {
        let pins: Vec<_> = p
            .pins
            .iter()
            .map(|pin| {
                serde_json::json!({
                    "number": pin.number,
                    "name": pin.name,
                    "kind": pin.kind.label(),
                    "color": pin.kind.color(),
                    "functions": pin.functions,
                })
            })
            .collect();
        serde_json::json!({ "rows": p.rows, "pins": pins })
    });

    Ok(Json(serde_json::json!({
        "id": id,
        "name": board.name,
        "vendor": board.vendor,
        "chip": board.fpga_device,
        "chip_family": board.fpga_family,
        "clock_mhz": board.clock_mhz,
        "price_usd": board.price_usd,
        "url": board.url,
        // How this board reports joules into the ledger (the abstraction —
        // not anchored to one part) + any carrier needed to reach the rail.
        "energy_source": tei_forge::ofpga_chipdb::boards::energy_source(board).tag(),
        "energy_how": tei_forge::ofpga_chipdb::boards::energy_source(board).label(),
        "carrier_note": tei_forge::ofpga_chipdb::boards::carrier_note(board),
        "pinout": pinout,
    })))
}

/// GET /api/forge/artifact?h=<sha256> — stream a produced UF2. The sha
/// is matched against the results dir, so only forge-produced files in
/// that dir are servable (no arbitrary path read).
async fn get_forge_artifact(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Result<impl IntoResponse, (StatusCode, String)> {
    let opts = state
        .forge
        .as_ref()
        .clone()
        .ok_or((StatusCode::SERVICE_UNAVAILABLE, "no build host".into()))?;
    let want = q
        .get("h")
        .cloned()
        .ok_or((StatusCode::BAD_REQUEST, "missing ?h=<sha256>".into()))?;
    if !want.chars().all(|c| c.is_ascii_hexdigit()) || want.len() != 64 {
        return Err((StatusCode::BAD_REQUEST, "h must be a 64-hex sha256".into()));
    }
    // Scan the results dir for a UF2 whose sha256 matches.
    let dir = &opts.results_dir;
    let entries = std::fs::read_dir(dir)
        .map_err(|e| (StatusCode::NOT_FOUND, format!("results: {e}")))?;
    for entry in entries.flatten() {
        let path = entry.path();
        // Forge artifacts are UF2 (RP-class) or raw BIN (DFU boards).
        let ext = path.extension().and_then(|e| e.to_str());
        if ext != Some("uf2") && ext != Some("bin") {
            continue;
        }
        if let Ok(bytes) = std::fs::read(&path) {
            if tei_forge::sha256_hex(&bytes) == want {
                let fname = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("teios.bin")
                    .to_string();
                let headers = [
                    (axum::http::header::CONTENT_TYPE, "application/octet-stream".to_string()),
                    (
                        axum::http::header::CONTENT_DISPOSITION,
                        format!("attachment; filename=\"{fname}\""),
                    ),
                ];
                return Ok((headers, bytes));
            }
        }
    }
    Err((StatusCode::NOT_FOUND, "no artifact with that hash".into()))
}

/// POST /api/calibration/reports — a device publishes a measured (or
/// proxy) J/op row: the tei-ledger CalibrationReport JSON shape
/// {board_id, substrate, primitive_id, n_ops, ledger, j_per_op}.
/// Appended as JSONL with a server timestamp; honest provenance is
/// preserved verbatim from the device.
async fn post_calibration_report(
    State(state): State<AppState>,
    Json(mut report): Json<serde_json::Value>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    for key in ["board_id", "substrate", "primitive_id", "n_ops", "ledger", "j_per_op"] {
        if report.get(key).is_none() {
            return Err((StatusCode::BAD_REQUEST, format!("missing field: {key}")));
        }
    }
    if !report["j_per_op"].is_number() || report["j_per_op"].as_f64().unwrap_or(-1.0) <= 0.0 {
        return Err((StatusCode::BAD_REQUEST, "j_per_op must be a positive number".into()));
    }
    report["received_unix_ms"] = serde_json::json!(
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis() as u64)
            .unwrap_or(0)
    );
    let line = serde_json::to_string(&report)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    {
        let _g = state.reports_lock.lock().unwrap();
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(state.reports_path.as_ref())
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("store: {e}")))?;
        writeln!(f, "{line}")
            .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("store: {e}")))?;
    }
    info!(
        "calibration report: {} {} prim={} {:.3e} J/op",
        report["board_id"].as_str().unwrap_or("?"),
        report["substrate"].as_str().unwrap_or("?"),
        report["primitive_id"],
        report["j_per_op"].as_f64().unwrap_or(0.0)
    );
    Ok(Json(serde_json::json!({ "stored": true })))
}

/// GET /api/calibration/reports?board_id=&substrate=&limit= — the
/// community J/op rows, newest last. The fabric hub's board cards and
/// Studio's cost-table browser read this.
async fn get_calibration_reports(
    State(state): State<AppState>,
    axum::extract::Query(q): axum::extract::Query<HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let limit: usize = q.get("limit").and_then(|s| s.parse().ok()).unwrap_or(200);
    let mut out = Vec::new();
    if let Ok(content) = std::fs::read_to_string(state.reports_path.as_ref()) {
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            if let Some(b) = q.get("board_id") {
                if v.get("board_id").and_then(|x| x.as_str()) != Some(b.as_str()) {
                    continue;
                }
            }
            if let Some(sub) = q.get("substrate") {
                if v.get("substrate").and_then(|x| x.as_str()) != Some(sub.as_str()) {
                    continue;
                }
            }
            out.push(v);
        }
    }
    let n = out.len();
    if n > limit {
        out.drain(..n - limit);
    }
    Json(serde_json::json!({ "count": out.len(), "reports": out }))
}

/// GET /api/hub — the calibration-hub aggregate that backs Studio's HUB
/// workspace ("what does THIS workload cost on THIS board"). Joins every
/// registry board (ofpga-chipdb) with the community calibration reports →
/// per-(board, primitive) best measured J/op, a lowest-joule leaderboard per
/// primitive, and coverage stats. Honest by construction: a cell exists only
/// where a device actually measured; the rest is awaiting bench calibration.
async fn get_hub(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::{HashMap, HashSet};
    use tei_forge::ofpga_chipdb::boards as cb;

    // A board is flashable if any of its aliases is a forge target.
    let forge_names: HashSet<&'static str> = tei_forge::TARGETS
        .iter()
        .filter_map(|t| tei_forge::board_info(t.id).map(|b| b.name))
        .collect();

    // Aggregate the best (lowest) measured J/op per (canonical board, primitive).
    // Device-reported board_id may be an alias → resolve to the canonical name.
    let mut best: HashMap<(String, u32), (f64, String, String, u64)> = HashMap::new();
    let mut reports_by_board: HashMap<String, usize> = HashMap::new();
    let mut reports_total = 0usize;
    if let Ok(content) = std::fs::read_to_string(state.reports_path.as_ref()) {
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(bid) = v.get("board_id").and_then(|x| x.as_str()) else {
                continue;
            };
            let jop = v.get("j_per_op").and_then(|x| x.as_f64()).unwrap_or(0.0);
            if !(jop.is_finite() && jop > 0.0) {
                continue;
            }
            let canon = cb::find_board(bid)
                .map(|b| b.name.to_string())
                .unwrap_or_else(|| bid.to_string());
            let prim = v.get("primitive_id").and_then(|x| x.as_u64()).unwrap_or(0) as u32;
            let src = v
                .get("ledger")
                .and_then(|l| l.get("joules_source"))
                .and_then(|x| x.as_str())
                .unwrap_or("table")
                .to_string();
            let sub = v
                .get("substrate")
                .and_then(|x| x.as_str())
                .unwrap_or("?")
                .to_string();
            let n = v.get("n_ops").and_then(|x| x.as_u64()).unwrap_or(0);
            reports_total += 1;
            *reports_by_board.entry(canon.clone()).or_insert(0) += 1;
            let e = best
                .entry((canon, prim))
                .or_insert((f64::INFINITY, String::new(), String::new(), 0));
            if jop < e.0 {
                *e = (jop, src, sub, n);
            }
        }
    }

    // Board cards — every board in the single registry.
    let boards: Vec<_> = cb::BOARDS
        .iter()
        .map(|b| {
            let es = cb::energy_source(b);
            serde_json::json!({
                "id": b.aliases.first().copied().unwrap_or(b.name),
                "name": b.name,
                "vendor": b.vendor,
                "chip": b.fpga_device,
                "category": format!("{:?}", b.category),
                "energy_source": es.tag(),
                "energy_how": es.label(),
                "forge": forge_names.contains(b.name),
                "report_count": reports_by_board.get(b.name).copied().unwrap_or(0),
            })
        })
        .collect();

    // Cells + per-primitive leaderboard (lowest-joule board wins).
    let mut cells = Vec::new();
    let mut prim_best: HashMap<u32, (String, f64)> = HashMap::new();
    for ((board, prim), (jop, src, sub, n)) in &best {
        cells.push(serde_json::json!({
            "board": board, "primitive_id": prim, "j_per_op": jop,
            "source": src, "substrate": sub, "n_ops": n,
        }));
        let e = prim_best.entry(*prim).or_insert((board.clone(), f64::INFINITY));
        if *jop < e.1 {
            *e = (board.clone(), *jop);
        }
    }
    let mut prim_ids: Vec<u32> = prim_best.keys().copied().collect();
    prim_ids.sort_unstable();
    let primitives: Vec<_> = prim_ids
        .iter()
        .map(|id| {
            let (board, jop) = prim_best.get(id).unwrap();
            let (name, family) = state
                .stack
                .get(*id)
                .map(|p| (p.name.clone(), p.family.clone()))
                .unwrap_or_else(|| (format!("primitive {id}"), String::new()));
            serde_json::json!({
                "id": id, "name": name, "family": family,
                "best_board": board, "best_j_per_op": jop,
            })
        })
        .collect();

    Json(serde_json::json!({
        "coverage": {
            "boards_total": cb::BOARDS.len(),
            "boards_measured": reports_by_board.len(),
            "reports_total": reports_total,
            "primitives_measured": prim_best.len(),
            "forge_targets": forge_names.len(),
        },
        "boards": boards,
        "primitives": primitives,
        "cells": cells,
    }))
}

/// GET /api/fleet — the deployment roster that backs Studio's FLEET
/// workspace: one entry per device (keyed by its reported `board_id`) that
/// has published a ledger home, with canonical identity, last-seen,
/// substrates exercised, report count, and latest J/op. Honest scope: a
/// `board_id` is a board *type* today — per-unit identity + OTA push +
/// remote provisioning arrive with the teiOS update agent (the runtime that
/// doesn't exist yet). This is the roster of who has reported, built over
/// the data that exists.
async fn get_fleet(State(state): State<AppState>) -> Json<serde_json::Value> {
    use std::collections::{BTreeSet, HashMap};
    use tei_forge::ofpga_chipdb::boards as cb;

    let forge_names: std::collections::HashSet<&'static str> = tei_forge::TARGETS
        .iter()
        .filter_map(|t| tei_forge::board_info(t.id).map(|b| b.name))
        .collect();

    struct Dev {
        reports: usize,
        substrates: BTreeSet<String>,
        primitives: BTreeSet<u32>,
        last_seen: u64,
        latest_jop: f64,
        latest_src: String,
    }
    let mut devs: HashMap<String, Dev> = HashMap::new();
    if let Ok(content) = std::fs::read_to_string(state.reports_path.as_ref()) {
        for line in content.lines() {
            let Ok(v) = serde_json::from_str::<serde_json::Value>(line) else {
                continue;
            };
            let Some(bid) = v.get("board_id").and_then(|x| x.as_str()) else {
                continue;
            };
            let ts = v.get("received_unix_ms").and_then(|x| x.as_u64()).unwrap_or(0);
            let d = devs.entry(bid.to_string()).or_insert_with(|| Dev {
                reports: 0,
                substrates: BTreeSet::new(),
                primitives: BTreeSet::new(),
                last_seen: 0,
                latest_jop: 0.0,
                latest_src: String::new(),
            });
            d.reports += 1;
            if let Some(s) = v.get("substrate").and_then(|x| x.as_str()) {
                d.substrates.insert(s.to_string());
            }
            if let Some(p) = v.get("primitive_id").and_then(|x| x.as_u64()) {
                d.primitives.insert(p as u32);
            }
            if ts >= d.last_seen {
                d.last_seen = ts;
                d.latest_jop = v.get("j_per_op").and_then(|x| x.as_f64()).unwrap_or(0.0);
                d.latest_src = v
                    .get("ledger")
                    .and_then(|l| l.get("joules_source"))
                    .and_then(|x| x.as_str())
                    .unwrap_or("table")
                    .to_string();
            }
        }
    }

    let mut roster: Vec<_> = devs
        .into_iter()
        .map(|(bid, d)| {
            let b = cb::find_board(&bid);
            serde_json::json!({
                "board_id": bid,
                "name": b.map(|b| b.name),
                "chip": b.map(|b| b.fpga_device),
                "energy_source": b.map(|b| cb::energy_source(b).tag()),
                "forge": b.map(|b| forge_names.contains(b.name)).unwrap_or(false),
                "reports": d.reports,
                "substrates": d.substrates.into_iter().collect::<Vec<_>>(),
                "primitives": d.primitives.into_iter().collect::<Vec<_>>(),
                "last_seen_ms": d.last_seen,
                "latest_j_per_op": d.latest_jop,
                "latest_source": d.latest_src,
            })
        })
        .collect();
    // Newest activity first.
    roster.sort_by(|a, b| {
        b["last_seen_ms"]
            .as_u64()
            .unwrap_or(0)
            .cmp(&a["last_seen_ms"].as_u64().unwrap_or(0))
    });

    Json(serde_json::json!({
        "devices": roster.len(),
        "roster": roster,
    }))
}

async fn post_dispatch(
    State(state): State<AppState>,
    Json(req): Json<DispatchRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let (substrates, calibrated) = state.dispatch_substrates(req.substrate_params.as_ref());
    let plan = dispatch(&state.stack, &req.workload, &substrates);
    let mut v = serde_json::to_value(&plan)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?;
    v["calibrated_defaults"] = serde_json::Value::Bool(calibrated);
    Ok(Json(v))
}

async fn post_dispatch_stream(
    State(state): State<AppState>,
    Json(req): Json<DispatchRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let workload = req.workload;
    let custom_params = req.substrate_params;
    // Per-invocation event pacing. 30ms × ~50 invocations = ~1.5s of visible
    // dispatch — enough for the UI to render incrementally. Override via env
    // for stress tests.
    let pace_ms: u64 = std::env::var("DISPATCH_STREAM_PACE_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let stack = state.stack.clone();
    let (substrates, calibrated_defaults) = state.dispatch_substrates(custom_params.as_ref());

    let stream = async_stream::stream! {
        let started_payload = serde_json::json!({
            "goal": workload.goal,
            "total_invocations": workload.invocations.len(),
            "constraints": workload.constraints,
            "calibrated_defaults": calibrated_defaults,
        });
        yield Ok(Event::default().event("started").data(started_payload.to_string()));

        let mut assignments = Vec::with_capacity(workload.invocations.len());
        for (idx, inv) in workload.invocations.iter().enumerate() {
            if let Some(a) = dispatch_invocation(&stack, inv, &substrates) {
                let payload = serde_json::json!({
                    "index": idx,
                    "total": workload.invocations.len(),
                    "assignment": &a,
                });
                yield Ok(Event::default()
                    .event("invocation")
                    .data(payload.to_string()));
                assignments.push(a);
            } else {
                let payload = serde_json::json!({
                    "index": idx,
                    "total": workload.invocations.len(),
                    "skipped_primitive_id": inv.primitive_id,
                });
                yield Ok(Event::default()
                    .event("skipped")
                    .data(payload.to_string()));
            }
            if pace_ms > 0 {
                tokio::time::sleep(Duration::from_millis(pace_ms)).await;
            }
        }

        let summary = summarize(&workload.goal, &workload.constraints, &substrates, &assignments);
        yield Ok(Event::default()
            .event("complete")
            .data(serde_json::to_string(&summary).unwrap_or_default()));
    };

    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Execute request — substrate-tagged simulator job, one variant per
/// native simulator column.
#[derive(Deserialize)]
#[serde(tag = "substrate", rename_all = "snake_case")]
enum ExecuteRequest {
    Stochastic(tei_sim_stochastic::StochasticJob),
    Spiking(tei_sim_spiking::SpikingJob),
    Crossbar(tei_sim_crossbar::CrossbarJob),
    Photonic(tei_sim_photonic::PhotonicJob),
    Gaussian(tei_sim_gaussian::GaussianJob),
    Circuit(tei_sim_circuit::CircuitJob),
    Field(tei_sim_field::FieldJob),
    Adiabatic(tei_sim_adiabatic::AdiabaticJob),
    Field3(tei_sim_field::Field3Job),
    Mnist(tei_sim_crossbar::mnist::MnistJob),
}

/// Run one executor on the blocking thread, forwarding progress ticks and
/// the final result over the SSE channel. `calibrate` prices the measured
/// ledger with the dialect's own constants (None for columns with no
/// dialect counterpart).
fn run_streaming<E, F>(
    exec: E,
    job: &E::Job,
    tx: &tokio::sync::mpsc::UnboundedSender<Event>,
    calibrate: F,
) where
    E: tei_sim_core::exec::Executor,
    F: FnOnce(&tei_sim_core::exec::ExecutionResult) -> Option<serde_json::Value>,
{
    let tx_progress = tx.clone();
    let mut on_progress = move |p: tei_sim_core::exec::Progress| {
        let payload = serde_json::json!({ "fraction": p.fraction, "metrics": p.metrics });
        let _ = tx_progress.send(Event::default().event("progress").data(payload.to_string()));
    };
    let result = exec.execute(job, &mut on_progress);
    let mut payload = serde_json::to_value(&result).unwrap_or_default();
    if let Some(cal) = calibrate(&result) {
        payload["calibration"] = cal;
    }
    let _ = tx.send(Event::default().event("result").data(payload.to_string()));
}

/// POST /api/execute — run a workload on a native simulator, streaming
/// progress over SSE: `started` → `progress`×N → `result`.
/// The simulator runs on a blocking thread; progress crosses to the SSE
/// stream over an unbounded channel.
async fn post_execute(
    State(state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();
    let stack = state.stack.clone();
    let subs = state.substrates.clone();

    tokio::task::spawn_blocking(move || {
        let _ = tx.send(Event::default().event("started").data("{}"));
        match req {
            ExecuteRequest::Stochastic(job) => {
                use tei_sim_stochastic::maxcut::ProblemSpec;
                let variables = match &job.problem {
                    ProblemSpec::Complete { n }
                    | ProblemSpec::Cycle { n }
                    | ProblemSpec::RandomRegular { n, .. } => *n,
                    ProblemSpec::CompleteBipartite { a, b } => a + b,
                    ProblemSpec::Petersen => 10,
                };
                let sweeps = job.schedule.sweeps;
                run_streaming(tei_sim_stochastic::StochasticExecutor, &job, &tx, |r| {
                    calib::stochastic(&stack, &subs, sweeps, variables, &r.ledger)
                })
            }
            ExecuteRequest::Spiking(job) => {
                let neurons: u64 = job.layers.iter().map(|l| l.n as u64).sum();
                let timesteps = (job.duration / job.dt).round() as u64;
                run_streaming(tei_sim_spiking::SpikingExecutor, &job, &tx, |r| {
                    let n_synapses = r.outputs.get("n_synapses").and_then(|v| v.as_u64())?;
                    calib::spiking(&stack, &subs, neurons, timesteps, n_synapses, &r.ledger)
                })
            }
            ExecuteRequest::Crossbar(job) => {
                let (rows, cols, n_queries) = (job.rows, job.cols, job.n_queries);
                run_streaming(tei_sim_crossbar::CrossbarExecutor, &job, &tx, |r| {
                    calib::crossbar(&stack, &subs, rows, cols, n_queries, &r.ledger)
                })
            }
            ExecuteRequest::Photonic(job) => {
                let (n, n_queries) = (job.n, job.n_queries);
                run_streaming(tei_sim_photonic::PhotonicExecutor, &job, &tx, |r| {
                    calib::photonic(&stack, &subs, n, n_queries, &r.ledger)
                })
            }
            ExecuteRequest::Gaussian(job) => {
                run_streaming(tei_sim_gaussian::GaussianExecutor, &job, &tx, |_| None)
            }
            ExecuteRequest::Circuit(job) => {
                run_streaming(tei_sim_circuit::CircuitExecutor, &job, &tx, |_| None)
            }
            ExecuteRequest::Field(job) => {
                run_streaming(tei_sim_field::FieldExecutor, &job, &tx, |_| None)
            }
            ExecuteRequest::Field3(job) => {
                run_streaming(tei_sim_field::Field3Executor, &job, &tx, |_| None)
            }
            ExecuteRequest::Adiabatic(job) => {
                run_streaming(tei_sim_adiabatic::AdiabaticExecutor, &job, &tx, |r| {
                    calib::adiabatic(&r.outputs)
                })
            }
            ExecuteRequest::Mnist(job) => {
                run_streaming(tei_sim_crossbar::mnist::MnistExecutor, &job, &tx, |r| {
                    calib::mnist_accuracy(&r.outputs)
                })
            }
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(event);
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
}

/// Pack a FieldJob into the exact f32 buffers the F4 WGSL kernels expect.
/// The browser WebGPU driver uploads these verbatim — the packing math
/// (CPML tables, ce = dt/ε, per-step source amplitudes) stays in Rust.
async fn post_field_gpu_pack(
    Json(job): Json<tei_sim_field::FieldJob>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    tei_sim_field::gpu::pack_job_json(&job)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, e))
}

/// The three WGSL kernel sources, straight from the crate (single source
/// of truth — the native validation suite proves these exact strings).
async fn get_field_gpu_shaders() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "update_h": tei_sim_field::gpu::UPDATE_H_WGSL,
        "update_e": tei_sim_field::gpu::UPDATE_E_WGSL,
        "inject": tei_sim_field::gpu::INJECT_WGSL,
    }))
}

async fn post_import_onnx(
    State(_state): State<AppState>,
    body: Bytes,
) -> Result<Json<tei_import::ImportReport>, (StatusCode, String)> {
    tei_import::parse_onnx(&body)
        .map(Json)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("ONNX parse failed: {e}")))
}

/// Streaming chunk response. The server returns one of these per `POST
/// /api/import/onnx/chunk` call: either `accepting` while more chunks are
/// expected or `done` with the final ImportReport.
#[derive(Serialize)]
#[serde(tag = "status", rename_all = "lowercase")]
enum ChunkResponse {
    Accepting { received: u32, total: u32 },
    Done { report: tei_import::ImportReport },
}

/// Chunked-upload ONNX import.
///
/// Clients split a large `.onnx` file into chunks (8 MB is a reasonable
/// default) and POST each one to this endpoint. Headers carry the upload
/// identity:
///   X-Upload-Id     unique string per upload (uuid is fine)
///   X-Chunk-Index   zero-based index
///   X-Chunk-Total   total expected chunks
///
/// Body is the raw chunk bytes (`Content-Type: application/octet-stream`).
///
/// While chunks are still missing the response is `{ status: "accepting",
/// received, total }`. When the last chunk arrives, the buffer is
/// concatenated and parsed; the response is `{ status: "done", report }`.
///
/// Bounded by MAX_UPLOAD_BYTES and MAX_CONCURRENT_UPLOADS; idle uploads are
/// dropped after UPLOAD_TTL.
async fn post_import_onnx_chunk(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: Bytes,
) -> Result<Json<ChunkResponse>, (StatusCode, String)> {
    let hdr_str = |name: &str| -> Option<String> {
        headers
            .get(name)
            .and_then(|h| h.to_str().ok())
            .map(|s| s.to_string())
    };
    let hdr_u32 = |name: &str| hdr_str(name).and_then(|s| s.parse::<u32>().ok());

    let upload_id = hdr_str("x-upload-id")
        .ok_or((StatusCode::BAD_REQUEST, "missing X-Upload-Id header".into()))?;
    let chunk_index = hdr_u32("x-chunk-index").ok_or((
        StatusCode::BAD_REQUEST,
        "missing/invalid X-Chunk-Index header".into(),
    ))?;
    let chunk_total = hdr_u32("x-chunk-total").ok_or((
        StatusCode::BAD_REQUEST,
        "missing/invalid X-Chunk-Total header".into(),
    ))?;
    if chunk_total == 0 || chunk_index >= chunk_total {
        return Err((
            StatusCode::BAD_REQUEST,
            format!("bad chunk indexing: {chunk_index}/{chunk_total}"),
        ));
    }

    let body_len = body.len();
    let full_bytes = {
        let mut map = state.uploads.map.lock().unwrap();
        // Sweep expired uploads.
        let now = Instant::now();
        map.retain(|_, b| now.duration_since(b.started) < UPLOAD_TTL);

        if map.len() >= MAX_CONCURRENT_UPLOADS && !map.contains_key(&upload_id) {
            return Err((
                StatusCode::TOO_MANY_REQUESTS,
                format!(
                    "too many concurrent uploads ({MAX_CONCURRENT_UPLOADS} max); retry shortly"
                ),
            ));
        }

        let buf = map
            .entry(upload_id.clone())
            .or_insert_with(|| UploadBuffer {
                chunks: BTreeMap::new(),
                total: chunk_total,
                received_bytes: 0,
                started: now,
            });

        if buf.total != chunk_total {
            let prev_total = buf.total;
            map.remove(&upload_id);
            return Err((
                StatusCode::BAD_REQUEST,
                format!(
                    "chunk_total disagreement: previous {prev_total} vs incoming {chunk_total}"
                ),
            ));
        }

        // Accept the chunk (idempotent on re-send).
        buf.received_bytes = buf.received_bytes.saturating_add(body_len);
        if buf.received_bytes > MAX_UPLOAD_BYTES {
            map.remove(&upload_id);
            return Err((
                StatusCode::PAYLOAD_TOO_LARGE,
                format!(
                    "upload exceeded server cap of {} MB",
                    MAX_UPLOAD_BYTES / (1024 * 1024)
                ),
            ));
        }
        buf.chunks.insert(chunk_index, body);

        // Final chunk? Snapshot bytes + drop buffer; otherwise yield progress.
        if (buf.chunks.len() as u32) == buf.total {
            let mut full = Vec::with_capacity(buf.received_bytes);
            for c in buf.chunks.values() {
                full.extend_from_slice(c);
            }
            map.remove(&upload_id);
            Some(full)
        } else {
            return Ok(Json(ChunkResponse::Accepting {
                received: buf.chunks.len() as u32,
                total: chunk_total,
            }));
        }
    };

    // We have the full file. Parse outside the lock.
    let bytes = full_bytes.unwrap();
    let report = tei_import::parse_onnx(&bytes)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("ONNX parse failed: {e}")))?;
    Ok(Json(ChunkResponse::Done { report }))
}

async fn not_found() -> impl IntoResponse {
    (StatusCode::NOT_FOUND, "not found")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("tei_serve=info".parse().unwrap())
                .add_directive("tower_http=info".parse().unwrap()),
        )
        .init();

    let stack_path =
        std::env::var("STACK_JSON_PATH").unwrap_or_else(|_| "data/stack.json".to_string());
    let stack = Stack::load_from_path(&stack_path)?;
    info!(
        "loaded periodic stack from {} ({} primitives, {} families)",
        stack_path,
        stack.count(),
        stack.data.families.len()
    );

    let substrates = Arc::new(default_substrates(stack.clone()));
    info!(
        "registered substrates: {}",
        substrates
            .iter()
            .map(|s| s.name())
            .collect::<Vec<_>>()
            .join(", ")
    );

    let calibration_path = std::path::PathBuf::from(
        std::env::var("CALIBRATION_PATH").unwrap_or_else(|_| "calibration.json".to_string()),
    );
    let calibration: Option<SubstrateParams> = std::fs::read_to_string(&calibration_path)
        .ok()
        .and_then(|s| serde_json::from_str(&s).ok());
    if calibration.is_some() {
        info!("loaded calibration from {}", calibration_path.display());
    }

    let reports_path = std::path::PathBuf::from(
        std::env::var("CALIBRATION_REPORTS_PATH")
            .unwrap_or_else(|_| "calibration-reports.jsonl".to_string()),
    );

    // Forge build host: enabled only where a cargo toolchain AND the
    // skeleton tree are present. FORGE_WORKSPACE_ROOT overrides discovery.
    let forge = {
        let root = std::env::var("FORGE_WORKSPACE_ROOT")
            .ok()
            .map(std::path::PathBuf::from)
            .or_else(|| {
                std::env::current_dir()
                    .ok()
                    .and_then(|d| tei_forge::workspace_root(&d))
            });
        match root {
            Some(r) if tei_forge::toolchain_available()
                && r.join("embedded/teios-app-rp2040/Cargo.toml").is_file() =>
            {
                info!("forge build host: {}", r.display());
                Some(tei_forge::BuildOpts::new(r))
            }
            _ => {
                info!("forge: no build host (no toolchain/skeletons) — /api/forge reports unavailable");
                None
            }
        }
    };

    let state = AppState {
        stack,
        substrates,
        uploads: Arc::new(UploadState::default()),
        calibration: Arc::new(Mutex::new(calibration)),
        calibration_path: Arc::new(calibration_path),
        reports_path: Arc::new(reports_path),
        reports_lock: Arc::new(Mutex::new(())),
        forge: Arc::new(forge),
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/stack", get(get_stack))
        .route("/api/substrates", get(list_substrates))
        .route("/api/presets", get(list_presets))
        .route("/api/dispatch", post(post_dispatch))
        .route(
            "/api/calibration",
            get(get_calibration)
                .post(post_calibration)
                .delete(delete_calibration),
        )
        .route(
            "/api/calibration/reports",
            get(get_calibration_reports).post(post_calibration_report),
        )
        .route("/api/hub", get(get_hub))
        .route("/api/fleet", get(get_fleet))
        .route("/api/forge/build", post(post_forge_build))
        .route("/api/forge/targets", get(get_forge_targets))
        .route("/api/forge/board", get(get_forge_board))
        .route("/api/forge/board-list", get(get_forge_board_list))
        .route("/api/forge/artifact", get(get_forge_artifact))
        .route("/api/dispatch/stream", post(post_dispatch_stream))
        .route("/api/execute", post(post_execute))
        .route("/api/field-gpu-pack", post(post_field_gpu_pack))
        .route("/api/field-gpu-shaders", get(get_field_gpu_shaders))
        .route("/api/import/onnx", post(post_import_onnx))
        .route("/api/import/onnx/chunk", post(post_import_onnx_chunk))
        // ONNX models can be hundreds of MB — lift the default 2 MB body limit.
        .layer(DefaultBodyLimit::max(512 * 1024 * 1024))
        .fallback(not_found)
        .layer(cors)
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(9651);
    let addr = format!("0.0.0.0:{port}");
    info!("listening on http://{addr}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
