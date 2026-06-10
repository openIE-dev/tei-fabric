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
    DispatchPlan, Preset, SubstrateParams, default_substrates, dispatch, dispatch_invocation,
    enumerate_presets, substrates_with_params, summarize,
};
use tei_ir::Workload;
use tei_stack::{Stack, StackData};
use tei_substrate_traits::Substrate;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

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

async fn post_dispatch(
    State(state): State<AppState>,
    Json(req): Json<DispatchRequest>,
) -> Result<Json<DispatchPlan>, (StatusCode, String)> {
    let substrates_owned;
    let substrates_ref: &[Arc<dyn Substrate>] = if let Some(p) = &req.substrate_params {
        substrates_owned = substrates_with_params(state.stack.clone(), p);
        &substrates_owned
    } else {
        &state.substrates
    };
    let plan = dispatch(&state.stack, &req.workload, substrates_ref);
    Ok(Json(plan))
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
    let substrates: Arc<Vec<Arc<dyn Substrate>>> = if let Some(p) = custom_params.as_ref() {
        Arc::new(substrates_with_params(stack.clone(), p))
    } else {
        state.substrates.clone()
    };

    let stream = async_stream::stream! {
        let started_payload = serde_json::json!({
            "goal": workload.goal,
            "total_invocations": workload.invocations.len(),
            "constraints": workload.constraints,
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
}

/// Run one executor on the blocking thread, forwarding progress ticks and
/// the final result over the SSE channel.
fn run_streaming<E: tei_sim_core::exec::Executor>(
    exec: E,
    job: &E::Job,
    tx: &tokio::sync::mpsc::UnboundedSender<Event>,
) {
    let tx_progress = tx.clone();
    let mut on_progress = move |p: tei_sim_core::exec::Progress| {
        let payload = serde_json::json!({ "fraction": p.fraction, "metrics": p.metrics });
        let _ = tx_progress.send(Event::default().event("progress").data(payload.to_string()));
    };
    let result = exec.execute(job, &mut on_progress);
    let _ = tx.send(
        Event::default()
            .event("result")
            .data(serde_json::to_string(&result).unwrap_or_default()),
    );
}

/// POST /api/execute — run a workload on a native simulator, streaming
/// progress over SSE: `started` → `progress`×N → `result`.
/// The simulator runs on a blocking thread; progress crosses to the SSE
/// stream over an unbounded channel.
async fn post_execute(
    State(_state): State<AppState>,
    Json(req): Json<ExecuteRequest>,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<Event>();

    tokio::task::spawn_blocking(move || {
        let _ = tx.send(Event::default().event("started").data("{}"));
        match req {
            ExecuteRequest::Stochastic(job) => {
                run_streaming(tei_sim_stochastic::StochasticExecutor, &job, &tx)
            }
            ExecuteRequest::Spiking(job) => {
                run_streaming(tei_sim_spiking::SpikingExecutor, &job, &tx)
            }
            ExecuteRequest::Crossbar(job) => {
                run_streaming(tei_sim_crossbar::CrossbarExecutor, &job, &tx)
            }
            ExecuteRequest::Photonic(job) => {
                run_streaming(tei_sim_photonic::PhotonicExecutor, &job, &tx)
            }
            ExecuteRequest::Gaussian(job) => {
                run_streaming(tei_sim_gaussian::GaussianExecutor, &job, &tx)
            }
            ExecuteRequest::Circuit(job) => {
                run_streaming(tei_sim_circuit::CircuitExecutor, &job, &tx)
            }
            ExecuteRequest::Field(job) => run_streaming(tei_sim_field::FieldExecutor, &job, &tx),
        }
    });

    let stream = async_stream::stream! {
        while let Some(event) = rx.recv().await {
            yield Ok(event);
        }
    };
    Sse::new(stream).keep_alive(KeepAlive::default())
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

    let state = AppState {
        stack,
        substrates,
        uploads: Arc::new(UploadState::default()),
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
        .route("/api/dispatch/stream", post(post_dispatch_stream))
        .route("/api/execute", post(post_execute))
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
