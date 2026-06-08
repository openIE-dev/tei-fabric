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
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Serialize;
use std::sync::Arc;
use tei_cost_surface::{DispatchPlan, default_substrates, dispatch};
use tei_ir::Workload;
use tei_stack::{Stack, StackData};
use tei_substrate_traits::Substrate;
use tower_http::cors::{Any, CorsLayer};
use tracing::info;

#[derive(Clone)]
struct AppState {
    stack: Arc<Stack>,
    substrates: Arc<Vec<Arc<dyn Substrate>>>,
}

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

async fn post_dispatch(
    State(state): State<AppState>,
    Json(workload): Json<Workload>,
) -> Result<Json<DispatchPlan>, (StatusCode, String)> {
    let plan = dispatch(&state.stack, &workload, &state.substrates);
    Ok(Json(plan))
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

    let state = AppState { stack, substrates };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/api/stack", get(get_stack))
        .route("/api/substrates", get(list_substrates))
        .route("/api/dispatch", post(post_dispatch))
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
