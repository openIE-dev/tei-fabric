//! F4 — wgpu compute-shader FDTD kernel (feature `gpu`).
//!
//! The **WGSL files in `src/gpu/shaders/` are the shared artifact**: this
//! module is the native host that proves them correct against the
//! validated f64 CPU core ([`crate::grid::Grid2d`]); a JS WebGPU driver
//! reuses the same `.wgsl` sources verbatim in the browser. Everything a
//! host must reproduce — bind-group indices, buffer packing, uniform
//! struct offsets, dispatch geometry — is documented in the canonical
//! table at the top of `shaders/update_h.wgsl` and mirrored here.
//!
//! ## Host contract (what a JS driver must replicate)
//!
//! One explicit pipeline layout shared by all four pipelines
//! (`update_h`, `update_e`, `inject`, `record_probe`), two bind groups:
//!
//! **group(0)** — static, bound once per pass:
//!
//! | binding | buffer  | WGSL access | f32 elems |
//! |---------|---------|-------------|-----------|
//! | 0 | `Params` uniform (48 B) | uniform | — |
//! | 1 | ez    | read_write | `nx*ny` |
//! | 2 | hx    | read_write | `nx*(ny-1)` |
//! | 3 | hy    | read_write | `(nx-1)*ny` |
//! | 4 | ce = dt/ε_r | read | `nx*ny` |
//! | 5 | psi_e (ψ_Ez_x ‖ ψ_Ez_y) | read_write | `2*nx*ny` |
//! | 6 | psi_h (ψ_Hx_y ‖ ψ_Hy_x) | read_write | `nx*(ny-1) + (nx-1)*ny` |
//! | 7 | coeff (packed CPML tables) | read | `3*(2nx-1) + 3*(2ny-1)` |
//! | 8 | trace | read_write | `steps` |
//!
//! `Params` (offsets in bytes): nx@0, ny@4, npml@8, src_kind@12, src_i@16,
//! src_j0@20, src_j1@24, probe_idx@28 (all u32), dt@32 (f32), 12 B zero pad
//! → 48 B total. `coeff` packing order (x tables then y tables, integer
//! "e" positions then half-integer "h" positions, computed in f64 on the
//! CPU with [`crate::grid::cpml_coeffs`] and cast to f32):
//! `inv_kx_e[nx] b_x_e[nx] c_x_e[nx] inv_kx_h[nx-1] b_x_h[nx-1]
//! c_x_h[nx-1] inv_ky_e[ny] b_y_e[ny] c_y_e[ny] inv_ky_h[ny-1]
//! b_y_h[ny-1] c_y_h[ny-1]`.
//!
//! **group(1)** — per-step `StepParams` uniform with `has_dynamic_offset =
//! true`, bound size 16 B, one 256-B-aligned slot per leapfrog step at
//! byte offset `256*step`: `{ amp: f32 @0, step: u32 @4, pad: 8 B }` with
//! `amp = amplitude·s((step+1)·dt)` evaluated on the CPU in f64 (Gaussian /
//! modulated-Gaussian / CW from [`crate::source::TimeProfile`]).
//!
//! **Dispatch geometry, per leapfrog step** (all steps in one compute
//! pass; bind group 1 rebound with dynamic offset `256*step`):
//!
//! 1. `update_h` — workgroup 16×16 → `(⌈nx/16⌉, ⌈ny/16⌉, 1)`
//! 2. `update_e` — workgroup 16×16 → `(⌈nx/16⌉, ⌈ny/16⌉, 1)`
//! 3. `inject`   — workgroup 64 → `(⌈(src_j1−src_j0+1)/64⌉, 1, 1)`, skipped when there is no source
//! 4. `record_probe` — workgroup 1 → `(1, 1, 1)`
//!
//! WGSL subtleties a JS host inherits for free but must not "fix":
//! updates are **in place** (no ping-pong — the H pass reads only Ez and
//! each thread owns its Hx/Hy/ψ cell; the E pass reads only Hx/Hy and
//! owns its Ez/ψ cells; WebGPU's implicit inter-dispatch ordering provides
//! the leapfrog sequencing). Field and ψ buffers rely on WebGPU's
//! guaranteed zero-initialization — the host uploads only `Params`, `ce`,
//! `coeff` and the `StepParams` slots. The CPML ψ recursion runs over the
//! *whole* grid: outside the PML strips the tables hold (1/κ, b, c) =
//! (1, 1, 0) so ψ stays 0 and the arithmetic matches the CPU's split
//! interior/strip passes. This host chunks steps into ≤256-step command
//! buffers purely to bound encoder size; chunking is not part of the
//! contract.
//!
//! Storage-buffer count is exactly 8 — the WebGPU default
//! `maxStorageBuffersPerShaderStage` — and the bound `StepParams` size is
//! 16 B with stride 256 (`minUniformBufferOffsetAlignment` default), so
//! the layout works against default WebGPU limits in every browser.
//!
//! Scope: point/line soft sources and the uniform/slab/waveguide ε layouts
//! of [`EpsSpec`](crate::EpsSpec). Mode sources and port monitors (F2) are
//! CPU-only; jobs carrying them are rejected with a clean error.

use crate::FieldJob;
use crate::grid::{cpml_coeffs, default_courant};
use crate::source::SourceShape;
#[cfg(feature = "gpu")]
use std::num::NonZeroU64;

#[cfg(feature = "gpu")]
mod sim;

#[cfg(feature = "gpu")]
pub use sim::{GpuSim, gpu_available};

/// Per-step uniform slot stride in bytes (WebGPU default
/// `minUniformBufferOffsetAlignment`).
pub const STEP_STRIDE: u64 = 256;

/// Bound size of the `StepParams` dynamic-offset uniform binding.
pub const STEP_PARAMS_SIZE: u64 = 16;

/// 2D update kernels use 16×16 workgroups.
pub const WORKGROUP_2D: u32 = 16;

/// The inject kernel uses 64-wide 1D workgroups.
pub const WORKGROUP_INJECT: u32 = 64;

/// WGSL source for the Hx/Hy update kernel (entry point `update_h`).
pub const UPDATE_H_WGSL: &str = include_str!("shaders/update_h.wgsl");
/// WGSL source for the Ez update kernel (entry point `update_e`).
pub const UPDATE_E_WGSL: &str = include_str!("shaders/update_e.wgsl");
/// WGSL source for source injection + probe recording (entry points
/// `inject` and `record_probe`).
pub const INJECT_WGSL: &str = include_str!("shaders/inject.wgsl");

/// `Params` uniform — layout documented in the module docs and in each
/// shader header. 48 bytes, `#[repr(C)]`, no implicit padding.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct Params {
    nx: u32,
    ny: u32,
    npml: u32,
    src_kind: u32,
    src_i: u32,
    src_j0: u32,
    src_j1: u32,
    probe_idx: u32,
    dt: f32,
    pad: [u32; 3],
}

/// `StepParams` slot — 16 bytes used of each 256-byte slot.
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct StepParams {
    amp: f32,
    step: u32,
    pad: [u32; 2],
}

/// Everything derived from a [`FieldJob`] on the CPU side, in the exact
/// packing the shaders expect (all f32).
struct JobData {
    params: Params,
    ce: Vec<f32>,
    coeff: Vec<f32>,
    /// amp slot per step: amplitude·s((n+1)·dt), f64-evaluated.
    amps: Vec<f32>,
    dt: f64,
    has_source: bool,
}

fn build_job_data(job: &FieldJob) -> Result<JobData, String> {
    if job.mode_source.is_some() {
        return Err("gpu: mode_source is CPU-only at F4".into());
    }
    if !job.ports.is_empty() {
        return Err("gpu: port monitors / S-parameters are CPU-only at F4".into());
    }
    let (nx, ny, npml) = (job.nx, job.ny, job.npml);
    if !(nx > 2 * npml + 2 && ny > 2 * npml + 2) {
        return Err("gpu: grid must leave interior cells beyond the CPML".into());
    }
    let courant = if job.courant > 0.0 {
        job.courant
    } else {
        default_courant()
    };
    let dt = courant / 2f64.sqrt();

    // ce = dt/eps_r, same precompute as Grid2d::new, cast to f32.
    let eps = job.eps.build(nx, ny);
    if eps.iter().any(|&e| e <= 0.0) {
        return Err("gpu: eps_r must be positive".into());
    }
    let ce: Vec<f32> = eps.iter().map(|&e| (dt / e) as f32).collect();

    // Packed CPML tables, same f64 evaluation as Grid2d::new.
    let p = &job.cpml;
    let mut coeff = Vec::with_capacity(3 * (2 * nx - 1) + 3 * (2 * ny - 1));
    let mut push_axis = |n: usize| {
        for half in [false, true] {
            let count = if half { n - 1 } else { n };
            for sel in 0..3 {
                for k in 0..count {
                    let pos = k as f64 + if half { 0.5 } else { 0.0 };
                    let (ik, b, c) = cpml_coeffs(pos, n, npml, dt, p);
                    coeff.push([ik, b, c][sel] as f32);
                }
            }
        }
    };
    push_axis(nx);
    push_axis(ny);

    // Source footprint + per-step amplitudes (f64 on the CPU).
    let (src_kind, src_i, src_j0, src_j1) = match &job.source {
        None => (0u32, 0u32, 0u32, 0u32),
        Some(s) => match s.shape {
            SourceShape::Point { i, j } => (1, i as u32, j as u32, j as u32),
            SourceShape::Line { i, j0, j1 } => (
                1,
                i as u32,
                j0.unwrap_or(1) as u32,
                j1.unwrap_or(ny - 2) as u32,
            ),
        },
    };
    let amps: Vec<f32> = (0..job.steps)
        .map(|n| match &job.source {
            Some(s) => (s.amplitude * s.time.eval((n + 1) as f64 * dt)) as f32,
            None => 0.0,
        })
        .collect();

    let (pi, pj) = job.probe.map(|p| (p[0], p[1])).unwrap_or((nx / 2, ny / 2));
    if pi >= nx || pj >= ny {
        return Err("gpu: probe out of bounds".into());
    }

    Ok(JobData {
        params: Params {
            nx: nx as u32,
            ny: ny as u32,
            npml: npml as u32,
            src_kind,
            src_i,
            src_j0,
            src_j1,
            probe_idx: (pi * ny + pj) as u32,
            dt: dt as f32,
            pad: [0; 3],
        },
        ce,
        coeff,
        amps,
        dt,
        has_source: src_kind != 0,
    })
}

/// JSON packing of [`build_job_data`] for external (browser WebGPU) hosts.
/// Same packing the native host uploads, so the WGSL contract holds.
pub fn pack_job_json(job: &FieldJob) -> Result<serde_json::Value, String> {
    let d = build_job_data(job)?;
    Ok(serde_json::json!({
        "params": {
            "nx": d.params.nx, "ny": d.params.ny, "npml": d.params.npml,
            "src_kind": d.params.src_kind, "src_i": d.params.src_i,
            "src_j0": d.params.src_j0, "src_j1": d.params.src_j1,
            "probe_idx": d.params.probe_idx, "dt": d.params.dt,
        },
        "ce": d.ce,
        "coeff": d.coeff,
        "amps": d.amps,
        "dt": d.dt,
        "has_source": d.has_source,
        "steps": d.amps.len(),
    }))
}

/// Bound size for the dynamic `StepParams` binding.
#[cfg(feature = "gpu")]
fn step_params_binding_size() -> NonZeroU64 {
    NonZeroU64::new(STEP_PARAMS_SIZE).unwrap()
}

/// Run a job on the GPU and return the probe trace — the GPU analogue of
/// what the CPU path records at the probe cell (one f32 sample per step,
/// recorded after source injection, probe defaulting to the grid center).
#[cfg(feature = "gpu")]
pub fn run_probe(job: &FieldJob) -> Result<Vec<f32>, String> {
    let mut sim = GpuSim::new(job)?;
    sim.step(job.steps)?;
    sim.read_trace()
}
