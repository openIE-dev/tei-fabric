// update_h.wgsl — F4 GPU FDTD: Hx/Hy half-step update from Ez, with the
// CPML ψ recursion folded in (Roden & Gedney 2000), 2D TEz, f32.
//
// ## Shared bind-group contract (identical in all three F4 shaders)
//
// Every F4 kernel is compiled against ONE explicit pipeline layout so a
// host (native wgpu or JS WebGPU) builds exactly two bind groups and
// reuses them for every dispatch. All arrays are tightly packed f32,
// row-major with y (j) contiguous: `ez[i*ny + j]`.
//
// ### group(0) — static state (one bind group for the whole run)
//
// | binding | var    | type                      | size (f32 elems)        | contents |
// |---------|--------|---------------------------|-------------------------|----------|
// | 0       | P      | uniform `Params`          | 48 bytes                | see below |
// | 1       | ez     | storage, read_write       | nx*ny                   | Ez(i,j) |
// | 2       | hx     | storage, read_write       | nx*(ny-1)               | Hx(i,j+1/2) |
// | 3       | hy     | storage, read_write       | (nx-1)*ny               | Hy(i+1/2,j) |
// | 4       | ce     | storage, read             | nx*ny                   | dt/eps_r per Ez cell |
// | 5       | psi_e  | storage, read_write       | 2*nx*ny                 | [0,n): psi_ez_x, [n,2n): psi_ez_y (n = nx*ny) |
// | 6       | psi_h  | storage, read_write       | nx*(ny-1) + (nx-1)*ny   | [0,hxlen): psi_hx_y, then psi_hy_x |
// | 7       | coeff  | storage, read             | 3*(2nx-1) + 3*(2ny-1)   | packed CPML tables, see below |
// | 8       | trace  | storage, read_write       | steps                   | probe trace, trace[step] |
//
// ### Params uniform (48 bytes, std140-compatible: 8 x u32, then f32 + 3 pad)
//
// | offset | field     | type | meaning |
// |--------|-----------|------|---------|
// | 0      | nx        | u32  | Ez points along x |
// | 4      | ny        | u32  | Ez points along y |
// | 8      | npml      | u32  | CPML thickness (informational; baked into coeff) |
// | 12     | src_kind  | u32  | 0 = no source, 1 = soft Ez source on column src_i, rows src_j0..=src_j1 |
// | 16     | src_i     | u32  | source column |
// | 20     | src_j0    | u32  | first driven row (point source: j0 == j1) |
// | 24     | src_j1    | u32  | last driven row, inclusive |
// | 28     | probe_idx | u32  | flat probe index i*ny + j |
// | 32     | dt        | f32  | time step (courant/sqrt(2)) |
// | 36..48 | pad       | 3 u32| zero |
//
// ### coeff packed layout (all f32, evaluated on CPU in f64 then cast)
//
// x tables first (e = integer positions, h = half-integer), then y:
//   [0,        nx)        inv_kx_e   = 1/kappa_x at i
//   [nx,       2nx)       b_x_e
//   [2nx,      3nx)       c_x_e
//   [3nx,            3nx+(nx-1))    inv_kx_h at i+1/2
//   [3nx+(nx-1),     3nx+2(nx-1))   b_x_h
//   [3nx+2(nx-1),    3nx+3(nx-1))   c_x_h
//   ybase = 3nx + 3(nx-1), then the same six tables with nx -> ny:
//   [ybase, ybase+ny) inv_ky_e ... etc., ending with c_y_h of length ny-1.
//
// ### group(1) — per-step params (dynamic-offset uniform)
//
// | binding | var | type | notes |
// |---------|-----|------|-------|
// | 0       | S   | uniform `StepParams`, has_dynamic_offset = true | bound size 16 bytes; slot stride 256 bytes |
//
// StepParams slot n (at byte offset 256*n): { amp: f32 = amplitude*s((n+1)*dt)
// precomputed on CPU, step: u32 = n, pad: 2 x u32 }. One slot per leapfrog step.
//
// ### Dispatch contract (per leapfrog step, all in one compute pass)
//
//   1. update_h   — workgroup 16x16, dispatch (ceil(nx/16), ceil(ny/16), 1)
//   2. update_e   — workgroup 16x16, dispatch (ceil(nx/16), ceil(ny/16), 1)
//   3. inject     — workgroup 64,    dispatch (ceil((src_j1-src_j0+1)/64), 1, 1); skip if src_kind == 0
//   4. record_probe — workgroup 1,   dispatch (1, 1, 1)
//
// Set bind group 1 with dynamic offset 256*step before the dispatches of
// each step. WebGPU's implicit ordering between dispatches provides the
// leapfrog sequencing; updates are IN PLACE (no ping-pong): the H pass
// only reads ez and each thread owns its hx/hy/psi cell, the E pass only
// reads hx/hy and each thread owns its ez/psi cell.
//
// Fields start at zero: WebGPU zero-initializes buffers — the host writes
// only ce, coeff, Params and the StepParams slots.

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
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> ez: array<f32>;
@group(0) @binding(2) var<storage, read_write> hx: array<f32>;
@group(0) @binding(3) var<storage, read_write> hy: array<f32>;
@group(0) @binding(6) var<storage, read_write> psi_h: array<f32>;
@group(0) @binding(7) var<storage, read> coeff: array<f32>;

// Bindings not statically used by an entry point (4 ce, 5 psi_e, 8 trace,
// group(1)) may be left undeclared; the indices that ARE used must match
// the shared table above.

fn inv_kx_h(i: u32) -> f32 { return coeff[3u * P.nx + i]; }
fn b_x_h(i: u32) -> f32 { return coeff[3u * P.nx + (P.nx - 1u) + i]; }
fn c_x_h(i: u32) -> f32 { return coeff[3u * P.nx + 2u * (P.nx - 1u) + i]; }
fn ybase() -> u32 { return 3u * P.nx + 3u * (P.nx - 1u); }
fn inv_ky_h(j: u32) -> f32 { return coeff[ybase() + 3u * P.ny + j]; }
fn b_y_h(j: u32) -> f32 { return coeff[ybase() + 3u * P.ny + (P.ny - 1u) + j]; }
fn c_y_h(j: u32) -> f32 { return coeff[ybase() + 3u * P.ny + 2u * (P.ny - 1u) + j]; }

// Hx <- Hx - dt*((dEz/dy)/kappa_y + psi);  psi <- b*psi + c*dEz/dy
// Hy <- Hy + dt*((dEz/dx)/kappa_x + psi);  psi <- b*psi + c*dEz/dx
// Outside the CPML strips the tables hold (1/kappa, b, c) = (1, 1, 0) and
// psi stays 0, so the uniform code path matches the CPU's split passes.
@compute @workgroup_size(16, 16)
fn update_h(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = P.nx;
    let ny = P.ny;

    // Hx at (i, j+1/2): valid for i < nx, j < ny-1.
    if (i < nx && j < ny - 1u) {
        let k = i * (ny - 1u) + j;
        let d = ez[i * ny + j + 1u] - ez[i * ny + j];
        let psi = b_y_h(j) * psi_h[k] + c_y_h(j) * d;
        psi_h[k] = psi;
        hx[k] = hx[k] - P.dt * (d * inv_ky_h(j) + psi);
    }

    // Hy at (i+1/2, j): valid for i < nx-1, j < ny.
    if (i < nx - 1u && j < ny) {
        let k = i * ny + j;
        let o = nx * (ny - 1u); // psi_hy_x starts after psi_hx_y
        let d = ez[(i + 1u) * ny + j] - ez[i * ny + j];
        let psi = b_x_h(i) * psi_h[o + k] + c_x_h(i) * d;
        psi_h[o + k] = psi;
        hy[k] = hy[k] + P.dt * (d * inv_kx_h(i) + psi);
    }
}
