// update_e.wgsl — F4 GPU FDTD: Ez full-step update from Hx/Hy with CPML
// psi recursion + 1/kappa stretching, inverse-eps multiply (ce = dt/eps_r
// storage buffer), PEC outer ring. 2D TEz, f32.
//
// BIND GROUP / BUFFER / DISPATCH CONTRACT: identical to the table at the
// top of update_h.wgsl (the canonical copy). Summary:
//   group(0): 0 Params uniform | 1 ez rw | 2 hx rw | 3 hy rw | 4 ce ro |
//             5 psi_e rw (psi_ez_x then psi_ez_y, nx*ny each) |
//             6 psi_h rw | 7 coeff ro (packed CPML tables) | 8 trace rw
//   group(1): 0 StepParams uniform, dynamic offset, stride 256, size 16
//   dispatch: workgroup 16x16 over (ceil(nx/16), ceil(ny/16), 1), after
//             update_h within the same step.
// Updates are in place — this pass reads only hx/hy and each thread owns
// exactly its own ez/psi_e cells.

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
@group(0) @binding(4) var<storage, read> ce: array<f32>;
@group(0) @binding(5) var<storage, read_write> psi_e: array<f32>;
@group(0) @binding(7) var<storage, read> coeff: array<f32>;

fn inv_kx_e(i: u32) -> f32 { return coeff[i]; }
fn b_x_e(i: u32) -> f32 { return coeff[P.nx + i]; }
fn c_x_e(i: u32) -> f32 { return coeff[2u * P.nx + i]; }
fn ybase() -> u32 { return 3u * P.nx + 3u * (P.nx - 1u); }
fn inv_ky_e(j: u32) -> f32 { return coeff[ybase() + j]; }
fn b_y_e(j: u32) -> f32 { return coeff[ybase() + P.ny + j]; }
fn c_y_e(j: u32) -> f32 { return coeff[ybase() + 2u * P.ny + j]; }

// Ez <- Ez + (dt/eps_r)*[ (dHy/dx)/kappa_x - (dHx/dy)/kappa_y + psi_x - psi_y ]
//   psi_x <- b_x_e*psi_x + c_x_e*dHy/dx
//   psi_y <- b_y_e*psi_y + c_y_e*dHx/dy
// The outermost ring (i = 0, i = nx-1, j = 0, j = ny-1) is PEC: never
// written, stays at its zero initialization. Outside the CPML strips
// (1/kappa, b, c) = (1, 1, 0) and psi stays 0, so the uniform code path
// reproduces the CPU's split interior + strip passes.
@compute @workgroup_size(16, 16)
fn update_e(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let j = gid.y;
    let nx = P.nx;
    let ny = P.ny;
    if (i == 0u || j == 0u || i >= nx - 1u || j >= ny - 1u) {
        return;
    }
    let idx = i * ny + j;
    let dhy = hy[idx] - hy[(i - 1u) * ny + j];
    let dhx = hx[i * (ny - 1u) + j] - hx[i * (ny - 1u) + j - 1u];

    let px = b_x_e(i) * psi_e[idx] + c_x_e(i) * dhy;
    psi_e[idx] = px;
    let n = nx * ny; // psi_ez_y starts after psi_ez_x
    let py = b_y_e(j) * psi_e[n + idx] + c_y_e(j) * dhx;
    psi_e[n + idx] = py;

    ez[idx] = ez[idx] + ce[idx] * (dhy * inv_kx_e(i) - dhx * inv_ky_e(j) + px - py);
}
