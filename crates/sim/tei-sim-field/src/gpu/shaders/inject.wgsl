// inject.wgsl — F4 GPU FDTD: soft source injection + probe recording.
// Two entry points sharing the layout of update_h.wgsl / update_e.wgsl:
//
//   `inject`       — adds StepParams.amp onto Ez along the source column
//                    (rows src_j0..=src_j1; a point source has j0 == j1).
//                    The time profile s(t) is evaluated on the CPU in f64
//                    and baked into amp = amplitude*s((step+1)*dt), one
//                    StepParams slot per step (stride 256 bytes, bound
//                    size 16, dynamic offset 256*step).
//                    Workgroup 64; dispatch ceil((j1-j0+1)/64) x 1 x 1.
//                    Skip the dispatch entirely when src_kind == 0.
//
//   `record_probe` — trace[step] = ez[probe_idx]. Workgroup 1; dispatch
//                    1 x 1 x 1 AFTER inject (the CPU reference records the
//                    probe after source injection each step).
//
// BIND GROUP / BUFFER CONTRACT: see the canonical table at the top of
// update_h.wgsl.

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

struct StepParams {
    amp: f32,
    step: u32,
    pad0: u32,
    pad1: u32,
}

@group(0) @binding(0) var<uniform> P: Params;
@group(0) @binding(1) var<storage, read_write> ez: array<f32>;
@group(0) @binding(8) var<storage, read_write> trace: array<f32>;
@group(1) @binding(0) var<uniform> S: StepParams;

@compute @workgroup_size(64)
fn inject(@builtin(global_invocation_id) gid: vec3<u32>) {
    if (P.src_kind == 0u) {
        return;
    }
    let j = P.src_j0 + gid.x;
    if (j > P.src_j1) {
        return;
    }
    let idx = P.src_i * P.ny + j;
    ez[idx] = ez[idx] + S.amp;
}

@compute @workgroup_size(1)
fn record_probe() {
    trace[S.step] = ez[P.probe_idx];
}
