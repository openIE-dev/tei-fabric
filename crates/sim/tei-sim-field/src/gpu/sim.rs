//! Native wgpu host for the F4 WGSL kernels — see the module docs of
//! [`crate::gpu`] for the bind-group/buffer/dispatch contract this code
//! implements (and that a JS WebGPU driver must mirror).

use super::{
    INJECT_WGSL, JobData, STEP_STRIDE, StepParams, UPDATE_E_WGSL, UPDATE_H_WGSL, WORKGROUP_2D,
    WORKGROUP_INJECT, build_job_data, step_params_binding_size,
};
use crate::FieldJob;

/// Steps encoded per command buffer (bounds encoder size; NOT part of the
/// host contract — any chunking gives identical results).
const CHUNK: usize = 256;

fn request_adapter() -> Result<wgpu::Adapter, String> {
    // Headless: no display handle needed for compute.
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
    pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        compatible_surface: None,
        force_fallback_adapter: false,
    }))
    .map_err(|e| format!("gpu: no wgpu adapter available: {e}"))
}

/// True when a wgpu adapter can be acquired on this machine (any backend,
/// headless). Tests use this to SKIP rather than fail on GPU-less CI.
pub fn gpu_available() -> bool {
    request_adapter().is_ok()
}

/// A [`FieldJob`] materialized on the GPU: create with [`GpuSim::new`],
/// advance with [`GpuSim::step`], read fields back with
/// [`GpuSim::read_ez`] / [`read_hx`](GpuSim::read_hx) /
/// [`read_hy`](GpuSim::read_hy) and the probe trace with
/// [`GpuSim::read_trace`].
pub struct GpuSim {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipeline_h: wgpu::ComputePipeline,
    pipeline_e: wgpu::ComputePipeline,
    pipeline_inject: wgpu::ComputePipeline,
    pipeline_probe: wgpu::ComputePipeline,
    bg0: wgpu::BindGroup,
    bg1: wgpu::BindGroup,
    buf_ez: wgpu::Buffer,
    buf_hx: wgpu::Buffer,
    buf_hy: wgpu::Buffer,
    buf_trace: wgpu::Buffer,
    staging: wgpu::Buffer,
    nx: usize,
    ny: usize,
    /// Time step (f64; the GPU runs the f32 cast).
    pub dt: f64,
    steps_total: usize,
    steps_done: usize,
    has_source: bool,
    inject_groups: u32,
    groups_x: u32,
    groups_y: u32,
}

impl GpuSim {
    /// Build the full GPU state for `job` (uniform/slab/waveguide ε via the
    /// job's own [`EpsSpec`](crate::EpsSpec); point/line sources; probe
    /// defaulting to the grid center). Fails cleanly when no adapter is
    /// present or the job needs CPU-only features (mode source, ports).
    pub fn new(job: &FieldJob) -> Result<Self, String> {
        let data = build_job_data(job)?;
        let adapter = request_adapter()?;
        let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
            label: Some("tei-sim-field F4"),
            ..Default::default()
        }))
        .map_err(|e| format!("gpu: request_device failed: {e}"))?;
        Self::with_device(device, queue, job, data)
    }

    fn with_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        job: &FieldJob,
        data: JobData,
    ) -> Result<Self, String> {
        let (nx, ny) = (job.nx, job.ny);
        let n_ez = nx * ny;
        let n_hx = nx * (ny - 1);
        let n_hy = (nx - 1) * ny;
        let steps_total = job.steps;

        let f32s = |n: usize| (n.max(1) * 4) as u64;
        let storage = |label: &str, size: u64, extra: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage: wgpu::BufferUsages::STORAGE | extra,
                mapped_at_creation: false,
            })
        };

        // Field + psi buffers rely on zero initialization.
        let buf_ez = storage("ez", f32s(n_ez), wgpu::BufferUsages::COPY_SRC);
        let buf_hx = storage("hx", f32s(n_hx), wgpu::BufferUsages::COPY_SRC);
        let buf_hy = storage("hy", f32s(n_hy), wgpu::BufferUsages::COPY_SRC);
        let buf_psi_e = storage("psi_e", f32s(2 * n_ez), wgpu::BufferUsages::empty());
        let buf_psi_h = storage("psi_h", f32s(n_hx + n_hy), wgpu::BufferUsages::empty());
        let buf_ce = storage("ce", f32s(n_ez), wgpu::BufferUsages::COPY_DST);
        let buf_coeff = storage(
            "coeff",
            f32s(data.coeff.len()),
            wgpu::BufferUsages::COPY_DST,
        );
        let buf_trace = storage("trace", f32s(steps_total), wgpu::BufferUsages::COPY_SRC);

        let buf_params = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("params"),
            size: std::mem::size_of::<super::Params>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let buf_step = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("step_params"),
            size: STEP_STRIDE * steps_total.max(1) as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let staging = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("staging"),
            size: f32s(n_ez.max(steps_total)),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        // Uploads: Params, ce, coeff, one StepParams slot per step.
        queue.write_buffer(&buf_params, 0, bytemuck::bytes_of(&data.params));
        queue.write_buffer(&buf_ce, 0, bytemuck::cast_slice(&data.ce));
        queue.write_buffer(&buf_coeff, 0, bytemuck::cast_slice(&data.coeff));
        let mut slots = vec![0u8; (STEP_STRIDE as usize) * steps_total.max(1)];
        for (n, &amp) in data.amps.iter().enumerate() {
            let s = StepParams {
                amp,
                step: n as u32,
                pad: [0; 2],
            };
            let at = n * STEP_STRIDE as usize;
            slots[at..at + 16].copy_from_slice(bytemuck::bytes_of(&s));
        }
        queue.write_buffer(&buf_step, 0, &slots);

        // Shared explicit layout — the canonical contract (see module docs).
        let storage_entry = |binding: u32, read_only: bool| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bgl0 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f4 group0"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                storage_entry(1, false), // ez
                storage_entry(2, false), // hx
                storage_entry(3, false), // hy
                storage_entry(4, true),  // ce
                storage_entry(5, false), // psi_e
                storage_entry(6, false), // psi_h
                storage_entry(7, true),  // coeff
                storage_entry(8, false), // trace
            ],
        });
        let bgl1 = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("f4 group1"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: Some(step_params_binding_size()),
                },
                count: None,
            }],
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("f4 layout"),
            bind_group_layouts: &[Some(&bgl0), Some(&bgl1)],
            immediate_size: 0,
        });

        let bg0 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("f4 group0"),
            layout: &bgl0,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buf_params.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: buf_ez.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: buf_hx.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: buf_hy.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: buf_ce.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: buf_psi_e.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: buf_psi_h.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: buf_coeff.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: buf_trace.as_entire_binding(),
                },
            ],
        });
        let bg1 = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("f4 group1"),
            layout: &bgl1,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buf_step,
                    offset: 0,
                    size: Some(step_params_binding_size()),
                }),
            }],
        });

        let pipeline = |label: &str, src: &str, entry: &str| {
            let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(src.into()),
            });
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                module: &module,
                entry_point: Some(entry),
                compilation_options: Default::default(),
                cache: None,
            })
        };
        let pipeline_h = pipeline("update_h", UPDATE_H_WGSL, "update_h");
        let pipeline_e = pipeline("update_e", UPDATE_E_WGSL, "update_e");
        let pipeline_inject = pipeline("inject", INJECT_WGSL, "inject");
        let pipeline_probe = pipeline("record_probe", INJECT_WGSL, "record_probe");

        let src_span = data.params.src_j1 - data.params.src_j0 + 1;
        Ok(Self {
            device,
            queue,
            pipeline_h,
            pipeline_e,
            pipeline_inject,
            pipeline_probe,
            bg0,
            bg1,
            buf_ez,
            buf_hx,
            buf_hy,
            buf_trace,
            staging,
            nx,
            ny,
            dt: data.dt,
            steps_total,
            steps_done: 0,
            has_source: data.has_source,
            inject_groups: src_span.div_ceil(WORKGROUP_INJECT),
            groups_x: (nx as u32).div_ceil(WORKGROUP_2D),
            groups_y: (ny as u32).div_ceil(WORKGROUP_2D),
        })
    }

    /// Steps run so far.
    pub fn steps_done(&self) -> usize {
        self.steps_done
    }

    /// Advance `n` leapfrog steps (H half-step, E full step, source
    /// injection, probe record — identical sequencing to the CPU loop).
    /// The run is capped at the job's `steps` (the trace and per-step
    /// uniform slots are sized for it).
    pub fn step(&mut self, n: usize) -> Result<(), String> {
        if self.steps_done + n > self.steps_total {
            return Err(format!(
                "gpu: step overrun: {} + {n} > job.steps = {}",
                self.steps_done, self.steps_total
            ));
        }
        let mut s = self.steps_done;
        let end = self.steps_done + n;
        while s < end {
            let chunk_end = (s + CHUNK).min(end);
            let mut enc = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("f4 steps"),
                });
            {
                let mut pass = enc.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("f4 steps"),
                    timestamp_writes: None,
                });
                pass.set_bind_group(0, &self.bg0, &[]);
                for step in s..chunk_end {
                    pass.set_bind_group(1, &self.bg1, &[(step as u64 * STEP_STRIDE) as u32]);
                    pass.set_pipeline(&self.pipeline_h);
                    pass.dispatch_workgroups(self.groups_x, self.groups_y, 1);
                    pass.set_pipeline(&self.pipeline_e);
                    pass.dispatch_workgroups(self.groups_x, self.groups_y, 1);
                    if self.has_source {
                        pass.set_pipeline(&self.pipeline_inject);
                        pass.dispatch_workgroups(self.inject_groups, 1, 1);
                    }
                    pass.set_pipeline(&self.pipeline_probe);
                    pass.dispatch_workgroups(1, 1, 1);
                }
            }
            self.queue.submit([enc.finish()]);
            s = chunk_end;
        }
        self.steps_done = end;
        Ok(())
    }

    fn read_buffer(&self, buf: &wgpu::Buffer, count: usize) -> Result<Vec<f32>, String> {
        if count == 0 {
            return Ok(Vec::new());
        }
        let bytes = (count * 4) as u64;
        let mut enc = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("f4 readback"),
            });
        enc.copy_buffer_to_buffer(buf, 0, &self.staging, 0, bytes);
        self.queue.submit([enc.finish()]);
        let slice = self.staging.slice(..bytes);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|e| format!("gpu: poll failed: {e}"))?;
        rx.recv()
            .map_err(|_| "gpu: map_async callback dropped".to_string())?
            .map_err(|e| format!("gpu: buffer map failed: {e}"))?;
        let out = bytemuck::cast_slice::<u8, f32>(&slice.get_mapped_range()).to_vec();
        self.staging.unmap();
        Ok(out)
    }

    /// Flat row-major Ez (`ez[i*ny + j]`, length nx·ny).
    pub fn read_ez(&self) -> Result<Vec<f32>, String> {
        self.read_buffer(&self.buf_ez, self.nx * self.ny)
    }

    /// Flat row-major Hx (length nx·(ny−1)).
    pub fn read_hx(&self) -> Result<Vec<f32>, String> {
        self.read_buffer(&self.buf_hx, self.nx * (self.ny - 1))
    }

    /// Flat row-major Hy (length (nx−1)·ny).
    pub fn read_hy(&self) -> Result<Vec<f32>, String> {
        self.read_buffer(&self.buf_hy, (self.nx - 1) * self.ny)
    }

    /// Probe trace recorded so far (one sample per completed step).
    pub fn read_trace(&self) -> Result<Vec<f32>, String> {
        let mut v = self.read_buffer(&self.buf_trace, self.steps_total)?;
        v.truncate(self.steps_done);
        Ok(v)
    }
}
