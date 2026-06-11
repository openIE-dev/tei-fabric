//! tei-sim-crossbar — CrossSim-class (Sandia) analog in-memory MVM simulator.
//!
//! Functional simulator for the in-memory-compute substrate column: a weight
//! matrix `W` (rows = inputs k, cols = outputs n) is mapped onto device
//! conductances and matrix-vector multiplies `y_j = Σᵢ xᵢ·W[i][j]` are
//! executed through a stack of device and peripheral non-idealities:
//!
//! - **programming variability** — per-device multiplicative lognormal error
//!   `G_prog = G_target · exp(σ_prog·Z)`, `Z ~ N(0,1)`. The lognormal form is
//!   the standard fit to ReRAM/PCM write distributions (see e.g. the CrossSim
//!   device models, Sandia, <https://cross-sim.sandia.gov>, and Wan et al.,
//!   "A compute-in-memory chip based on resistive random-access memory",
//!   Nature 608, 2022).
//! - **read noise** — per-read additive Gaussian on each device current with
//!   standard deviation proportional to the device conductance
//!   (`σ_i = σ_read·|G_i|`), i.e. multiplicative shot/thermal read noise.
//! - **conductance drift** — PCM power law `G(t) = G0·(t/t0)^(−ν)`
//!   (Ielmini et al., IEEE TED 2007; Le Gallo & Sebastian, "An overview of
//!   phase-change memory device physics", J. Phys. D 53, 213002, 2020),
//!   applied deterministically through an `age = t/t0` parameter.
//! - **DAC quantization** — `b_in`-bit uniform mid-rise quantizer over the
//!   input full scale ±`input_range`, with clipping.
//! - **ADC transfer** — `b_out`-bit uniform quantizer over a configurable
//!   range with clipping and optional bow-shaped INL. Ideal quantization SNR
//!   for a full-scale sinusoid is the classic `6.02·b + 1.76 dB`
//!   (Bennett, "Spectra of quantized signals", Bell Syst. Tech. J. 27, 1948).
//! - **IR drop** — three fidelity modes; see [`IrDropMode`]. The exact
//!   resistive-mesh mode builds each tile's parasitic wire network as a
//!   resistor netlist and solves its DC operating point through
//!   `tei-sim-circuit` (roadmap §2 cross-crate flow / §3.5 M1), factoring
//!   the MNA matrix once per programmed tile and re-solving per input vector.
//!
//! Matrices larger than the physical array are tiled `⌈k/size⌉ × ⌈n/size⌉`
//! with partial sums accumulated in the digital domain; the ADC fires once
//! per (row-tile, output column), so a full MVM costs `n·⌈k/size⌉` ADC
//! samples and `k·n` MACs — both counted in the [`EventLedger`].
//!
//! **Signed-weight convention.** Hardware realizes signed weights as a
//! balanced differential conductance pair `(G⁺, G⁻)` read on complementary
//! columns. This crate collapses the pair into one *effective signed
//! conductance* `G = G⁺ − G⁻` per cell, with all multiplicative device
//! effects (lognormal programming, drift, read noise scaled by |G|) applied
//! to the effective value. The differential periphery is a circuit-level
//! detail owned by `tei-sim-circuit`.
//!
//! Validation (tests/analytic.rs): analytic + published only — independent
//! read-noise variance propagation σ_y² = Σ xᵢ²σᵢ², quantization SNR
//! 6.02·b + 1.76 dB, drift-exponent recovery, tiling exactness, lognormal
//! mean exp(μ + σ²/2), seed determinism.

pub mod idx;
pub mod mlp;
pub mod mnist;

use serde::{Deserialize, Serialize};
use tei_sim_core::exec::{ExecutionResult, Executor, Progress};
use tei_sim_core::ledger::EventLedger;
use tei_sim_core::rng::Rng;

// ───────────────────────────── device model ─────────────────────────────

/// ADC transfer parameters: uniform `bits`-bit quantizer over ±`range`
/// (output-domain units — internally the column current is normalized by the
/// weight→conductance scale before digitization, which is mathematically
/// identical to an ADC range of `range·g_scale` amperes) with clipping, plus
/// an optional bow-shaped integral non-linearity.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct AdcParams {
    pub bits: u32,
    /// Full-scale: codes span [−range, +range]; inputs outside clip.
    pub range: f64,
    /// Peak INL in LSB. Modeled as a half-sine bow over the code axis,
    /// `e(c) = inl_lsb·Δ·sin(π·c/(2^b − 1))` — the standard low-order INL
    /// signature of flash/SAR converters. 0 disables it.
    #[serde(default)]
    pub inl_lsb: f64,
}

/// IR-drop fidelity modes for the parasitic wire resistance of the array.
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum IrDropMode {
    /// Zero wire resistance — currents sum losslessly.
    #[default]
    Ideal,
    /// First-order closed form. Each cell (i, j) of a physical tile of
    /// `r` rows sees, to first order, its own series wire resistance
    ///
    /// ```text
    /// R_path(i, j) = R_wire · ((j + 1) + (r − i))
    /// ```
    ///
    /// — `j + 1` row-wire segments from the input driver at the left edge
    /// to the cell, plus `r − i` column-wire segments from the cell down to
    /// the sense amplifier (virtual ground) at the bottom edge. The device
    /// branch then behaves as `G` in series with `R_path`, i.e. an effective
    /// conductance
    ///
    /// ```text
    /// G_eff = G / (1 + |G|·R_path)
    /// ```
    ///
    /// **Approximation:** this treats every cell's current path as
    /// independent, ignoring the voltage drops caused by *other* cells'
    /// currents sharing the same wire segments. It is exact for a single
    /// active device and first-order accurate when the aggregate drop is
    /// small (`N·Ḡ·R_wire ≪ 1`); it underestimates the degradation of a
    /// fully-active array. The coupled solve is [`IrDropMode::ExactMesh`].
    FirstOrder { r_wire: f64 },
    /// Exact resistive-mesh solve: every wire segment is a resistor and the
    /// coupled network is solved by full MNA through `tei-sim-circuit`
    /// (docs/SIM-ROADMAP.md §2 cross-crate flow, §3.3, §3.5 M1/M4).
    ///
    /// **Mesh model** (per physical tile of `r` rows × `c` cols, same wire
    /// geometry as [`IrDropMode::FirstOrder`]): each row wire is a chain of
    /// `r_wire` segments from the driver at the left edge through its
    /// crosspoints; each column wire is a chain of `r_wire` segments from its
    /// crosspoints down to the sense node at the bottom edge — an ideal TIA,
    /// i.e. virtual ground, so the termination *is* circuit ground. Row
    /// inputs are ideal voltage sources.
    ///
    /// **Signed weights.** The crate's collapsed signed conductance
    /// `G = G⁺ − G⁻` is un-collapsed for the mesh: each logical column is a
    /// balanced differential pair of physical column wires, the device
    /// `|G|` sits on the `+` wire when `G > 0` and on the `−` wire when
    /// `G < 0`, and the logical sense current is `I⁺ − I⁻`. (A literal
    /// negative resistor would *amplify* under IR drop — `G/(1 − |G|R)` —
    /// which is unphysical; the differential network reproduces the
    /// `G/(1 + |G|R_path)` single-device closed form exactly.)
    ///
    /// `r_wire = 0` elides the mesh and behaves as [`IrDropMode::Ideal`],
    /// matching `FirstOrder { r_wire: 0.0 }`.
    ///
    /// **Cost.** The tile's MNA matrix depends only on the programmed
    /// conductances, so it is factored once at programming time and each
    /// query is a cached-LU re-solve (`tei-sim-circuit::LinearDcSolver`);
    /// one solve per (tile, MVM), counted in `EventLedger::mesh_solves`.
    /// Tiles are capped at [`EXACT_MESH_MAX_ARRAY`]² devices — see the
    /// constant's docs.
    ///
    /// Serde note: this variant gained `r_wire` when the mode was
    /// implemented; jobs must now spell `{"exact_mesh": {"r_wire": …}}`
    /// (the old bare `"exact_mesh"` string only ever panicked). `Ideal` and
    /// `FirstOrder` encodings are unchanged.
    ExactMesh { r_wire: f64 },
}

/// Largest `array_size` accepted for [`IrDropMode::ExactMesh`].
///
/// A fully-populated 64×64 tile is 8,320 MNA unknowns (64 drivers + 2·64²
/// mesh nodes + 64 source branch rows). Measured on the M4 sparse Markowitz
/// LU (release, Apple Silicon; `bench_exact_mesh_factor_and_solve_64`):
/// ~2.3 s to factor — paid **once** per programmed tile — then ~0.4 ms per
/// query re-solve, so steady-state queries are cheap. Fill growth makes the
/// one-time factorization balloon super-linearly past this size, while
/// wire-resistance current *redistribution* — the physics the mode exists
/// for — is fully visible well below it; larger logical matrices should
/// simply tile (`array_size ≤ 64`), exactly as physical chips do.
pub const EXACT_MESH_MAX_ARRAY: usize = 64;

/// Full device + periphery parameter set for a crossbar.
#[derive(Debug, Clone, Deserialize, Serialize)]
#[serde(default)]
pub struct DeviceParams {
    /// Maximum device conductance magnitude, siemens. The largest |weight|
    /// maps to this; everything else scales linearly.
    pub g_max: f64,
    /// Lognormal programming-error σ (0 = perfect write).
    pub sigma_prog: f64,
    /// Relative per-read noise: each read draws `N(0, (σ_read·|G|)²)` on the
    /// device conductance (0 = noiseless read).
    pub sigma_read: f64,
    /// PCM drift exponent ν in `G(t) = G0·(t/t0)^(−ν)` (0 = no drift).
    pub drift_nu: f64,
    /// Normalized age `t/t0` at read time (1 = freshly programmed).
    pub age: f64,
    /// Input DAC resolution; `None` = ideal analog input.
    pub dac_bits: Option<u32>,
    /// Input full scale ±`input_range` for the DAC.
    pub input_range: f64,
    /// Output ADC; `None` = ideal analog readout.
    pub adc: Option<AdcParams>,
    /// Parasitic wire-resistance fidelity.
    pub ir_drop: IrDropMode,
}

impl Default for DeviceParams {
    fn default() -> Self {
        Self {
            g_max: 100e-6, // 100 µS — typical ReRAM LRS scale.
            sigma_prog: 0.0,
            sigma_read: 0.0,
            drift_nu: 0.0,
            age: 1.0,
            dac_bits: None,
            input_range: 1.0,
            adc: None,
            ir_drop: IrDropMode::Ideal,
        }
    }
}

/// Uniform mid-rise quantizer over ±range with clipping: step Δ = 2R/2^b,
/// reconstruction levels at (c + ½)Δ − R for codes c ∈ [0, 2^b).
fn quantize_uniform(v: f64, range: f64, bits: u32) -> f64 {
    let levels = (1u64 << bits) as f64;
    let step = 2.0 * range / levels;
    let code = ((v + range) / step).floor().clamp(0.0, levels - 1.0);
    (code + 0.5) * step - range
}

/// ADC transfer: uniform quantization + clipping + optional INL bow.
fn adc_transfer(v: f64, p: &AdcParams) -> f64 {
    let levels = (1u64 << p.bits) as f64;
    let step = 2.0 * p.range / levels;
    let code = ((v + p.range) / step).floor().clamp(0.0, levels - 1.0);
    let mut out = (code + 0.5) * step - p.range;
    if p.inl_lsb != 0.0 {
        out += p.inl_lsb * step * (std::f64::consts::PI * code / (levels - 1.0)).sin();
    }
    out
}

// ───────────────────────────── crossbar array ─────────────────────────────

/// One physical tile: a contiguous block of the weight matrix programmed
/// onto a ≤ `array_size`² device array.
#[derive(Debug, Clone)]
struct Tile {
    row0: usize,
    col0: usize,
    rows: usize,
    cols: usize,
    /// Programmed effective signed conductance G0 (post-lognormal), row-major.
    g: Vec<f64>,
}

/// The parasitic resistive mesh of one programmed tile, with its MNA matrix
/// factored once — the engine behind [`IrDropMode::ExactMesh`].
///
/// Network (see [`IrDropMode::ExactMesh`] for the physics): per row, a
/// voltage-source-driven chain of `r_wire` segments through the row's
/// crosspoint nodes; per logical column, a differential pair of column-wire
/// chains terminated at ground (the ideal-TIA virtual ground); per device,
/// `|G|` linking its row node to its `+` or `−` column node by sign.
/// Pass-through nodes (zero-conductance crosspoints, device-free column
/// levels) are merged into longer wire segments — they carry no branch
/// current of their own, so the solve is unchanged and the system shrinks.
///
/// Column currents are read as `Σ_devices G·(v_row − v_col)` rather than from
/// the termination-segment voltage drop: by KCL the sums are identical, but
/// the device form stays well-conditioned as `r_wire → 0` (each `v_row −
/// v_col` is O(drive), never a difference of near-equal tiny numbers).
#[derive(Debug, Clone)]
struct TileMesh {
    /// Factor-once / solve-many DC solver over the tile netlist; one
    /// voltage source per tile row, in row order.
    solver: tei_sim_circuit::LinearDcSolver,
    /// One tap per programmed device: (logical column, signed aged
    /// conductance G, row-wire node, column-wire node). The device current
    /// contribution to its logical column is `G·(v(row) − v(col))`.
    taps: Vec<(usize, f64, usize, usize)>,
    cols: usize,
}

impl TileMesh {
    /// Build and factor the mesh for `tile`, conductances aged by `drift`.
    /// `r_wire` must be positive (callers elide the mesh at 0).
    fn build(tile: &Tile, drift: f64, r_wire: f64) -> Self {
        use tei_sim_circuit::{Netlist, Waveform};
        debug_assert!(r_wire > 0.0);
        let (rows, cols) = (tile.rows, tile.cols);
        let mut net = Netlist::new();
        let mut next_node = 0usize;

        // Row drivers: vsource k = row k, the order LinearDcSolver::solve
        // expects its per-row drive voltages in.
        let mut drivers = Vec::with_capacity(rows);
        for i in 0..rows {
            next_node += 1;
            net.vsource(&format!("v{i}"), next_node, 0, Waveform::Dc { v: 0.0 });
            drivers.push(next_node);
        }

        // Row wires: driver →(j+1 segments)→ first device, then segment runs
        // between devices; the stub past the last device carries no current
        // and is dropped. Records each device's row node.
        let mut row_node = vec![0usize; rows * cols];
        for i in 0..rows {
            let (mut prev, mut prev_j) = (drivers[i], -1i64);
            for j in 0..cols {
                if tile.g[i * cols + j] == 0.0 {
                    continue;
                }
                next_node += 1;
                let seg = (j as i64 - prev_j) as f64 * r_wire;
                net.resistor(&format!("rw{i}_{j}"), prev, next_node, seg);
                row_node[i * cols + j] = next_node;
                (prev, prev_j) = (next_node, j as i64);
            }
        }

        // Column wires (one chain per polarity) + devices. Level i sits
        // i segments below the top, (rows − i) above the ground termination.
        let mut taps = Vec::with_capacity(rows * cols);
        for j in 0..cols {
            for sign in [1.0f64, -1.0] {
                let tag = if sign > 0.0 { 'p' } else { 'n' };
                let mut prev: Option<(usize, usize)> = None; // (node, level)
                for i in 0..rows {
                    let g = tile.g[i * cols + j] * drift;
                    if g == 0.0 || (g > 0.0) != (sign > 0.0) {
                        continue;
                    }
                    next_node += 1;
                    net.resistor(
                        &format!("d{tag}{i}_{j}"),
                        row_node[i * cols + j],
                        next_node,
                        1.0 / g.abs(),
                    );
                    if let Some((pn, pi)) = prev {
                        let seg = (i - pi) as f64 * r_wire;
                        net.resistor(&format!("cw{tag}{i}_{j}"), pn, next_node, seg);
                    }
                    taps.push((j, g, row_node[i * cols + j], next_node));
                    prev = Some((next_node, i));
                }
                if let Some((pn, pi)) = prev {
                    let term = (rows - pi) as f64 * r_wire;
                    net.resistor(&format!("ct{tag}{j}"), pn, 0, term);
                }
            }
        }

        let solver = tei_sim_circuit::LinearDcSolver::new(&net)
            .expect("crossbar IR-drop mesh is a grounded linear network; must factor");
        Self { solver, taps, cols }
    }

    /// One coupled DC solve: drive the rows with `x` (volts) and return every
    /// logical column's sense current `I⁺ − I⁻` in amperes.
    fn column_currents(&self, x: &[f64]) -> Vec<f64> {
        let sol = self.solver.solve(x);
        let mut currents = vec![0.0; self.cols];
        for &(j, g, rn, cn) in &self.taps {
            // Device |G| sits row→col on the polarity wire matching
            // sign(G); its logical-column contribution is G·(v_r − v_c)
            // for both signs (the − wire's current enters as −I⁻).
            currents[j] += g * (sol.node(rn) - sol.node(cn));
        }
        currents
    }
}

/// A weight matrix mapped onto tiled crossbar arrays with a programmable
/// device model. Construction *programs* the devices (drawing lognormal
/// write errors from `rng`); [`CrossbarArray::mvm`] then executes noisy
/// matrix-vector products and [`CrossbarArray::ideal_mvm`] the exact
/// digital reference.
#[derive(Debug, Clone)]
pub struct CrossbarArray {
    rows: usize,
    cols: usize,
    array_size: usize,
    params: DeviceParams,
    /// Weight → conductance scale, S per weight unit: g_max / max|w|.
    g_scale: f64,
    /// Ideal weights, row-major (the digital reference).
    w_ideal: Vec<f64>,
    tiles: Vec<Tile>,
    /// Factored IR-drop meshes, parallel to `tiles`. Non-empty exactly when
    /// `ir_drop` is `ExactMesh` with `r_wire > 0` (built — and the MNA LU
    /// paid — once, at programming time).
    meshes: Vec<TileMesh>,
}

impl CrossbarArray {
    /// Program `weights` (row-major, `rows × cols`) onto tiled physical
    /// arrays of side `array_size`. Lognormal write errors are drawn from
    /// `rng` in a fixed tile-major, row-major device order, so identical
    /// seeds program identical arrays.
    pub fn program(
        weights: &[f64],
        rows: usize,
        cols: usize,
        array_size: usize,
        params: DeviceParams,
        rng: &mut Rng,
    ) -> Self {
        assert_eq!(weights.len(), rows * cols, "weights must be rows×cols");
        assert!(array_size > 0, "array_size must be positive");
        if let IrDropMode::ExactMesh { r_wire } = params.ir_drop {
            assert!(
                r_wire.is_finite() && r_wire >= 0.0,
                "ExactMesh r_wire must be finite and ≥ 0, got {r_wire}"
            );
            assert!(
                r_wire == 0.0 || array_size <= EXACT_MESH_MAX_ARRAY,
                "ExactMesh tiles are capped at array_size {EXACT_MESH_MAX_ARRAY} \
                 (got {array_size}): the per-tile MNA factorization grows \
                 super-linearly in fill — tile larger matrices instead"
            );
        }
        let w_max = weights.iter().fold(0.0f64, |m, &w| m.max(w.abs()));
        let g_scale = if w_max > 0.0 {
            params.g_max / w_max
        } else {
            params.g_max
        };

        let row_tiles = rows.div_ceil(array_size);
        let col_tiles = cols.div_ceil(array_size);
        let mut tiles = Vec::with_capacity(row_tiles * col_tiles);
        for tr in 0..row_tiles {
            for tc in 0..col_tiles {
                let row0 = tr * array_size;
                let col0 = tc * array_size;
                let trows = array_size.min(rows - row0);
                let tcols = array_size.min(cols - col0);
                let mut g = Vec::with_capacity(trows * tcols);
                for i in 0..trows {
                    for j in 0..tcols {
                        let target = weights[(row0 + i) * cols + (col0 + j)] * g_scale;
                        // G_prog = G_target · exp(σ_prog·Z): multiplicative
                        // lognormal on the magnitude, sign preserved.
                        let prog = if params.sigma_prog > 0.0 {
                            target * (params.sigma_prog * rng.normal()).exp()
                        } else {
                            target
                        };
                        g.push(prog);
                    }
                }
                tiles.push(Tile {
                    row0,
                    col0,
                    rows: trows,
                    cols: tcols,
                    g,
                });
            }
        }

        // ExactMesh: build and LU-factor each tile's parasitic network now —
        // the matrix depends only on the programmed (and aged) conductances,
        // so every later query re-solves against the cached factorization.
        let meshes = match params.ir_drop {
            IrDropMode::ExactMesh { r_wire } if r_wire > 0.0 => {
                let drift = if params.drift_nu == 0.0 {
                    1.0
                } else {
                    params.age.powf(-params.drift_nu)
                };
                tiles
                    .iter()
                    .map(|t| TileMesh::build(t, drift, r_wire))
                    .collect()
            }
            _ => Vec::new(),
        };

        Self {
            rows,
            cols,
            array_size,
            params,
            g_scale,
            w_ideal: weights.to_vec(),
            tiles,
            meshes,
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }

    pub fn cols(&self) -> usize {
        self.cols
    }

    /// Weight → conductance scale (siemens per weight unit).
    pub fn g_scale(&self) -> f64 {
        self.g_scale
    }

    /// PCM drift factor `(t/t0)^(−ν)` at the configured age.
    fn drift_factor(&self) -> f64 {
        if self.params.drift_nu == 0.0 {
            1.0
        } else {
            self.params.age.powf(-self.params.drift_nu)
        }
    }

    /// Effective signed conductance of device (row, col) at read time:
    /// programmed value (lognormal included) aged by the drift power law.
    pub fn conductance(&self, row: usize, col: usize) -> f64 {
        assert!(row < self.rows && col < self.cols);
        let tr = row / self.array_size;
        let tc = col / self.array_size;
        let col_tiles = self.cols.div_ceil(self.array_size);
        let tile = &self.tiles[tr * col_tiles + tc];
        let (i, j) = (row - tile.row0, col - tile.col0);
        tile.g[i * tile.cols + j] * self.drift_factor()
    }

    /// Exact digital reference `y_j = Σᵢ xᵢ·W[i][j]` — no device model.
    pub fn ideal_mvm(&self, x: &[f64]) -> Vec<f64> {
        assert_eq!(x.len(), self.rows);
        let mut y = vec![0.0; self.cols];
        for (i, &xi) in x.iter().enumerate() {
            let row = &self.w_ideal[i * self.cols..(i + 1) * self.cols];
            for (yj, &w) in y.iter_mut().zip(row) {
                *yj += xi * w;
            }
        }
        y
    }

    /// Noisy MVM through the full non-ideality stack. Per row-tile partial
    /// sums are digitized (one ADC sample per tile × output column) and
    /// accumulated in the digital domain. Ledger: `macs += rows·cols`,
    /// `adc_samples += cols·⌈rows/array_size⌉`, and under
    /// [`IrDropMode::ExactMesh`] one `mesh_solves` per (tile, MVM) — the
    /// `macs` convention is unchanged (the mesh realizes the same MACs,
    /// coupled by the wire network).
    pub fn mvm(&self, x: &[f64], rng: &mut Rng, ledger: &mut EventLedger) -> Vec<f64> {
        assert_eq!(x.len(), self.rows);

        // DAC: quantize the input vector once; every tile in a row-block is
        // driven by the same quantized line voltages.
        let xq: Vec<f64> = match self.params.dac_bits {
            Some(b) => x
                .iter()
                .map(|&v| quantize_uniform(v, self.params.input_range, b))
                .collect(),
            None => x.to_vec(),
        };

        let drift = self.drift_factor();
        let mut y = vec![0.0; self.cols];
        for (ti, tile) in self.tiles.iter().enumerate() {
            let xs = &xq[tile.row0..tile.row0 + tile.rows];
            // ExactMesh: one coupled DC solve per (tile, query) yields every
            // column's noiseless sense current at once. The mesh replaces
            // only the ideal-current computation — read noise and the
            // ADC/digital stages below apply identically in the same order.
            let mesh_currents = self.meshes.get(ti).map(|m| {
                ledger.mesh_solves += 1;
                m.column_currents(xs)
            });
            for j in 0..tile.cols {
                // Analog column current, normalized by g_scale into output
                // (weight·input) units — identical math, friendlier numbers.
                let mut acc = 0.0;
                match &mesh_currents {
                    Some(currents) => {
                        acc = currents[j];
                        // Per-read conductance noise σ = σ_read·|G| on each
                        // device, same statistics and draw order as the
                        // uncoupled modes (noise × IR-drop cross terms are
                        // second-order small and not modeled).
                        if self.params.sigma_read > 0.0 {
                            for (i, &xi) in xs.iter().enumerate() {
                                let g0 = tile.g[i * tile.cols + j] * drift;
                                acc += xi * self.params.sigma_read * g0.abs() * rng.normal();
                            }
                        }
                    }
                    None => {
                        for (i, &xi) in xs.iter().enumerate() {
                            let g0 = tile.g[i * tile.cols + j] * drift;
                            let g_eff = match self.params.ir_drop {
                                IrDropMode::Ideal => g0,
                                IrDropMode::FirstOrder { r_wire } => {
                                    // See IrDropMode::FirstOrder for the derivation.
                                    let r_path = r_wire * ((j + 1) + (tile.rows - i)) as f64;
                                    g0 / (1.0 + g0.abs() * r_path)
                                }
                                // Mesh elided (r_wire = 0): lossless wires.
                                IrDropMode::ExactMesh { .. } => g0,
                            };
                            // Per-read conductance noise σ = σ_read·|G| (a device
                            // property — scaled by the aged programmed conductance).
                            let g_read = if self.params.sigma_read > 0.0 {
                                g_eff + self.params.sigma_read * g0.abs() * rng.normal()
                            } else {
                                g_eff
                            };
                            acc += xi * g_read;
                        }
                    }
                }
                ledger.macs += tile.rows as u64;
                ledger.adc_samples += 1;
                let mut partial = acc / self.g_scale;
                if let Some(adc) = &self.params.adc {
                    partial = adc_transfer(partial, adc);
                }
                y[tile.col0 + j] += partial;
            }
        }
        y
    }
}

// ───────────────────────────── executor ─────────────────────────────

/// Job spec accepted by the crossbar executor (mirrors /api/execute):
/// a `rows × cols` random weight matrix is programmed onto `array_size`
/// tiles with the given device model, then `n_queries` random-input MVMs
/// run and the noisy outputs are scored against the digital reference.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CrossbarJob {
    pub rows: usize,
    pub cols: usize,
    #[serde(default = "default_array_size")]
    pub array_size: usize,
    #[serde(default)]
    pub device: DeviceParams,
    pub n_queries: u64,
    #[serde(default)]
    pub seed: u64,
}

fn default_array_size() -> usize {
    256
}

pub struct CrossbarExecutor;

impl Executor for CrossbarExecutor {
    type Job = CrossbarJob;

    fn execute(&self, job: &Self::Job, on_progress: &mut dyn FnMut(Progress)) -> ExecutionResult {
        let t0 = std::time::Instant::now();
        // Validate job-level ExactMesh constraints here so HTTP callers get an
        // error payload instead of a panicked worker (program() asserts).
        if let IrDropMode::ExactMesh { r_wire } = job.device.ir_drop {
            if !(r_wire.is_finite() && r_wire >= 0.0) {
                return ExecutionResult {
                    ledger: EventLedger::default(),
                    outputs: serde_json::json!({
                        "error": format!("exact_mesh r_wire must be finite and ≥ 0, got {r_wire}")
                    }),
                };
            }
            if r_wire > 0.0 && job.array_size > EXACT_MESH_MAX_ARRAY {
                return ExecutionResult {
                    ledger: EventLedger::default(),
                    outputs: serde_json::json!({
                        "error": format!(
                            "exact_mesh tiles are capped at array_size {EXACT_MESH_MAX_ARRAY}                              (got {}); set array_size ≤ {EXACT_MESH_MAX_ARRAY}",
                            job.array_size
                        )
                    }),
                };
            }
        }
        let mut rng = Rng::new(job.seed);

        // Random weights in [−1, 1], programmed once.
        let weights: Vec<f64> = (0..job.rows * job.cols)
            .map(|_| 2.0 * rng.f64() - 1.0)
            .collect();
        let array = CrossbarArray::program(
            &weights,
            job.rows,
            job.cols,
            job.array_size,
            job.device.clone(),
            &mut rng,
        );

        let mut ledger = EventLedger::default();
        let mut sum_err2 = 0.0;
        let mut sum_sig2 = 0.0;
        let mut elements = 0u64;
        let report_every = (job.n_queries / 100).max(1);

        for q in 0..job.n_queries {
            let x: Vec<f64> = (0..job.rows).map(|_| 2.0 * rng.f64() - 1.0).collect();
            let y_ideal = array.ideal_mvm(&x);
            let y_noisy = array.mvm(&x, &mut rng, &mut ledger);
            for (yn, yi) in y_noisy.iter().zip(&y_ideal) {
                let e = yn - yi;
                sum_err2 += e * e;
                sum_sig2 += yi * yi;
            }
            elements += job.cols as u64;

            if (q + 1) % report_every == 0 || q + 1 == job.n_queries {
                let rms = (sum_err2 / elements as f64).sqrt();
                on_progress(Progress {
                    fraction: (q + 1) as f64 / job.n_queries as f64,
                    metrics: serde_json::json!({
                        "query": q + 1,
                        "rms_error": rms,
                        "snr_db": snr_db(sum_sig2, sum_err2),
                    }),
                });
            }
        }
        ledger.wall_seconds = Some(t0.elapsed().as_secs_f64());

        let rms_error = (sum_err2 / elements.max(1) as f64).sqrt();
        ExecutionResult {
            ledger: ledger.clone(),
            outputs: serde_json::json!({
                "rows": job.rows,
                "cols": job.cols,
                "array_size": job.array_size,
                "n_queries": job.n_queries,
                "rms_error": rms_error,
                "snr_db": snr_db(sum_sig2, sum_err2),
                "macs": ledger.macs,
                "adc_samples": ledger.adc_samples,
                "mesh_solves": ledger.mesh_solves,
            }),
        }
    }
}

/// SNR in dB; `None` (JSON null) when the error is exactly zero.
fn snr_db(sig2: f64, err2: f64) -> Option<f64> {
    (err2 > 0.0).then(|| 10.0 * (sig2 / err2).log10())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mid-rise quantizer: full-scale endpoints clip to the outermost
    /// reconstruction levels and 0 maps within Δ/2.
    #[test]
    fn quantizer_basics() {
        let step = 2.0 / 16.0; // b=4, range 1
        assert!((quantize_uniform(1.0, 1.0, 4) - (1.0 - step / 2.0)).abs() < 1e-15);
        assert!((quantize_uniform(-1.0, 1.0, 4) - (-1.0 + step / 2.0)).abs() < 1e-15);
        assert!(quantize_uniform(0.0, 1.0, 4).abs() <= step / 2.0 + 1e-15);
        // Clipping beyond full scale.
        assert!((quantize_uniform(5.0, 1.0, 4) - (1.0 - step / 2.0)).abs() < 1e-15);
    }

    /// First-order IR drop strictly reduces output magnitude and converges
    /// to ideal as R_wire → 0.
    #[test]
    fn ir_drop_first_order_sanity() {
        let w = vec![1.0; 64];
        let x = vec![1.0; 8];
        let mk = |r_wire: f64| {
            let params = DeviceParams {
                ir_drop: if r_wire == 0.0 {
                    IrDropMode::Ideal
                } else {
                    IrDropMode::FirstOrder { r_wire }
                },
                ..Default::default()
            };
            let arr = CrossbarArray::program(&w, 8, 8, 8, params, &mut Rng::new(1));
            arr.mvm(&x, &mut Rng::new(2), &mut EventLedger::default())[0]
        };
        let ideal = mk(0.0);
        let small = mk(1e-3); // |G|·R_path ~ 1e-9 — negligible
        let large = mk(200.0); // |G|·R_path ~ 2e-4·per-segment — visible
        assert!((small - ideal).abs() / ideal < 1e-6, "{small} vs {ideal}");
        assert!(large < ideal, "{large} !< {ideal}");
        assert!(large > 0.9 * ideal, "first-order should stay small here");
    }

    /// IrDropMode serde: the pre-existing `Ideal`/`FirstOrder` encodings are
    /// unchanged and `ExactMesh` round-trips with its `r_wire`.
    #[test]
    fn ir_drop_serde_encodings() {
        let ideal: IrDropMode = serde_json::from_str(r#""ideal""#).unwrap();
        assert!(matches!(ideal, IrDropMode::Ideal));
        let fo: IrDropMode = serde_json::from_str(r#"{"first_order":{"r_wire":2.5}}"#).unwrap();
        assert!(matches!(fo, IrDropMode::FirstOrder { r_wire } if r_wire == 2.5));
        let em: IrDropMode = serde_json::from_str(r#"{"exact_mesh":{"r_wire":1.5}}"#).unwrap();
        assert!(matches!(em, IrDropMode::ExactMesh { r_wire } if r_wire == 1.5));
        let s = serde_json::to_string(&IrDropMode::ExactMesh { r_wire: 1.5 }).unwrap();
        assert_eq!(s, r#"{"exact_mesh":{"r_wire":1.5}}"#);
    }

    /// CrossbarJob round-trips through serde with defaults filled in.
    #[test]
    fn job_serde_defaults() {
        let job: CrossbarJob =
            serde_json::from_str(r#"{"rows": 64, "cols": 32, "n_queries": 10}"#).unwrap();
        assert_eq!(job.array_size, 256);
        assert_eq!(job.seed, 0);
        assert!(matches!(job.device.ir_drop, IrDropMode::Ideal));
        let s = serde_json::to_string(&job).unwrap();
        let back: CrossbarJob = serde_json::from_str(&s).unwrap();
        assert_eq!(back.rows, 64);
    }
}
