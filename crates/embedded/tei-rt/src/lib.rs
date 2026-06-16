//! teiOS runtime kernel — the `no_std`, allocation-free dispatch +
//! calibration core.
//!
//! ## What this is
//!
//! Every `teios-*` board image today **hardcodes** the same loop: run the
//! primitive on substrate A, run it on substrate B, cross-check, emit a
//! dispatch line. That loop, factored out, is the runtime the
//! EMBEDDED-ROADMAP §3.5 calls **teiOS**:
//!
//! > a `no_std` TEI executor — instrumented runs, the embedded ledger,
//! > lowest-joule dispatch, a calibration agent.
//!
//! [`Runtime`] is exactly that core. You hand it the board's **substrates**
//! (each a named way to compute a primitive) and its **cost table** (the
//! price list, [`tei_ledger::CostTable`]); it:
//!
//! 1. **dispatches** — picks the lowest-joule substrate for a primitive
//!    ([`tei_ledger::CostTable::cheapest`], the rule the fabric uses);
//! 2. **instruments** — times the run on a [`CycleSource`], reads an optional
//!    [`EnergyMeter`], and produces an [`EventLedger`] with honest
//!    provenance ([`JoulesSource`]);
//! 3. **calibrates** — folds the *measured* joules back into the cost table,
//!    so the next dispatch is priced on reality, not the shipped guess.
//!
//! ## What this is not (yet)
//!
//! The contract types live in `tei-ledger`; this crate is the executor over
//! them. It does **not** own task scheduling — async per-task instrumentation
//! (the Embassy `trace` hooks) is a separate binding. And it is
//! **host-tested, not hardware-tested**: the dispatch/ledger/calibration
//! logic is exercised here with fakes; running it on real silicon (and the
//! board's real substrates + meter) is the bench step the `teios-*` images
//! gate. The kernel is real; the silicon is pending.

#![cfg_attr(not(test), no_std)]
#![forbid(unsafe_code)]

pub use tei_ledger::{
    CostEntry, CostTable, CalibrationReport, CycleSource, EnergyMeter, EventLedger, JoulesSource,
};

/// How a board computes one primitive on one substrate.
///
/// The work signature is `fn(&[u8]) -> u32` — the embedded profile's first
/// primitives (hash/CRC/checksum) all fold a byte workload to a 32-bit
/// result, which is also exactly what makes a cross-substrate [`check`]
/// possible. Richer result types are a later generalization; this matches
/// every `teios-*` image shipping today.
///
/// [`check`]: Runtime::check
#[derive(Clone, Copy)]
pub struct Substrate {
    /// Substrate name, fabric conventions: `"cpu@150mhz"`, `"pio"`,
    /// `"crca"`, `"dma-sniffer"`, … Matches [`CostEntry::substrate`].
    pub id: &'static str,
    /// The Periodic Stack primitive this substrate computes.
    pub primitive_id: u32,
    /// The work: fold the byte workload to a 32-bit result. A function
    /// pointer (not a closure) keeps the registry allocation-free.
    pub run: fn(&[u8]) -> u32,
}

/// The outcome of one dispatched run.
#[derive(Debug, Clone)]
pub struct Run {
    /// Which substrate actually ran (the dispatched, lowest-joule one).
    pub substrate: &'static str,
    /// The primitive's 32-bit result (for cross-checks / app use).
    pub result: u32,
    /// The priced ledger for this run.
    pub ledger: EventLedger,
}

/// The teiOS runtime: a fixed substrate set + a mutable cost table.
///
/// `N` is the cost-table capacity (primitives × substrates — small).
pub struct Runtime<'s, const N: usize> {
    substrates: &'s [Substrate],
    costs: CostTable<N>,
}

impl<'s, const N: usize> Runtime<'s, N> {
    /// Build a runtime over a board's substrates and its (possibly empty,
    /// possibly shipped-Table-tier) cost table.
    pub fn new(substrates: &'s [Substrate], costs: CostTable<N>) -> Self {
        Self { substrates, costs }
    }

    /// Read-only view of the live cost table (re-priced as runs measure).
    pub fn costs(&self) -> &CostTable<N> {
        &self.costs
    }

    /// The substrate teiOS would dispatch a primitive to **right now**: the
    /// lowest-joule priced substrate, or — before any price exists (cold
    /// start) — the first registered substrate for that primitive.
    pub fn dispatch(&self, primitive_id: u32) -> Option<&'s Substrate> {
        if let Some(e) = self.costs.cheapest(primitive_id) {
            if let Some(s) = self
                .substrates
                .iter()
                .find(|s| s.primitive_id == primitive_id && s.id == e.substrate)
            {
                return Some(s);
            }
        }
        self.substrates
            .iter()
            .find(|s| s.primitive_id == primitive_id)
    }

    /// Run a primitive on its dispatched substrate, producing a priced
    /// [`Run`]. When `meter` reports joules, the ledger is `Measured` and the
    /// cost table is **re-priced** from that measurement (so the next
    /// [`dispatch`] reflects reality). `n_ops` is the operation count the
    /// J/op is divided by (e.g. bytes hashed).
    ///
    /// Returns `None` only if no substrate is registered for the primitive.
    ///
    /// [`dispatch`]: Runtime::dispatch
    pub fn run<C: CycleSource>(
        &mut self,
        primitive_id: u32,
        data: &[u8],
        n_ops: u64,
        cycles: &C,
        meter: Option<&mut dyn EnergyMeter>,
    ) -> Option<Run> {
        // Copy out of the (immutable) substrate slice so the table can be
        // re-priced (mutable) without a borrow conflict.
        let sub = self.dispatch(primitive_id)?;
        let (id, f) = (sub.id, sub.run);
        let run = exec(id, f, primitive_id, data, n_ops, cycles, meter);
        self.reprice(&run, n_ops);
        Some(run)
    }

    /// The **calibration agent**: run *every* registered substrate for a
    /// primitive, measure each, and fold the result into the cost table.
    /// After this, [`dispatch`] picks the truly-cheapest substrate rather
    /// than the shipped Table-tier guess. Returns the number of substrates
    /// measured.
    ///
    /// [`dispatch`]: Runtime::dispatch
    pub fn calibrate<C: CycleSource>(
        &mut self,
        primitive_id: u32,
        data: &[u8],
        n_ops: u64,
        cycles: &C,
        mut meter: Option<&mut dyn EnergyMeter>,
    ) -> usize {
        let subs = self.substrates; // slice ref is Copy → frees `self` for &mut
        let mut measured = 0;
        for s in subs.iter().filter(|s| s.primitive_id == primitive_id) {
            let run = exec(s.id, s.run, primitive_id, data, n_ops, cycles, meter.as_deref_mut());
            self.reprice(&run, n_ops);
            measured += 1;
        }
        measured
    }

    /// Fold a run's measured joules into the cost table (no-op if the run
    /// had no joules or zero ops — we never invent a price).
    fn reprice(&mut self, run: &Run, n_ops: u64) {
        if let Some(j) = run.ledger.joules {
            if n_ops > 0 {
                let _ = self.costs.upsert(CostEntry {
                    primitive_id: self
                        .substrates
                        .iter()
                        .find(|s| s.id == run.substrate)
                        .map(|s| s.primitive_id)
                        .unwrap_or(0),
                    substrate: run.substrate,
                    j_per_op: j / n_ops as f64,
                    source: run.ledger.joules_source,
                });
            }
        }
    }

    /// Cross-check two substrate results — the `check` line the harnesses
    /// emit when racing a hardware substrate against the software one.
    pub fn check(a: u32, b: u32) -> bool {
        a == b
    }
}

/// Run one substrate's work, instrumented: reset the meter, time the call on
/// the cycle source, read joules, build the ledger with honest provenance.
fn exec<C: CycleSource>(
    id: &'static str,
    f: fn(&[u8]) -> u32,
    _primitive_id: u32,
    data: &[u8],
    n_ops: u64,
    cycles: &C,
    // `+ '_`: the trait-object lifetime is independent of the (short) borrow,
    // so a reborrow from the calibration loop is accepted.
    mut meter: Option<&mut (dyn EnergyMeter + '_)>,
) -> Run {
    if let Some(m) = meter.as_deref_mut() {
        m.reset();
    }
    let c0 = cycles.now();
    let result = f(data);
    let mut ledger = EventLedger::new(JoulesSource::Table);
    ledger.cycles = cycles.delta(c0);
    let _ = n_ops;
    if let Some(m) = meter.as_deref_mut() {
        if let Some(j) = m.joules() {
            ledger.joules = Some(j);
            ledger.joules_source = JoulesSource::Measured;
        }
    }
    Run {
        substrate: id,
        result,
        ledger,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A deterministic cycle source: each `now()` advances a fixed step so
    /// `delta` is predictable in tests.
    struct StepCycles {
        cell: core::cell::Cell<u64>,
        step: u64,
    }
    impl StepCycles {
        fn new(step: u64) -> Self {
            Self {
                cell: core::cell::Cell::new(0),
                step,
            }
        }
    }
    impl CycleSource for StepCycles {
        fn now(&self) -> u64 {
            let v = self.cell.get();
            self.cell.set(v + self.step);
            v
        }
        fn delta(&self, start: u64) -> u64 {
            self.cell.get().wrapping_sub(start)
        }
    }

    /// A meter that returns a fixed joules reading (one per substrate, in
    /// registration order) so calibration is deterministic.
    struct FixedMeter {
        readings: &'static [f64],
        i: usize,
    }
    impl EnergyMeter for FixedMeter {
        fn reset(&mut self) {}
        fn joules(&mut self) -> Option<f64> {
            let v = self.readings.get(self.i).copied();
            self.i += 1;
            v
        }
    }

    // Two substrates for primitive 36 (hash): a "software" sum and a
    // "hardware" sum that returns the same value (so check() passes).
    fn sw(d: &[u8]) -> u32 {
        d.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32))
    }
    fn hw(d: &[u8]) -> u32 {
        d.iter().fold(0u32, |a, &b| a.wrapping_add(b as u32))
    }
    const SUBS: &[Substrate] = &[
        Substrate { id: "cpu", primitive_id: 36, run: sw },
        Substrate { id: "accel", primitive_id: 36, run: hw },
    ];

    #[test]
    fn cold_dispatch_picks_first_registered() {
        let rt: Runtime<'_, 8> = Runtime::new(SUBS, CostTable::new());
        // No prices yet → first registered substrate for the primitive.
        assert_eq!(rt.dispatch(36).unwrap().id, "cpu");
        assert!(rt.dispatch(99).is_none());
    }

    #[test]
    fn shipped_table_steers_dispatch() {
        let mut costs = CostTable::<8>::new();
        costs
            .upsert(CostEntry { primitive_id: 36, substrate: "cpu", j_per_op: 9.0e-6, source: JoulesSource::Table })
            .unwrap();
        costs
            .upsert(CostEntry { primitive_id: 36, substrate: "accel", j_per_op: 4.0e-6, source: JoulesSource::Table })
            .unwrap();
        let rt: Runtime<'_, 8> = Runtime::new(SUBS, costs);
        // Cheapest priced substrate wins, even before any measurement.
        assert_eq!(rt.dispatch(36).unwrap().id, "accel");
    }

    #[test]
    fn run_instruments_a_ledger_and_checks() {
        let cyc = StepCycles::new(1000);
        let mut rt: Runtime<'_, 8> = Runtime::new(SUBS, CostTable::new());
        let data = [1u8, 2, 3, 4];
        let a = rt.run(36, &data, 4, &cyc, None).unwrap();
        let b = rt.run(36, &data, 4, &cyc, None).unwrap();
        assert!(a.ledger.cycles > 0);
        assert!(Runtime::<8>::check(a.result, b.result));
        assert_eq!(a.result, 10); // 1+2+3+4
    }

    #[test]
    fn calibration_reprices_and_flips_dispatch() {
        let cyc = StepCycles::new(1000);
        let mut rt: Runtime<'_, 8> = Runtime::new(SUBS, CostTable::new());
        // Cold: dispatch → "cpu" (first registered).
        assert_eq!(rt.dispatch(36).unwrap().id, "cpu");
        // Calibrate: measure cpu @ 8 µJ then accel @ 2 µJ over the 4 ops.
        let mut meter = FixedMeter { readings: &[8.0e-6, 2.0e-6], i: 0 };
        let data = [1u8, 2, 3, 4];
        let n = rt.calibrate(36, &data, 4, &cyc, Some(&mut meter));
        assert_eq!(n, 2);
        // Both entries are now Measured-tier and priced from reality…
        let cpu = rt.costs().for_primitive(36).find(|e| e.substrate == "cpu").unwrap();
        assert_eq!(cpu.source, JoulesSource::Measured);
        assert!((cpu.j_per_op - 2.0e-6).abs() < 1e-12); // 8µJ / 4 ops
        // …and dispatch now flips to the truly-cheapest substrate.
        assert_eq!(rt.dispatch(36).unwrap().id, "accel");
    }

    #[test]
    fn run_after_calibration_emits_measured_ledger() {
        let cyc = StepCycles::new(500);
        let mut rt: Runtime<'_, 8> = Runtime::new(SUBS, CostTable::new());
        let data = [9u8; 16];
        let mut meter = FixedMeter { readings: &[1.6e-5], i: 0 };
        let run = rt.run(36, &data, 16, &cyc, Some(&mut meter)).unwrap();
        assert_eq!(run.ledger.joules_source, JoulesSource::Measured);
        assert_eq!(run.ledger.joules, Some(1.6e-5));
        // The cost table learned this substrate's price.
        let e = rt.costs().cheapest(36).unwrap();
        assert_eq!(e.source, JoulesSource::Measured);
        assert!((e.j_per_op - 1.0e-6).abs() < 1e-12); // 16µJ / 16 ops
    }
}
