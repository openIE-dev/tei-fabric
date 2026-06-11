//! Native ↔ wasm identity against the committed reference fixture.
//!
//! `tests/fixtures/native-ref.tsv` is the output of
//! `cargo run --release -p tei-sim-wasm --example reference` on a native
//! host. This test runs the same fixture jobs and asserts every result
//! field except `ledger.wall_seconds` matches exactly:
//!
//! - natively (`cargo test -p tei-sim-wasm`) it proves the fixture is
//!   current — regenerate it if sim outputs intentionally change;
//! - on wasm (`wasm-pack test --node crates/sim/tei-sim-wasm -- --test
//!   identity`) it proves the actual .wasm reproduces the native numbers
//!   bit for bit — the RNG's documented WASM-identical determinism.

use tei_sim_wasm::{FIXTURE_JOBS, run_fixture};

const FIXTURE: &str = include_str!("fixtures/native-ref.tsv");

/// Result JSON → Value with `ledger.wall_seconds` nulled (measured wall
/// time on native hosts, `null` on wasm — excluded from identity).
fn normalized(result_json: &str) -> serde_json::Value {
    let mut v: serde_json::Value = serde_json::from_str(result_json).expect("result parses");
    v["ledger"]["wall_seconds"] = serde_json::Value::Null;
    v
}

#[cfg_attr(not(target_arch = "wasm32"), test)]
#[cfg_attr(target_arch = "wasm32", wasm_bindgen_test::wasm_bindgen_test)]
fn matches_native_reference_fixture() {
    let mut checked = 0;
    for line in FIXTURE.lines() {
        let (name, reference) = line.split_once('\t').expect("name\\tjson lines");
        let (_, column, job) = FIXTURE_JOBS
            .iter()
            .find(|(n, _, _)| *n == name)
            .unwrap_or_else(|| panic!("fixture {name} has no job entry"));

        let mut ticks = 0u64;
        let result = run_fixture(column, job, |_| ticks += 1);
        assert!(ticks > 0, "{name}: progress ticks were forwarded");
        assert_eq!(
            normalized(&result),
            normalized(reference),
            "{name}: this build diverges from the committed native reference"
        );
        checked += 1;
    }
    assert_eq!(checked, FIXTURE_JOBS.len(), "every fixture job checked");
}
