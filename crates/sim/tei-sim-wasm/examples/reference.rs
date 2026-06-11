//! Native reference dumper for the native-vs-wasm identity check.
//!
//! Prints one `name\tresult_json` line per [`tei_sim_wasm::FIXTURE_JOBS`]
//! entry. The committed copy lives at `tests/fixtures/native-ref.tsv` —
//! regenerate it after any change that intentionally alters sim outputs:
//!
//! ```text
//! cargo run --release -p tei-sim-wasm --example reference \
//!     > crates/sim/tei-sim-wasm/tests/fixtures/native-ref.tsv
//! ```
//!
//! `tests/identity.rs` then proves the actual .wasm reproduces it:
//! `wasm-pack test --node crates/sim/tei-sim-wasm -- --test identity`.

use tei_sim_wasm::{FIXTURE_JOBS, run_fixture};

fn main() {
    for (name, column, job) in FIXTURE_JOBS {
        let mut ticks = 0u64;
        let result = run_fixture(column, job, |_| ticks += 1);
        eprintln!("{name}: {ticks} progress ticks");
        println!("{name}\t{result}");
    }
}
