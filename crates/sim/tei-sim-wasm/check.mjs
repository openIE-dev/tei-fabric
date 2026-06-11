// Native-vs-wasm identity check for tei-sim-wasm.
//
//   cargo run --release -p tei-sim-wasm --example reference > native-ref.tsv
//   wasm-pack build crates/sim/tei-sim-wasm --target nodejs --release --out-dir <node-pkg-dir>
//   node crates/sim/tei-sim-wasm/check.mjs native-ref.tsv <node-pkg-dir>
//
// Runs the same fixture jobs as examples/reference.rs inside the actual
// .wasm (nodejs target), then asserts deep strict equality of every result
// field except ledger.wall_seconds (measured natively, null on wasm). The
// RNG is hand-rolled and documented WASM-identical — this proves it.

import { readFileSync } from "node:fs";
import { createRequire } from "node:module";
import assert from "node:assert/strict";

const [refPath, pkgDir] = process.argv.slice(2);
const require = createRequire(import.meta.url);
const wasm = require(`${pkgDir}/tei_sim_wasm.js`);

// Keep in lockstep with examples/reference.rs.
const JOBS = [
  ["stochastic_petersen", "stochastic",
   '{"problem":{"kind":"petersen"},"schedule":{"sweeps":2000,"beta0":0.1,"beta1":5.0},"seed":42}'],
  ["stochastic_tempering_rr40", "stochastic",
   '{"problem":{"kind":"random_regular","n":40,"degree":3,"seed":7},"schedule":{"sweeps":1500,"beta0":0.1,"beta1":5.0},"seed":42,"tempering":{"replicas":4,"beta_min":0.1,"beta_max":6.0,"swap_interval":10}}'],
  ["adiabatic_charge_recovery", "adiabatic",
   '{"cell":{"kind":"charge_recovery","r_ohm":1000.0,"c_f":1e-9,"v":1.0},"ratios":[1.0,3.1623,10.0,31.623,100.0,316.23,1000.0]}'],
];

const native = new Map(
  readFileSync(refPath, "utf8").trim().split("\n").map((line) => {
    const tab = line.indexOf("\t");
    return [line.slice(0, tab), JSON.parse(line.slice(tab + 1))];
  }),
);

console.log(`wasm version: ${wasm.version()}`);
let failures = 0;
for (const [name, column, job] of JOBS) {
  let ticks = 0;
  let firstTick = null;
  const run = column === "stochastic" ? wasm.run_stochastic : wasm.run_adiabatic;
  const result = JSON.parse(run(job, (tick) => {
    if (ticks === 0) firstTick = JSON.parse(tick);
    ticks += 1;
  }));
  const ref = native.get(name);
  assert.ok(ref, `missing native reference for ${name}`);

  // wall_seconds: measured natively, null on wasm — excluded from identity.
  assert.equal(result.ledger.wall_seconds, null, "wasm wall_seconds is null");
  const nativeWall = ref.ledger.wall_seconds;
  ref.ledger.wall_seconds = null;
  result.ledger.wall_seconds = null;

  try {
    assert.deepStrictEqual(result, ref);
    const o = result.outputs;
    const headline = column === "stochastic"
      ? `best_cut=${o.best_cut} (known_optimum=${o.known_optimum}) best_energy=${o.best_energy} sweeps=${result.ledger.sweeps} flips=${result.ledger.flips}`
      : `fitted_loglog_slope=${o.fitted_loglog_slope} total_joules=${result.ledger.joules} steps=${result.ledger.sweeps} first_e_diss_j=${o.curve[0].e_diss_j}`;
    console.log(`PASS ${name}: ${headline}; ${ticks} ticks (first fraction=${firstTick.fraction}); native wall_seconds=${nativeWall}`);
  } catch (e) {
    failures += 1;
    console.error(`FAIL ${name}: ${e.message}`);
  }
}
if (failures) {
  console.error(`${failures} job(s) diverged`);
  process.exit(1);
}
console.log("native and wasm results are IDENTICAL (all fields except wall_seconds)");
