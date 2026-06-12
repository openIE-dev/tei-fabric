# TEI Studio — design constitution

**Status**: synthesis of the 2026-06-12 three-sweep survey (web embedded
tooling · studio HMI patterns · AI-in-the-tool), ~250 sources, adversarially
fact-checked. Companion to EMBEDDED-ROADMAP.md (§3.5 product trio, §8
web-only doctrine). The live preview at fabric.thermoedge.ai/studio is the
seed this document grows.

The one-line thesis the whole survey converged on: **every winning tool
keeps execution deterministic and uses the web (and AI) to collapse the
distance between an action and its visible consequence.** Studio's job is
to make joules that consequence.

---

## 1. The first-run grammar (the five beats)

Every winner (ESPHome Web, setup.particle.io, MakeCode, ViperIDE,
code.circuitpython.org) follows the same script. Studio's version:

1. **Plug in → one big CONNECT button.** The browser port-picker is the
   only chooser. One Connect button, three transports behind it
   (WebSerial / WebUSB / fallback ladder), capability-detected — the
   code.circuitpython.org pattern.
2. **Auto-identify the board** (PICOBOOT/ESP-ROM/VID-PID → an
   EMBEDDED-TARGETS entry): "Pico 2 · W1 · ready to flash teiOS". Never
   a 400-board dropdown.
3. **Flash a known-good default image before discussing anything else**
   (ESPHome's "Prepare for first use").
4. **Provision in the same flow** — Improv Serial on the same CDC port
   the ledger uses; the *device* scans WiFi and the browser shows its
   results (setup.particle.io). For W2/UF2/Safari installs: credentials
   + fabric token baked into the downloaded image (balena/CircuitPython
   settings.toml move) so even a drag-drop install boots provisioned.
5. **End on proof-of-life: the page itself flips Live** when the first
   ledger line arrives (Viam). Flash → connect → stream is one unbroken
   sequence on one screen. Never "now open the console."

**Auth rule**: flash + live ledger require no account, ever. Identity is
requested at exactly one moment — clicking "Publish calibration to the
fabric" — and Golioth-style, the board entry + token are minted in that
same motion.

**The capability ladder is invisible** (MakeCode): one FLASH button whose
mechanism upgrades (UF2 download/drag → WebUSB one-click → teiProbe
bridge) without the UI changing. Users never read a compatibility matrix.

**Distribution**: the forge publishes ESP-Web-Tools-style JSON manifests;
`<tei-install-button manifest=…>` is an embeddable component, so every
board card on the fabric hub — and any partner page or tutorial — carries
a working "Flash teiOS" button (the Adafruit lesson: the flash button
lives on the board's catalog page, not in a separate tool).

## 2. Information architecture

**The left rail IS the pipeline**: Connect → Flash → Ledger → Calibrate →
Publish, in order, with completion states. Navigation doubles as progress.
Everything past Flash is skippable — the live ledger IS the product;
Calibrate/Publish unlock progressively behind it (roadmap doctrine,
confirmed by Edge Impulse's wizard anatomy). Unavailable stages are
greyed-with-reason ("not supported on this board"), never hidden. An
optional first-run wizard overlays the rail; it is never the architecture.

**The persistent provenance bar** (Edge Impulse's budget bar, re-aimed):
every screen carries the connected board, its calibration freshness, and
measured-vs-table J/op status with the `joules_source` badge. This is the
single pattern that makes "joules as a first-class resource" *felt*.
Every action gets priced against it live — including pre-flight estimates
("this calibration ≈ 40 s, ~2.1 J") before anything runs.

**The result pane is always on** (Teachable Machine): the streaming
ledger never leaves the screen — it runs beside flashing, beside
calibration (whose progress is rows appearing in the live table, not a
spinner), beside the cost-table browser.

**Anti-spec** (PlatformIO Home / nRF Connect for Desktop): no
interstitial home screen, no news feed before the trace, no
launcher-that-installs-tools, no network-gated first paint. The URL is
the launcher; `/studio` renders the console instantly and attaches the
stream async. Long-running jobs get a permalink immediately (Qualcomm),
so feedback survives the tab.

## 3. The ledger console HMI (energy-trace spec)

The convergent grammar of PPK2 + Joulescope + Otii, upgraded by the fact
that our event stream is *structured*:

- **Trace**: power/current vs time, filled area, log-Y toggle (sleep↔
  active spans decades), **min/max envelope + mean** rendering at
  zoom-out (1-px spikes must survive — Joulescope's honesty rule),
  newest-at-right live scroll, **pause-to-inspect that never stops
  acquisition**, minimap viewport, cursor-anchored zoom.
- **Region → joules**: one drag → band with two grippable pin handles +
  Δt; a **fixed stat strip** (never a tooltip) with avg · max · Δt ·
  charge · joules for WINDOW and SELECTION scopes, carrying the
  provenance badge.
- **Selections are first-class**: named, colored, annotated ("FFT on
  PIO", "sleep"), persistent — and **the analysis gesture is the
  contribution gesture**: "Publish selection as calibration" turns the
  selected region directly into the `/api/calibration` POST (Otii's
  get-from-selection, aimed at the fabric).
- **The ledger lane**: ledger events (primitive start/stop, dispatch
  decisions, sleep transitions, checks) render as a synchronized track
  under the trace with **bidirectional selection** — click a row, the
  trace highlights; select a region, the rows highlight. Otii syncs
  UART text; we sync structured events. Strictly better.
- **Substrate comparison**: overlay runs color-coded by substrate (M33
  vs Hazard3 vs PIO, same primitive), Alt-drag time-align, comparison
  table per named selection → the table's rows ARE the per-substrate
  J/op entries the dispatcher consumes.
- **The joule odometer**: session accumulator, always visible, explicit
  reset — "this board has spent 1.34 J since flash." Vitals come free
  (Particle): teiOS emits a standard vitals line unprompted; the console
  renders uptime/heap/joule-totals with zero configuration.
- **Anomalies are issues, not log lines** (Memfault): deterministic
  detectors group failed checks and dispatch regressions into
  deduplicated issues with trend sparklines; reboot boundaries render as
  visible gaps (energy continuity must never interpolate across a reset).

## 4. The community calibration store (the fabric hub's economy)

- **Public-by-default IS the free tier** (Roboflow's mechanic): free
  Studio auto-publishes calibrations; privacy (private cost tables, org
  workspaces) is the future paywall.
- **Zero-form publishing**: the POST body is pre-filled from the named
  selection; publishing is one click on the work surface.
- **Every board is a card rendered from the same JSON `/api/dispatch`
  consumes** (HF frontmatter / Qualcomm perf.yaml): machine-readable
  artifact first, page second. Accept **no-fork PR-style community
  submissions** of third-party measurements — the open wedge Qualcomm's
  one-way tables conspicuously lack.
- **Social actions are functional** (Roboflow stars-as-checkpoints):
  starring a community calibration makes it your board's dispatch
  default; use counters surface trust.
- **Publish honest numbers** — CPU fallbacks, Table-tier rows, n_ops,
  board revision, provenance columns. Qualcomm publishes its 58 ms CPU
  fallbacks; credibility compounds from candor. Leaderboards ("lowest
  J/FFT per board dollar") give contributors a reason to measure.

## 5. The AI layer

Mid-2026 verdict: the industry converged on napkin's four-tier invariant
— **LLMs translate in (NL→grammar) and narrate out (evidence-linked
explanation); execution stays deterministic.** Grafana emits PromQL,
Honeycomb emits queries, k8sgpt narrates analyzer output, Arduino's
assistant never drives upload. The architecture is validated; we extend
it.

**The router** (per surface):
- **Tier-0** — the deterministic command grammar (`flash pico2`,
  `calibrate fft on pio`, `compare cpu vs pio for fft`). The contract.
- **Tier-0.5** *(new)* — Chrome/Edge built-in Prompt API with JSON-schema
  `responseConstraint` (Gemini Nano / Phi-4-mini): zero-download NL→
  grammar for the majority browser.
- **Tier-1** — in-browser WebGPU model, opt-in: **LFM2.5-350M Q4 ONNX
  (276 MB)**, officially aimed at structured output/tool use; upgrade
  path = a FunctionGemma-270M fine-tune on the TEI grammar once it
  stabilizes (closed-grammar fine-tunes jump 58→85% on function calling).
  Grammar-constrained decoding via WebLLM+xgrammar, or validate-and-retry
  against the parser.
- **Tier-2** — Claude API for deep narration/planning.
- **Tier-3** *(new transport)* — an MCP server exposing `tei` ops
  (the Nordic/ESP-IDF/Wokwi precedent), so users' own agents become a
  free integration surface — with flash/erase as elicitation-gated tools.

**v1 features, ranked** (effort → payoff):
1. **Command-K: NL → grammar** — fuzzy in, the parsed exact command
   rendered for confirmation, Enter executes. Never a chat bubble.
2. **Dispatch-decision + anomaly explanation** — *the unowned niche*
   (Memfault has no LLM features; nobody explains scheduler decisions).
   Precondition is deterministic: the dispatcher logs a structured
   decision record (candidates, J/op each, chosen, margin); detectors
   flag anomalies statistically; the LLM only narrates from the record,
   linking the exact ledger rows. Click an anomaly marker on the joule
   trace → an evidence-linked card. **"The energy ledger that explains
   itself."**
3. **Calibration-plan generation** — "characterize this board" → a
   sidebar plan document (sequenced runs, per-step projected J/time,
   enumerable from stack.json × the board's substrates), approve-then-
   run with live check-off. Roboflow's sample-before-batch: run one
   primitive first, show the projected total, then commit.
4. **Dispatch advisor** — ships v1 as a *deterministic* inline badge
   ("PIO measured 0.31× — switch?"); LLM narration later. Ghost-text
   register: glanceable, dismissible, never modal.
5. **NL → forge recipe** — v2, generate-diff-commit shape (v0/bolt).

**Physical-action rules** (the risk taxonomy + Claude Code ask-rule
semantics): reads auto-run; calibration runs are preview-then-one-click;
**flash, erase, power-cycle, publish are ask-rules that override any
allowlist** — dialogs state board id, image name+hash, what's
overwritten. The AI can *stage* a flash, never initiate one; the browser
port-picker stays as the final physical-consent layer by design. AI
provenance mirrors ledger provenance: anything AI-proposed carries
`origin: ai-proposed, approved-by: user` (Edge Impulse's `ai-labeled`
discipline).

**Deterministic forever**: the grammar and its executor; the dispatch
rule (lowest measured joule wins — auditable arithmetic, the product
thesis); anomaly *detection*; the ledger and its provenance fields (AI
never writes a ledger row); the dispatch decision record the explainer
narrates from.

## 6. Zero-hardware first run

The SIMULATE button grows up: Wokwi proved simulator-first pedagogy, and
tei-sim-wasm already runs fabric columns in the browser. A simulated
teiOS board (the wasm build emitting the same JSON-lines protocol)
makes the entire tutorial — connect, ledger, selection, publish-dry-run
— work on a phone with no board shipped. The Edge Impulse QR trick
generalizes: a QR hands the session to a phone for boards connected to
another machine.

## 7. Build order (v1.1 from the live preview)

1. Region-select → joules stat strip + named selections on the console
   (the analysis/contribution gesture).
2. The five-beat flash flow for Pico 2 (WebUSB PICOBOOT) + ESP32
   (esptool-js), ending on the Live flip; `<tei-install-button>` +
   forge manifests.
3. The provenance bar + ledger event lane with bidirectional selection.
4. Improv Serial provisioning card + credential-baked W2 images;
   publish-from-selection wiring to `/api/calibration` (auth minted at
   that moment, Golioth-style).
5. Command-K with Tier-0 grammar + Tier-0.5 Prompt API; the dispatch
   decision record in teiOS; deterministic anomaly markers.
6. Anomaly/dispatch explanation cards (Tier-1/2); calibration-plan
   sidebar; board cards on the fabric hub rendered from store JSON.

Each step ends demoable on the live page; none requires an account
before step 4's publish moment.
