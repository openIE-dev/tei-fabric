//! Analytic + published-number validation per docs/SIM-ROADMAP.md §3.2.
//!
//! No foreign-tool fixtures — every expected value below is a closed-form
//! solution of the LIF model or a qualitative bracket of the Brunel 2000
//! analytic phase diagram. Time is in seconds, voltage in arbitrary
//! self-consistent units (mV-like), current scaled so R = 1.

use tei_sim_core::exec::{Executor, Progress};
use tei_sim_core::rng::Rng;
use tei_sim_spiking::lif::{self, NeuronParams};
use tei_sim_spiking::network::NetworkBuilder;
use tei_sim_spiking::stdp::{StdpConfig, StdpState};
use tei_sim_spiking::{ConnSpec, LayerSpec, SpikingExecutor, SpikingJob};

fn base_params() -> NeuronParams {
    NeuronParams {
        tau: 0.020, // 20 ms
        v_rest: 0.0,
        v_reset: 0.0,
        v_th: 20.0,
        t_ref: 0.0,
        r: 1.0,
    }
}

/// (a) Membrane trajectory under constant subthreshold current matches the
/// closed-form charging curve v(t) = v_rest + R·I·(1 − exp(−t/τ)) to ~1e-9.
#[test]
fn membrane_trajectory_matches_closed_form() {
    let p = base_params();
    let dt = 1e-4; // 0.1 ms
    let i = 15.0; // R·I = 15 < v_th = 20 → subthreshold
    let n_steps = 4000; // 400 ms ≫ τ
    let trace = lif::membrane_trace(&p, i, dt, n_steps, p.v_rest);

    for &k in &[1usize, 10, 50, 200, 1000, 2000, 3999] {
        let t = (k + 1) as f64 * dt; // membrane_trace[k] is v after k+1 steps
        let v_sim = trace[k];
        let v_exact = p.v_rest + p.r * i * (1.0 - (-t / p.tau).exp());
        let rel = (v_sim - v_exact).abs() / v_exact.abs();
        assert!(
            rel < 1e-9,
            "step {k}: sim {v_sim} vs exact {v_exact} (rel {rel:e})"
        );
    }
}

/// (b) f–I curve: measured firing rate matches f = 1/T (t_ref = 0) within the
/// dt-induced quantization band. The discrete detector fires at the first
/// timestep ≥ T, so the measured period is ceil(T/dt)·dt ∈ [T, T+dt), i.e. the
/// rate lies in (1/(T+dt), 1/T].
#[test]
fn fi_curve_within_quantization_band() {
    let p = base_params();
    let dt = 1e-4;
    let n_steps = 200_000; // 20 s — plenty of cycles
    for &i in &[25.0, 30.0, 40.0, 60.0, 100.0] {
        let t_period = p.analytic_period(i).expect("suprathreshold");
        let f_hi = 1.0 / t_period; // measured period ≥ T
        let f_lo = 1.0 / (t_period + dt); // measured period < T + dt
        let f_sim = lif::measured_rate(&p, i, dt, n_steps);
        assert!(
            f_sim <= f_hi + 1e-6 && f_sim >= f_lo - 1e-6,
            "I={i}: f_sim {f_sim} not in [{f_lo}, {f_hi}] (T={t_period})"
        );
    }
}

/// (c) Refractory cap: with huge drive and t_ref > 0 the rate is bounded by
/// 1/t_ref and approaches it. With t_ref a whole number of timesteps the period
/// is exactly (t_ref/dt + 1)·dt, so 1/(t_ref + dt) ≤ f ≤ 1/t_ref.
#[test]
fn refractory_rate_cap() {
    let dt = 1e-4;
    let t_ref = 0.005; // 5 ms = 50 dt
    let p = NeuronParams {
        t_ref,
        ..base_params()
    };
    let i = 1.0e6; // enormous drive: one step suffices to cross threshold
    let n_steps = 200_000;
    let f_sim = lif::measured_rate(&p, i, dt, n_steps);
    let cap = 1.0 / t_ref;
    assert!(f_sim <= cap + 1e-9, "rate {f_sim} exceeds cap {cap}");
    // Close to the cap: period = t_ref + dt ⇒ f = 1/(t_ref+dt).
    let expected = 1.0 / (t_ref + dt);
    assert!(
        (f_sim - expected).abs() < 1e-6,
        "rate {f_sim} vs expected {expected}"
    );
    assert!(
        f_sim > 1.0 / (t_ref + 2.0 * dt),
        "rate {f_sim} too far below cap"
    );
}

/// Build a Brunel-style sparse random E/I network with relative inhibition `g`
/// and run it. Returns (mean_rate_hz, mean_cv_isi).
///
/// Parameters follow Brunel 2000 (delta synapses, mV/ms scaled to s):
/// N_E = 1000, N_I = 250, ε = 0.1 connectivity, τ_m = 20 ms, V_th = 20,
/// V_reset = 10, t_ref = 2 ms, delay = 1.5 ms, J = 0.1, external Poisson drive.
fn run_brunel(g: f64, seed: u64) -> (f64, f64) {
    let dt = 1e-4; // 0.1 ms
    let tau = 0.020;
    let j = 0.5;
    let delay = 15; // 1.5 ms / dt
    let n_e = 1000;
    let n_i = 250;
    let eps = 0.1;
    // External drive at ~the threshold rate (η ≈ 1): with delta synapses the
    // mean membrane from external input is J·ν_ext·τ, so ν_ext = η·V_th/(J·τ).
    // Sitting the mean drive near threshold makes the network fluctuation-driven,
    // which is what exposes the low-rate irregular (inhibition-dominated) vs
    // high-rate (excitation-dominated) bracket on this small network.
    let eta = 1.0;
    let nu_ext = eta * 20.0 / (j * tau);

    let neuron = NeuronParams {
        tau,
        v_rest: 0.0,
        v_reset: 10.0,
        v_th: 20.0,
        t_ref: 0.002,
        r: 1.0,
    };

    let mut b = NetworkBuilder::new(dt);
    let exc = tei_sim_spiking::PopulationSpec {
        name: "E".into(),
        n: n_e,
        params: neuron.clone(),
        i_ext: 0.0,
        poisson_rate: nu_ext,
        poisson_weight: j,
    };
    let inh = tei_sim_spiking::PopulationSpec {
        name: "I".into(),
        n: n_i,
        params: neuron,
        i_ext: 0.0,
        poisson_rate: nu_ext,
        poisson_weight: j,
    };
    let pe = b.add_population(exc);
    let pi = b.add_population(inh);

    let mut rng = Rng::new(seed);
    // Excitatory outgoing: weight +J.
    b.connect_random(pe, pe, eps, j, delay, &mut rng);
    b.connect_random(pe, pi, eps, j, delay, &mut rng);
    // Inhibitory outgoing: weight −g·J.
    b.connect_random(pi, pe, eps, -g * j, delay, &mut rng);
    b.connect_random(pi, pi, eps, -g * j, delay, &mut rng);

    let net = b.build();
    let n_steps = 6000; // 600 ms — enough ISIs per neuron for a stable CV
    let mut nop = |_: Progress| {};
    let res = net.run(n_steps, &mut rng, &mut nop);
    (res.mean_rate_hz(), res.mean_cv_isi())
}

/// (d) Brunel 2000 sanity: strong inhibition ⇒ low, irregular rate (CV > 0.5);
/// weak inhibition ⇒ substantially higher rate. A robust qualitative bracket
/// of the published phase diagram, deterministic for a fixed seed.
#[test]
fn brunel_inhibition_brackets_rate() {
    let (rate_strong, cv_strong) = run_brunel(8.0, 12345);
    let (rate_weak, _cv_weak) = run_brunel(2.0, 12345);

    // Inhibition-dominated: low mean rate, irregular spiking.
    assert!(
        rate_strong < 50.0,
        "strong-inhibition rate {rate_strong} Hz not low"
    );
    assert!(
        cv_strong > 0.5,
        "strong-inhibition CV {cv_strong} not irregular (>0.5)"
    );
    // Weak inhibition: substantially higher rate.
    assert!(
        rate_weak > 3.0 * rate_strong,
        "weak rate {rate_weak} not ≫ strong rate {rate_strong}"
    );
}

/// (e) Ledger consistency: sops counted by the simulator equals the sum over
/// every spike of the firing neuron's out-degree, on a tiny deterministic
/// network with no randomness in connectivity.
#[test]
fn ledger_sops_equals_spikes_times_fanout() {
    let dt = 1e-4;
    // Three neurons; neuron 0 driven suprathreshold, 1 & 2 subthreshold but
    // receive from 0. Build explicit synapses with known fan-out.
    let p_drive = NeuronParams {
        v_reset: 0.0,
        ..base_params()
    };
    let p_quiet = NeuronParams {
        v_th: 1.0e9, // never fires on its own or from input
        ..base_params()
    };

    let mut b = NetworkBuilder::new(dt);
    let driver = b.add_population(tei_sim_spiking::PopulationSpec {
        name: "driver".into(),
        n: 1,
        params: p_drive,
        i_ext: 40.0, // suprathreshold
        poisson_rate: 0.0,
        poisson_weight: 0.0,
    });
    let _quiet = b.add_population(tei_sim_spiking::PopulationSpec {
        name: "quiet".into(),
        n: 2,
        params: p_quiet,
        i_ext: 0.0,
        poisson_rate: 0.0,
        poisson_weight: 0.0,
    });
    let _ = driver;
    // Neuron 0 → neuron 1 and neuron 0 → neuron 2: out-degree(0) = 2.
    b.connect_explicit(0, 1, 0.5, 1);
    b.connect_explicit(0, 2, 0.5, 1);

    let net = b.build();
    assert_eq!(net.out_degree(0), 2);
    let mut rng = Rng::new(1);
    let mut nop = |_: Progress| {};
    let res = net.run(5000, &mut rng, &mut nop);

    // Only neuron 0 fires; out-degree 2.
    let expected_sops: u64 = (0..net.n)
        .map(|i| res.spike_times[i].len() as u64 * net.out_degree(i) as u64)
        .sum();
    assert!(res.ledger.spikes > 0, "driver never fired");
    assert_eq!(res.ledger.sops, expected_sops, "sops bookkeeping mismatch");
    // Neurons 1 and 2 never fire (threshold is astronomically high).
    assert_eq!(res.spike_times[1].len(), 0);
    assert_eq!(res.spike_times[2].len(), 0);
    // Each driver spike contributes its fan-out of 2.
    assert_eq!(res.ledger.sops, res.ledger.spikes * 2);
}

/// (f) Determinism: identical job + seed ⇒ identical spike count and ledger.
#[test]
fn determinism_same_seed_same_spikes() {
    let job = SpikingJob {
        layers: vec![
            LayerSpec {
                name: "E".into(),
                n: 200,
                tau: 0.020,
                v_rest: 0.0,
                v_reset: 10.0,
                v_th: 20.0,
                t_ref: 0.002,
                r: 1.0,
                i_ext: 0.0,
                poisson_rate: 2.0 * 20.0 / (0.1 * 0.020),
                poisson_weight: 0.1,
            },
            LayerSpec {
                name: "I".into(),
                n: 50,
                tau: 0.020,
                v_rest: 0.0,
                v_reset: 10.0,
                v_th: 20.0,
                t_ref: 0.002,
                r: 1.0,
                i_ext: 0.0,
                poisson_rate: 2.0 * 20.0 / (0.1 * 0.020),
                poisson_weight: 0.1,
            },
        ],
        connections: vec![
            ConnSpec {
                pre: 0,
                post: 0,
                probability: 0.1,
                weight: 0.1,
                delay: 15,
            },
            ConnSpec {
                pre: 0,
                post: 1,
                probability: 0.1,
                weight: 0.1,
                delay: 15,
            },
            ConnSpec {
                pre: 1,
                post: 0,
                probability: 0.1,
                weight: -0.5,
                delay: 15,
            },
            ConnSpec {
                pre: 1,
                post: 1,
                probability: 0.1,
                weight: -0.5,
                delay: 15,
            },
        ],
        duration: 0.2,
        dt: 1e-4,
        seed: 777,
    };

    let exec = SpikingExecutor;
    let mut nop = |_: Progress| {};
    let a = exec.execute(&job, &mut nop);
    let b = exec.execute(&job, &mut nop);
    assert_eq!(a.ledger.spikes, b.ledger.spikes, "spike counts differ");
    assert_eq!(a.ledger.sops, b.ledger.sops, "sops differ");
    assert!(a.ledger.spikes > 0, "no spikes — drive too weak");
    assert_eq!(
        a.outputs["raster_sample"], b.outputs["raster_sample"],
        "rasters differ"
    );
}

/// STDP window (Bi & Poo 1998): pre-before-post potentiates, post-before-pre
/// depresses, the window is asymmetric and decays exponentially, and the online
/// trace implementation reproduces the closed-form window for an isolated pair.
#[test]
fn stdp_window_asymmetry_and_traces() {
    let cfg = StdpConfig {
        a_plus: 0.01,
        a_minus: 0.012, // depression slightly stronger (Bi & Poo asymmetry)
        tau_plus: 0.017,
        tau_minus: 0.034,
        w_min: 0.0,
        w_max: 1.0,
    };

    // Sign/shape of the analytic window.
    assert!(cfg.window(0.005) > 0.0, "pre-before-post must potentiate");
    assert!(cfg.window(-0.005) < 0.0, "post-before-pre must depress");
    assert!(
        cfg.window(0.005) > cfg.window(0.015),
        "potentiation must decay with |Δt|"
    );
    assert!(
        cfg.window(-0.005).abs() > cfg.window(-0.015).abs(),
        "depression magnitude must decay with |Δt|"
    );

    // Online traces reproduce the window for isolated pairs.
    for &dt in &[0.002_f64, 0.005, 0.010, 0.020, -0.005, -0.012] {
        let mut st = StdpState::new(cfg.clone(), 0.5);
        if dt > 0.0 {
            // pre at 0, post at dt.
            st.on_pre(0.0);
            st.on_post(dt);
        } else {
            // post at 0, pre at |dt|.
            st.on_post(0.0);
            st.on_pre(-dt);
        }
        let dw = st.weight - 0.5;
        let expected = cfg.window(dt);
        assert!(
            (dw - expected).abs() < 1e-12,
            "Δt={dt}: trace Δw {dw} vs window {expected}"
        );
    }
}
