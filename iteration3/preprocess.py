"""
preprocess.py
=============
Builds the measurement model for the IEEE 14-bus system (pandapower `case14`)
and generates labeled time-series data for two scenario classes:

  1. Normal operation      -- state (bus voltage angles) follows a smooth
                               AR(1) process around the base operating point.
  2. Stealthy FDIA         -- an attacker injects a bias vector a = H @ c
                               into the measurements, where H is the DC
                               state-estimation Jacobian and c is an
                               arbitrary attacker-chosen state perturbation.
                               Because `a` lives exactly in the column space
                               of H, the classic bad-data residual test
                               (||z - H theta_hat||) does NOT change under
                               attack -- this is the "unobservable" stealthy
                               construction from Liu et al. 2011, which is
                               why a purely residual-based defense fails and
                               a learned spatio-temporal detector (CNN+LSTM,
                               as in Niu et al. 2019) is needed.

TWIST vs. the paper: alongside the raw measurement vector z_t, we also
compute an `action_t` feature -- a heuristic "LLM-agent" corrective
generation-redispatch command that reflects local generator telemetry
(e.g. a direct meter/AGC feed at each generation asset) rather than the
centralized SCADA feed that the state estimator (and the attacker) reads.
This mirrors how real FDIAs work in the literature: attackers target the
*measurements feeding the state estimator*, not every independent sensor
in the substation. So z_t can be quietly poisoned while action_t (driven
by the true local state) stays clean. Individually, neither signal is
enough to reliably catch a subtle, low-magnitude stealthy attack -- the
poisoned z_t looks like noise, and action_t alone has no notion of what
the state estimator currently believes. Jointly, though, an attack shows
up as a growing *disagreement* between what the grid claims (z_t) and
what is actually happening (action_t) -- exactly the feature vector
GridLock's ML layer needs to reason about.

Output: data/dataset.npz with arrays
  X_meas   (N, window, meas_dim)   -- normalized measurement time series
  X_action (N, window, action_dim) -- normalized proposed-action time series
  y        (N,)                    -- 1 if the window's *current* (last)
                                       timestep is under attack, else 0
  meta     dict of shapes / normalization stats (saved separately as .npz keys)
"""
import os
import numpy as np
import pandapower as pp
import pandapower.networks as pn

RNG_SEED = 0
WINDOW = 20          # timesteps of history fed to the LSTM
STRIDE = 2           # stride between consecutive windows when slicing episodes
N_EPISODES = 60
EPISODE_LEN = 220
ATTACK_PROB_PER_EPISODE = 0.55
ATTACK_DUR_RANGE = (15, 45)
MEAS_NOISE_STD = 0.01
STATE_AR_RHO = 0.9
STATE_AR_NOISE_STD = 0.02
ATTACK_C_MAGNITUDE = (0.035, 0.09)  # attacker's fake state offset -- close to normal
                                     # AR(1) noise scale so raw-measurement detection
                                     # alone is genuinely hard, not trivial
ATTACK_SPARSITY = 0.35              # fraction of non-slack buses the attacker corrupts
ACTION_GAIN = 3.0
ACTION_NOISE_STD = 0.03


def build_jacobian():
    """Return (H, b, theta0, gen_bus_idx, noref_idx) for IEEE 14-bus DC model.

    Measurement vector z = H @ theta[noref] + b, where:
      - the first 14 rows of H are bus power-injection sensitivities (Bbus)
      - the remaining 20 rows are branch (line/trafo) flow sensitivities (Bf)
    theta0 is the converged base-case voltage-angle vector (radians).
    """
    net = pn.case14()
    pp.rundcpp(net)
    internal = net._ppc["internal"]
    Bbus = np.asarray(internal["Bbus"].todense())
    Bf = np.asarray(internal["Bf"].todense())
    Pbusinj = np.asarray(internal["Pbusinj"]).flatten()
    Pfinj = np.asarray(internal["Pfinj"]).flatten()
    ref = int(internal["ref"][0])

    n_bus = Bbus.shape[0]
    noref_idx = [i for i in range(n_bus) if i != ref]

    H = np.vstack([Bbus[:, noref_idx], Bf[:, noref_idx]])
    b = np.concatenate([Pbusinj, Pfinj])

    theta0_deg = net.res_bus.va_degree.values
    theta0 = np.deg2rad(theta0_deg)

    gen_bus_idx = net.gen.bus.values.astype(int).tolist()
    return H, b, theta0, gen_bus_idx, noref_idx, ref


def simulate_episode(H, b, theta0, gen_bus_idx, noref_idx, rng, episode_len):
    """Simulate one episode: AR(1) state trajectory, optional stealthy FDIA window.

    Returns:
      Z        (T, meas_dim)   noisy measurements
      A        (T, action_dim) heuristic proposed-action features
      attacked (T,)            binary per-timestep attack label
    """
    n_state = len(noref_idx)
    theta0_noref = theta0[noref_idx]

    # --- normal AR(1) state trajectory around the base operating point ---
    theta = np.zeros((episode_len, n_state))
    theta[0] = theta0_noref
    for t in range(1, episode_len):
        theta[t] = (
            theta0_noref
            + STATE_AR_RHO * (theta[t - 1] - theta0_noref)
            + rng.normal(0, STATE_AR_NOISE_STD, size=n_state)
        )

    attacked = np.zeros(episode_len, dtype=np.int64)
    attack_bias = np.zeros((episode_len, H.shape[0]))

    if rng.random() < ATTACK_PROB_PER_EPISODE:
        dur = rng.integers(*ATTACK_DUR_RANGE)
        start = rng.integers(0, max(1, episode_len - dur))
        end = start + dur
        attacked[start:end] = 1

        # Sparse attacker-chosen fake state offset c (stealthy: a = H @ c)
        n_targets = max(1, int(ATTACK_SPARSITY * n_state))
        target_idx = rng.choice(n_state, size=n_targets, replace=False)
        c = np.zeros(n_state)
        mags = rng.uniform(*ATTACK_C_MAGNITUDE, size=n_targets)
        signs = rng.choice([-1.0, 1.0], size=n_targets)
        c[target_idx] = mags * signs
        a = H @ c  # lives in col-space of H -> invisible to residual test

        # ramp the injection in/out over a few steps for a touch of realism
        ramp = np.ones(dur)
        ramp_len = min(4, dur // 2)
        if ramp_len > 0:
            ramp[:ramp_len] = np.linspace(0.2, 1.0, ramp_len)
            ramp[-ramp_len:] = np.linspace(1.0, 0.2, ramp_len)
        attack_bias[start:end] = np.outer(ramp, a)

    # --- measurements ---
    Z_clean = theta @ H.T + b
    Z = Z_clean + attack_bias + rng.normal(0, MEAS_NOISE_STD, size=Z_clean.shape)

    # --- heuristic "LLM agent" proposed action ---
    # Modeled as a local-telemetry-driven proportional redispatch signal at
    # each gen bus: it reacts to the TRUE injection at that bus (as if read
    # from a local meter/AGC feed independent of the centralized SCADA/state
    # -estimation channel that `Z` represents), plus its own sensor noise.
    # This is what makes the feature genuinely informative jointly with Z:
    # under a stealthy FDIA, Z drifts from the true state while A does not,
    # so the *pair* diverges even though neither alone need look anomalous.
    n_bus = len(theta0)
    H_bus = H[:n_bus, :]  # bus-injection rows of the Jacobian
    true_inj = theta @ H_bus.T + b[:n_bus]  # (T, n_bus), computed from TRUE theta (no attack_bias)
    A = np.zeros((episode_len, len(gen_bus_idx)))
    for gi, bus in enumerate(gen_bus_idx):
        A[:, gi] = -ACTION_GAIN * true_inj[:, bus]
    A += rng.normal(0, ACTION_NOISE_STD, size=A.shape)

    return Z, A, attacked


def slice_windows(Z, A, attacked, window, stride):
    T = Z.shape[0]
    Xz, Xa, y = [], [], []
    for end in range(window, T + 1, stride):
        start = end - window
        Xz.append(Z[start:end])
        Xa.append(A[start:end])
        y.append(attacked[end - 1])
    return np.stack(Xz), np.stack(Xa), np.array(y)


def main():
    rng = np.random.default_rng(RNG_SEED)
    H, b, theta0, gen_bus_idx, noref_idx, ref = build_jacobian()
    print(f"[preprocess] measurement dim={H.shape[0]}, state dim={H.shape[1]}, "
          f"gen buses={gen_bus_idx}")

    all_Xz, all_Xa, all_y = [], [], []
    n_attack_episodes = 0
    for ep in range(N_EPISODES):
        Z, A, attacked = simulate_episode(H, b, theta0, gen_bus_idx, noref_idx, rng, EPISODE_LEN)
        if attacked.any():
            n_attack_episodes += 1
        Xz, Xa, y = slice_windows(Z, A, attacked, WINDOW, STRIDE)
        all_Xz.append(Xz)
        all_Xa.append(Xa)
        all_y.append(y)

    X_meas = np.concatenate(all_Xz, axis=0).astype(np.float32)
    X_action = np.concatenate(all_Xa, axis=0).astype(np.float32)
    y = np.concatenate(all_y, axis=0).astype(np.int64)

    print(f"[preprocess] {n_attack_episodes}/{N_EPISODES} episodes contained an attack")
    print(f"[preprocess] windows: {X_meas.shape[0]}, positive rate: {y.mean():.3f}")
    print(f"[preprocess] X_meas {X_meas.shape}, X_action {X_action.shape}")

    # normalize (z-score) using training-style global stats; store stats for reuse
    meas_mean, meas_std = X_meas.mean(axis=(0, 1)), X_meas.std(axis=(0, 1)) + 1e-8
    act_mean, act_std = X_action.mean(axis=(0, 1)), X_action.std(axis=(0, 1)) + 1e-8
    X_meas_n = (X_meas - meas_mean) / meas_std
    X_action_n = (X_action - act_mean) / act_std

    os.makedirs("data", exist_ok=True)
    np.savez(
        "data/dataset.npz",
        X_meas=X_meas_n,
        X_action=X_action_n,
        y=y,
        meas_mean=meas_mean, meas_std=meas_std,
        act_mean=act_mean, act_std=act_std,
        meas_dim=H.shape[0], action_dim=len(gen_bus_idx), window=WINDOW,
    )
    print("[preprocess] saved data/dataset.npz")


if __name__ == "__main__":
    main()
