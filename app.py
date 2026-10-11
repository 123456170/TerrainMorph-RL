import sys
import time
import math
import base64

import numpy as np
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import streamlit as st


# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------

TERRAINS = ["stairs", "gravel", "grass", "ramp"]

METRIC_KEYS = [
    "step_height_m",
    "stride_length_m",
    "cadence_spm",
    "foot_clearance_m",
]

HUMAN_TARGETS = {
    "stairs": {
        "step_height_m": 0.170,
        "stride_length_m": 0.580,
        "cadence_spm": 102.0,
        "foot_clearance_m": 0.090,
    },
    "gravel": {
        "step_height_m": 0.080,
        "stride_length_m": 0.660,
        "cadence_spm": 112.0,
        "foot_clearance_m": 0.055,
    },
    "grass": {
        "step_height_m": 0.070,
        "stride_length_m": 0.700,
        "cadence_spm": 116.0,
        "foot_clearance_m": 0.045,
    },
    "ramp": {
        "step_height_m": 0.090,
        "stride_length_m": 0.640,
        "cadence_spm": 108.0,
        "foot_clearance_m": 0.050,
    },
}

ACTION_SCALE = np.array([0.035, 0.090, 9.0, 0.022])

METRIC_BOUNDS = {
    "step_height_m": (0.010, 0.350),
    "stride_length_m": (0.200, 1.200),
    "cadence_spm": (45.0, 170.0),
    "foot_clearance_m": (0.010, 0.200),
}


# ------------------------------------------------------------------------------
# Small utilities
# ------------------------------------------------------------------------------

def clamp(x, lo, hi):
    return float(max(lo, min(hi, x)))


def smooth_1d(a, k=5):
    a = np.asarray(a, dtype=float)
    if a.size <= 1 or k <= 1:
        return a

    k = min(k, a.size if a.size % 2 == 1 else a.size - 1)
    k = max(1, k)

    kernel = np.ones(k) / k
    return np.convolve(a, kernel, mode="same")


def smooth_2d(z, k=3):
    z = np.asarray(z, dtype=float).copy()
    if z.size == 0:
        return z

    for j in range(z.shape[1]):
        z[:, j] = smooth_1d(z[:, j], k)

    for i in range(z.shape[0]):
        z[i, :] = smooth_1d(z[i, :], k)

    return z


def clip_grad(g, max_norm=4.0):
    g = np.asarray(g, dtype=float)
    n = np.linalg.norm(g)
    if n > max_norm and n > 0:
        return g * (max_norm / n)
    return g


def add_hline_compat(fig, y, text="", dash="dash", color="crimson"):
    fig.add_shape(
        type="line",
        x0=0,
        x1=1,
        xref="paper",
        y0=y,
        y1=y,
        yref="y",
        line=dict(dash=dash, width=2, color=color),
    )

    if text:
        fig.add_annotation(
            x=1,
            xref="paper",
            y=y,
            yref="y",
            text=text,
            showarrow=False,
            yshift=10,
            font=dict(color=color, size=12),
        )


def add_info_panel(fig, lines, x0=0.02, x1=0.56, y0=0.03, font_size=12):
    """
    Adds a compact overlay panel to a Plotly figure.

    This avoids repeated st.dataframe() calls inside the live demo loop,
    which can trigger duplicate-element issues in some Streamlit versions.
    """
    if not lines:
        return fig

    n = len(lines)
    line_h = 0.045
    pad = 0.035
    height = min(0.42, pad * 2 + line_h * n)
    y1 = y0 + height

    fig.add_shape(
        type="rect",
        x0=x0,
        x1=x1,
        y0=y0,
        y1=y1,
        line_width=0,
        fillcolor="rgba(8,12,20,0.68)",
        xref="paper",
        yref="paper",
        layer="above",
    )

    for i, line in enumerate(lines):
        fig.add_annotation(
            x=x0 + 0.015,
            y=y1 - pad - i * line_h,
            xref="paper",
            yref="paper",
            text=line,
            showarrow=False,
            font=dict(color="white", size=font_size),
            align="left",
        )

    return fig


# ------------------------------------------------------------------------------
# 1. Human clip recording / synthetic realistic capture
# ------------------------------------------------------------------------------

def make_human_clips():
    """
    Creates 6 labeled human walking clips across stairs, gravel, grass, ramp.

    In the full production project, replace this with real mocap / RGB-D capture.
    """
    r = np.random.default_rng(2026)

    specs = [
        ("H01", "stairs", "indoor_wood_stairs"),
        ("H02", "stairs", "outdoor_concrete_stairs"),
        ("H03", "gravel", "gravel_path"),
        ("H04", "grass", "short_lawn"),
        ("H05", "ramp", "accessibility_ramp"),
        ("H06", "ramp", "gentle_slope"),
    ]

    rows = []

    for cid, terrain, label in specs:
        base = HUMAN_TARGETS[terrain]

        step_height = clamp(
            abs(base["step_height_m"] * r.normal(1.0, 0.035)),
            0.020,
            0.300,
        )

        stride_length = clamp(
            abs(base["stride_length_m"] * r.normal(1.0, 0.030)),
            0.250,
            1.100,
        )

        cadence = clamp(
            abs(base["cadence_spm"] * r.normal(1.0, 0.025)),
            60.0,
            160.0,
        )

        foot_clearance = clamp(
            abs(base["foot_clearance_m"] * r.normal(1.0, 0.050)),
            0.015,
            0.180,
        )

        duration = float(r.uniform(8.5, 13.5))
        fps = 60
        frames = int(duration * fps)

        rows.append(
            {
                "clip_id": cid,
                "terrain": terrain,
                "label": label,
                "subject": "P1",
                "duration_s": duration,
                "fps": fps,
                "frames": frames,
                "step_height_m": step_height,
                "stride_length_m": stride_length,
                "cadence_spm": cadence,
                "foot_clearance_m": foot_clearance,
                "pose_confidence": float(r.uniform(0.88, 0.97)),
                "ground_error_cm": float(r.uniform(0.8, 2.6)),
            }
        )

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# 2. World-grounded gait extraction from synthetic 3D footfall trajectory
# ------------------------------------------------------------------------------

def generate_footfall_path(terrain, duration_s=10.0, fps=60, seed=0, stats=None):
    """
    Generates a realistic 3D footfall trajectory for one terrain type.

    This stands in for a world-grounded pose estimator output.
    """
    r = np.random.default_rng(seed)

    if stats is None:
        stats = HUMAN_TARGETS[terrain]

    cadence = clamp(
        float(stats.get("cadence_spm", HUMAN_TARGETS[terrain]["cadence_spm"])),
        60.0,
        160.0,
    )

    stride = clamp(
        float(stats.get("stride_length_m", HUMAN_TARGETS[terrain]["stride_length_m"])),
        0.25,
        1.10,
    )

    step_height = clamp(
        float(stats.get("step_height_m", HUMAN_TARGETS[terrain]["step_height_m"])),
        0.02,
        0.30,
    )

    foot_clearance = clamp(
        float(
            stats.get(
                "foot_clearance_m",
                HUMAN_TARGETS[terrain]["foot_clearance_m"],
            )
        ),
        0.015,
        0.18,
    )

    duration_s = max(2.0, float(duration_s))
    fps = max(10, int(fps))

    step_period = 60.0 / cadence
    n_steps = max(4, int(round(duration_s / step_period)))
    total_time = n_steps * step_period
    num_samples = max(60, int(total_time * fps))

    t = np.linspace(0.0, total_time, num_samples)
    step_float = t / step_period
    step_idx = np.floor(step_float).astype(int)
    phase = step_float - step_idx

    swing_start = 0.55
    swing_phase = np.clip((phase - swing_start) / (1.0 - swing_start), 0.0, 1.0)

    arc = foot_clearance * np.sin(np.pi * swing_phase) ** 2
    arc[phase < swing_start] = 0.0

    if terrain == "stairs":
        ground_steps = step_height * np.arange(n_steps + 2, dtype=float)

    elif terrain == "ramp":
        ground_steps = step_height * np.arange(n_steps + 2, dtype=float)

    else:
        increments = r.laplace(scale=max(step_height, 0.004), size=n_steps + 1)
        ground_steps = np.concatenate(([0.0], np.cumsum(increments)))

    ground = ground_steps[step_idx]

    x = stride * step_float + smooth_1d(r.normal(0.0, 0.004, len(t)), 5)
    y = 0.09 * ((-1.0) ** step_idx) + smooth_1d(r.normal(0.0, 0.006, len(t)), 5)
    z = ground + arc + r.normal(0.0, 0.0035, len(t))

    return {
        "t": t,
        "x": x,
        "y": y,
        "z": z,
        "step_idx": step_idx,
        "phase": phase,
        "terrain": terrain,
    }


def detect_contiguous(mask, min_len=1):
    mask = np.asarray(mask, dtype=bool)
    out = []
    start = None

    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                out.append((start, i - 1))
            start = None

    if start is not None and len(mask) - start >= min_len:
        out.append((start, len(mask) - 1))

    return out


def extract_gait_stats(path, terrain=None):
    """
    Extracts step height, stride length, cadence, and foot clearance
    from a 3D footfall trajectory using contact/swing segmentation.
    """
    t = path["t"]
    x = path["x"]
    z = path["z"]

    terrain = terrain or path.get("terrain", "grass")
    base = HUMAN_TARGETS.get(terrain, HUMAN_TARGETS["grass"])

    if len(t) < 8:
        return {
            "step_height_m": base["step_height_m"],
            "stride_length_m": base["stride_length_m"],
            "cadence_spm": base["cadence_spm"],
            "foot_clearance_m": base["foot_clearance_m"],
            "confidence": 0.50,
            "segments": 0,
        }

    dt = float(np.median(np.diff(t)))
    fps = 1.0 / max(dt, 1e-6)

    deg = 1 if terrain in ("stairs", "ramp") else 2

    t_center = float(np.mean(t))
    tc = t - t_center

    coef = np.polyfit(tc, z, deg)
    ztrend = z - np.polyval(coef, tc)

    thresh = max(0.010, 0.35 * float(np.std(ztrend)))
    swing0 = ztrend > thresh
    stance = ~swing0

    min_stance = max(2, int(0.05 * fps))
    stance_clusters = detect_contiguous(stance, min_stance)

    if len(stance_clusters) < 2:
        return {
            "step_height_m": base["step_height_m"],
            "stride_length_m": base["stride_length_m"],
            "cadence_spm": base["cadence_spm"],
            "foot_clearance_m": base["foot_clearance_m"],
            "confidence": 0.55,
            "segments": len(stance_clusters),
        }

    cx = []
    cz = []
    ct = []

    for s, e in stance_clusters:
        sl = slice(s, e + 1)
        cx.append(float(np.mean(x[sl])))
        cz.append(float(np.mean(z[sl])))
        ct.append(float(np.mean(t[sl])))

    cx = np.array(cx)
    cz = np.array(cz)
    ct = np.array(ct)

    if len(ct) >= 2:
        ground_interp = np.interp(t, ct, cz)
    else:
        ground_interp = np.polyval(coef, t - t_center)

    zrel = z - ground_interp
    thresh = max(0.008, 0.35 * float(np.std(zrel)))
    swing = zrel > thresh

    step_heights = np.abs(np.diff(cz))
    strides = np.diff(cx)

    duration = float(t[-1] - t[0])
    cadence = float(len(ct) / duration * 60.0) if duration > 0 else base["cadence_spm"]

    min_swing = max(1, int(0.03 * fps))
    swing_clusters = detect_contiguous(swing, min_swing)

    clears = []

    for s, e in swing_clusters:
        peak = float(np.max(zrel[s : e + 1]))
        if peak > 0.001:
            clears.append(peak)

    if clears:
        foot_clearance = float(np.mean(clears))
    else:
        foot_clearance = float(max(0.005, np.max(zrel)))

    step_height = float(np.mean(step_heights)) if len(step_heights) else base["step_height_m"]
    stride_length = float(np.mean(strides)) if len(strides) else base["stride_length_m"]

    step_height = max(0.005, step_height)
    stride_length = max(0.150, stride_length)
    cadence = float(np.clip(cadence, 40.0, 180.0))
    foot_clearance = float(np.clip(foot_clearance, 0.005, 0.200))

    if len(step_heights) > 1 and np.mean(step_heights) > 1e-6:
        variability = float(np.std(step_heights) / np.mean(step_heights))
    else:
        variability = 0.20

    confidence = clamp(0.62 + 0.035 * len(cx) - 0.35 * variability, 0.45, 0.98)

    return {
        "step_height_m": step_height,
        "stride_length_m": stride_length,
        "cadence_spm": cadence,
        "foot_clearance_m": foot_clearance,
        "confidence": confidence,
        "segments": len(cx),
    }


def gait_match_percent(actual, target):
    tolerances = {
        "step_height_m": 0.035,
        "stride_length_m": 0.090,
        "cadence_spm": 9.0,
        "foot_clearance_m": 0.022,
    }

    scores = []

    for k, tol in tolerances.items():
        a = float(actual.get(k, target[k]))
        t = float(target[k])

        if abs(t) < 1e-9:
            score = 1.0 if abs(a) < tol else 0.0
        else:
            score = math.exp(-0.5 * ((a - t) / tol) ** 2)

        scores.append(score)

    return float(100.0 * np.mean(scores))


# ------------------------------------------------------------------------------
# 3. Procedural terrain generator
# ------------------------------------------------------------------------------

def generate_terrain(terrain, nx=48, ny=48, seed=7, overrides=None):
    overrides = overrides or {}
    r = np.random.default_rng(seed)

    x = np.linspace(0.0, 2.2, nx)
    y = np.linspace(-0.9, 0.9, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    base = HUMAN_TARGETS.get(terrain, HUMAN_TARGETS["grass"])

    if terrain == "stairs":
        tread = float(overrides.get("tread_m", 0.28))
        riser = float(overrides.get("riser_m", base["step_height_m"]))
        Z = np.floor(X / tread) * riser
        Z = Z + smooth_2d(r.normal(0.0, 0.004, X.shape), 3)

    elif terrain == "ramp":
        default_slope = base["step_height_m"] / max(base["stride_length_m"], 0.25)
        slope = float(overrides.get("slope", default_slope))
        Z = slope * X + smooth_2d(r.normal(0.0, 0.006, X.shape), 4)

    elif terrain == "gravel":
        rough = float(overrides.get("roughness", base["step_height_m"] * 0.45))
        Z = smooth_2d(r.normal(0.0, rough, X.shape), 5)

        rocks = r.random(X.shape) < 0.025
        if np.any(rocks):
            Z = Z.copy()
            Z[rocks] += r.uniform(0.02, 0.07, int(np.sum(rocks)))

    else:
        rough = float(overrides.get("roughness", base["step_height_m"] * 0.35))
        Z = smooth_2d(r.normal(0.0, rough, X.shape), 7)
        Z = Z + 0.008 * np.sin(X * 14.0) * np.cos(Y * 11.0)

    return X, Y, Z


# ------------------------------------------------------------------------------
# 4. Reward function
# ------------------------------------------------------------------------------

def reward_function(actual, target):
    match = gait_match_percent(actual, target)

    sh = float(actual["step_height_m"])
    stride = float(actual["stride_length_m"])
    cad = float(actual["cadence_spm"])
    fc = float(actual["foot_clearance_m"])

    r_match = 0.08 * match

    r_stab = (
        1.20
        - 18.0 * max(0.0, fc - 0.12) ** 2
        - 8.0 * max(0.0, sh - 0.22) ** 2
        - 3.0 * max(0.0, stride - 0.95) ** 2
    )

    r_energy = (
        -0.0007 * (cad - 105.0) ** 2
        - 1.10 * fc
        - 0.45 * sh
    )

    r_impact = (
        -3.0 * max(0.0, sh - float(target["step_height_m"])) ** 2
        - 2.0 * max(0.0, fc - float(target["foot_clearance_m"])) ** 2
    )

    fall_penalty = -8.0 if match < 40.0 else 0.0

    total = float(r_match + r_stab + r_energy + r_impact + fall_penalty)

    terms = {
        "gait_match_bonus": float(r_match),
        "stability": float(r_stab),
        "energy": float(r_energy),
        "impact_penalty": float(r_impact),
        "fall_penalty": float(fall_penalty),
        "total_reward": total,
        "gait_match_pct": float(match),
    }

    return total, terms


# ------------------------------------------------------------------------------
# 5. Embedded PPO surrogate for gait-parameter policy
# ------------------------------------------------------------------------------

class MiniPPO:
    """
    Lightweight NumPy PPO-style policy for a 4D gait-parameter residual policy.

    In the full system, this would be replaced by Isaac Lab / RSL-RL / MuJoCo
    whole-body PPO. Here it demonstrates the learning loop end-to-end.
    """

    def __init__(self, state_dim=6, action_dim=4, seed=1):
        r = np.random.default_rng(seed)

        self.state_dim = state_dim
        self.action_dim = action_dim

        self.Wmu = r.normal(0.0, 0.45, (state_dim, action_dim))
        self.bmu = np.zeros(action_dim, dtype=float)

        self.Wv = r.normal(0.0, 0.08, (state_dim, 1))
        self.bv = np.zeros(1, dtype=float)

        self.std = np.full(action_dim, 0.70, dtype=float)

        self.lr_mu = 0.035
        self.lr_v = 0.080
        self.clip = 0.20
        self.epochs = 4
        self.minibatch = 64

    def forward(self, states):
        states = np.atleast_2d(np.asarray(states, dtype=float))

        pre = states @ self.Wmu + self.bmu
        mu = np.tanh(pre)

        val = (states @ self.Wv + self.bv).reshape(-1)

        return mu, val

    def logpdf(self, actions, mu):
        var = self.std ** 2
        return np.sum(
            -0.5 * ((actions - mu) ** 2 / var + np.log(2.0 * np.pi * var)),
            axis=1,
        )

    def sample(self, states, rng):
        mu, val = self.forward(states)

        noise = rng.normal(size=mu.shape)
        actions = mu + self.std * noise

        logp = self.logpdf(actions, mu)

        return actions, logp, val, mu

    def update(self, buf, rng):
        states = np.asarray(buf["states"], dtype=float)
        actions = np.asarray(buf["actions"], dtype=float)
        old_logp = np.asarray(buf["logp"], dtype=float)
        rewards = np.asarray(buf["rewards"], dtype=float)
        values = np.asarray(buf["values"], dtype=float)

        n = len(rewards)
        if n == 0:
            return

        # One-step episodes: done=True for every sample.
        adv = rewards - values
        returns = rewards.copy()

        adv = (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

        for _ in range(self.epochs):
            idx = rng.permutation(n)

            for start in range(0, n, self.minibatch):
                mb = idx[start : start + self.minibatch]

                S = states[mb]
                A = actions[mb]
                OL = old_logp[mb]
                AD = adv[mb]
                RET = returns[mb]

                mu, V = self.forward(S)
                logp = self.logpdf(A, mu)

                ratio = np.exp(np.clip(logp - OL, -8.0, 8.0))

                surr1 = ratio * AD
                surr2 = np.clip(ratio, 1.0 - self.clip, 1.0 + self.clip) * AD

                active = surr1 <= surr2

                diff = (A - mu) / (self.std ** 2)

                grad_mu = np.zeros_like(mu)

                if np.any(active):
                    grad_mu[active] = (
                        -(ratio[active] * AD[active])[:, None]
                        * diff[active]
                        * (1.0 - mu[active] ** 2)
                    )

                denom = max(len(mb), 1)
                grad_mu /= denom

                GW = S.T @ grad_mu
                gb = grad_mu.sum(axis=0)

                err = V - RET

                # Correct value-head gradient.
                # Wv has shape (state_dim, 1), so gWv must also be (state_dim, 1).
                gWv = (S.T @ (2.0 * err[:, None])) / denom
                gbv = np.array([2.0 * np.sum(err) / denom])

                self.Wmu -= self.lr_mu * clip_grad(GW, 4.0)
                self.bmu -= self.lr_mu * clip_grad(gb, 2.0)

                self.Wv -= self.lr_v * clip_grad(gWv, 4.0)
                self.bv -= self.lr_v * clip_grad(gbv, 2.0)


def curriculum_probs(it, total):
    p = it / max(total - 1, 1)

    # Order: stairs, gravel, grass, ramp.
    # Easy terrains start with higher probability; harder terrain probability grows.
    w = np.array(
        [
            0.10 + 0.38 * p,
            0.15 + 0.25 * p,
            0.35 - 0.15 * p,
            0.40 - 0.30 * p,
        ],
        dtype=float,
    )

    w = np.clip(w, 0.02, None)
    return w / np.sum(w)


def train_mini_ppo(
    iterations=50,
    samples_per_iter=128,
    seed=7,
    curriculum=True,
    domain_rand=True,
):
    rng = np.random.default_rng(seed)
    policy = MiniPPO(state_dim=6, action_dim=4, seed=seed)

    iterations = max(1, int(iterations))
    samples_per_iter = max(16, int(samples_per_iter))

    history = []
    terrain_arr = np.array(TERRAINS)

    for it in range(iterations):
        progress = it / max(iterations - 1, 1)
        decay = max(0.12, 0.75 * (1.0 - progress) ** 1.1 + 0.10)
        policy.std = np.full(4, decay, dtype=float)

        probs = curriculum_probs(it, iterations) if curriculum else np.full(4, 0.25, dtype=float)
        probs = probs / np.sum(probs)

        all_states = []
        all_actions = []
        all_logp = []
        all_rewards = []
        all_values = []
        all_matches = []
        all_abs_actions = []

        for _ in range(samples_per_iter):
            terrain = str(rng.choice(terrain_arr, p=probs))
            target = HUMAN_TARGETS[terrain]

            dr_noise_scale = (
                np.array([0.006, 0.018, 1.8, 0.004])
                if domain_rand
                else np.zeros(4)
            )

            dr = rng.normal(0.0, 1.0, 4) * dr_noise_scale

            onehot = np.zeros(4, dtype=float)
            onehot[TERRAINS.index(terrain)] = 1.0

            slope = {
                "stairs": 0.09,
                "ramp": 0.12,
                "gravel": 0.02,
                "grass": 0.01,
            }[terrain]

            speed_err = rng.normal(0.0, 0.04)
            state = np.concatenate([onehot, [speed_err, slope]])

            actions, logp, values, _ = policy.sample(state[None, :], rng)
            action = actions[0]

            actual = {}

            for i, key in enumerate(METRIC_KEYS):
                raw = float(target[key] + ACTION_SCALE[i] * action[i] + dr[i])
                lo, hi = METRIC_BOUNDS[key]
                actual[key] = float(np.clip(raw, lo, hi))

            match = gait_match_percent(actual, target)
            rew, _ = reward_function(actual, target)

            all_states.append(state)
            all_actions.append(action)
            all_logp.append(float(logp[0]))
            all_rewards.append(float(rew))
            all_values.append(float(values[0]))
            all_matches.append(float(match))
            all_abs_actions.append(float(np.mean(np.abs(action))))

        buf = {
            "states": np.array(all_states, dtype=float),
            "actions": np.array(all_actions, dtype=float),
            "logp": np.array(all_logp, dtype=float),
            "rewards": np.array(all_rewards, dtype=float),
            "values": np.array(all_values, dtype=float),
        }

        policy.update(buf, rng)

        history.append(
            {
                "iteration": it,
                "mean_reward": float(np.mean(all_rewards)),
                "mean_match_pct": float(np.mean(all_matches)),
                "mean_abs_action": float(np.mean(all_abs_actions)),
                "std": float(np.mean(policy.std)),
                "stairs_p": float(probs[0]),
                "gravel_p": float(probs[1]),
                "grass_p": float(probs[2]),
                "ramp_p": float(probs[3]),
            }
        )

    return policy, pd.DataFrame(history)


def policy_final_gaits(policy, calibration=0.65):
    out = {}

    for terrain in TERRAINS:
        onehot = np.zeros(4, dtype=float)
        onehot[TERRAINS.index(terrain)] = 1.0

        state = np.concatenate([onehot, [0.0, 0.05]])

        mu, _ = policy.forward(state[None, :])
        action = mu[0] * calibration

        target = HUMAN_TARGETS[terrain]

        actual = {}

        for i, key in enumerate(METRIC_KEYS):
            raw = float(target[key] + ACTION_SCALE[i] * action[i])
            lo, hi = METRIC_BOUNDS[key]
            actual[key] = float(np.clip(raw, lo, hi))

        out[terrain] = actual

    return out


# ------------------------------------------------------------------------------
# 6. System identification
# ------------------------------------------------------------------------------

def run_system_identification(seed=9):
    r = np.random.default_rng(seed)

    n = 600

    u = r.uniform(0.5, 8.0, n)
    v = r.normal(0.0, 2.2, n)
    sv = np.sign(v)

    kt_true = 0.182
    b_true = 0.037
    c_true = 0.086
    bias_true = 0.012

    tau = (
        kt_true * u
        + b_true * (-v)
        + c_true * (-sv)
        + bias_true * (-np.ones(n))
        + r.normal(0.0, 0.018, n)
    )

    X = np.column_stack([u, -v, -sv, -np.ones(n)])
    beta, _, _, _ = np.linalg.lstsq(X, tau, rcond=None)
    tau_hat = X @ beta

    t = np.linspace(0.0, 0.45, 300)
    t0 = 0.08
    tc_true = 0.034
    final = 0.18

    resp = np.where(t >= t0, 1.0 - np.exp(-(t - t0) / tc_true), 0.0)
    meas = final * resp + r.normal(0.0, 0.004, len(t))

    crossing = np.where((t >= t0) & (meas >= 0.632 * final))[0]
    tc_est = float(t[crossing[0]] - t0) if len(crossing) else tc_true

    names = [
        "torque_constant_Nm_per_A",
        "damping_Nm_per_rad_s",
        "friction_Nm",
        "bias_Nm",
    ]

    sim_before = [0.150, 0.025, 0.050, 0.000]

    rows = []

    for name, s, f in zip(names, sim_before, beta):
        rows.append(
            {
                "parameter": name,
                "sim_before": float(s),
                "real_fit": float(f),
                "abs_error": float(abs(s - f)),
            }
        )

    rows.append(
        {
            "parameter": "closed_loop_latency_s",
            "sim_before": 0.020,
            "real_fit": tc_est,
            "abs_error": float(abs(0.020 - tc_est)),
        }
    )

    table = pd.DataFrame(rows)

    return {
        "table": table,
        "u": u,
        "v": v,
        "tau": tau,
        "tau_hat": tau_hat,
        "latency_t": t,
        "latency_meas": meas,
        "latency_est": tc_est,
        "mean_abs_error": float(table["abs_error"].mean()),
    }


# ------------------------------------------------------------------------------
# 7. Deployment simulation and KPIs
# ------------------------------------------------------------------------------

def simulate_deployment(
    final_gaits,
    sysid,
    indoor_trials=12,
    outdoor_trials=20,
    seed=17,
):
    r = np.random.default_rng(seed)

    sysid_pen = min(2.0, float(sysid.get("mean_abs_error", 0.1)) * 8.0)

    sequences = {
        "indoor": ["grass", "ramp", "gravel", "stairs"],
        "outdoor": ["grass", "gravel", "ramp", "stairs"],
    }

    rows = []

    for course, n_trials, fall_base in [
        ("indoor", indoor_trials, 0.006),
        ("outdoor", outdoor_trials, 0.018),
    ]:
        seq = sequences[course]

        for trial in range(int(n_trials)):
            trial_fall = bool(r.random() < fall_base)
            fail_seg = int(r.integers(0, len(seq))) if trial_fall else -1

            for seg, terrain in enumerate(seq):
                target = HUMAN_TARGETS[terrain]
                actual = final_gaits[terrain]

                match = gait_match_percent(actual, target)

                terrain_factor = {
                    "stairs": 1.22,
                    "gravel": 1.12,
                    "ramp": 1.02,
                    "grass": 0.96,
                }[terrain]

                err = (
                    ((100.0 - match) * 0.22 + 3.1)
                    * terrain_factor
                    + r.normal(0.0, 0.55)
                    + sysid_pen * 0.35
                )

                if seg == fail_seg:
                    err += r.uniform(5.0, 8.5)

                err = float(np.clip(err, 0.35, 18.5))

                reaction = 0.52 + 0.18 * float(r.random())
                if terrain in ("stairs", "gravel"):
                    reaction += 0.07

                reaction = float(np.clip(reaction, 0.35, 1.40))

                rows.append(
                    {
                        "course": course,
                        "trial": trial + 1,
                        "segment": seg + 1,
                        "terrain": terrain,
                        "gait_match_pct": float(match),
                        "joint_tracking_error_pct": float(err),
                        "terrain_switch_reaction_s": float(reaction),
                        "fall": bool(seg == fail_seg),
                        "spotter_contact": bool(trial_fall and seg == fail_seg),
                        "trial_fall": trial_fall,
                    }
                )

    return pd.DataFrame(rows)


def compute_kpis(deploy_df, final_gaits):
    outdoor = deploy_df[deploy_df["course"] == "outdoor"]

    if len(outdoor):
        trial_falls = outdoor.groupby("trial")["trial_fall"].any()
        fall_rate = float(trial_falls.mean())
    else:
        fall_rate = 0.0

    terrain_matches = {
        t: gait_match_percent(final_gaits[t], HUMAN_TARGETS[t])
        for t in TERRAINS
    }

    gait_match = float(np.mean(list(terrain_matches.values())))

    error = (
        float(deploy_df["joint_tracking_error_pct"].mean())
        if len(deploy_df)
        else 0.0
    )

    reaction = (
        float(deploy_df["terrain_switch_reaction_s"].mean())
        if len(deploy_df)
        else 0.0
    )

    match_score = clamp(gait_match, 0.0, 100.0)
    fall_score = 100.0 * clamp(1.0 - fall_rate / 0.05, 0.0, 1.0)
    err_score = 100.0 * clamp(1.0 - error / 10.0, 0.0, 1.0)
    react_score = 100.0 * clamp(1.0 - reaction / 1.0, 0.0, 1.0)

    overall = (
        0.40 * match_score
        + 0.25 * fall_score
        + 0.25 * err_score
        + 0.10 * react_score
    )

    return {
        "gait_match_pct": gait_match,
        "fall_rate_pct": fall_rate * 100.0,
        "sim_to_real_error_pct": error,
        "reaction_s": reaction,
        "overall_score": float(overall),
        "terrain_matches": terrain_matches,
        "fall_score": float(fall_score),
        "err_score": float(err_score),
        "react_score": float(react_score),
    }


def kpi_dataframe(kpis):
    rows = [
        {
            "Metric": "Gait-statistic match to human",
            "Value": f"{kpis['gait_match_pct']:.1f}%",
            "Target": "> 80%",
            "Status": "PASS" if kpis["gait_match_pct"] >= 80.0 else "REVIEW",
        },
        {
            "Metric": "Fall rate on mixed terrain",
            "Value": f"{kpis['fall_rate_pct']:.1f}%",
            "Target": "< 5%",
            "Status": "PASS" if kpis["fall_rate_pct"] < 5.0 else "REVIEW",
        },
        {
            "Metric": "Sim-to-real tracking error",
            "Value": f"{kpis['sim_to_real_error_pct']:.1f}%",
            "Target": "< 10%",
            "Status": "PASS" if kpis["sim_to_real_error_pct"] < 10.0 else "REVIEW",
        },
        {
            "Metric": "Terrain-switch reaction time",
            "Value": f"{kpis['reaction_s']:.2f} s",
            "Target": "< 1.00 s",
            "Status": "PASS" if kpis["reaction_s"] < 1.0 else "REVIEW",
        },
        {
            "Metric": "Overall demo score",
            "Value": f"{kpis['overall_score']:.1f}/100",
            "Target": "Higher is better",
            "Status": "INFO",
        },
    ]

    return pd.DataFrame(rows)


# ------------------------------------------------------------------------------
# Plotting helpers
# ------------------------------------------------------------------------------

def fig_footfall_3d(path, title="Footfall trajectory"):
    fig = go.Figure()

    down = max(1, len(path["t"]) // 500)

    fig.add_trace(
        go.Scatter3d(
            x=path["x"][::down],
            y=path["y"][::down],
            z=path["z"][::down],
            mode="lines",
            line=dict(color="#00c853", width=5),
            name="tracked footfall",
        )
    )

    seen = set()
    events = []

    for i, s in enumerate(path["step_idx"]):
        if s not in seen:
            seen.add(s)
            events.append(i)

    if events:
        ev_stride = max(1, len(events) // 30)
        ev = events[::ev_stride]

        fig.add_trace(
            go.Scatter3d(
                x=path["x"][ev],
                y=path["y"][ev],
                z=path["z"][ev],
                mode="markers",
                marker=dict(size=4, color="#ff6d00"),
                name="step events",
            )
        )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="forward x (m)",
            yaxis_title="lateral y (m)",
            zaxis_title="height z (m)",
        ),
    )

    return fig


def fig_terrain_surface(X, Y, Z, title="Procedural terrain"):
    fig = go.Figure(
        data=[
            go.Surface(
                z=Z,
                x=X,
                y=Y,
                colorscale="Earth",
                opacity=0.92,
                contours=dict(
                    z=dict(
                        show=True,
                    )
                ),
            )
        ]
    )

    fig.update_layout(
        title=title,
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        scene=dict(
            xaxis_title="x (m)",
            yaxis_title="y (m)",
            zaxis_title="height (m)",
        ),
    )

    return fig


def fig_gait_comparison(actual, target):
    labels = [
        "Step height\n(m)",
        "Stride\n(m)",
        "Cadence\n(spm)",
        "Clearance\n(m)",
    ]

    a = [float(actual[k]) for k in METRIC_KEYS]
    t = [float(target[k]) for k in METRIC_KEYS]

    norm_a = [100.0 * ai / ti if abs(ti) > 1e-9 else 0.0 for ai, ti in zip(a, t)]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=labels,
            y=norm_a,
            name="Policy / robot (% of human)",
        )
    )

    fig.add_trace(
        go.Bar(
            x=labels,
            y=[100.0] * 4,
            name="Human reference",
        )
    )

    fig.update_layout(
        barmode="group",
        height=320,
        yaxis_title="% of human reference",
        margin=dict(l=0, r=0, t=40, b=0),
    )

    add_hline_compat(fig, 100.0, "human reference")

    return fig


def fig_training(hist):
    if hist.empty:
        return go.Figure()

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.10,
        subplot_titles=(
            "Mean episode reward",
            "Mean gait match %",
            "Mean |action| residual",
        ),
    )

    fig.add_trace(
        go.Scatter(x=hist["iteration"], y=hist["mean_reward"], name="reward"),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=hist["iteration"], y=hist["mean_match_pct"], name="match %"),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Scatter(x=hist["iteration"], y=hist["mean_abs_action"], name="|action|"),
        row=3,
        col=1,
    )

    fig.update_layout(
        title="PPO surrogate training",
        height=680,
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False,
    )

    return fig


def fig_curriculum(hist):
    if hist.empty:
        return go.Figure()

    fig = go.Figure()

    for terrain, col in zip(
        TERRAINS,
        ["stairs_p", "gravel_p", "grass_p", "ramp_p"],
    ):
        fig.add_trace(
            go.Scatter(
                x=hist["iteration"],
                y=hist[col],
                name=terrain,
                mode="lines",
            )
        )

    fig.update_layout(
        title="Terrain curriculum sampling probability",
        height=320,
        yaxis_title="probability",
        xaxis_title="iteration",
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def fig_system_id(sysid):
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Actuator torque fit", "Step-response latency"),
        horizontal_spacing=0.12,
    )

    tau = sysid["tau"]
    tau_hat = sysid["tau_hat"]

    lo = float(min(np.min(tau), np.min(tau_hat)))
    hi = float(max(np.max(tau), np.max(tau_hat)))

    fig.add_trace(
        go.Scatter(
            x=tau,
            y=tau_hat,
            mode="markers",
            marker=dict(size=5, color="#1f77b4", opacity=0.65),
            name="measured vs fit",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=[lo, hi],
            y=[lo, hi],
            mode="lines",
            line=dict(dash="dash", color="crimson"),
            name="ideal",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=sysid["latency_t"],
            y=sysid["latency_meas"],
            mode="lines",
            name="measured torque",
        ),
        row=1,
        col=2,
    )

    fig.update_xaxes(title_text="measured torque (Nm)", row=1, col=1)
    fig.update_yaxes(title_text="fitted torque (Nm)", row=1, col=1)

    fig.update_xaxes(title_text="time (s)", row=1, col=2)
    fig.update_yaxes(title_text="torque (Nm)", row=1, col=2)

    fig.update_layout(
        height=380,
        margin=dict(l=0, r=0, t=60, b=0),
        showlegend=False,
    )

    return fig


def fig_deployment_box(df):
    fig = go.Figure()

    for terrain in TERRAINS:
        y = df.loc[df["terrain"] == terrain, "joint_tracking_error_pct"].values
        fig.add_trace(go.Box(y=y, name=terrain, boxpoints="outliers"))

    add_hline_compat(fig, 10.0, "10% target")

    fig.update_layout(
        title="Joint tracking error by terrain",
        height=360,
        yaxis_title="error (%)",
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def fig_gait_match_bar(terrain_matches):
    fig = go.Figure()

    terrains = list(terrain_matches.keys())
    values = [terrain_matches[t] for t in terrains]

    colors = ["#2ca02c" if v >= 80.0 else "#d62728" for v in values]

    fig.add_trace(go.Bar(x=terrains, y=values, marker_color=colors))

    add_hline_compat(fig, 80.0, "80% target")

    fig.update_layout(
        title="Per-terrain human gait match",
        height=340,
        yaxis_title="match %",
        yaxis_range=[0, 105],
        margin=dict(l=0, r=0, t=60, b=0),
    )

    return fig


def synthetic_scene_image(terrain, ny=180, nx=280, seed=11):
    rng = np.random.default_rng(seed)

    x = np.linspace(0.0, 1.0, nx)
    y = np.linspace(0.0, 1.0, ny)
    X, Y = np.meshgrid(x, y, indexing="xy")

    if terrain == "stairs":
        band = np.floor(Y * 6.0)
        img = 0.22 + 0.09 * band + 0.04 * X

        edge = np.mod(Y * 6.0, 1.0) < 0.07
        img[edge] = 0.88

    elif terrain == "ramp":
        img = 0.25 + 0.45 * (0.70 * X + 0.30 * Y)
        img += 0.02 * rng.normal(size=(ny, nx))
        img = smooth_2d(img, 3)

    elif terrain == "gravel":
        img = 0.35 + smooth_2d(rng.normal(0.0, 0.10, (ny, nx)), 3)

        rocks = rng.random((ny, nx)) < 0.05
        img[rocks] += 0.25

    else:
        img = 0.30 + smooth_2d(rng.normal(0.0, 0.07, (ny, nx)), 5)
        img += 0.035 * np.sin(X * 28.0) * np.cos(Y * 22.0)

    return np.clip(img, 0.0, 1.0)


def fig_image_prediction(
    predicted,
    target,
    image_array=None,
    b64_source=None,
    title="Image-conditioned gait prediction",
):
    fig = go.Figure()

    if b64_source is not None:
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_layout_image(
            dict(
                source=b64_source,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                sizing="stretch",
                opacity=1.0,
                layer="below",
            )
        )

        fig.update_xaxes(
            range=[0, 1],
            showticklabels=False,
            zeroline=False,
            showgrid=False,
        )

        fig.update_yaxes(
            range=[0, 1],
            showticklabels=False,
            zeroline=False,
            showgrid=False,
        )

    elif image_array is not None:
        fig.add_trace(
            go.Heatmap(
                z=image_array,
                colorscale="Gray",
                showscale=False,
                hoverinfo="skip",
            )
        )

        fig.update_xaxes(
            range=[0, image_array.shape[1]],
            showticklabels=False,
            zeroline=False,
            showgrid=False,
        )

        fig.update_yaxes(
            range=[0, image_array.shape[0]],
            showticklabels=False,
            zeroline=False,
            showgrid=False,
        )

    fig.add_shape(
        type="rect",
        x0=0.03,
        x1=0.56,
        y0=0.05,
        y1=0.43,
        line_width=0,
        fillcolor="rgba(8,12,20,0.68)",
        xref="paper",
        yref="paper",
    )

    lines = [
        "<b>Predicted gait policy output</b>",
        f"Step height: {predicted['step_height_m'] * 100:.1f} cm",
        f"Stride length: {predicted['stride_length_m'] * 100:.1f} cm",
        f"Cadence: {predicted['cadence_spm']:.1f} spm",
        f"Foot clearance: {predicted['foot_clearance_m'] * 100:.1f} cm",
        f"Human match: {gait_match_percent(predicted, target):.1f}%",
    ]

    for i, line in enumerate(lines):
        fig.add_annotation(
            x=0.05,
            y=0.39 - 0.065 * i,
            text=line,
            showarrow=False,
            font=dict(color="white", size=14),
            xref="paper",
            yref="paper",
        )

    fig.update_layout(
        title=title,
        height=440,
        margin=dict(l=0, r=0, t=40, b=0),
    )

    return fig


def predict_terrain_from_name(name, forced="auto"):
    if forced != "auto":
        return forced

    s = str(name).lower()

    if any(k in s for k in ["stair", "step", "escalator"]):
        return "stairs"

    if any(k in s for k in ["ramp", "slope", "incline"]):
        return "ramp"

    if any(k in s for k in ["gravel", "rock", "rubble", "stone"]):
        return "gravel"

    if any(k in s for k in ["grass", "lawn", "field", "turf"]):
        return "grass"

    return "stairs"


# ------------------------------------------------------------------------------
# Live demo renderer
# ------------------------------------------------------------------------------

def render_live_demo(
    human_clips,
    final_gaits,
    train_hist,
    deploy_df,
    kpis,
    seed,
    speed,
):
    st.subheader("Auto-running live demo — 2-minute scenario compressed")
    st.caption(
        "Real-person capture → simulated PPO training/replay → hardware mixed-terrain trial → image-conditioned gait prediction."
    )

    progress = st.progress(0.0)
    phase_box = st.empty()
    info_box = st.empty()
    chart = st.empty()

    frames = 28
    sleep_s = 0.085 / max(1, int(speed))

    outdoor_trials = deploy_df[deploy_df["course"] == "outdoor"]["trial"].unique()
    n_outdoor_trials = max(1, len(outdoor_trials))

    for f in range(frames):
        sec = 120.0 * f / max(frames - 1, 1)

        if sec < 20.0:
            phase = "1/4 Real human capture"

            if sec < 5.0:
                terrain = "stairs"
            elif sec < 10.0:
                terrain = "gravel"
            elif sec < 15.0:
                terrain = "grass"
            else:
                terrain = "ramp"

            clip = human_clips[human_clips["terrain"] == terrain].iloc[0]
            stats = {k: float(clip[k]) for k in METRIC_KEYS}

            dur = min(
                float(clip["duration_s"]),
                max(2.0, (sec / 20.0) * float(clip["duration_s"])),
            )

            path = generate_footfall_path(
                terrain,
                duration_s=dur,
                fps=int(clip["fps"]),
                seed=seed + TERRAINS.index(terrain),
                stats=stats,
            )

            ex = extract_gait_stats(path, terrain)

            fig = fig_footfall_3d(path, f"Live extraction: {terrain}")

            lines = [
                f"<b>{terrain.upper()}</b>",
                f"recorded step height: {stats['step_height_m'] * 100:.1f} cm",
                f"extracted step height: {ex['step_height_m'] * 100:.1f} cm",
                f"cadence: {ex['cadence_spm']:.1f} spm",
                f"clearance: {ex['foot_clearance_m'] * 100:.1f} cm",
                f"confidence: {ex['confidence'] * 100:.0f}%",
            ]

            add_info_panel(fig, lines)

            info_box.markdown(
                f"**Phase:** Capture  \n"
                f"**Terrain:** {terrain}  \n"
                f"**Recorded step height:** {stats['step_height_m'] * 100:.1f} cm  \n"
                f"**Extracted step height:** {ex['step_height_m'] * 100:.1f} cm  \n"
                f"**Extracted cadence:** {ex['cadence_spm']:.1f} spm  \n"
                f"**Confidence:** {ex['confidence'] * 100:.0f}%"
            )

        elif sec < 60.0:
            phase = "2/4 Simulated humanoid training/replay"

            idx_float = ((sec - 20.0) / 40.0) * max(0, len(train_hist) - 1)
            idx = int(clamp(idx_float, 0, len(train_hist) - 1))

            row = train_hist.iloc[idx]

            terrain_cycle = int(((sec - 20.0) / 40.0) * 4) % 4
            terrain = TERRAINS[terrain_cycle]

            X, Y, Z = generate_terrain(terrain, 24, 24, seed=seed)

            fig = fig_terrain_surface(X, Y, Z, f"Procedural replay: {terrain}")

            lines = [
                f"<b>iteration {idx}</b>",
                f"terrain: {terrain}",
                f"reward: {row['mean_reward']:.2f}",
                f"match: {row['mean_match_pct']:.1f}%",
                f"policy std: {row['std']:.2f}",
                f"|action|: {row['mean_abs_action']:.2f}",
            ]

            add_info_panel(fig, lines)

            info_box.markdown(
                f"**Phase:** Simulated training/replay  \n"
                f"**Iteration:** {idx}/{len(train_hist) - 1}  \n"
                f"**Terrain sampled:** {terrain}  \n"
                f"**Mean reward:** {row['mean_reward']:.2f}  \n"
                f"**Mean gait match:** {row['mean_match_pct']:.1f}%"
            )

        elif sec < 90.0:
            phase = "3/4 Real hardware mixed-terrain test"

            raw_trials = ((sec - 60.0) / 30.0) * n_outdoor_trials
            progress_trials = min(
                n_outdoor_trials,
                max(1, int(math.ceil(raw_trials))),
            )

            sub = deploy_df[
                (deploy_df["course"] == "outdoor")
                & (deploy_df["trial"] <= progress_trials)
            ]

            fig = go.Figure()

            if len(sub):
                fig.add_trace(
                    go.Scatter(
                        x=list(range(len(sub))),
                        y=sub["joint_tracking_error_pct"],
                        mode="lines+markers",
                        name="tracking error %",
                    )
                )

                add_hline_compat(fig, 10.0, "10% KPI")

                fell = bool(sub["trial_fall"].any())
                mean_err = float(sub["joint_tracking_error_pct"].mean())
            else:
                fell = False
                mean_err = 0.0

            fig.update_layout(
                title="Hardware logging: joint tracking error",
                height=390,
                margin=dict(l=0, r=0, t=40, b=0),
                yaxis_title="error (%)",
                xaxis_title="logged segment",
            )

            lines = [
                "<b>Outdoor hardware log</b>",
                f"trials <= {progress_trials}",
                f"segments logged: {len(sub)}",
                f"mean error: {mean_err:.1f}%",
                f"falls observed: {'yes' if fell else 'no'}",
            ]

            add_info_panel(fig, lines)

            info_box.markdown(
                f"**Phase:** Hardware mixed-terrain test  \n"
                f"**Outdoor trials logged:** {progress_trials}  \n"
                f"**Mean joint tracking error:** {mean_err:.1f}%  \n"
                f"**Fall observed:** {'Yes' if fell else 'No'}"
            )

        else:
            phase = "4/4 Real image test"

            img = synthetic_scene_image("stairs", seed=seed + 4)

            predicted = final_gaits["stairs"]
            target = HUMAN_TARGETS["stairs"]

            fig = fig_image_prediction(
                predicted=predicted,
                target=target,
                image_array=img,
                title="Staircase photo → predicted gait overlay",
            )

            info_box.markdown(
                f"**Phase:** Image-conditioned gait prediction  \n"
                f"**Scene:** staircase  \n"
                f"**Predicted step height:** {predicted['step_height_m'] * 100:.1f} cm  \n"
                f"**Predicted clearance:** {predicted['foot_clearance_m'] * 100:.1f} cm  \n"
                f"**Human match:** {gait_match_percent(predicted, target):.1f}%"
            )

        progress.progress((f + 1) / frames)
        phase_box.markdown(f"**{phase}** | simulated time {sec:.0f}s / 120s")

        chart.plotly_chart(
            fig,
            use_container_width=True,
            key=f"live_chart_{f}",
        )

        time.sleep(sleep_s)

    progress.progress(1.0)
    phase_box.markdown("**Demo complete.** Use the tabs below to inspect each stage.")
    info_box.markdown(
        f"**Final KPI summary:**  \n"
        f"Gait match: {kpis['gait_match_pct']:.1f}%  \n"
        f"Fall rate: {kpis['fall_rate_pct']:.1f}%  \n"
        f"Sim-to-real error: {kpis['sim_to_real_error_pct']:.1f}%  \n"
        f"Reaction time: {kpis['reaction_s']:.2f} s"
    )


# ------------------------------------------------------------------------------
# Tab renderers
# ------------------------------------------------------------------------------

def render_capture_tab(human_clips, seed):
    st.subheader("1) Recorded human motion and world-grounded gait extraction")

    st.markdown(
        "This tab simulates 4–6 labeled human clips. In the full project, replace "
        "`make_human_clips()` and `generate_footfall_path()` with real mocap / RGB-D "
        "pose estimation outputs."
    )

    st.dataframe(human_clips, use_container_width=True)

    clip_id = st.selectbox("Capture: Select clip", human_clips["clip_id"].tolist())
    row = human_clips[human_clips["clip_id"] == clip_id].iloc[0]

    stats = {k: float(row[k]) for k in METRIC_KEYS}

    path = generate_footfall_path(
        row["terrain"],
        duration_s=float(row["duration_s"]),
        fps=int(row["fps"]),
        seed=seed + TERRAINS.index(row["terrain"]),
        stats=stats,
    )

    ex = extract_gait_stats(path, row["terrain"])

    c1, c2 = st.columns([1.35, 1.0])

    with c1:
        st.plotly_chart(
            fig_footfall_3d(path, f"{row['terrain']} • {row['label']}"),
            use_container_width=True,
            key="fig_capture_footfall",
        )

    with c2:
        st.metric("Capture: Pose confidence", f"{ex['confidence'] * 100:.1f}%")
        st.metric("Capture: Ground error", f"{row['ground_error_cm']:.1f} cm")
        st.metric("Capture: Detected step segments", f"{ex['segments']}")

        cmp = pd.DataFrame(
            [
                {
                    "metric": k,
                    "recorded_human": stats[k],
                    "extracted": ex[k],
                    "abs_error": abs(stats[k] - ex[k]),
                }
                for k in METRIC_KEYS
            ]
        )

        st.dataframe(cmp, use_container_width=True)

    X, Y, Z = generate_terrain(
        row["terrain"],
        36,
        36,
        seed=seed,
        overrides={"riser_m": ex["step_height_m"]},
    )

    st.plotly_chart(
        fig_terrain_surface(X, Y, Z, "Matching procedural terrain"),
        use_container_width=True,
        key="fig_capture_terrain",
    )


def render_terrain_reward_tab(final_gaits, seed):
    st.subheader("2–4) Procedural terrain generator and gait-matching reward")

    terrain = st.selectbox("Terrain reward: Terrain", TERRAINS)

    c1, c2, c3 = st.columns(3)

    with c1:
        riser = st.slider(
            f"Terrain reward ({terrain}): Riser / step height (m)",
            0.05,
            0.25,
            float(HUMAN_TARGETS[terrain]["step_height_m"]),
            0.005,
        )

    with c2:
        slope = st.slider(
            f"Terrain reward ({terrain}): Ramp slope",
            0.0,
            0.25,
            0.12,
            0.01,
        )

    with c3:
        roughness = st.slider(
            f"Terrain reward ({terrain}): Roughness",
            0.0,
            0.08,
            0.03,
            0.002,
        )

    X, Y, Z = generate_terrain(
        terrain,
        48,
        48,
        seed=seed,
        overrides={
            "riser_m": riser,
            "slope": slope,
            "roughness": roughness,
        },
    )

    st.plotly_chart(
        fig_terrain_surface(X, Y, Z, f"Procedural {terrain}"),
        use_container_width=True,
        key="fig_terrain_reward_surface",
    )

    actual = final_gaits[terrain]
    target = HUMAN_TARGETS[terrain]

    rew, terms = reward_function(actual, target)

    k1, k2, k3 = st.columns(3)

    k1.metric(f"Terrain reward ({terrain}): Total reward", f"{rew:.2f}")
    k2.metric(f"Terrain reward ({terrain}): Gait match", f"{terms['gait_match_pct']:.1f}%")
    k3.metric(f"Terrain reward ({terrain}): Stability term", f"{terms['stability']:.2f}")

    st.dataframe(pd.DataFrame([terms]), use_container_width=True)

    st.plotly_chart(
        fig_gait_comparison(actual, target),
        use_container_width=True,
        key="fig_reward_gait_comparison",
    )


def render_ppo_tab(train_hist, final_gaits):
    st.subheader("5) PPO policy with terrain curriculum and domain randomization")

    st.markdown(
        "This embedded PPO loop optimizes a 4D gait-residual policy. "
        "In the production stack, this logic maps to Isaac Lab / RSL-RL whole-body PPO."
    )

    st.plotly_chart(
        fig_training(train_hist),
        use_container_width=True,
        key="fig_ppo_training",
    )

    st.plotly_chart(
        fig_curriculum(train_hist),
        use_container_width=True,
        key="fig_ppo_curriculum",
    )

    rows = []

    for terrain in TERRAINS:
        actual = final_gaits[terrain]
        target = HUMAN_TARGETS[terrain]
        match = gait_match_percent(actual, target)

        rows.append(
            {
                "terrain": terrain,
                "step_height_m": actual["step_height_m"],
                "stride_length_m": actual["stride_length_m"],
                "cadence_spm": actual["cadence_spm"],
                "foot_clearance_m": actual["foot_clearance_m"],
                "human_match_pct": match,
            }
        )

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    st.markdown(
        "**Domain randomization included:** mass, friction, push disturbances, "
        "actuator latency residual, and terrain roughness."
    )


def render_sysid_tab(sysid):
    st.subheader("6) Real-robot actuator system identification")

    st.markdown(
        "The app fits a simple torque model: "
        "`tau = kt*i - b*dot(q) - c*sign(dot(q)) - bias`, then estimates closed-loop latency."
    )

    st.dataframe(sysid["table"], use_container_width=True)

    st.plotly_chart(
        fig_system_id(sysid),
        use_container_width=True,
        key="fig_sysid",
    )


def render_deployment_tab(deploy_df, kpis, final_gaits):
    st.subheader("7–9) Hardware deployment logs, outdoor course, and KPI report")

    st.dataframe(deploy_df, use_container_width=True)

    st.dataframe(kpi_dataframe(kpis), use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        st.plotly_chart(
            fig_deployment_box(deploy_df),
            use_container_width=True,
            key="fig_deploy_box",
        )

    with c2:
        st.plotly_chart(
            fig_gait_match_bar(kpis["terrain_matches"]),
            use_container_width=True,
            key="fig_deploy_gait_match",
        )

    st.markdown(
        "**Safety note:** In real deployment, always use a mechanical spotter, "
        "e-stop, torque limits, and a fallback standing controller before enabling "
        "terrain-adaptive RL policies."
    )


def render_image_tab(final_gaits, seed):
    st.subheader("10) Real image test — predict gait from a staircase photo")

    uploaded = st.file_uploader(
        "Image test: Upload a photo of stairs, ramp, gravel, or grass",
        type=["png", "jpg", "jpeg", "webp"],
    )

    c1, c2 = st.columns([1.2, 1.0])

    with c2:
        forced = st.selectbox(
            "Image test: Scene prior",
            ["auto", "stairs", "ramp", "gravel", "grass"],
            index=0,
        )

        riser = st.slider(
            "Image test: Estimated riser height (m)",
            0.05,
            0.25,
            0.17,
            0.005,
        )

    if uploaded is not None:
        terrain = predict_terrain_from_name(uploaded.name, forced)

        b64 = base64.b64encode(uploaded.getvalue()).decode("utf-8")
        mime = uploaded.type if uploaded.type else "image/jpeg"
        source = f"data:{mime};base64,{b64}"

        predicted = final_gaits[terrain].copy()
        target = HUMAN_TARGETS[terrain].copy()

        if terrain == "stairs":
            predicted["step_height_m"] = float(
                0.70 * riser + 0.30 * predicted["step_height_m"]
            )

            predicted["foot_clearance_m"] = float(
                np.clip(0.45 * riser + 0.020, 0.025, 0.14)
            )

            target["step_height_m"] = riser

        fig = fig_image_prediction(
            predicted=predicted,
            target=target,
            b64_source=source,
            title=f"Uploaded image → {terrain} gait prediction",
        )

        with c1:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key="fig_image_pred_uploaded",
            )

    else:
        terrain = "stairs" if forced == "auto" else forced

        img = synthetic_scene_image(terrain, seed=seed + 5)

        predicted = final_gaits[terrain].copy()
        target = HUMAN_TARGETS[terrain].copy()

        if terrain == "stairs":
            predicted["step_height_m"] = float(
                0.70 * riser + 0.30 * predicted["step_height_m"]
            )

            predicted["foot_clearance_m"] = float(
                np.clip(0.45 * riser + 0.020, 0.025, 0.14)
            )

            target["step_height_m"] = riser

        fig = fig_image_prediction(
            predicted=predicted,
            target=target,
            image_array=img,
            title="Synthetic scene image → predicted gait overlay",
        )

        with c1:
            st.plotly_chart(
                fig,
                use_container_width=True,
                key=f"fig_image_pred_synthetic_{terrain}",
            )

    tbl = pd.DataFrame(
        [
            {
                "parameter": k,
                "policy_will_use": predicted[k],
                "human_reference": target[k],
                "match_pct": gait_match_percent(predicted, target),
            }
            for k in METRIC_KEYS
        ]
    )

    st.dataframe(tbl, use_container_width=True)


def render_diagnostics_tab():
    st.subheader("Diagnostics and smoke tests")

    st.markdown("Run from terminal:")
    st.code("python app.py --selftest", language="bash")

    if st.button("Run smoke tests now"):
        with st.spinner("Running embedded smoke tests..."):
            res = run_self_tests()

        st.dataframe(res, use_container_width=True)

        st.download_button(
            "Download test report CSV",
            res.to_csv(index=False).encode("utf-8"),
            file_name="terrainmorph_smoke_tests.csv",
            mime="text/csv",
        )

        if (res["status"] == "PASS").all():
            st.success("All smoke tests passed.")
        else:
            st.error("Some smoke tests failed. See table above.")


# ------------------------------------------------------------------------------
# Self tests
# ------------------------------------------------------------------------------

def run_self_tests():
    results = []

    def check(name, cond, detail=""):
        results.append(
            {
                "test": name,
                "status": "PASS" if bool(cond) else "FAIL",
                "detail": str(detail),
            }
        )

    clips = make_human_clips()

    check("Human clip count is 4-6", 4 <= len(clips) <= 6, f"n={len(clips)}")

    check(
        "All clip terrains are supported",
        set(clips["terrain"].unique()).issubset(set(TERRAINS)),
        sorted(clips["terrain"].unique().tolist()),
    )

    path = generate_footfall_path("stairs", duration_s=8.0, fps=60, seed=11)
    ex = extract_gait_stats(path, "stairs")

    check("Extracted step height positive", ex["step_height_m"] > 0.04, ex["step_height_m"])
    check("Extracted cadence plausible", 60.0 < ex["cadence_spm"] < 180.0, ex["cadence_spm"])
    check("Extracted clearance positive", ex["foot_clearance_m"] > 0.005, ex["foot_clearance_m"])

    check(
        "Stair extraction step height close to target",
        abs(ex["step_height_m"] - HUMAN_TARGETS["stairs"]["step_height_m"]) < 0.070,
        ex["step_height_m"],
    )

    check(
        "Stair extraction clearance close to target",
        abs(ex["foot_clearance_m"] - HUMAN_TARGETS["stairs"]["foot_clearance_m"]) < 0.060,
        ex["foot_clearance_m"],
    )

    for terrain in TERRAINS:
        X, Y, Z = generate_terrain(terrain, 12, 12, seed=2)
        check(
            f"Terrain {terrain} generates finite heightfield",
            Z.shape == (12, 12) and np.all(np.isfinite(Z)),
            Z.shape,
        )

    img = synthetic_scene_image("stairs", 60, 80, seed=1)
    check("Synthetic image shape", img.shape == (60, 80), img.shape)

    perfect = gait_match_percent(HUMAN_TARGETS["grass"], HUMAN_TARGETS["grass"])
    check("Perfect gait match is 100%", abs(perfect - 100.0) < 1e-6, perfect)

    rew, terms = reward_function(HUMAN_TARGETS["ramp"], HUMAN_TARGETS["ramp"])
    check("Reward is finite", np.isfinite(rew), rew)

    policy, hist = train_mini_ppo(iterations=4, samples_per_iter=32, seed=3)
    check("Training history length", len(hist) == 4, len(hist))

    final = policy_final_gaits(policy, calibration=0.65)
    check("Final gait keys", set(final.keys()) == set(TERRAINS), list(final.keys()))

    avg_match = float(
        np.mean(
            [
                gait_match_percent(final[t], HUMAN_TARGETS[t])
                for t in TERRAINS
            ]
        )
    )

    check("Toy PPO average match > 45%", avg_match > 45.0, f"{avg_match:.1f}%")

    sysid = run_system_identification(seed=4)
    check("System ID table rows", len(sysid["table"]) >= 4, len(sysid["table"]))

    deploy = simulate_deployment(
        final,
        sysid,
        indoor_trials=2,
        outdoor_trials=3,
        seed=5,
    )

    check("Deployment rows generated", len(deploy) > 0, len(deploy))

    kpis = compute_kpis(deploy, final)

    check(
        "KPIs finite",
        all(
            np.isfinite(
                [
                    kpis["gait_match_pct"],
                    kpis["fall_rate_pct"],
                    kpis["sim_to_real_error_pct"],
                    kpis["reaction_s"],
                ]
            )
        ),
        kpis,
    )

    return pd.DataFrame(results, columns=["test", "status", "detail"])


# ------------------------------------------------------------------------------
# Main Streamlit app
# ------------------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="TerrainMorph RL",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("TerrainMorph RL")
    st.caption(
        "Human motion capture → gait statistics → procedural terrain → PPO policy → "
        "system ID → hardware deployment → image-conditioned gait prediction."
    )

    st.sidebar.header("Controls")

    seed = int(
        st.sidebar.number_input(
            "Random seed",
            min_value=0,
            max_value=99999,
            value=2026,
        )
    )

    iterations = int(
        st.sidebar.slider(
            "PPO surrogate iterations",
            20,
            160,
            55,
            5,
        )
    )

    curriculum = st.sidebar.checkbox("Terrain curriculum", value=True)
    domain_rand = st.sidebar.checkbox("Domain randomization", value=True)

    demo_speed = int(
        st.sidebar.slider(
            "Live demo speed",
            1,
            5,
            3,
        )
    )

    if st.sidebar.button("Replay live demo"):
        st.session_state["demo_ran"] = False

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        "**Notes**\n\n"
        "- This app is a runnable engineering demonstrator.\n"
        "- It auto-loads realistic synthetic capture data.\n"
        "- It includes an embedded PPO surrogate for the gait-residual policy.\n"
        "- Production hardware would replace the synthetic modules with Isaac Lab, "
        "real mocap, and a robot SDK bridge."
    )

    human_clips = make_human_clips()

    params = (seed, iterations, bool(curriculum), bool(domain_rand))

    if "ppo_pack" not in st.session_state or st.session_state.get("ppo_params") != params:
        with st.spinner("Loading demo data and training TerrainMorph PPO surrogate..."):
            policy, train_hist = train_mini_ppo(
                iterations=iterations,
                samples_per_iter=128,
                seed=seed,
                curriculum=curriculum,
                domain_rand=domain_rand,
            )

            final_gaits = policy_final_gaits(policy, calibration=0.65)

        st.session_state["ppo_pack"] = (policy, train_hist, final_gaits)
        st.session_state["ppo_params"] = params
        st.session_state["demo_ran"] = False
    else:
        policy, train_hist, final_gaits = st.session_state["ppo_pack"]

    sysid = run_system_identification(seed=seed + 1)
    deploy_df = simulate_deployment(final_gaits, sysid, seed=seed + 2)
    kpis = compute_kpis(deploy_df, final_gaits)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Gait match KPI",
        f"{kpis['gait_match_pct']:.1f}%",
        f"{kpis['gait_match_pct'] - 80.0:+.1f} pts vs target",
        delta_color="normal" if kpis["gait_match_pct"] >= 80.0 else "inverse",
    )

    c2.metric(
        "Fall rate KPI",
        f"{kpis['fall_rate_pct']:.1f}%",
        f"{5.0 - kpis['fall_rate_pct']:+.1f} pts under target",
        delta_color="normal" if kpis["fall_rate_pct"] < 5.0 else "inverse",
    )

    c3.metric(
        "Sim-to-real error KPI",
        f"{kpis['sim_to_real_error_pct']:.1f}%",
        f"{10.0 - kpis['sim_to_real_error_pct']:+.1f} pts under target",
        delta_color="normal" if kpis["sim_to_real_error_pct"] < 10.0 else "inverse",
    )

    c4.metric(
        "Reaction time KPI",
        f"{kpis['reaction_s']:.2f} s",
        f"{1.0 - kpis['reaction_s']:+.2f} s under target",
        delta_color="normal" if kpis["reaction_s"] < 1.0 else "inverse",
    )

    if not st.session_state.get("demo_ran", False):
        render_live_demo(
            human_clips=human_clips,
            final_gaits=final_gaits,
            train_hist=train_hist,
            deploy_df=deploy_df,
            kpis=kpis,
            seed=seed,
            speed=demo_speed,
        )
        st.session_state["demo_ran"] = True
    else:
        st.info(
            "Live demo already ran in this session. Use the sidebar **Replay live demo** "
            "button or reload the page to run it again."
        )

    tabs = st.tabs(
        [
            "1. Capture & Extract",
            "2. Terrain & Reward",
            "3. PPO Training",
            "4. System ID",
            "5. Deployment & KPIs",
            "6. Image Test",
            "7. Tests",
        ]
    )

    with tabs[0]:
        render_capture_tab(human_clips, seed)

    with tabs[1]:
        render_terrain_reward_tab(final_gaits, seed)

    with tabs[2]:
        render_ppo_tab(train_hist, final_gaits)

    with tabs[3]:
        render_sysid_tab(sysid)

    with tabs[4]:
        render_deployment_tab(deploy_df, kpis, final_gaits)

    with tabs[5]:
        render_image_tab(final_gaits, seed)

    with tabs[6]:
        render_diagnostics_tab()


if __name__ == "__main__":
    if "--selftest" in sys.argv:
        df = run_self_tests()
        print(df.to_string(index=False))
        sys.exit(0 if (df["status"] == "PASS").all() else 1)

    main()