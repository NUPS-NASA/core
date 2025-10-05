
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX-based detector for "low-valued interval" (at most one box-shaped dip) in noisy scatter plots.

Usage:
  python dip_detector_jax.py /path/to/folder \
     --time-col TIME --y-col Y \
     --out-dir /path/to/output

If column names are omitted, the script will auto-detect sensible numeric columns.
The script will recursively search for all .csv files under the given folder.

For each CSV:
  * Detect if there is at most one "dip" interval where y is noticeably lower than surrounding points.
  * Minimize false negatives (prefer to flag a candidate dip rather than miss it).
  * Robust to heavy noise (esp. near edges) and huge y-distribution spread (treat as y=a when "filled rectangle").
  * Save an annotated PNG with the detected interval (if any) and add a row to a summary CSV.
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- JAX imports
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
    import optax
except Exception as e:
    print("ERROR: This script requires JAX and Optax.\n"
          "Install with: pip install --upgrade 'jax[cpu]' optax", file=sys.stderr)
    raise

# --------------- Utilities ---------------

def robust_mad(x: np.ndarray) -> float:
    """Median absolute deviation -> robust sigma estimate."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)

def find_columns(df: pd.DataFrame, time_col: Optional[str], y_col: Optional[str]) -> Tuple[str, str]:
    """Heuristically pick time and y columns if not provided."""
    if time_col is not None and y_col is not None:
        return time_col, y_col

    # Normalize column names
    candidates = {c.lower(): c for c in df.columns}

    time_aliases = ["time", "t", "x", "timestamp", "frame", "hour"]
    y_aliases = ["flux", "y", "value", "relative flux", "detrended", "intensity"]

    def guess_from_aliases(aliases):
        for a in aliases:
            # exact match
            if a in candidates:
                return candidates[a]
            # startswith match
            for k, v in candidates.items():
                if k.startswith(a):
                    return v
        return None

    tc = time_col or guess_from_aliases(time_aliases)
    yc = y_col or guess_from_aliases(y_aliases)

    # Fallback: first two numeric columns
    if tc is None or yc is None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) >= 2:
            if tc is None: tc = numeric_cols[0]
            if yc is None: yc = numeric_cols[1]

    if tc is None or yc is None:
        raise ValueError("Could not infer time/y columns. Please pass --time-col and --y-col.")

    return tc, yc

def rectangular_y_distribution(y: np.ndarray, bins: int = 50) -> float:
    """
    Heuristic "rectangularness" score in [0,1]. Higher means the points
    fill much of the vertical range somewhat uniformly (treat as y ~ a).
    """
    if len(y) < 10:
        return 0.0
    hist, _ = np.histogram(y, bins=bins)
    occupied = np.sum(hist > 0)
    occ_frac = occupied / bins  # fraction of bins with any points

    # If the spread is very wide relative to median absolute deviation, further boost the score.
    spread = np.percentile(y, 97.5) - np.percentile(y, 2.5)
    sigma = robust_mad(y)
    wide = np.clip(spread / (sigma + 1e-12), 0, 20) / 20.0  # 0..1
    # Combine (tunable)
    score = 0.6 * occ_frac + 0.4 * wide
    return float(score)

# --------------- Smooth box model in JAX ---------------

def soft_box(t: jnp.ndarray, c: jnp.ndarray, w: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth indicator of interval centered at c with width w.
    Returns values ~1 inside, ~0 outside, with smooth edges controlled by tau.
    """
    left = c - 0.5 * w
    right = c + 0.5 * w
    s1 = jax.nn.sigmoid((t - left) / tau)
    s2 = jax.nn.sigmoid((t - right) / tau)
    return jnp.clip(s1 - s2, 0.0, 1.0)

def huber_loss(residuals: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    """Elementwise Huber loss; delta is the transition between L2 and L1."""
    abs_r = jnp.abs(residuals)
    quad = 0.5 * (abs_r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quad, lin)

@jit
def objective(params: Dict[str, jnp.ndarray],
              t: jnp.ndarray,
              y: jnp.ndarray,
              w_weights: jnp.ndarray,
              tau: jnp.ndarray,
              delta: jnp.ndarray,
              w_min: jnp.ndarray,
              w_max: jnp.ndarray,
              lam_width: jnp.ndarray,
              lam_amp: jnp.ndarray) -> jnp.ndarray:
    """
    Objective: weighted Huber loss between y and (a - d * box(t; c, w)) plus soft regularization.
    params: {'a','d_raw','c_sig','w_sig'}
    Constraints:
      d = softplus(d_raw)  >= 0, amplitude = -d
      c in [tmin, tmax] via sigmoid on c_sig
      w in [w_min, w_max] via sigmoid on w_sig
    """
    tmin = t.min()
    tmax = t.max()
    a = params["a"]
    d = jax.nn.softplus(params["d_raw"])  # depth >= 0 (we subtract it inside interval)
    c = tmin + (tmax - tmin) * jax.nn.sigmoid(params["c_sig"])
    w = w_min + (w_max - w_min) * jax.nn.sigmoid(params["w_sig"])

    box = soft_box(t, c, w, tau)  # ~1 inside
    yhat = a - d * box

    res = (y - yhat) * w_weights
    loss = jnp.sum(huber_loss(res, delta))

    # Regularization: prefer non-tiny widths; and avoid absurd depth
    width_reg = lam_width * jnp.exp(-(w / (w_min + 1e-6)))
    amp_reg = lam_amp * (d ** 2)
    return loss + width_reg + amp_reg

def optimize_dip(t: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    """
    Optimize parameters using gradient descent with a few random restarts.
    Returns dictionary with best parameters and diagnostics.
    """
    key = jax.random.PRNGKey(seed)

    t_j = jnp.asarray(t)
    y_j = jnp.asarray(y)

    # robust scales & weights (downweight edges)
    sigma = np.float32(robust_mad(y))
    delta = np.float32(1.345 * sigma)  # Huber delta
    # Edge weights decrease near the ends (help with boundary noise)
    n = len(t)
    edge = np.linspace(0, 1, n)
    w_edge = (1 - np.exp(-5 * np.minimum(edge, 1 - edge)))  # 0 in extreme edges, ~1 middle
    w_weights = jnp.asarray(0.25 + 0.75 * w_edge)           # keep floor weight 0.25

    # Smoothness for box edges (fraction of range)
    tau = 0.01 * (t_j.max() - t_j.min() + 1e-12)

    # Width constraints
    w_min = 0.05 * (t_j.max() - t_j.min() + 1e-12)
    w_max = 0.80 * (t_j.max() - t_j.min() + 1e-12)

    # Opt hyperparams
    lam_width = 1.0
    lam_amp = 1e-4
    steps = 1000
    lr = 0.02

    # initialize
    y_med = float(np.median(y))
    c0 = t[np.argmin(y)]
    w0 = np.clip(4 * np.median(np.diff(np.sort(t))), w_min, w_max)  # start narrow-ish
    # Parameterize: a, d_raw, c_sig, w_sig
    def pack(a, d, c, w):
        c_sig = np.log((c - t.min()) / (t.max() - c) + 1e-9)
        w_sig = np.log((w - w_min) / (w_max - w) + 1e-9)
        d_raw = np.log(np.exp(max(0.0, d)) - 1 + 1e-12)  # inverse softplus
        return {"a": np.float32(a), "d_raw": np.float32(d_raw),
                "c_sig": np.float32(c_sig), "w_sig": np.float32(w_sig)}

    # Random restarts + a deterministic init
    inits = [pack(y_med, max(0.0, y_med - float(np.min(y))), c0, w0)]
    for i in range(5):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        a_i = y_med + 0.1 * sigma * float(jax.random.normal(k1))
        d_i = abs(0.5 * sigma * float(jax.random.normal(k2))) + 0.1 * sigma
        c_i = float(t.min() + (t.max() - t.min()) * jax.random.uniform(k3))
        w_i = float(w_min + (w_max - w_min) * jax.random.uniform(k4))
        inits.append(pack(a_i, d_i, c_i, w_i))

    # Optimizer
    opt = optax.adam(lr)

    best = None
    best_val = np.inf

    obj = lambda p: objective(p, t_j, y_j, jnp.asarray(w_weights),
                              jnp.asarray(tau), jnp.asarray(delta),
                              jnp.asarray(w_min), jnp.asarray(w_max),
                              jnp.asarray(lam_width), jnp.asarray(lam_amp))

    for init in inits:
        params = {k: jnp.asarray(v) for k, v in init.items()}
        opt_state = opt.init(params)

        @jit
        def step(p, s):
            val, grads = value_and_grad(obj)(p)
            updates, s = opt.update(grads, s, p)
            p = optax.apply_updates(p, updates)
            return p, s, val

        val = None
        for _ in range(steps):
            params, opt_state, val = step(params, opt_state)

        val_np = float(val)
        if val_np < best_val:
            best_val = val_np
            best = {k: float(v) for k, v in params.items()}

    # Decode best params to real values
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    a = best["a"]
    d = float(np.log1p(np.exp(best["d_raw"])))  # softplus
    c = tmin + (tmax - tmin) * (1 / (1 + np.exp(-best["c_sig"])))
    w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-best["w_sig"])))

    # Diagnostics
    box = 1 / (1 + np.exp(-(t - (c - 0.5*w)) / float(tau))) - 1 / (1 + np.exp(-(t - (c + 0.5*w)) / float(tau)))
    yhat = a - d * box

    # baseline-only loss (same Huber & weights), optimizing 'a' analytically ~ median
    # to keep it robust and comparable
    a0 = float(np.median(y))
    def huber_np(r, delta):
        ar = np.abs(r)
        quad = 0.5 * (ar ** 2)
        lin = delta * (ar - 0.5 * delta)
        return np.where(ar <= delta, quad, lin)
    loss_base = float(np.sum(huber_np((y - a0) * np.asarray(w_weights), float(delta))))
    loss_model = float(np.sum(huber_np((y - yhat) * np.asarray(w_weights), float(delta))))

    improvement = max(0.0, (loss_base - loss_model) / (loss_base + 1e-12))
    snr = float(d / (robust_mad(y) + 1e-12))

    return {
        "a": float(a),
        "depth": float(d),
        "center": float(c),
        "width": float(w),
        "tau": float(tau),
        "yhat": yhat.astype(np.float32),
        "loss_base": loss_base,
        "loss_model": loss_model,
        "improvement": improvement,
        "snr": snr,
        "box": box.astype(np.float32),
    }

def detect_on_xy(t: np.ndarray, y: np.ndarray,
                 prefer_sensitivity: bool = True) -> Dict[str, Any]:
    """
    Run detection on a single time series (t, y).
    Returns dict with 'has_dip' and interval if present.
    """
    # Pre-clean
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]
    y = y[mask]

    # Sort by time
    order = np.argsort(t)
    t = t[order]
    y = y[order]

    # Heuristic: if the vertical distribution is "rectangular", declare no dip.
    rect_score = rectangular_y_distribution(y, bins=50)
    is_rect = rect_score >= 0.75  # conservative
    result: Dict[str, Any] = {"has_dip": False, "reason": None}

    if is_rect:
        result.update({
            "has_dip": False,
            "reason": f"Vertical distribution looks 'filled' (score={rect_score:.2f}); treat as y=a.",
            "a": float(np.median(y)),
            "depth": 0.0,
            "center": np.nan,
            "width": 0.0,
            "snr": 0.0,
            "improvement": 0.0,
        })
        return result

    # Optimize dip model
    fit = optimize_dip(t, y)

    # Decision policy (favor sensitivity to reduce false negatives)
    # Low thresholds + require a minimal duration (>= 4 samples or >= 5% of span)
    min_points = max(4, int(0.05 * len(t)))
    inside = (t >= (fit["center"] - 0.5 * fit["width"])) & (t <= (fit["center"] + 0.5 * fit["width"]))
    support = int(np.sum(inside))

    snr_thr = 0.8 if prefer_sensitivity else 1.1
    imp_thr = 0.02 if prefer_sensitivity else 0.05

    has_dip = (fit["snr"] >= snr_thr) and (fit["improvement"] >= imp_thr) and (support >= min_points)

    result.update({
        "has_dip": bool(has_dip),
        "reason": None if has_dip else "Insufficient SNR/improvement/support for a reliable interval.",
        "a": fit["a"],
        "depth": fit["depth"],
        "center": fit["center"],
        "width": fit["width"],
        "snr": fit["snr"],
        "improvement": fit["improvement"],
    })
    return result

def plot_and_save(t: np.ndarray, y: np.ndarray, res: Dict[str, Any], out_png: Path, title: str = "") -> None:
    plt.figure(figsize=(9, 5.2))
    plt.scatter(t, y, s=18)
    plt.xlabel("Time")
    plt.ylabel("Y")
    if title:
        plt.title(title)

    if res["has_dip"] and np.isfinite(res["center"]):
        s = res["center"] - 0.5 * res["width"]
        e = res["center"] + 0.5 * res["width"]
        plt.axvspan(s, e, alpha=0.2, label="Detected interval")
        # baseline
        plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")
        plt.text(s, res["a"], f"  depth≈{res['depth']:.4g}, SNR≈{res['snr']:.2f}, imp≈{100*res['improvement']:.1f}%",
                 va="bottom", ha="left")
        plt.legend(frameon=False, loc="best")
    else:
        # baseline only
        plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")
        if res.get("reason"):
            plt.text(np.min(t), np.max(y), res["reason"], va="top", ha="left")
        plt.legend(frameon=False, loc="best")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def load_csv(path: Path, time_col: Optional[str], y_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(path)
    tc, yc = find_columns(df, time_col, y_col)
    t = df[tc].to_numpy(dtype=float)
    y = df[yc].to_numpy(dtype=float)
    return t, y

def process_folder(root: Path, out_dir: Path,
                   time_col: Optional[str], y_col: Optional[str],
                   prefer_sensitivity: bool = True) -> Path:
    results: List[Dict[str, Any]] = []
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {root}", file=sys.stderr)

    for p in csv_files:
        try:
            t, y = load_csv(p, time_col, y_col)
            res = detect_on_xy(t, y, prefer_sensitivity=prefer_sensitivity)
            png_path = out_dir / p.with_suffix(".png").name
            title = f"{p.name}"
            plot_and_save(t, y, res, png_path, title=title)
            results.append({
                "file": str(p),
                "has_dip": res["has_dip"],
                "baseline_a": res["a"],
                "center": res["center"],
                "width": res["width"],
                "depth": res["depth"],
                "snr": res["snr"],
                "improvement": res["improvement"],
                "note": res.get("reason", ""),
                "image": str(png_path),
            })
            print(f"[OK] {p.name}: has_dip={res['has_dip']} "
                  f"center={res['center']:.3g} width={res['width']:.3g} "
                  f"depth={res['depth']:.3g} SNR={res['snr']:.2f} imp={100*res['improvement']:.1f}%")
        except Exception as e:
            print(f"[ERROR] {p}: {e}", file=sys.stderr)

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dip_summary.csv"
    pd.DataFrame(results).to_csv(summary_path, index=False)
    return summary_path

def main():
    parser = argparse.ArgumentParser(description="Detect low-valued intervals in scatter charts (JAX).")
    parser.add_argument("folder", type=str, help="Root folder to search recursively for CSV files.")
    parser.add_argument("--time-col", type=str, default=None, help="Name of time/x column (optional).")
    parser.add_argument("--y-col", type=str, default=None, help="Name of y/flux column (optional).")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for images and summary CSV.")
    parser.add_argument("--strict", action="store_true", help="Be stricter (fewer false positives, maybe more false negatives).")

    args = parser.parse_args()
    root = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "_dip_outputs")

    summary_path = process_folder(root, out_dir, args.time_col, args.y_col,
                                  prefer_sensitivity=(not args.strict))

    print(f"\nSummary written to: {summary_path}\nImages saved to: {out_dir}")

if __name__ == "__main__":
    main()
