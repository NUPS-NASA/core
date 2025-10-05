
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
JAX-based detector for a single "low-valued interval" (box-shaped dip) in noisy scatter plots.

Goals
-----
- Minimize false negatives (prefer to flag a plausible dip rather than miss it).
- Be robust to heavy noise (esp. near x-edges) and huge y spread.
- If the vertical distribution looks "filled rectangle", treat as y≈a (no dip).
- Save annotated PNG per CSV and a summary CSV aggregating results.

Usage
-----
python dip_detector_jax.py /path/to/folder \
  --time-col TIME --y-col Y \
  --out-dir /path/to/output

Options
-------
--strict            : stricter decision rule (fewer FPs, possibly more FNs)
--imp-override 0.08 : improvement threshold for FN-minimizing OR rule
--snr 0.8           : base SNR threshold (sensitivity mode)
--imp 0.02          : base improvement threshold (sensitivity mode)

Requires
--------
pip install -U "jax[cpu]" optax pandas matplotlib
(Or CUDA JAX if you have GPU.)

Author: ChatGPT
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- JAX / Optax ----
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    import optax
except Exception as e:
    raise SystemExit(
        "This script requires JAX and Optax.\n"
        "Install with: pip install --upgrade 'jax[cpu]' optax\n"
        f"Import error: {e}"
    )

# ---------------- Utilities ----------------

def robust_mad(x: np.ndarray) -> float:
    """Median absolute deviation scaled to ~sigma (robust)."""
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)

def find_columns(df: pd.DataFrame, time_col: Optional[str], y_col: Optional[str]) -> Tuple[str, str]:
    """Heuristically pick time and y columns if not provided."""
    if time_col is not None and y_col is not None:
        return time_col, y_col

    candidates = {c.lower(): c for c in df.columns}

    time_aliases = ["time since first frame", "time", "t", "x", "timestamp", "frame", "hour", "jd"]
    y_aliases = ["relative flux", "flux", "y", "value", "intensity", "detrended"]

    def guess(aliases):
        for a in aliases:
            if a in candidates:
                return candidates[a]
            for k, v in candidates.items():
                if k.startswith(a):
                    return v
        return None

    tc = time_col or guess(time_aliases)
    yc = y_col or guess(y_aliases)

    if tc is None or yc is None:
        num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(num_cols) >= 2:
            tc = tc or num_cols[0]
            yc = yc or num_cols[1]

    if tc is None or yc is None:
        raise ValueError("Could not infer time/y columns. Pass --time-col and --y-col.")

    return tc, yc

def rectangular_y_distribution(y: np.ndarray, bins: int = 50) -> float:
    """
    Heuristic 'rectangularness' score in [0,1]. Higher => y values fill the
    vertical range like a rectangle (treat as y≈a).
    """
    if len(y) < 10:
        return 0.0
    hist, _ = np.histogram(y, bins=bins)
    occ_frac = (hist > 0).mean()
    spread = np.percentile(y, 97.5) - np.percentile(y, 2.5)
    sigma = robust_mad(y)
    wide = np.clip(spread / (sigma + 1e-12), 0, 20) / 20.0
    return float(0.6 * occ_frac + 0.4 * wide)

# ---------------- Smooth box model (JAX) ----------------

def soft_box_jax(t: jnp.ndarray, c: jnp.ndarray, w: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    """
    Smooth indicator of interval centered at c with width w.
    Returns ~1 inside, ~0 outside, with smooth edges controlled by tau.
    """
    left = c - 0.5 * w
    right = c + 0.5 * w
    s1 = jax.nn.sigmoid((t - left) / tau)
    s2 = jax.nn.sigmoid((t - right) / tau)
    return jnp.clip(s1 - s2, 0.0, 1.0)

def huber_loss(residuals: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
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
    """Weighted Huber loss between y and yhat = a - d * soft_box + regularization."""
    tmin = t.min()
    tmax = t.max()

    a = params["a"]
    d = jax.nn.softplus(params["d_raw"])  # depth >= 0
    c = tmin + (tmax - tmin) * jax.nn.sigmoid(params["c_sig"])  # c in [tmin, tmax]
    w = w_min + (w_max - w_min) * jax.nn.sigmoid(params["w_sig"])  # w in [w_min, w_max]

    box = soft_box_jax(t, c, w, tau)
    yhat = a - d * box

    res = (y - yhat) * w_weights
    loss = jnp.sum(huber_loss(res, delta))

    width_reg = lam_width * jnp.exp(-(w / (w_min + 1e-6)))  # discourage super tiny widths
    amp_reg = lam_amp * (d ** 2)                            # mild amplitude regularization
    return loss + width_reg + amp_reg

def optimize_dip(t: np.ndarray, y: np.ndarray, seed: int = 0) -> Dict[str, Any]:
    """
    Optimize parameters with random restarts. Returns model params and diagnostics.
    """
    key = jax.random.PRNGKey(seed)

    t_j = jnp.asarray(t)
    y_j = jnp.asarray(y)

    # robust scale & edge weights
    sigma = float(robust_mad(y))
    delta = np.float32(1.345 * sigma)

    n = len(t)
    edge = np.linspace(0, 1, n)
    w_edge = (1 - np.exp(-5 * np.minimum(edge, 1 - edge)))     # near-edges downweighted
    w_weights = jnp.asarray(0.25 + 0.75 * w_edge)

    # Smoothness
    tspan = float(np.max(t) - np.min(t))
    tau = float(max(1e-9, 0.01 * (tspan + 1e-12)))              # avoid 0

    # width constraints
    w_min = 0.05 * (tspan + 1e-12)
    w_max = 0.80 * (tspan + 1e-12)

    # optimization hyperparams
    lam_width = 1.0
    lam_amp = 1e-4
    steps = 1000
    lr = 0.02

    y_med = float(np.median(y))

    # Stable parameterization helpers
    def safe_logit(p: float, eps: float = 1e-6) -> float:
        p = np.clip(p, eps, 1 - eps)
        return float(np.log(p / (1 - p)))

    def inv_softplus(d: float, eps: float = 1e-12) -> float:
        d = max(0.0, d)
        return float(np.log(np.expm1(d) + eps))

    # pack/unpack
    tmin = float(np.min(t))
    tmax = float(np.max(t))
    span = max(tmax - tmin, 1e-9)

    def pack(a: float, d: float, c: float, w: float):
        frac_c = (c - tmin) / span
        alpha = (w - w_min) / max(w_max - w_min, 1e-9)
        c_sig = safe_logit(frac_c)
        w_sig = safe_logit(alpha)
        d_raw = inv_softplus(d)
        return {"a": np.float32(a), "d_raw": np.float32(d_raw),
                "c_sig": np.float32(c_sig), "w_sig": np.float32(w_sig)}

    # init near the deepest point + random restarts
    c0 = t[np.argmin(y)]
    w0 = float(np.clip(4 * np.median(np.diff(np.sort(t))), w_min, w_max))
    inits = [pack(y_med, max(0.0, y_med - float(np.min(y))), float(c0), float(w0))]

    for _ in range(5):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        a_i = y_med + 0.1 * sigma * float(jax.random.normal(k1))
        d_i = abs(0.5 * sigma * float(jax.random.normal(k2))) + 0.1 * sigma
        c_i = float(tmin + span * jax.random.uniform(k3))
        w_i = float(w_min + (w_max - w_min) * jax.random.uniform(k4))
        inits.append(pack(a_i, d_i, c_i, w_i))

    opt = optax.adam(lr)

    # build objective
    obj = lambda p: objective(
        p, t_j, y_j, jnp.asarray(w_weights),
        jnp.asarray(tau), jnp.asarray(delta),
        jnp.asarray(w_min), jnp.asarray(w_max),
        jnp.asarray(lam_width), jnp.asarray(lam_amp)
    )

    best = None
    best_val = np.inf

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

    # Decode best
    a = best["a"]
    d = float(np.log1p(np.exp(best["d_raw"])))  # softplus
    c = tmin + span * (1 / (1 + np.exp(-best["c_sig"])))
    w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-best["w_sig"])))

    # Predictions (numpy, numerically stable)
    tau_safe = float(max(tau, 1e-9 * (span + 1.0)))
    z1 = (t - (c - 0.5 * w)) / tau_safe
    z2 = (t - (c + 0.5 * w)) / tau_safe
    z1 = np.clip(z1, -60, 60)
    z2 = np.clip(z2, -60, 60)
    s1 = 1.0 / (1.0 + np.exp(-z1))
    s2 = 1.0 / (1.0 + np.exp(-z2))
    box = s1 - s2
    yhat = a - d * box

    # Comparable baseline loss with Huber (a fixed at robust median)
    a0 = float(np.median(y))
    def huber_np(r, delta):
        ar = np.abs(r); quad = 0.5*(ar**2); lin = delta*(ar-0.5*delta)
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
                 prefer_sensitivity: bool = True,
                 imp_override: float = 0.08,
                 base_snr: float = 0.8,
                 base_imp: float = 0.02) -> Dict[str, Any]:
    """
    Run detection on a single time series (t, y).
    prefer_sensitivity=True uses FN-minimizing rule:
      has_dip = (SNR>=base_snr AND improvement>=base_imp AND support ok)
                OR (improvement>=imp_override AND support ok)
    """
    # Clean/sort
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    order = np.argsort(t); t = t[order]; y = y[order]

    # Rectangular distribution => treat as y=a
    rect_score = rectangular_y_distribution(y, bins=50)
    is_rect = rect_score >= 0.75

    # Optimize dip model
    fit = optimize_dip(t, y)

    # Support points inside the interval
    min_points = max(4, int(0.05 * len(t)))
    inside = (t >= (fit["center"] - 0.5 * fit["width"])) & (t <= (fit["center"] + 0.5 * fit["width"]))
    support = int(np.sum(inside))

    # Base thresholds
    if not prefer_sensitivity:
        base_snr = 1.1
        base_imp = 0.05

    rule_and = (fit["snr"] >= base_snr) and (fit["improvement"] >= base_imp) and (support >= min_points)
    rule_or = (fit["improvement"] >= imp_override) and (support >= min_points)

    has_dip = (not is_rect) and (rule_and or (prefer_sensitivity and rule_or))

    reason = None
    if is_rect:
        reason = f"Vertical distribution looks filled (score={rect_score:.2f}); treat as y=a."
    elif not has_dip:
        reason = "Insufficient SNR/improvement/support for a reliable interval."

    return {
        "has_dip": bool(has_dip),
        "reason": reason,
        "a": fit["a"],
        "depth": fit["depth"],
        "center": fit["center"],
        "width": fit["width"],
        "snr": fit["snr"],
        "improvement": fit["improvement"],
        "rect_score": rect_score,
        "support_points": support,
    }

def plot_and_save(t: np.ndarray, y: np.ndarray, res: Dict[str, Any], out_png: Path, title: str = "") -> None:
    plt.figure(figsize=(9, 5.2))
    plt.scatter(t, y, s=18)
    plt.xlabel("Time")
    plt.ylabel("Y")
    if title: plt.title(title)

    # Baseline
    plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")

    if res["has_dip"] and np.isfinite(res["center"]):
        s = res["center"] - 0.5 * res["width"]
        e = res["center"] + 0.5 * res["width"]
        plt.axvspan(s, e, alpha=0.2, label="Detected interval")
        plt.text(s, res["a"],
                 f"  depth≈{res['depth']:.4g}, SNR≈{res['snr']:.2f}, imp≈{100*res['improvement']:.1f}%",
                 va="bottom", ha="left")
    else:
        if res.get("reason"):
            # top-left note
            xmin, xmax = np.min(t), np.max(t)
            ymax = np.max(y)
            plt.text(xmin, ymax, res["reason"], va="top", ha="left")

    plt.legend(frameon=False, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.close()

def load_csv(path: Path, time_col: Optional[str], y_col: Optional[str]) -> Tuple[np.ndarray, np.ndarray, str, str]:
    df = pd.read_csv(path)
    tc, yc = find_columns(df, time_col, y_col)
    t = df[tc].to_numpy(dtype=float)
    y = df[yc].to_numpy(dtype=float)
    return t, y, tc, yc

def process_folder(root: Path, out_dir: Path,
                   time_col: Optional[str], y_col: Optional[str],
                   prefer_sensitivity: bool = True,
                   imp_override: float = 0.08,
                   base_snr: float = 0.8,
                   base_imp: float = 0.02) -> Path:
    rows: List[Dict[str, Any]] = []
    csv_files = sorted(root.rglob("*.csv"))
    if not csv_files:
        print(f"No CSV files found under {root}")

    for p in csv_files:
        try:
            t, y, tc, yc = load_csv(p, time_col, y_col)
            res = detect_on_xy(
                t, y,
                prefer_sensitivity=prefer_sensitivity,
                imp_override=imp_override,
                base_snr=base_snr,
                base_imp=base_imp,
            )
            png_path = out_dir / p.with_suffix(".png").name
            plot_and_save(t, y, res, png_path, title=f"{p.name}")

            rows.append({
                "file": str(p),
                "has_dip": res["has_dip"],
                "baseline_a": res["a"],
                "center": res["center"],
                "width": res["width"],
                "depth": res["depth"],
                "snr": res["snr"],
                "improvement": res["improvement"],
                "support_points": res["support_points"],
                "rect_score": res["rect_score"],
                "note": res.get("reason", ""),
                "image": str(png_path),
                "time_col": tc, "y_col": yc,
            })

            print(f"[OK] {p.name}: has_dip={res['has_dip']} "
                  f"center={res['center']:.6g} width={res['width']:.6g} "
                  f"depth={res['depth']:.3g} SNR={res['snr']:.2f} "
                  f"imp={100*res['improvement']:.1f}% support={res['support_points']}")

        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dip_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path

def main():
    ap = argparse.ArgumentParser(description="Detect low-valued intervals in scatter charts (JAX).")
    ap.add_argument("folder", type=str, help="Root folder to search recursively for CSV files.")
    ap.add_argument("--time-col", type=str, default=None, help="Name of time/x column (optional).")
    ap.add_argument("--y-col", type=str, default=None, help="Name of y/flux column (optional).")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for images and summary CSV.")
    ap.add_argument("--strict", action="store_true", help="Be stricter (fewer FPs, maybe more FNs).")
    ap.add_argument("--imp-override", type=float, default=0.08, help="Improvement threshold for FN-minimizing OR rule.")
    ap.add_argument("--snr", type=float, default=0.8, help="Base SNR threshold for AND rule (sensitivity mode).")
    ap.add_argument("--imp", type=float, default=0.02, help="Base improvement threshold for AND rule (sensitivity mode).")

    args = ap.parse_args()
    root = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "_dip_outputs")

    summary_path = process_folder(
        root, out_dir, args.time_col, args.y_col,
        prefer_sensitivity=(not args.strict),
        imp_override=args.imp_override,
        base_snr=args.snr,
        base_imp=args.imp
    )
    print(f"\nSummary written to: {summary_path}\nImages saved to: {out_dir}")

if __name__ == "__main__":
    main()
