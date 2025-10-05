
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dip detector for scatter charts.

Core idea:
- Model lower (q10) and upper (q90) envelopes vs x using quantile regression.
- Flag a dip interval only if:
  * lower envelope drops significantly vs its baseline (robust threshold),
  * the shape is valley-like (recovers on both sides),
  * the top is not missing (upper envelope doesn't drop similarly),
  * and distribution isn't a "filled rectangle" (huge near-constant vertical spread).

Implements a JAX RBF-quantile-regression version (preferred) and a NumPy fallback.

Usage:
    python dip_detector.py <root_folder_with_csvs> --out dip_results

You can tweak sensitivity:
    --depth_k 0.1 --min_width_frac 0.02

CSV assumptions:
- Has a header. The first two numeric columns are used as (x, y) by default,
  or columns named 'x' and 'y' (case-insensitive) if present.

Outputs:
- Annotated PNG per CSV in the output folder.
- summary.csv with detection results.

Author: ChatGPT
"""
import argparse
import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Try to import JAX
USE_JAX = True
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
except Exception:
    USE_JAX = False

# -----------------------------
# Data utilities
# -----------------------------
def load_xy_from_csv(path: Path):
    df = pd.read_csv(path)
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(num_cols) < 2:
        raise ValueError(f"{path} must have at least two numeric columns.")
    # Prefer explicit x,y columns
    x_col = None
    y_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("x", "x_value", "xval") and pd.api.types.is_numeric_dtype(df[c]):
            x_col = c
        if lc in ("y", "y_value", "yval") and pd.api.types.is_numeric_dtype(df[c]):
            y_col = c
    if x_col is None or y_col is None:
        x_col, y_col = num_cols[0], num_cols[1]
    x = df[x_col].astype(float).to_numpy()
    y = df[y_col].astype(float).to_numpy()
    m = np.isfinite(x) & np.isfinite(y)
    return x[m], y[m], (x_col, y_col)

def robust_spread(y):
    q25, q75 = np.percentile(y, [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        return float(np.std(y) + 1e-12)
    return float(iqr)

# -----------------------------
# Quantile curve fitters
# -----------------------------
def fit_quantile_curve_jax(x, y, tau=0.1, num_centers=30, l2=1e-2, iters=2000, lr=0.05, seed=0):
    key = jax.random.PRNGKey(seed)
    xmin, xmax = float(np.min(x)), float(np.max(x))
    xscale = xmax - xmin if xmax > xmin else 1.0
    xn = (x - xmin) / xscale

    centers = jnp.linspace(0.0, 1.0, num_centers)
    lengthscale = 0.08

    def design(xn_vec):
        xn_vec = jnp.atleast_1d(xn_vec)
        diffs = xn_vec[:, None] - centers[None, :]
        phi = jnp.exp(-0.5 * (diffs / lengthscale) ** 2)
        phi = jnp.concatenate([jnp.ones((phi.shape[0], 1)), phi], axis=1)
        return phi

    Xphi = design(jnp.asarray(xn))
    K = Xphi.shape[1]

    w = jax.random.normal(key, (K,)) * 0.01
    yj = jnp.asarray(y)

    def predict(w, xn_vec):
        return design(xn_vec) @ w

    def pinball_loss(residual, tau):
        return jnp.maximum(tau * residual, (tau - 1.0) * residual)

    @jit
    def loss_fn(w):
        pred = Xphi @ w
        res = yj - pred
        loss = jnp.mean(pinball_loss(res, tau)) + l2 * jnp.sum(w[1:] ** 2)
        return loss

    value_and_grad_fn = value_and_grad(loss_fn)
    w_curr = w
    _lr = lr
    for i in range(iters):
        val, g = value_and_grad_fn(w_curr)
        w_curr = w_curr - _lr * g
        if (i + 1) % 500 == 0:
            _lr *= 0.5

    def f(x_grid):
        xg = jnp.asarray(x_grid)
        xgn = (xg - xmin) / xscale
        return np.array(predict(w_curr, xgn))
    return f

def fit_quantile_curve_numpy(x, y, tau=0.1, bins=120, smooth=7):
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]

    edges = np.linspace(0, 1, bins + 1)
    qs = np.quantile(x_sorted, edges)

    x_centers, y_q = [], []
    for i in range(bins):
        left, right = qs[i], qs[i + 1]
        m = (x_sorted >= left) & (x_sorted <= right)
        if np.any(m):
            x_centers.append(0.5 * (left + right))
            y_q.append(np.quantile(y_sorted[m], tau))
    x_centers = np.array(x_centers)
    y_q = np.array(y_q)
    if smooth > 1 and len(y_q) > 0:
        k = np.ones(smooth) / smooth
        y_q = np.convolve(y_q, k, mode='same')

    def f(xg):
        return np.interp(xg, x_centers, y_q, left=y_q[0], right=y_q[-1])
    return f

# -----------------------------
# Dip detection
# -----------------------------
def detect_dip_interval(
    x, y, *,
    fig_path=None, title=None,
    tau_lower=0.1, tau_upper=0.9,
    depth_k=0.12,          # threshold scale vs robust spread
    min_width_frac=0.03,   # min width vs x-range
    use_numpy_fallback=not USE_JAX,
    numpy_bins=120,
    numpy_smooth=7,
    verbose=False
):
    s = robust_spread(y)
    if USE_JAX and not use_numpy_fallback:
        q_lo = fit_quantile_curve_jax(x, y, tau=tau_lower, num_centers=35, l2=1e-2, iters=1500, lr=0.05, seed=0)
        q_hi = fit_quantile_curve_jax(x, y, tau=tau_upper, num_centers=35, l2=1e-2, iters=1500, lr=0.05, seed=1)
        method = "jax_rbf_quantile"
    else:
        q_lo = fit_quantile_curve_numpy(x, y, tau=tau_lower, bins=numpy_bins, smooth=numpy_smooth)
        q_hi = fit_quantile_curve_numpy(x, y, tau=tau_upper, bins=numpy_bins, smooth=numpy_smooth)
        method = "numpy_bin_quantile"

    x_lo, x_hi = np.percentile(x, [1, 99])
    grid = np.linspace(x_lo, x_hi, 400)
    lo = q_lo(grid)
    hi = q_hi(grid)

    # Rectangle-like rejection
    y_range = np.percentile(y, 99.5) - np.percentile(y, 0.5)
    spread_median = float(np.median(hi - lo))
    rectangle_like = (spread_median > 0.8 * y_range)
    if rectangle_like:
        if verbose:
            print("rectangle-like distribution -> no dip")
        pos = False; xs = xe = None; depth = 0.0; notes = "rectangle-like"
    else:
        lo_base = float(np.median(lo))
        d = lo_base - lo

        mad_lo = float(np.median(np.abs(lo - np.median(lo))) + 1e-12)
        depth_thresh = max(depth_k * s, 2.0 * mad_lo)

        mask = d > depth_thresh
        segs = []
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                segs.append((i, j))
                i = j + 1
            else:
                i += 1

        xs = xe = None
        depth = 0.0
        pos = False
        notes = ""

        if segs:
            rng = float(np.max(x) - np.min(x))
            best = None
            for (a, b) in segs:
                width = grid[b] - grid[a]
                if width < min_width_frac * rng:
                    continue
                area = float(np.trapz(d[a:b+1], grid[a:b+1]))
                min_depth = float(np.max(d[a:b+1]))
                if (best is None) or (area > best[0]):
                    best = (area, min_depth, a, b)
            if best is not None:
                area, min_depth, a, b = best
                left_ok = (a > 5) and (d[a - 5] < 0.6 * depth_thresh)
                right_ok = (b < len(d) - 6) and (d[b + 5] < 0.6 * depth_thresh)
                valley_ok = bool(left_ok and right_ok)

                hi_base = float(np.median(hi))
                t = hi_base - hi
                top_drop = float(np.median(t[a:b+1]))
                bot_drop = float(np.median(d[a:b+1]))
                top_ratio = top_drop / (bot_drop + 1e-12)
                top_missing_ok = top_ratio < 0.6

                if valley_ok and (min_depth > depth_thresh) and top_missing_ok:
                    pos = True
                    xs, xe = float(grid[a]), float(grid[b])
                    depth = float(min_depth)
                else:
                    notes = f"rejected: valley_ok={valley_ok}, min_depth={min_depth:.3g}, top_ratio={top_ratio:.2g}"

    if fig_path is not None:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, s=6, alpha=0.7)
        plt.plot(grid, lo, linewidth=2, alpha=0.9, label=f"q{int(tau_lower*100)}")
        plt.plot(grid, hi, linewidth=2, alpha=0.9, label=f"q{int(tau_upper*100)}")
        if pos and (xs is not None):
            plt.axvspan(xs, xe, alpha=0.15, label="dip")
        if title:
            plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    return {
        "is_positive": bool(pos),
        "x_start": None if xs is None else float(xs),
        "x_end": None if xe is None else float(xe),
        "depth": float(depth),
        "method": method,
        "notes": notes,
    }

# -----------------------------
# Batch processing
# -----------------------------
def process_directory(root_dir, out_dir, **kwargs):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in root.rglob("*.csv"):
        try:
            x, y, cols = load_xy_from_csv(csv_path)
            img_path = out / f"{csv_path.stem}_annotated.png"
            res = detect_dip_interval(
                x, y, fig_path=str(img_path), title=csv_path.name, **kwargs
            )
            rows.append({
                "file": str(csv_path),
                "x_col": cols[0],
                "y_col": cols[1],
                **res,
                "image_path": str(img_path),
            })
            print(f"[OK] {csv_path} -> {res['is_positive']} interval=({res['x_start']}, {res['x_end']})")
        except Exception as e:
            print(f"[ERR] {csv_path}: {e}", file=sys.stderr)
            rows.append({
                "file": str(csv_path),
                "x_col": None, "y_col": None,
                "is_positive": None,
                "x_start": None, "x_end": None,
                "depth": None,
                "method": "jax" if USE_JAX else "numpy",
                "notes": f"error: {e}",
                "image_path": "",
            })
    df = pd.DataFrame(rows)
    sum_csv = Path(out_dir) / "summary.csv"
    df.to_csv(sum_csv, index=False)
    print(f"\nSummary written to: {sum_csv}")
    return df

# -----------------------------
# CLI
# -----------------------------
def main():
    ap = argparse.ArgumentParser(description="Scatter dip detector (JAX/NumPy)")
    ap.add_argument("root", type=str, help="Root folder to recursively search for CSV files.")
    ap.add_argument("--out", type=str, default="dip_results", help="Output folder for images and summary.csv")
    ap.add_argument("--tau_lower", type=float, default=0.10, help="Lower quantile for bottom envelope (e.g., 0.10)")
    ap.add_argument("--tau_upper", type=float, default=0.90, help="Upper quantile for top envelope (e.g., 0.90)")
    ap.add_argument("--depth_k", type=float, default=0.12, help="Depth threshold coefficient relative to robust spread")
    ap.add_argument("--min_width_frac", type=float, default=0.03, help="Minimum interval width as fraction of x-range")
    ap.add_argument("--numpy_bins", type=int, default=120, help="NumPy fallback: number of x-bins")
    ap.add_argument("--numpy_smooth", type=int, default=7, help="NumPy fallback: moving-average window")
    ap.add_argument("--force_numpy", action="store_true", help="Force NumPy fallback even if JAX is available")
    ap.add_argument("--verbose", action="store_true", help="Verbose logs")
    args = ap.parse_args()

    kwargs = dict(
        tau_lower=args.tau_lower,
        tau_upper=args.tau_upper,
        depth_k=args.depth_k,
        min_width_frac=args.min_width_frac,
        use_numpy_fallback=args.force_numpy,
        numpy_bins=args.numpy_bins,
        numpy_smooth=args.numpy_smooth,
        verbose=args.verbose
    )
    process_directory(args.root, args.out, **kwargs)

if __name__ == "__main__":
    main()
