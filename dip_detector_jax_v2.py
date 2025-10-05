# We'll implement a JAX-based quantile regression approach to detect "dip" intervals in scatter plots.
# Then we'll run it on the three provided CSVs and save annotated images for each.
import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Try to import JAX; if unavailable, we'll fallback to numpy-only quantile smoothing.
use_jax = True
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, value_and_grad
except Exception as e:
    print("JAX not available; falling back to NumPy. Reason:", e)
    use_jax = False

# -----------------------------
# Utility functions
# -----------------------------
def load_xy_from_csv(path: Path):
    df = pd.read_csv(path)
    # Identify numeric columns
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not num_cols or len(num_cols) < 2:
        raise ValueError(f"File {path} doesn't have at least two numeric columns.")
    # Prefer columns named x/y if present
    x_col = None
    y_col = None
    for c in df.columns:
        lc = str(c).strip().lower()
        if lc in ("x", "x_value", "xval") and pd.api.types.is_numeric_dtype(df[c]):
            x_col = c
        if lc in ("y", "y_value", "yval") and pd.api.types.is_numeric_dtype(df[c]):
            y_col = c
    if x_col is None or y_col is None:
        # Just take the first two numeric columns as (x, y)
        x_col, y_col = num_cols[0], num_cols[1]
    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col].to_numpy(dtype=float)
    # Remove NaNs
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    return x, y, (x_col, y_col)

def robust_spread(y):
    # Use IQR as robust spread
    q25, q75 = np.percentile(y, [25, 75])
    return q75 - q25

# -----------------------------
# JAX Quantile Regression with RBF basis
# -----------------------------
def fit_quantile_curve_jax(x, y, tau=0.1, num_centers=30, l2=1e-2, iters=2000, lr=0.05, seed=0):
    """
    Fit a smooth quantile curve q_tau(x) ≈ y using RBF basis and pinball loss with JAX.
    Returns callable f(x_grid)->q_tau.
    """
    key = jax.random.PRNGKey(seed)
    # Normalize x to [0, 1] for stable lengthscale
    xmin, xmax = np.min(x), np.max(x)
    xscale = xmax - xmin if xmax > xmin else 1.0
    xn = (x - xmin) / xscale

    # RBF centers across [0,1]
    centers = jnp.linspace(0.0, 1.0, num_centers)
    lengthscale = 0.08  # heuristic; relatively smooth

    def design(xn_vec):
        # Phi: [N, K]
        xn_vec = jnp.atleast_1d(xn_vec)
        diffs = xn_vec[:, None] - centers[None, :]
        phi = jnp.exp(-0.5 * (diffs / lengthscale) ** 2)
        # Add bias term
        phi = jnp.concatenate([jnp.ones((phi.shape[0], 1)), phi], axis=1)
        return phi

    Xphi = design(jnp.asarray(xn))
    K = Xphi.shape[1]

    # Initialize weights
    w = jax.random.normal(key, (K,)) * 0.01

    def predict(w, xn_vec):
        return design(xn_vec) @ w

    def pinball_loss(residual, tau):
        return jnp.maximum(tau * residual, (tau - 1.0) * residual)

    yj = jnp.asarray(y)

    @jit
    def loss_fn(w):
        pred = Xphi @ w
        res = yj - pred
        loss = jnp.mean(pinball_loss(res, tau)) + l2 * jnp.sum(w[1:] ** 2)
        return loss

    value_and_grad_fn = value_and_grad(loss_fn)

    w_curr = w
    for i in range(iters):
        val, g = value_and_grad_fn(w_curr)
        w_curr = w_curr - lr * g
        # Optional: simple learning rate decay
        if (i+1) % 500 == 0:
            lr *= 0.5

    def f(x_grid):
        xg = jnp.asarray(x_grid)
        xgn = (xg - xmin) / xscale
        return np.array(predict(w_curr, xgn))  # return NumPy array

    return f

def fit_quantile_curve_numpy(x, y, tau=0.1, bins=120, smooth=5):
    """
    Fallback: bin-based lower quantile curve using NumPy, with convolution smoothing.
    """
    order = np.argsort(x)
    x_sorted, y_sorted = x[order], y[order]
    # Bin by quantiles of x
    edges = np.linspace(0, 1, bins + 1)
    qs = np.quantile(x_sorted, edges)
    # For each bin, compute quantile of y
    x_centers = []
    y_q = []
    for i in range(bins):
        left, right = qs[i], qs[i+1]
        m = (x_sorted >= left) & (x_sorted <= right)
        if not np.any(m):
            continue
        x_centers.append(0.5 * (left + right))
        y_q.append(np.quantile(y_sorted[m], tau))
    x_centers = np.array(x_centers)
    y_q = np.array(y_q)
    # Smooth with simple moving average
    if smooth > 1 and len(y_q) > 0:
        k = np.ones(smooth) / smooth
        y_q = np.convolve(y_q, k, mode='same')
    # Interpolate back to x grid
    def f(xg):
        return np.interp(xg, x_centers, y_q, left=y_q[0], right=y_q[-1])
    return f

# -----------------------------
# Dip detection
# -----------------------------
def detect_dip_interval(x, y, fig_path=None, title=None, verbose=False):
    """
    Detect a "dip" interval where lower-quantile curve q10(x) drops significantly below baseline.
    Returns dict with fields: is_positive, x_start, x_end, depth, method, notes
    """
    # Robust scale
    s = robust_spread(y)
    if s <= 0:
        s = np.std(y) + 1e-9

    # Fit quantile curves
    if use_jax:
        q10_func = fit_quantile_curve_jax(x, y, tau=0.1, num_centers=35, l2=1e-2, iters=1500, lr=0.05, seed=0)
        q90_func = fit_quantile_curve_jax(x, y, tau=0.9, num_centers=35, l2=1e-2, iters=1500, lr=0.05, seed=1)
        method = "jax_rbf_quantile"
    else:
        q10_func = fit_quantile_curve_numpy(x, y, tau=0.1, bins=120, smooth=7)
        q90_func = fit_quantile_curve_numpy(x, y, tau=0.9, bins=120, smooth=7)
        method = "numpy_bin_quantile"

    # Evaluate on grid (exclude extreme x noise by focusing on 1%..99%)
    x_lo, x_hi = np.percentile(x, [1, 99])
    grid = np.linspace(x_lo, x_hi, 400)
    q10 = q10_func(grid)
    q90 = q90_func(grid)

    # "Rectangle fill" check — if vertical spread is huge and roughly constant, treat as baseline (no dip).
    y_range = np.percentile(y, 99.5) - np.percentile(y, 0.5)
    spread_median = np.median(q90 - q10)
    rectangle_like = (spread_median > 0.8 * y_range)  # largely fills vertical range
    if rectangle_like:
        is_positive = False
        x_start = x_end = None
        depth = 0.0
        notes = "rectangle-like distribution → treat as y=a"
    else:
        # Baseline for q10: use median of q10 across grid
        q10_base = np.median(q10)
        d = q10_base - q10  # positive where local bottom is below baseline

        # Threshold for a meaningful dip
        # Use robust variability of q10 curve
        mad_q10 = np.median(np.abs(q10 - np.median(q10))) + 1e-9
        # Depth threshold scales with data spread; keep it relatively sensitive to minimize FN
        depth_thresh = max(0.12 * s, 2.0 * mad_q10)

        # Identify regions above threshold
        mask = d > depth_thresh
        # Find contiguous segments
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

        x_start = x_end = None
        depth = 0.0
        is_positive = False
        notes = ""

        if segs:
            # Score segments by "area" (integrated depth) and minimum depth
            best = None
            for (a, b) in segs:
                area = np.trapz(d[a:b+1], grid[a:b+1])
                min_depth = np.max(d[a:b+1])  # peak drop relative to baseline
                width = grid[b] - grid[a]
                # Basic width requirement to avoid tiny noise
                width_ok = width >= 0.03 * (np.max(x) - np.min(x))
                if width_ok:
                    if (best is None) or (area > best[0]):
                        best = (area, min_depth, a, b)
            if best is not None:
                area, min_depth, a, b = best

                # Ensure V/U shape: q10 should recover on both sides (valley)
                left_ok = (a > 5) and (d[a-5] < depth_thresh * 0.6)
                right_ok = (b < len(d) - 6) and (d[b+5] < depth_thresh * 0.6)
                valley_ok = left_ok and right_ok

                # Top-missing check: ensure q90 doesn't drop similarly
                q90_base = np.median(q90)
                t = q90_base - q90
                top_drop = np.median(t[a:b+1])
                bot_drop = np.median(d[a:b+1])
                top_ratio = top_drop / (bot_drop + 1e-9)

                top_missing_ok = top_ratio < 0.6  # bottom drop dominates

                if valley_ok and min_depth > depth_thresh and top_missing_ok:
                    is_positive = True
                    x_start, x_end = grid[a], grid[b]
                    depth = float(min_depth)
                else:
                    notes = f"rejected: valley_ok={valley_ok}, min_depth={min_depth:.3g}, top_ratio={top_ratio:.2g}"

    # Plot and save
    if fig_path is not None:
        plt.figure(figsize=(8, 5))
        plt.scatter(x, y, s=6, alpha=0.7)
        plt.plot(grid, q10, linewidth=2, alpha=0.9)
        plt.plot(grid, q90, linewidth=2, alpha=0.9)
        if is_positive and (x_start is not None):
            plt.axvspan(x_start, x_end, alpha=0.15)
        if title:
            plt.title(title)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.tight_layout()
        plt.savefig(fig_path, dpi=150)
        plt.close()

    return {
        "is_positive": bool(is_positive),
        "x_start": None if x_start is None else float(x_start),
        "x_end": None if x_end is None else float(x_end),
        "depth": float(depth),
        "method": "jax" if use_jax else "numpy",
        "notes": notes,
    }

# -----------------------------
# Batch processing over a directory
# -----------------------------
def process_directory(root_dir, out_dir):
    root = Path(root_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for csv_path in root.rglob("*.csv"):
        try:
            x, y, cols = load_xy_from_csv(csv_path)
            img_path = out / f"{csv_path.stem}_annotated.png"
            res = detect_dip_interval(x, y, fig_path=img_path, title=csv_path.name)
            rows.append({
                "file": str(csv_path),
                "x_col": cols[0],
                "y_col": cols[1],
                **res,
                "image_path": str(img_path),
            })
        except Exception as e:
            rows.append({
                "file": str(csv_path),
                "x_col": None,
                "y_col": None,
                "is_positive": None,
                "x_start": None,
                "x_end": None,
                "depth": None,
                "method": "jax" if use_jax else "numpy",
                "notes": f"error: {e}",
                "image_path": "",
            })
    return pd.DataFrame(rows)

# -----------------------------
# Run on the three provided CSVs and show a summary table
# -----------------------------
base_dir = "./result/positive"
out_dir = "./result/res"
files = ["star0053.csv", "star0129.csv", "star0159.csv"]
# Ensure output directory exists
Path(out_dir).mkdir(parents=True, exist_ok=True)

summary_rows = []
for fname in files:
    fpath = Path(base_dir) / fname
    if not fpath.exists():
        summary_rows.append({"file": str(fpath), "status": "missing"})
        continue
    x, y, cols = load_xy_from_csv(fpath)
    img_path = Path(out_dir) / f"{fpath.stem}_annotated.png"
    res = detect_dip_interval(x, y, fig_path=img_path, title=fpath.name)
    summary_rows.append({
        "file": str(fpath),
        "x_col": cols[0],
        "y_col": cols[1],
        **res,
        "image_path": str(img_path),
        "status": "ok",
    })

summary_df = pd.DataFrame(summary_rows)
import caas_jupyter_tools
caas_jupyter_tools.display_dataframe_to_user("Dip detection summary", summary_df)

print("Saved images in:", out_dir)
print("You can download them individually, e.g.:")
for row in summary_rows:
    if "image_path" in row and row.get("status") == "ok":
        print(row["image_path"])
