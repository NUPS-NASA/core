
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid detector (coarse NumPy grid + JAX refine) for a single "low-valued interval" (box-shaped dip) in noisy scatter plots.
Designed to minimize false negatives while remaining robust.

Key ideas
---------
1) Coarse grid search (NumPy) quickly finds a plausible dip even if gradient descent struggles.
2) JAX optimizer refines from multiple inits, including the coarse best; but detection can be decided by the coarse result
   when it is strong enough (FN-minimizing guard).
3) If y-values fill a rectangle vertically, treat as y≈a (no dip).

Usage
-----
python dip_detector_jax.py /path/to/folder \
  --time-col TIME --y-col Y \
  --out-dir /path/to/output

Thresholds
----------
- Sensitivity mode (default): rule = (SNR>=snr AND improvement>=imp AND support ok) OR (improvement>=imp_override AND support ok)
  Defaults: snr=0.8, imp=0.02, imp_override=0.08
- Strict mode: snr=1.1, imp=0.05, imp_override ignored

Install
-------
pip install -U "jax[cpu]" optax pandas matplotlib

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

# ---- Try JAX / Optax (used for refinement) ----
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    import optax
    HAVE_JAX = True
except Exception as e:
    print("[WARN] JAX not available; running in coarse-grid mode only. "
          "Install with: pip install --upgrade 'jax[cpu]' optax")
    HAVE_JAX = False

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

# ---------------- Coarse grid search (NumPy) ----------------

def soft_box_np(t: np.ndarray, c: float, wv: float, tau: float) -> np.ndarray:
    taus = max(tau, 1e-9 * (t.max() - t.min() + 1))
    z1 = (t - (c - 0.5 * wv)) / taus
    z2 = (t - (c + 0.5 * wv)) / taus
    z1 = np.clip(z1, -60, 60)
    z2 = np.clip(z2, -60, 60)
    s1 = 1.0 / (1.0 + np.exp(-z1))
    s2 = 1.0 / (1.0 + np.exp(-z2))
    return s1 - s2

def huber_np(r, delta):
    ar = np.abs(r); quad = 0.5*(ar**2); lin = delta*(ar-0.5*delta)
    return np.where(ar <= delta, quad, lin)

def coarse_grid_search(t: np.ndarray, y: np.ndarray,
                       centers: int = 80, widths: int = 25, depths: int = 40) -> Dict[str, float]:
    """FN-oriented coarse search for c,w,d using weighted Huber loss and fixed a=median(y)."""
    sigma = robust_mad(y)
    delta = 1.345 * sigma

    n = len(t)
    edge = np.linspace(0, 1, n)
    w_edge = (1 - np.exp(-5 * np.minimum(edge, 1 - edge)))
    w = 0.25 + 0.75 * w_edge

    a0 = float(np.median(y))
    loss_base = float(np.sum(huber_np((y - a0) * w, delta)))

    tspan = (t.max() - t.min())
    tau = 0.01 * tspan

    C = np.linspace(t.min(), t.max(), centers)
    W = np.linspace(0.05 * tspan, 0.70 * tspan, widths)
    dmax = max(1e-6, a0 - np.min(y))
    D = np.linspace(0.0, dmax * 2.0, depths)

    best = {"loss": np.inf}
    for wv in W:
        for c in C:
            box = soft_box_np(t, c, wv, tau)
            yhat_all = a0 - np.outer(D, box)           # [D, N]
            res = (y - yhat_all) * w                   # [D, N]
            L = np.sum(huber_np(res, delta), axis=1)   # [D]
            idx = int(np.argmin(L))
            Lmin = float(L[idx])
            if Lmin < best["loss"]:
                best = {
                    "loss": Lmin, "center": float(c), "width": float(wv),
                    "depth": float(D[idx]), "a": a0
                }

    improvement = max(0.0, (loss_base - best["loss"]) / (loss_base + 1e-12))
    snr = best["depth"] / (sigma + 1e-12)

    return {
        "a": best["a"],
        "center": best["center"],
        "width": best["width"],
        "depth": best["depth"],
        "improvement": float(improvement),
        "snr": float(snr),
        "loss_base": float(loss_base),
        "loss_model": float(best["loss"]),
        "tau": float(tau),
    }

# ---------------- JAX refine ----------------

def soft_box_jax(t: jnp.ndarray, c: jnp.ndarray, w: jnp.ndarray, tau: jnp.ndarray) -> jnp.ndarray:
    left = c - 0.5 * w
    right = c + 0.5 * w
    s1 = jax.nn.sigmoid((t - left) / tau)
    s2 = jax.nn.sigmoid((t - right) / tau)
    return jnp.clip(s1 - s2, 0.0, 1.0)

def huber_jax(residuals: jnp.ndarray, delta: jnp.ndarray) -> jnp.ndarray:
    abs_r = jnp.abs(residuals)
    quad = 0.5 * (abs_r ** 2)
    lin = delta * (abs_r - 0.5 * delta)
    return jnp.where(abs_r <= delta, quad, lin)

if HAVE_JAX:
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
        tmin = t.min()
        tmax = t.max()

        a = params["a"]
        d = jax.nn.softplus(params["d_raw"])
        c = tmin + (tmax - tmin) * jax.nn.sigmoid(params["c_sig"])
        w = w_min + (w_max - w_min) * jax.nn.sigmoid(params["w_sig"])

        box = soft_box_jax(t, c, w, tau)
        yhat = a - d * box

        res = (y - yhat) * w_weights
        loss = jnp.sum(huber_jax(res, delta))

        width_reg = lam_width * jnp.exp(-(w / (w_min + 1e-6)))
        amp_reg = lam_amp * (d ** 2)
        return loss + width_reg + amp_reg

def jax_refine(t: np.ndarray, y: np.ndarray, seed: int, coarse: Dict[str, float]) -> Dict[str, Any]:
    if not HAVE_JAX:
        return coarse | {"refined": False}

    key = jax.random.PRNGKey(seed)

    t_j = jnp.asarray(t)
    y_j = jnp.asarray(y)

    sigma = float(robust_mad(y))
    delta = np.float32(1.345 * sigma)

    n = len(t)
    edge = np.linspace(0, 1, n)
    w_edge = (1 - np.exp(-5 * np.minimum(edge, 1 - edge)))
    w_weights = jnp.asarray(0.25 + 0.75 * w_edge)

    tmin = float(np.min(t)); tmax = float(np.max(t))
    span = float(max(tmax - tmin, 1e-9))

    tau = float(max(1e-9, 0.01 * (span + 1e-12)))
    w_min = 0.05 * (span + 1e-12)
    w_max = 0.80 * (span + 1e-12)

    lam_width = 1.0; lam_amp = 1e-4
    steps = 1000; lr = 0.02

    # Stable inverse maps
    def safe_logit(p: float, eps: float = 1e-6) -> float:
        p = np.clip(p, eps, 1 - eps); return float(np.log(p / (1 - p)))
    def inv_softplus(d: float, eps: float = 1e-12) -> float:
        d = max(0.0, d); return float(np.log(np.expm1(d) + eps))

    # Seed from coarse + random restarts
    def pack(a: float, d: float, c: float, w: float):
        frac_c = (c - tmin) / span
        alpha = (w - w_min) / max(w_max - w_min, 1e-9)
        c_sig = safe_logit(frac_c)
        w_sig = safe_logit(alpha)
        d_raw = inv_softplus(d)
        return {"a": np.float32(a), "d_raw": np.float32(d_raw),
                "c_sig": np.float32(c_sig), "w_sig": np.float32(w_sig)}

    a0 = float(np.median(y))

    inits: List[Dict[str, float]] = [pack(coarse["a"], coarse["depth"], coarse["center"], coarse["width"])]
    for _ in range(5):
        key, k1, k2, k3, k4 = jax.random.split(key, 5)
        a_i = a0 + 0.1 * sigma * float(jax.random.normal(k1))
        d_i = abs(0.5 * sigma * float(jax.random.normal(k2))) + 0.1 * sigma
        c_i = float(tmin + span * jax.random.uniform(k3))
        w_i = float(w_min + (w_max - w_min) * jax.random.uniform(k4))
        inits.append(pack(a_i, d_i, c_i, w_i))

    opt = optax.adam(lr)
    obj = lambda p: objective(
        p, t_j, y_j, jnp.asarray(w_weights),
        jnp.asarray(tau), jnp.asarray(delta),
        jnp.asarray(w_min), jnp.asarray(w_max),
        jnp.asarray(lam_width), jnp.asarray(lam_amp)
    )

    best_params = None; best_val = np.inf
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
            best_params = {k: float(v) for k, v in params.items()}

    # Decode
    a = best_params["a"]
    d = float(np.log1p(np.exp(best_params["d_raw"])))  # softplus
    c = tmin + span * (1 / (1 + np.exp(-best_params["c_sig"])))
    w = w_min + (w_max - w_min) * (1 / (1 + np.exp(-best_params["w_sig"])))

    # Predictions (stable)
    tau_safe = float(max(tau, 1e-9 * (span + 1.0)))
    z1 = (t - (c - 0.5 * w)) / tau_safe
    z2 = (t - (c + 0.5 * w)) / tau_safe
    z1 = np.clip(z1, -60, 60); z2 = np.clip(z2, -60, 60)
    s1 = 1.0 / (1.0 + np.exp(-z1)); s2 = 1.0 / (1.0 + np.exp(-z2))
    box = s1 - s2
    yhat = a - d * box

    # Comparable losses to coarse
    sigma = robust_mad(y); delta = 1.345 * sigma
    n = len(t); edge = np.linspace(0, 1, n); w_edge = (1 - np.exp(-5 * np.minimum(edge, 1 - edge)))
    w_weights = 0.25 + 0.75 * w_edge
    a0 = float(np.median(y))

    def huber_np(r, dlt): ar = np.abs(r); quad = 0.5*(ar**2); lin = dlt*(ar-0.5*dlt); return np.where(ar<=dlt, quad, lin)
    loss_base = float(np.sum(huber_np((y - a0) * w_weights, delta)))
    loss_model = float(np.sum(huber_np((y - yhat) * w_weights, delta)))
    improvement = max(0.0, (loss_base - loss_model) / (loss_base + 1e-12))
    snr = float(d / (sigma + 1e-12))

    return {
        "a": float(a), "center": float(c), "width": float(w), "depth": float(d),
        "improvement": float(improvement), "snr": float(snr),
        "loss_base": float(loss_base), "loss_model": float(loss_model),
        "refined": True
    }

# ---------------- Detection & I/O ----------------

def detect_on_xy(t: np.ndarray, y: np.ndarray,
                 prefer_sensitivity: bool = True,
                 imp_override: float = 0.08,
                 base_snr: float = 0.8,
                 base_imp: float = 0.02,
                 grid_centers: int = 80,
                 grid_widths: int = 25,
                 grid_depths: int = 40,
                 seed: int = 0) -> Dict[str, Any]:
    """Run detection (grid -> optional refine) and decide with FN-minimizing policy."""
    mask = np.isfinite(t) & np.isfinite(y)
    t = t[mask]; y = y[mask]
    order = np.argsort(t); t = t[order]; y = y[order]

    rect_score = rectangular_y_distribution(y, bins=50)
    is_rect = rect_score >= 0.80  # slightly stricter, only reject when clearly rectangle-like

    # 1) Coarse search (robust baseline)
    coarse = coarse_grid_search(t, y, centers=grid_centers, widths=grid_widths, depths=grid_depths)

    # Support based on coarse interval
    min_points = max(4, int(0.05 * len(t)))
    inside_coarse = (t >= (coarse["center"] - 0.5 * coarse["width"])) & (t <= (coarse["center"] + 0.5 * coarse["width"]))
    support = int(np.sum(inside_coarse))

    # Decide early (FN-minimizing guard)
    rule_or = (coarse["improvement"] >= imp_override) and (support >= min_points)
    if prefer_sensitivity and (not is_rect) and rule_or:
        return {
            "has_dip": True, "reason": None,
            "a": coarse["a"], "depth": coarse["depth"], "center": coarse["center"], "width": coarse["width"],
            "snr": coarse["snr"], "improvement": coarse["improvement"],
            "rect_score": rect_score, "support_points": support, "source": "coarse"
        }

    # 2) Otherwise, refine with JAX (if available)
    refined = jax_refine(t, y, seed=seed, coarse=coarse) if HAVE_JAX else coarse | {"refined": False}

    # Decide with AND / OR depending on mode
    snr_thr = 1.1 if (not prefer_sensitivity) else base_snr
    imp_thr = 0.05 if (not prefer_sensitivity) else base_imp

    # Adaptive SNR relaxation with improvement (optional; helps FN further)
    if prefer_sensitivity:
        snr_thr = float(np.interp(refined["improvement"], [0.00, 0.08, 0.20], [snr_thr, 0.55, 0.35]))

    inside = (t >= (refined["center"] - 0.5 * refined["width"])) & (t <= (refined["center"] + 0.5 * refined["width"]))
    support_ref = int(np.sum(inside))

    rule_and = (refined["snr"] >= snr_thr) and (refined["improvement"] >= imp_thr) and (support_ref >= min_points)
    rule_or2 = prefer_sensitivity and (refined["improvement"] >= imp_override) and (support_ref >= min_points)

    has_dip = (not is_rect) and (rule_and or rule_or2)

    reason = None
    if is_rect:
        reason = f"Vertical distribution looks filled (score={rect_score:.2f}); treat as y≈a."
    elif not has_dip:
        reason = "Insufficient SNR/improvement/support for a reliable interval."

    return {
        "has_dip": bool(has_dip),
        "reason": reason,
        "a": refined["a"],
        "depth": refined["depth"],
        "center": refined["center"],
        "width": refined["width"],
        "snr": refined["snr"],
        "improvement": refined["improvement"],
        "rect_score": rect_score,
        "support_points": support_ref,
        "source": "refined" if HAVE_JAX else "coarse",
    }

def plot_and_save(t: np.ndarray, y: np.ndarray, res: Dict[str, Any], out_png: Path, title: str = "") -> None:
    plt.figure(figsize=(9, 5.2))
    plt.scatter(t, y, s=18)
    plt.xlabel("Time")
    plt.ylabel("Y")
    if title: plt.title(title)

    plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")

    if res["has_dip"] and np.isfinite(res["center"]):
        s = res["center"] - 0.5 * res["width"]
        e = res["center"] + 0.5 * res["width"]
        plt.axvspan(s, e, alpha=0.2, label=f"Detected interval ({res.get('source','')})")
        plt.text(s, res["a"],
                 f"depth≈{res['depth']:.4g}, SNR≈{res['snr']:.2f}, imp≈{100*res['improvement']:.1f}%",
                 va="bottom", ha="left")
    else:
        if res.get("reason"):
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
                   base_imp: float = 0.02,
                   grid_centers: int = 80,
                   grid_widths: int = 25,
                   grid_depths: int = 40,
                   seed: int = 0) -> Path:
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
                grid_centers=grid_centers,
                grid_widths=grid_widths,
                grid_depths=grid_depths,
                seed=seed,
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
                "source": res.get("source","")
            })

            print(f"[OK] {p.name}: has_dip={res['has_dip']} "
                  f"center={res['center']:.6g} width={res['width']:.6g} "
                  f"depth={res['depth']:.4g} SNR={res['snr']:.2f} "
                  f"imp={100*res['improvement']:.1f}% support={res['support_points']} src={res.get('source','')}")

        except Exception as e:
            print(f"[ERROR] {p}: {e}")

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "dip_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    return summary_path

def main():
    ap = argparse.ArgumentParser(description="Detect low-valued intervals in scatter charts (hybrid grid+JAX).")
    ap.add_argument("folder", type=str, help="Root folder to search recursively for CSV files.")
    ap.add_argument("--time-col", type=str, default=None, help="Name of time/x column (optional).")
    ap.add_argument("--y-col", type=str, default=None, help="Name of y/flux column (optional).")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for images and summary CSV.")
    ap.add_argument("--strict", action="store_true", help="Be stricter (fewer FPs, maybe more FNs).")
    ap.add_argument("--imp-override", type=float, default=0.08, help="Improvement threshold for FN-minimizing OR rule.")
    ap.add_argument("--snr", type=float, default=0.8, help="Base SNR threshold for AND rule (sensitivity mode).")
    ap.add_argument("--imp", type=float, default=0.02, help="Base improvement threshold for AND rule (sensitivity mode).")
    ap.add_argument("--grid-centers", type=int, default=80, help="Coarse grid: number of center samples.")
    ap.add_argument("--grid-widths", type=int, default=25, help="Coarse grid: number of width samples.")
    ap.add_argument("--grid-depths", type=int, default=40, help="Coarse grid: number of depth samples.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for JAX refinement.")

    args = ap.parse_args()
    root = Path(args.folder).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "_dip_outputs")

    summary_path = process_folder(
        root, out_dir, args.time_col, args.y_col,
        prefer_sensitivity=(not args.strict),
        imp_override=args.imp_override,
        base_snr=args.snr,
        base_imp=args.imp,
        grid_centers=args.grid_centers,
        grid_widths=args.grid_widths,
        grid_depths=args.grid_depths,
        seed=args.seed,
    )
    print(f"\nSummary written to: {summary_path}\nImages saved to: {out_dir}")

if __name__ == "__main__":
    main()
