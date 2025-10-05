"""
Hybrid Scatter Classifier: Robust algorithmic first-pass + JAX ML backstop

What it does
------------
- Input: one scatter sample as two 1D arrays x, y (same length).
- Goal: classify into {negative: flat/rectangular; positive: V-shaped} robustly
  even with wide distributions, heavy edge noise, and occasional outliers.

Pipeline
--------
1) Robust preprocessing:
   - robust scaling by median/MAD (per-axis)
   - edge down-weighting by x-quantiles (default: keep 10%-90% core heavier)
   - optional binning to summarize (x-bin, median(y), IQR)

2) Algorithmic decision (model selection):
   - Model A (negative): robust linear (intercept + slope) with Huber IRLS
   - Model B (positive): robust "hinge" piecewise linear at x=c with Huber IRLS
     y = a + b1 * (c - x)_+ + b2 * (x - c)_+
   - Compare robust losses with a small complexity penalty on Model B.
   - Score s = (L_A - L_B_adj) / max(L_A, tiny). If |s| > s_thresh ⇒ decide.

3) ML backstop (only for ambiguous |s|<=s_thresh):
   - Engineer robust features from the binned/smoothed medians:
     s, log(L_A), log(L_B), left/right slopes, depth ratio, curvature proxy,
     corr(x,y), central/edge density ratio, median IQR, etc.
   - Train a small logistic regression in JAX (full-batch, L2, JIT).

4) Outputs:
   - label, score s, (optional) ML prob, hinge center c_hat, diagnostics.
   - bootstrap_predict(...) provides uncertainty via resampling.

Usage
-----
from hybrid_scatter_classifier import HybridScatterClassifier, synth_dataset

# Train with synthetic data (example)
clf = HybridScatterClassifier()
Xtrain, ytrain = synth_dataset(n_samples=200, n_points=800, seed=0)
clf.fit(Xtrain, ytrain, train_mode='ambiguous_only')  # uses JAX for backstop if installed

# Predict on a new sample
x, y, label_true = Xtrain[0][0], Xtrain[0][1], ytrain[0]
pred = clf.predict(x, y, return_features=True)
print(pred['label'], pred['score'], pred.get('ml_proba', None))

Notes
-----
- JAX is only required for the ML backstop. If JAX is unavailable, the algorithmic
  decision still works; the classifier will skip ML and rely on s-thresholds.
- This module avoids heavyweight dependencies; only numpy/scipy-like IRLS is coded by hand.
"""
from __future__ import annotations

import csv
import math
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Union

import numpy as np

# Optional JAX (for ML backstop)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
    _JAX_AVAILABLE = True
except Exception:
    _JAX_AVAILABLE = False
    jax = None
    jnp = None


PathLike = Union[str, Path]


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def load_scatter_csv(path: PathLike) -> Tuple[np.ndarray, np.ndarray]:
    """Load a scatter sample from a CSV file as two 1D numpy arrays (x, y)."""
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"CSV file not found: {file_path}")

    rows = []
    with file_path.open("r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append(row)

    if not rows:
        raise ValueError(f"CSV file is empty: {file_path}")

    header_offset = 0
    first_two = rows[0][:2]
    if len(first_two) < 2 or not all(_is_number(cell) for cell in first_two):
        header_offset = 1

    data = []
    for row in rows[header_offset:]:
        if len(row) < 2:
            continue
        try:
            x_val = float(row[0])
            y_val = float(row[1])
        except ValueError:
            continue
        data.append((x_val, y_val))

    if not data:
        raise ValueError(f"No numeric x/y pairs found in {file_path}")

    arr = np.asarray(data, dtype=float)
    mask = np.isfinite(arr[:, 0]) & np.isfinite(arr[:, 1])
    arr = arr[mask]

    if arr.size == 0:
        raise ValueError(f"All x/y values were non-finite in {file_path}")

    return arr[:, 0], arr[:, 1]


# ---------------------------
# Robust stats & utilities
# ---------------------------
def median(x: np.ndarray) -> float:
    return float(np.median(x))


def mad(x: np.ndarray, c: float = 1.4826) -> float:
    m = median(x)
    return float(c * np.median(np.abs(x - m)) + 1e-12)


def iqr(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1 + 1e-12)


def robust_scale(u: np.ndarray) -> Tuple[np.ndarray, float, float]:
    m = median(u)
    s = mad(u)
    return (u - m) / s, m, s


def weights_by_quantile(x: np.ndarray, q_low=0.1, q_high=0.9) -> np.ndarray:
    """Inside [q_low, q_high] weight=1; linearly decay to 0 at min/max outside."""
    lo = np.quantile(x, q_low)
    hi = np.quantile(x, q_high)
    x_min, x_max = np.min(x), np.max(x)
    w = np.ones_like(x, dtype=float)
    left = x < lo
    right = x > hi
    # Avoid division by zero
    denom_l = max(lo - x_min, 1e-9)
    denom_r = max(x_max - hi, 1e-9)
    w[left] = np.clip((x[left] - x_min) / denom_l, 0.0, 1.0)
    w[right] = np.clip((x_max - x[right]) / denom_r, 0.0, 1.0)
    return w


def weighted_median(x: np.ndarray, w: np.ndarray) -> float:
    idx = np.argsort(x)
    x_sorted = x[idx]
    w_sorted = w[idx]
    cum = np.cumsum(w_sorted) / (np.sum(w_sorted) + 1e-12)
    j = np.searchsorted(cum, 0.5, side="left")
    j = np.clip(j, 0, len(x_sorted) - 1)
    return float(x_sorted[j])


def running_median(y: np.ndarray, k: int = 3) -> np.ndarray:
    """Simple running median with odd kernel size k."""
    k = int(max(1, k))
    if k % 2 == 0:
        k += 1
    pad = k // 2
    ypad = np.pad(y, (pad, pad), mode='edge')
    out = np.empty_like(y, dtype=float)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+k])
    return out


def bin_stats(x: np.ndarray,
              y: np.ndarray,
              n_bins: int = 40,
              weights: Optional[np.ndarray] = None,
              min_count: int = 8) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return bin centers, median y, IQR y, and bin weights (counts * mean w)."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if weights is None:
        weights = np.ones_like(x, dtype=float)
    # fixed-width bins
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max:
        x_max = x_min + 1.0
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    y_med = []
    y_iqr = []
    w_bin = []
    cts = []

    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i + 1]) if i < n_bins - 1 else (x >= edges[i]) & (x <= edges[i + 1])
        if not np.any(m):
            y_med.append(np.nan)
            y_iqr.append(np.nan)
            w_bin.append(0.0)
            cts.append(0)
            continue
        # apply weights
        ww = weights[m]
        xx = x[m]
        yy = y[m]
        # Weighted median via replication-like approach (approx): use unweighted median but robust to outliers already.
        y_med.append(np.median(yy))
        y_iqr.append(iqr(yy))
        w_bin.append(float(np.mean(ww) * len(ww)))
        cts.append(int(np.sum(m)))

    centers = centers.astype(float)
    y_med = np.asarray(y_med, dtype=float)
    y_iqr = np.asarray(y_iqr, dtype=float)
    w_bin = np.asarray(w_bin, dtype=float)
    cts = np.asarray(cts, dtype=int)

    # drop bins with too few points or NaNs
    good = (~np.isnan(y_med)) & (cts >= min_count)
    return centers[good], y_med[good], y_iqr[good], w_bin[good]


# ---------------------------
# Robust regression (IRLS)
# ---------------------------
def huber_weights(r: np.ndarray, delta: float) -> np.ndarray:
    a = np.abs(r)
    w = np.ones_like(r, dtype=float)
    m = a > delta
    w[m] = (delta / (a[m] + 1e-12))
    return w


def robust_wls(X: np.ndarray,
               y: np.ndarray,
               base_w: Optional[np.ndarray] = None,
               delta: float = 1.0,
               n_iter: int = 30,
               eps: float = 1e-9) -> Tuple[np.ndarray, float]:
    """
    IRLS with Huber weights. Returns (beta, final_loss).
    Loss is sum base_w * huber(r).
    """
    X = np.asarray(X, float)
    y = np.asarray(y, float).ravel()
    n, d = X.shape
    if base_w is None:
        base_w = np.ones(n, float)
    base_w = base_w.ravel()

    # init via ordinary WLS
    W = np.diag(base_w + eps)
    beta = np.linalg.lstsq(X.T @ W @ X + 1e-6 * np.eye(d), X.T @ W @ y, rcond=None)[0]

    for _ in range(n_iter):
        r = y - X @ beta
        w = huber_weights(r, delta)
        W = np.diag(base_w * w + eps)
        beta_new = np.linalg.lstsq(X.T @ W @ X + 1e-6 * np.eye(d), X.T @ W @ y, rcond=None)[0]
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    # final loss
    r = y - X @ beta
    loss = np.sum(base_w * (np.where(np.abs(r) <= delta, 0.5 * r**2, delta * (np.abs(r) - 0.5 * delta))))
    return beta, float(loss)


# ---------------------------
# Model A / Model B fitting
# ---------------------------
@dataclass
class ModelAFit:
    beta: np.ndarray           # [a, b]
    loss: float


@dataclass
class ModelBFit:
    beta: np.ndarray           # [a, b1, b2]
    c: float
    loss: float


def fit_model_A(xb: np.ndarray,
                yb: np.ndarray,
                wb: np.ndarray,
                delta: float) -> ModelAFit:
    # Design: [1, x]
    X = np.column_stack([np.ones_like(xb), xb])
    # IQR-based weight stabilization: higher weight if bin variability smaller
    iqr_b = np.maximum(iqr(yb) * np.ones_like(yb), 1e-6)  # rough proxy
    base_w = wb * (1.0 / iqr_b)
    beta, loss = robust_wls(X, yb, base_w=base_w, delta=delta, n_iter=50)
    return ModelAFit(beta=beta, loss=loss)


def fit_model_B_hinge(xb: np.ndarray,
                      yb: np.ndarray,
                      wb: np.ndarray,
                      delta: float,
                      c_grid: Optional[np.ndarray] = None) -> ModelBFit:
    if c_grid is None:
        # focus search on 35~65 percentile to avoid edges dominated by noise
        lo, hi = np.quantile(xb, [0.35, 0.65])
        c_grid = np.linspace(lo, hi, 11)
    best = None
    iqr_b = np.maximum(iqr(yb) * np.ones_like(yb), 1e-6)
    base_w_root = wb * (1.0 / iqr_b)

    for c in c_grid:
        h1 = np.maximum(c - xb, 0.0)
        h2 = np.maximum(xb - c, 0.0)
        X = np.column_stack([np.ones_like(xb), h1, h2])
        beta, loss = robust_wls(X, yb, base_w=base_w_root, delta=delta, n_iter=60)
        if (best is None) or (loss < best.loss):
            best = ModelBFit(beta=beta, c=float(c), loss=float(loss))
    return best


def model_selection(x: np.ndarray,
                    y: np.ndarray,
                    n_bins: int = 40,
                    q_low: float = 0.1,
                    q_high: float = 0.9,
                    delta: float = 1.2,
                    penalty_lambda: float = 0.1,
                    smooth_k: int = 3) -> Tuple[ModelAFit, ModelBFit, Dict[str, float], Dict[str, np.ndarray]]:
    """
    Returns model_A, model_B, metrics, and intermediates for feature engineering.
    """
    # Robust scaling per-axis
    xs, xm, xscl = robust_scale(np.asarray(x).ravel())
    ys, ym, yscl = robust_scale(np.asarray(y).ravel())

    # Edge weights
    w = weights_by_quantile(xs, q_low=q_low, q_high=q_high)

    # Bin + summarize
    xb, y_med, y_iqr, w_bin = bin_stats(xs, ys, n_bins=n_bins, weights=w, min_count=6)
    if len(xb) < max(10, n_bins // 5):
        # fallback: use all points (may be slower), but still robust with weights
        xb = xs
        y_med = ys
        y_iqr = np.ones_like(y_med) * iqr(ys)
        w_bin = w

    # Smooth medians (running median)
    y_med_s = running_median(y_med, k=smooth_k)

    # Fit models on binned-smooth series
    A = fit_model_A(xb, y_med_s, w_bin, delta=delta)
    B = fit_model_B_hinge(xb, y_med_s, w_bin, delta=delta)

    # Complexity penalty on B
    L_A = A.loss
    L_B = B.loss * (1.0 + penalty_lambda)

    s = (L_A - L_B) / (max(L_A, 1e-9))
    metrics = dict(L_A=L_A, L_B=L_B, score=s, xm=xm, xs=xscl, ym=ym, ys=yscl)

    intermediates = dict(xb=xb, yb=y_med_s, y_iqr=y_iqr, w_bin=w_bin,
                         xs=xs, ys=ys, w=w, c_hat=B.c, betaA=A.beta, betaB=B.beta)
    return A, B, metrics, intermediates


# ---------------------------
# Feature engineering
# ---------------------------
def quadratic_fit_coeff(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> np.ndarray:
    X = np.column_stack([np.ones_like(x), x, x**2])
    if w is None:
        w = np.ones_like(x, float)
    W = np.diag(w)
    beta = np.linalg.lstsq(X.T @ W @ X + 1e-6 * np.eye(3), X.T @ W @ y, rcond=None)[0]
    return beta  # [a, b, c] with c as curvature proxy


def engineered_features(A: ModelAFit,
                        B: ModelBFit,
                        metrics: Dict[str, float],
                        intermediates: Dict[str, np.ndarray]) -> np.ndarray:
    xb = intermediates['xb']
    yb = intermediates['yb']
    y_iqr = intermediates['y_iqr']
    w_bin = intermediates['w_bin']
    xs = intermediates['xs']
    ys = intermediates['ys']
    w = intermediates['w']
    c = intermediates['c_hat']

    # Score and losses
    s = metrics['score']
    L_A = metrics['L_A']
    L_B = metrics['L_B']

    # Left/right slopes implied by hinge
    a, b1, b2 = B.beta
    sL = -b1
    sR = b2

    # Depth ratio
    depth = np.median(yb) - np.min(yb)
    depth_ratio = depth / (mad(yb) + 1e-9)

    # Quadratic curvature
    qcoef = quadratic_fit_coeff(xb, yb, w_bin)
    curvature = qcoef[2]  # >0 suggests convex (V-like)

    # Correlation
    corr = float(np.corrcoef(xs, ys)[0, 1]) if len(xs) > 2 else 0.0

    # Central/edge density ratio
    w_core = weights_by_quantile(xs, 0.2, 0.8)
    core = np.mean(w_core > 0.99)
    edge = 1.0 - core + 1e-6
    density_ratio = core / edge

    # Median bin IQR (stability)
    med_iqr = float(np.median(y_iqr)) if len(y_iqr) else float(iqr(ys))

    feats = np.array([
        s,
        math.log(L_A + 1e-9),
        math.log(L_B + 1e-9),
        sL, sR,
        depth_ratio,
        curvature,
        corr,
        density_ratio,
        med_iqr
    ], dtype=float)

    return feats


# ---------------------------
# JAX Logistic Regression
# ---------------------------
class JAXLogReg:
    def __init__(self, lr: float = 0.05, l2: float = 1e-3, steps: int = 1500, seed: int = 0):
        if not _JAX_AVAILABLE:
            raise ImportError("JAX not available. Please `pip install jax jaxlib` to use ML backstop.")
        self.lr = lr
        self.l2 = l2
        self.steps = steps
        self.rng = jax.random.PRNGKey(seed)
        self.w = None
        self.b = None

    @staticmethod
    @jit
    def _sigmoid(z):
        return 1.0 / (1.0 + jnp.exp(-z))

    def _loss(self, params, X, y):
        w, b = params
        z = X @ w + b
        p = self._sigmoid(z)
        # Binary cross entropy + L2
        ce = -jnp.mean(y * jnp.log(p + 1e-9) + (1 - y) * jnp.log(1 - p + 1e-9))
        reg = 0.5 * self.l2 * jnp.sum(w * w)
        return ce + reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = jnp.array(X, dtype=jnp.float32)
        y = jnp.array(y, dtype=jnp.float32).reshape(-1, 1)
        n, d = X.shape
        self.w = jnp.zeros((d, 1), dtype=jnp.float32)
        self.b = jnp.array(0.0, dtype=jnp.float32)

        params = (self.w, self.b)
        loss_grad = jit(grad(self._loss, argnums=0))

        for _ in range(self.steps):
            g_w, g_b = loss_grad(params, X, y)
            self.w = self.w - self.lr * g_w
            self.b = self.b - self.lr * g_b
            params = (self.w, self.b)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = jnp.array(X, dtype=jnp.float32)
        z = X @ self.w + self.b
        p = self._sigmoid(z)
        return np.asarray(p).reshape(-1)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)
        return (p >= threshold).astype(int)


# ---------------------------
# Main classifier
# ---------------------------
class HybridScatterClassifier:
    def __init__(self,
                 n_bins: int = 40,
                 q_low: float = 0.1,
                 q_high: float = 0.9,
                 huber_delta: float = 1.2,
                 penalty_lambda: float = 0.1,
                 s_thresh: float = 0.15,
                 jax_params: Optional[dict] = None):
        self.n_bins = n_bins
        self.q_low = q_low
        self.q_high = q_high
        self.huber_delta = huber_delta
        self.penalty_lambda = penalty_lambda
        self.s_thresh = s_thresh
        self._ml = None
        self._feat_names = ["score", "log_LA", "log_LB", "slope_left", "slope_right",
                            "depth_ratio", "curvature", "corr", "density_ratio", "median_iqr"]
        self._jax_params = jax_params or dict(lr=0.05, l2=1e-3, steps=1500, seed=0)

    def _first_stage(self, x: np.ndarray, y: np.ndarray):
        A, B, metrics, inter = model_selection(
            x, y,
            n_bins=self.n_bins,
            q_low=self.q_low,
            q_high=self.q_high,
            delta=self.huber_delta,
            penalty_lambda=self.penalty_lambda,
            smooth_k=3,
        )
        s = metrics['score']
        label = 'positive' if s > self.s_thresh else ('negative' if s < -self.s_thresh else 'ambiguous')
        return A, B, metrics, inter, label

    def _featurize(self, A, B, metrics, inter) -> np.ndarray:
        return engineered_features(A, B, metrics, inter)

    def fit(self,
            samples: List[Tuple[np.ndarray, np.ndarray]],
            labels: List[int],
            train_mode: str = 'ambiguous_only') -> "HybridScatterClassifier":
        """
        samples: list of (x,y) arrays. labels: 1 for positive (V), 0 for negative (flat).
        train_mode: 'ambiguous_only' uses only |s|<=s_thresh; 'all' uses every sample.
        """
        feats = []
        ys = []
        used = 0
        for (x, y), lab in zip(samples, labels):
            A, B, metrics, inter, lab_stage1 = self._first_stage(x, y)
            use = (train_mode != 'ambiguous_only') or (lab_stage1 == 'ambiguous')
            if use:
                f = self._featurize(A, B, metrics, inter)
                feats.append(f)
                ys.append(int(lab))
                used += 1

        if used == 0:
            # fallback: if no ambiguous, use all
            for (x, y), lab in zip(samples, labels):
                A, B, metrics, inter, _ = self._first_stage(x, y)
                feats.append(self._featurize(A, B, metrics, inter))
                ys.append(int(lab))

        X = np.vstack(feats)
        y = np.array(ys, int)

        if _JAX_AVAILABLE:
            self._ml = JAXLogReg(**self._jax_params)
            self._ml.fit(X, y.reshape(-1, 1))
        else:
            self._ml = None  # still usable; just no ML backstop
        return self

    def predict(self, x: np.ndarray, y: np.ndarray, return_features: bool = False) -> Dict[str, object]:
        A, B, metrics, inter, lab_stage1 = self._first_stage(x, y)
        out = dict(stage1_label=lab_stage1, score=metrics['score'], c_hat=float(inter['c_hat']))
        if lab_stage1 == 'ambiguous' and self._ml is not None:
            f = self._featurize(A, B, metrics, inter).reshape(1, -1)
            p = float(self._ml.predict_proba(f)[0])
            label = 'positive' if p >= 0.5 else 'negative'
            out.update(label=label, ml_proba=p)
            if return_features:
                out['features'] = dict(zip(self._feat_names, f.ravel().tolist()))
        else:
            out['label'] = 'positive' if lab_stage1 == 'positive' else 'negative'
            if return_features:
                f = self._featurize(A, B, metrics, inter)
                out['features'] = dict(zip(self._feat_names, f.ravel().tolist()))
        return out

    def bootstrap_predict(self, x: np.ndarray, y: np.ndarray, B: int = 50, seed: int = 0) -> Dict[str, float]:
        rng = np.random.default_rng(seed)
        labels = []
        for _ in range(B):
            idx = rng.integers(0, len(x), size=len(x))
            xi, yi = x[idx], y[idx]
            p = self.predict(xi, yi)
            labels.append(1 if p['label'] == 'positive' else 0)
        prob_pos = float(np.mean(labels))
        return dict(prob_positive=prob_pos, n_bootstrap=B)

    # Convenience wrappers for batch prediction
    def predict_batch(self, samples: List[Tuple[np.ndarray, np.ndarray]]) -> List[Dict[str, object]]:
        return [self.predict(x, y) for (x, y) in samples]


# ---------------------------
# Safe-mode classifier
# ---------------------------
class SafeHybridScatterClassifier(HybridScatterClassifier):
    """FN-averse wrapper around :class:`HybridScatterClassifier` with conformal calibration."""

    def __init__(
        self,
        n_bins: int = 40,
        q_low: float = 0.1,
        q_high: float = 0.9,
        huber_delta: float = 1.2,
        penalty_lambda: float = 0.1,
        base_s_thresh: float = 0.15,
        jax_params: Optional[dict] = None,
    ) -> None:
        super().__init__(
            n_bins=n_bins,
            q_low=q_low,
            q_high=q_high,
            huber_delta=huber_delta,
            penalty_lambda=penalty_lambda,
            s_thresh=base_s_thresh,
            jax_params=jax_params,
        )
        self.base_s_thresh = base_s_thresh
        self.t_neg: Optional[float] = None
        self.p_neg_cap: Optional[float] = None
        self.alpha: Optional[float] = None
        self.n_calib_pos: Optional[int] = None

    def fit_ml(
        self,
        samples: List[Tuple[np.ndarray, np.ndarray]],
        labels: List[int],
        use_only_ambiguous: bool = True,
    ) -> "SafeHybridScatterClassifier":
        train_mode = "ambiguous_only" if use_only_ambiguous else "all"
        super().fit(samples, labels, train_mode=train_mode)
        return self

    def conformal_calibrate(
        self,
        calib_samples: List[Tuple[np.ndarray, np.ndarray]],
        calib_labels: List[int],
        alpha: float = 0.0,
    ) -> Dict[str, Optional[float]]:
        pos_scores: List[float] = []
        pos_probs: List[float] = []
        for (x, y), lab in zip(calib_samples, calib_labels):
            if lab != 1:
                continue
            A, B, metrics, inter, _ = self._first_stage(x, y)
            pos_scores.append(float(metrics["score"]))
            if self._ml is not None:
                feat = self._featurize(A, B, metrics, inter).reshape(1, -1)
                pos_probs.append(float(self._ml.predict_proba(feat)[0]))

        if not pos_scores:
            raise ValueError("Calibration set must include at least one positive sample.")

        q = float(np.clip(alpha, 0.0, 1.0 - 1e-9))
        self.t_neg = float(np.quantile(np.array(pos_scores, dtype=float), q))
        self.p_neg_cap = (
            float(np.quantile(np.array(pos_probs, dtype=float), q)) if pos_probs else None
        )
        self.alpha = alpha
        self.n_calib_pos = len(pos_scores)
        return dict(
            t_neg=self.t_neg,
            p_neg_cap=self.p_neg_cap,
            alpha=self.alpha,
            n_calib_pos=self.n_calib_pos,
        )

    def predict(
        self,
        x: np.ndarray,
        y: np.ndarray,
        return_features: bool = False,
        allow_review: bool = False,
    ) -> Dict[str, object]:
        A, B, metrics, inter, stage1_label = self._first_stage(x, y)
        feats = self._featurize(A, B, metrics, inter)
        ml_proba: Optional[float] = None
        if self._ml is not None:
            ml_proba = float(self._ml.predict_proba(feats.reshape(1, -1))[0])

        score = float(metrics["score"])
        t_neg = self.t_neg if self.t_neg is not None else -1.0
        neg_ok = score <= t_neg
        if neg_ok and self.p_neg_cap is not None:
            if ml_proba is None:
                neg_ok = False
            else:
                neg_ok = ml_proba <= self.p_neg_cap

        label = "negative" if neg_ok else "positive"
        rule = "NEG_safe" if neg_ok else "FN_guard"
        if not neg_ok and allow_review and stage1_label == "ambiguous":
            label = "review"

        out: Dict[str, object] = dict(
            label=label,
            stage1_label=stage1_label,
            score=score,
            c_hat=float(inter["c_hat"]),
            rule=rule,
        )
        if ml_proba is not None:
            out["ml_proba"] = ml_proba
        if return_features:
            out["features"] = dict(zip(self._feat_names, feats.ravel().tolist()))
        if self.t_neg is not None:
            out["t_neg"] = self.t_neg
        if self.p_neg_cap is not None:
            out["p_neg_cap"] = self.p_neg_cap
        return out


# ---------------------------
# Synthetic data for testing
# ---------------------------
def synth_flat(n: int = 800,
               noise_y: float = 0.4,
               rect_height: float = 2.0,
               edge_noise: float = 0.8,
               seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, size=n))
    # 'Rectangular' spread around baseline 0 with some height
    y = rng.uniform(-rect_height/2, rect_height/2, size=n)
    # Add central small slope sometimes
    y += rng.normal(0, noise_y, size=n)
    # Edge noise stronger at both ends
    k = int(0.1 * n)
    idx_left = np.arange(k)
    idx_right = np.arange(n - k, n)
    y[idx_left] += rng.normal(0, edge_noise, size=k)
    y[idx_right] += rng.normal(0, edge_noise, size=k)
    return x, y


def synth_v(n: int = 800,
            depth: float = 2.0,
            noise_y: float = 0.35,
            edge_noise: float = 0.9,
            c: float = 0.0,
            asym: float = 0.0,
            seed: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.sort(rng.uniform(-3, 3, size=n))
    # V shape: y = depth * |x - c| + noise; asym skews slopes
    left_slope = depth * (1.0 + max(0.0, -asym))
    right_slope = depth * (1.0 + max(0.0, asym))
    y = np.where(x < c, left_slope * (c - x), right_slope * (x - c))
    y += rng.normal(0, noise_y, size=n)
    # stronger edge noise
    k = int(0.1 * n)
    idx_left = np.arange(k)
    idx_right = np.arange(n - k, n)
    y[idx_left] += rng.normal(0, edge_noise, size=k)
    y[idx_right] += rng.normal(0, edge_noise, size=k)
    return x, y


def synth_dataset(n_samples: int = 200,
                  n_points: int = 800,
                  seed: int = 0,
                  p_positive: float = 0.5) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int]]:
    rng = np.random.default_rng(seed)
    samples = []
    labels = []
    for i in range(n_samples):
        is_pos = rng.uniform() < p_positive
        if is_pos:
            x, y = synth_v(n=n_points,
                           depth=rng.uniform(1.2, 2.8),
                           noise_y=rng.uniform(0.25, 0.55),
                           edge_noise=rng.uniform(0.6, 1.0),
                           c=rng.uniform(-0.5, 0.5),
                           asym=rng.uniform(-0.6, 0.6),
                           seed=rng.integers(1e9))
            lab = 1
        else:
            x, y = synth_flat(n=n_points,
                              noise_y=rng.uniform(0.25, 0.55),
                              rect_height=rng.uniform(1.2, 2.8),
                              edge_noise=rng.uniform(0.6, 1.0),
                              seed=rng.integers(1e9))
            lab = 0
        samples.append((x, y))
        labels.append(lab)
    return samples, labels


def load_labeled_scatter_directory(
    root: PathLike,
    positive_subdir: str = "positive",
    negative_subdir: str = "negative",
    pattern: str = "*.csv",
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], List[int], List[Path]]:
    """Load labeled scatter samples from a directory with ``positive``/``negative`` subfolders."""
    base = Path(root)
    pos_dir = base / positive_subdir
    neg_dir = base / negative_subdir

    if not pos_dir.is_dir() or not neg_dir.is_dir():
        raise FileNotFoundError(
            f"Expected subdirectories '{positive_subdir}' and '{negative_subdir}' inside {base}"
        )

    samples: List[Tuple[np.ndarray, np.ndarray]] = []
    labels: List[int] = []
    paths: List[Path] = []

    for label, folder in ((1, pos_dir), (0, neg_dir)):
        csv_files = sorted(p for p in folder.rglob(pattern) if p.is_file())
        if not csv_files:
            raise ValueError(f"No CSV files found in {folder}")
        for csv_file in csv_files:
            x_vals, y_vals = load_scatter_csv(csv_file)
            samples.append((x_vals, y_vals))
            labels.append(label)
            paths.append(csv_file)

    if not samples:
        raise ValueError(f"No samples collected from {base}")

    return samples, labels, paths


__all__ = [
    "HybridScatterClassifier",
    "SafeHybridScatterClassifier",
    "load_labeled_scatter_directory",
    "load_scatter_csv",
    "JAXLogReg",
    "ModelAFit",
    "ModelBFit",
    "engineered_features",
    "model_selection",
    "synth_dataset",
    "synth_flat",
    "synth_v",
]
