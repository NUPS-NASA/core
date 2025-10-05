#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Flat-Bottom (Plateau-at-Minimum) Detector — polished plotting + FITS banner

What's new (per request)
------------------------
1) **색상 고정**: 메인 플롯(보정 산점), 추세선(이동평균), 탐지 구간 음영을 서로 **다른 고정 색상**으로 통일.
2) **--y-raw-col**: raw 시리즈를 플롯에 얇게 오버레이(탐지엔 미사용).
3) **--fits-dir**: 해당 CSV의 번호(%04d 등)와 매칭되는 FITS 헤더를 읽어 메타(RIGHT_LINES) 구성.
4) **--banner / --logo-path**: Matplotlib 플롯을 이미지로 만든 뒤, 상단 배너(좌상단 로고, 중앙 제목, 우상단 메타)를 합성하여
   `plots_bannered` 폴더에 `{원본이름}_bannered.확장자`로 저장. 시작 시 폴더 내용 전체 삭제.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import shutil
import hashlib
import re
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------------------------- Global plot colors --------------------------- #
# (Requested: fixed distinct colors per layer, not per file)
COLOR_MAIN   = (0.298, 0.471, 0.659)  # blue-ish for corrected scatter
COLOR_TREND  = (0.961, 0.521, 0.094)  # orange for moving avg
COLOR_REGION = (0.329, 0.643, 0.294)  # green for detected span
COLOR_RAW    = (0.6,   0.6,   0.6  )  # grey for raw overlay (smoothed)

# --------------------------- Robust helpers --------------------------- #

def mad(x: np.ndarray) -> float:
    med = np.median(x)
    return float(np.median(np.abs(x - med)))

def robust_scale(x: np.ndarray, eps: float = 1e-12) -> float:
    return 1.4826 * mad(x) + eps

def moving_average_centered(y: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average using reflect padding; output has same length as y."""
    N = len(y)
    if window <= 1 or N == 0:
        return y.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    ypad = np.pad(y, pad_width=pad, mode="reflect")
    k = np.ones(window, dtype=float) / window
    conv = np.convolve(ypad, k, mode="valid")
    if len(conv) != N:  # fallback safety
        conv = np.convolve(y, k, mode="same")
        conv = conv[:N]
    return conv

def boolean_runs(mask: np.ndarray) -> List[Tuple[int, int]]:
    """Return inclusive (start, end) index pairs for contiguous True runs."""
    if mask.size == 0:
        return []
    diff = np.diff(mask.astype(int))
    starts = list(np.where(diff == 1)[0] + 1)
    ends = list(np.where(diff == -1)[0])
    if mask[0]:
        starts = [0] + starts
    if mask[-1]:
        ends = ends + [len(mask) - 1]
    return list(zip(starts, ends))


# --------------------------- Data classes ----------------------------- #

@dataclass
class PlateauSegment:
    start_idx: int
    end_idx: int  # inclusive
    start_x: float
    end_x: float
    x_span: float
    mean_y: float
    min_y: float
    max_abs_slope: float
    mean_abs_slope: float
    mean_abs_curv: float
    score: float  # generic score/improve used internally

@dataclass
class DetectionParams:
    # smoothing & flatness thresholds
    window: int = 20
    slope_mad_mult: float = 1.0
    curv_mad_mult: float = 1.0

    # legacy/heuristic thresholds (kept for compatibility/fallbacks)
    y_thresh_mode: str = "both"  # {"min+mad","percentile","both"}
    y_quantile: float = 0.10
    y_mad_mult: float = 0.5

    # size constraints
    min_points: int = 5
    min_x_span_frac: float = 0.05
    edge_margin_points: int = 3

    # triple-tangent stationary detection (fallback)
    slope_zero_mad_mult: float = 0.5
    stationary_min_sep_pts: int = 3
    use_legacy_mask: bool = False
    contrast_min: float = 0.0

    # model selection (plateau vs quadratic)
    use_model_selection: bool = True
    model_improve_min: float = 0.10
    model_contrast_min: float = 0.00

    # prefer wider & deeper in candidate scoring
    width_pref: float = 0.5
    depth_pref: float = 0.5
    widen_rel_tol: float = 0.02

    # FP suppression: depth/shoulders filters (hard accept criteria)
    depth_min: float = 0.15
    depth_bilateral_min: float = 0.08
    shoulder_win_frac: float = 0.5
    slope_sign_frac_min: float = 0.6
    inside_flat_frac_min: float = 0.6
    baseline_line_depth_min: float = 0.0

    # edge-monotone rejection (from boundaries straight into plateau)
    edge_monotone_frac_min: float = 0.75
    edge_drop_min: float = 0.50
    # scope: only consider edge-monotone if plateau is near edges
    edge_monotone_scope_pts: int = 12
    edge_monotone_scope_x_frac: float = 0.06

    # hard padding from x-ends
    plateau_pad_points: int = 8
    plateau_pad_x_frac: float = 0.02


# ---------------------------- FITS helpers ---------------------------- #

def _read_fits_header(fits_path: Path) -> Optional[Dict[str, Any]]:
    try:
        from astropy.io import fits  # pip install astropy
        with fits.open(fits_path) as hdul:
            hdr = dict(hdul[0].header)
            if not hdr and len(hdul) > 1:
                hdr = dict(hdul[1].header)
            return hdr
    except Exception:
        return None

def _safe_get(h: Dict[str, Any], keys: List[str]) -> Optional[Any]:
    if not h:
        return None
    for k in keys:
        if k in h:
            return h[k]
        for hk in h.keys():
            if hk.upper() == k.upper():
                return h[hk]
    return None

def _parse_iso_datetime(s: str):
    import datetime as dt
    try:
        s = s.strip().replace('T', ' ')
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                return dt.datetime.strptime(s, fmt).replace(tzinfo=dt.timezone.utc)
            except ValueError:
                pass
    except Exception:
        pass
    return None

def _mjd_to_utc_datetime(mjd: float):
    import datetime as dt
    base = dt.datetime(1858, 11, 17, 0, 0, 0, tzinfo=dt.timezone.utc)
    return base + dt.timedelta(days=float(mjd))

def _deg_to_hms(ra_deg: float) -> str:
    try:
        total_hours = (float(ra_deg) / 15.0) % 24.0
        h = int(total_hours); m = int((total_hours - h) * 60)
        s = (total_hours - h - m/60.0) * 3600.0
        return f"{h:02d}:{m:02d}:{s:05.2f}"
    except Exception:
        return "N/A"

def _deg_to_dms(dec_deg: float) -> str:
    try:
        sign = '+' if float(dec_deg) >= 0 else '-'
        val = abs(float(dec_deg))
        d = int(val); m = int((val - d) * 60)
        s = (val - d - m/60.0) * 3600.0
        return f"{sign}{d:02d}:{m:02d}:{s:05.2f}"
    except Exception:
        return "N/A"

def _sexagesimal_to_deg(text: str, is_ra: bool) -> Optional[float]:
    try:
        t = text.strip()
        t = re.sub(r'[hHdD]', ':', t); t = re.sub(r'[mM]', ':', t); t = re.sub(r'[sS]', '', t)
        t = t.replace(' ', ':')
        parts = [p for p in t.split(':') if p]
        if len(parts) < 2:
            return None
        a = float(parts[0]); b = float(parts[1]); c = float(parts[2]) if len(parts) > 2 else 0.0
        if is_ra:
            hours = abs(a) + b/60.0 + c/3600.0
            if a < 0: hours = -hours
            return hours * 15.0
        else:
            sign = -1.0 if str(parts[0]).strip().startswith('-') else 1.0
            deg = abs(a) + b/60.0 + c/3600.0
            return sign * deg
    except Exception:
        return None

def _build_right_lines_from_header(hdr: Optional[Dict[str, Any]]) -> List[str]:
    import datetime as dt
    if not hdr:
        return ["Obs: N/A", "Dur: N/A / Exp: N/A", "Filter: N/A", "RA/Dec: N/A"]
    # Obs time
    obs_dt = None
    date_obs = _safe_get(hdr, ["DATE-OBS"]); time_obs = _safe_get(hdr, ["TIME-OBS"])
    if isinstance(date_obs, str):
        obs_dt = _parse_iso_datetime(date_obs if not time_obs else f"{date_obs} {time_obs}")
    if obs_dt is None:
        mjd_obs = _safe_get(hdr, ["MJD-OBS"])
        try:
            if mjd_obs is not None:
                obs_dt = _mjd_to_utc_datetime(float(mjd_obs))
        except Exception:
            pass
    obs_str = obs_dt.strftime("%Y-%m-%d %H:%M (UT)") if obs_dt else "N/A"
    # Duration
    duration_sec = None
    for key in ["ELAPTIME", "DURATION", "TELAPSE", "TEXPTIME", "TOTALEXP", "TOTTIME"]:
        v = _safe_get(hdr, [key])
        try:
            if v is not None:
                duration_sec = float(v); break
        except Exception: pass
    if duration_sec is None:
        date_end = _safe_get(hdr, ["DATE-END"])
        if isinstance(date_end, str) and obs_dt:
            end_dt = _parse_iso_datetime(date_end)
            if end_dt: duration_sec = (end_dt - obs_dt).total_seconds()
        if duration_sec is None:
            mjd_end = _safe_get(hdr, ["MJD-END"]); mjd_obs = _safe_get(hdr, ["MJD-OBS"])
            try:
                if mjd_end is not None and mjd_obs is not None:
                    duration_sec = (float(mjd_end) - float(mjd_obs)) * 86400.0
            except Exception: pass
    if isinstance(duration_sec, (int, float)) and duration_sec > 0:
        hours = duration_sec / 3600.0
        dur_str = f"{hours:.1f}h" if hours >= 1.0 else f"{duration_sec:.0f}s"
    else:
        dur_str = "N/A"
    # Exposure
    exp = _safe_get(hdr, ["EXPTIME", "EXPOSURE", "ITIME", "EXPTime"])
    try:
        exp_str = f"{float(exp):g}s" if exp is not None else "N/A"
    except Exception:
        exp_str = "N/A"
    # Filter
    filt = _safe_get(hdr, ["FILTER", "FILTER1", "FILTER2", "FILTERS"])
    filt_str = " ".join(map(str, filt)) if isinstance(filt, (list, tuple)) else (str(filt) if filt is not None else "N/A")
    # RA/Dec
    ra_deg = dec_deg = None
    ra_str = dec_str = "N/A"
    ra_txt = _safe_get(hdr, ["OBJCTRA", "RA_OBJ", "RA-TARG", "RASTRNG"])
    dec_txt = _safe_get(hdr, ["OBJCTDEC", "DEC_OBJ", "DEC-TARG", "DECSTRNG"])
    if isinstance(ra_txt, str) and isinstance(dec_txt, str):
        ra_deg = _sexagesimal_to_deg(ra_txt, is_ra=True)
        dec_deg = _sexagesimal_to_deg(dec_txt, is_ra=False)
        ra_str, dec_str = ra_txt.strip(), dec_txt.strip()
    if ra_deg is None:
        v = _safe_get(hdr, ["RA", "CRVAL1", "OBJRA"])
        try:
            if v is not None: ra_deg = float(v)
        except Exception: pass
    if dec_deg is None:
        v = _safe_get(hdr, ["DEC", "CRVAL2", "OBJDEC"])
        try:
            if v is not None: dec_deg = float(v)
        except Exception: pass
    if not (isinstance(ra_txt, str) and isinstance(dec_txt, str)):
        if ra_deg is not None: ra_str = _deg_to_hms(ra_deg)
        if dec_deg is not None: dec_str = _deg_to_dms(dec_deg)
    return [f"{obs_str}", f"Dur: {dur_str} / Exp: {exp_str}", f"Filter: {filt_str}", f"RA/Dec: {ra_str} {dec_str}"]

def _extract_number_from_stem(stem: str) -> Optional[str]:
    m = re.search(r'(\d{4,})', stem)  # prefer 4+ digits (e.g., %04d)
    return m.group(1) if m else None

def _find_matching_fits(fits_dir: Optional[Path], data_path: Path) -> Optional[Path]:
    if not fits_dir:
        return None
    try:
        if not fits_dir.exists():
            return None
    except Exception:
        return None
    num = _extract_number_from_stem(data_path.stem)
    patterns = ["*.fits", "*.fit", "*.fits.fz", "*.fz"]
    if num:
        for pat in patterns:
            for fp in fits_dir.rglob(pat):
                if num in fp.stem:
                    return fp
    # fallback: any one file
    for pat in patterns:
        cands = list(fits_dir.rglob(pat))
        if cands:
            return cands[0]
    return None


# ------------------------- Model selection pieces ---------------------- #

def _fit_quadratic(x: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    X = np.vstack([x**2, x, np.ones_like(x)]).T
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    sse = float(np.sum((y - yhat) ** 2))
    return sse, beta, yhat

def _best_plateau_piecewise(
    x: np.ndarray,
    y: np.ndarray,
    min_points: int,
    edge_margin: int,
    min_x_frac: float,
    width_pref: float,
    depth_pref: float,
    widen_rel_tol: float,
    pad_pts: int,
    pad_x_frac: float,
) -> Optional[Dict[str, Any]]:
    n = len(x)
    if n < min_points + 2 * edge_margin + 1:
        return None

    xspan = float(x[-1] - x[0])
    y_scale = robust_scale(y)

    def refit_for_span(s: int, e: int) -> Dict[str, Any]:
        # design matrix for piecewise [1, left_term, right_term]
        rows: List[List[float]] = []
        rhs: List[float] = []

        if s > 0:
            xs = x[s]
            for i in range(0, s):
                rows.append([1.0, (x[i] - xs), 0.0])
                rhs.append(float(y[i]))
        for i in range(s, e + 1):
            rows.append([1.0, 0.0, 0.0])
            rhs.append(float(y[i]))
        if e < n - 1:
            xe = x[e]
            for i in range(e + 1, n):
                rows.append([1.0, 0.0, (x[i] - xe)])
                rhs.append(float(y[i]))

        A = np.asarray(rows, dtype=float)
        b = np.asarray(rhs, dtype=float)
        params, *_ = np.linalg.lstsq(A, b, rcond=None)
        y0, mL, mR = params.tolist()

        yhat = np.empty_like(y, dtype=float)
        if s > 0:
            yhat[:s] = y0 + mL * (x[:s] - x[s])
        yhat[s : e + 1] = y0
        if e < n - 1:
            yhat[e + 1 :] = y0 + mR * (x[e + 1 :] - x[e])

        sse = float(np.sum((y - yhat) ** 2))

        # compute contrast (depth) vs surroundings
        halfw = max(min_points, (e - s) // 2)
        Ls, Le = max(edge_margin, s - halfw), s
        Rs, Re = e + 1, min(n - edge_margin, e + 1 + halfw)
        if Le - Ls >= min_points and Re - Rs >= min_points:
            left_med = float(np.median(y[Ls:Le]))
            right_med = float(np.median(y[Rs:Re]))
            inside_med = float(np.median(y[s : e + 1]))
            surround = min(left_med, right_med)
            contrast = (surround - inside_med) / (y_scale + 1e-12)
        else:
            contrast = 0.0

        width = float(x[e] - x[s])
        width_norm = width / (xspan + 1e-12)
        sse_norm = sse / ((y_scale**2) * n + 1e-12)
        score = -sse_norm + width_pref * width_norm + depth_pref * max(0.0, contrast)

        return {
            "s": s,
            "e": e,
            "y0": y0,
            "mL": mL,
            "mR": mR,
            "sse": sse,
            "yhat": yhat,
            "contrast": contrast,
            "width": width,
            "width_norm": width_norm,
            "sse_norm": sse_norm,
            "score": score,
        }

    best: Optional[Dict[str, Any]] = None

    # brute-force candidate search with hard padding
    for s in range(edge_margin, n - edge_margin - min_points):
        for e in range(s + min_points - 1, n - edge_margin):
            # padding constraints
            if (s < pad_pts) or ((n - 1 - e) < pad_pts):
                continue
            if (x[s] - x[0]) < (pad_x_frac * xspan) or (x[-1] - x[e]) < (pad_x_frac * xspan):
                continue
            if (x[e] - x[s]) < (min_x_frac * xspan):
                continue

            try:
                cand = refit_for_span(s, e)
            except np.linalg.LinAlgError:
                continue

            if (best is None) or (cand["score"] > best["score"]):
                best = cand

    if best is None:
        return None

    # greedy widening within tolerance
    improved = True
    while improved:
        improved = False
        # extend left
        if best["s"] > max(edge_margin, pad_pts):
            s_new, e_new = best["s"] - 1, best["e"]
            if (x[e_new] - x[s_new]) >= (min_x_frac * xspan) and (x[s_new] - x[0]) >= (pad_x_frac * xspan):
                cand = refit_for_span(s_new, e_new)
                if cand["sse"] <= best["sse"] * (1.0 + widen_rel_tol):
                    best = cand
                    improved = True
                    continue
        # extend right
        if best["e"] < min(n - edge_margin - 1, n - 1 - pad_pts):
            s_new, e_new = best["s"], best["e"] + 1
            if (x[e_new] - x[s_new]) >= (min_x_frac * xspan) and (x[-1] - x[e_new]) >= (pad_x_frac * xspan):
                cand = refit_for_span(s_new, e_new)
                if cand["sse"] <= best["sse"] * (1.0 + widen_rel_tol):
                    best = cand
                    improved = True
                    continue

    return best


# --------------------- FP-suppression filters ------------------------- #

def _depth_shoulders_filter(
    x: np.ndarray,
    y_s: np.ndarray,
    s: int,
    e: int,
    slope: np.ndarray,
    y_scale: float,
    params: DetectionParams,
) -> Tuple[bool, Dict[str, float]]:
    n = len(x)
    width_pts = e - s + 1
    halfw = max(params.min_points, int(params.shoulder_win_frac * width_pts))

    Ls, Le = max(params.edge_margin_points, s - halfw), s
    Rs, Re = e + 1, min(n - params.edge_margin_points, e + 1 + halfw)
    if Le - Ls < params.min_points or Re - Rs < params.min_points:
        return False, {"reason": "shoulder_windows_too_small"}

    inside_med = float(np.median(y_s[s : e + 1]))
    left_med = float(np.median(y_s[Ls:Le]))
    right_med = float(np.median(y_s[Rs:Re]))

    gapL = (left_med - inside_med) / (y_scale + 1e-12)
    gapR = (right_med - inside_med) / (y_scale + 1e-12)
    depth_ok = (min(gapL, gapR) >= params.depth_min) and \
               (gapL >= params.depth_bilateral_min) and (gapR >= params.depth_bilateral_min)

    # slope-based checks
    left_sl = slope[Ls:Le]
    right_sl = slope[Rs:Re]
    inside_sl = slope[s : e + 1]

    left_neg_frac = float(np.mean(left_sl < 0)) if len(left_sl) else 0.0
    right_pos_frac = float(np.mean(right_sl > 0)) if len(right_sl) else 0.0
    slope_sign_ok = (left_neg_frac >= params.slope_sign_frac_min) and \
                    (right_pos_frac >= params.slope_sign_frac_min)

    slope_scale = robust_scale(slope)
    inside_flat_frac = float(np.mean(np.abs(inside_sl) <= params.slope_mad_mult * slope_scale))
    inside_flat_ok = inside_flat_frac >= params.inside_flat_frac_min

    # linear baseline depth at center
    xc = 0.5 * (x[s] + x[e])
    # left line
    XL = np.vstack([x[Ls:Le], np.ones(Le - Ls)]).T
    betaL, *_ = np.linalg.lstsq(XL, y_s[Ls:Le], rcond=None)
    yL_mid = float(betaL[0] * xc + betaL[1])
    # right line
    XR = np.vstack([x[Rs:Re], np.ones(Re - Rs)]).T
    betaR, *_ = np.linalg.lstsq(XR, y_s[Rs:Re], rcond=None)
    yR_mid = float(betaR[0] * xc + betaR[1])
    baseline_mid = min(yL_mid, yR_mid)
    base_gap = (baseline_mid - inside_med) / (y_scale + 1e-12)
    baseline_ok = base_gap >= params.baseline_line_depth_min

    ok = depth_ok and slope_sign_ok and inside_flat_ok and baseline_ok
    return ok, {
        "gapL": gapL,
        "gapR": gapR,
        "left_neg_frac": left_neg_frac,
        "right_pos_frac": right_pos_frac,
        "inside_flat_frac": inside_flat_frac,
        "base_gap": base_gap,
    }

def _edge_monotone_reject(
    x: np.ndarray,
    y_s: np.ndarray,
    s: int,
    e: int,
    slope: np.ndarray,
    y_scale: float,
    params: DetectionParams,
) -> Tuple[bool, Dict[str, float]]:
    """
    Reject if the curve descends (or ascends) into the plateau directly from the left (or right) boundary,
    but only if the plateau touches the scope near the edges (points or x-fraction).
    """
    n = len(x)
    xspan = float(x[-1] - x[0]) if n > 1 else 1.0
    inside_med = float(np.median(y_s[s : e + 1]))

    left_scope  = (s <= params.edge_monotone_scope_pts) or ((x[s] - x[0]) <= params.edge_monotone_scope_x_frac * xspan)
    right_scope = ((n - 1 - e) <= params.edge_monotone_scope_pts) or ((x[-1] - x[e]) <= params.edge_monotone_scope_x_frac * xspan)

    # left side
    if s > 0:
        left_neg_frac = float(np.mean(slope[:s] < 0))
        left_edge_med = float(np.median(y_s[:s]))
        left_drop = (left_edge_med - inside_med) / (y_scale + 1e-12)
    else:
        left_neg_frac, left_drop = 1.0, float("inf")

    # right side
    if e < n - 1:
        right_pos_frac = float(np.mean(slope[e + 1 :] > 0))
        right_edge_med = float(np.median(y_s[e + 1 :]))
        right_drop = (right_edge_med - inside_med) / (y_scale + 1e-12)
    else:
        right_pos_frac, right_drop = 1.0, float("inf")

    left_reject  = left_scope  and (left_neg_frac  >= params.edge_monotone_frac_min) and (left_drop  >= params.edge_drop_min)
    right_reject = right_scope and (right_pos_frac >= params.edge_monotone_frac_min) and (right_drop >= params.edge_drop_min)

    reject = left_reject or right_reject
    return reject, {
        "left_neg_frac": left_neg_frac,
        "left_drop": left_drop,
        "right_pos_frac": right_pos_frac,
        "right_drop": right_drop,
        "left_reject": left_reject,
        "right_reject": right_reject,
    }


# ---------------------- Triple-tangent fallback ------------------------ #

def stationary_points_from_slope(
    slope: np.ndarray,
    slope_scale: float,
    edge_margin_points: int,
    zero_mult: float,
    min_sep: int,
) -> List[int]:
    n = len(slope)
    thr = zero_mult * slope_scale + 1e-12
    small = np.abs(slope) <= thr
    runs = boolean_runs(small)
    cand: List[int] = []
    for s, e in runs:
        i = s + int(np.argmin(np.abs(slope[s : e + 1])))
        if i <= edge_margin_points or i >= n - 1 - edge_margin_points:
            continue
        cand.append(i)
    # sign changes
    sign = np.sign(slope)
    sc = np.where(sign[:-1] * sign[1:] < 0)[0]
    for idx in sc:
        i = idx if abs(slope[idx]) <= abs(slope[idx + 1]) else idx + 1
        if i <= edge_margin_points or i >= n - 1 - edge_margin_points:
            continue
        cand.append(i)
    cand = sorted(set(cand))
    filtered: List[int] = []
    for i in cand:
        if not filtered or i - filtered[-1] >= min_sep:
            filtered.append(i)
    return filtered

def detect_flat_bottom_triple(
    x: np.ndarray,
    y: np.ndarray,
    params: DetectionParams,
) -> Tuple[Optional[PlateauSegment], Dict[str, Any]]:
    n = len(x)
    if n < max(7, params.window):
        return None, {"reason": "too_few_points"}

    # smoothing & derivatives
    y_s = moving_average_centered(y, params.window)
    dx = np.gradient(x)
    eps = 1e-12
    slope = np.gradient(y_s) / (dx + eps)
    curv = np.gradient(slope) / (dx + eps)

    y_scale = robust_scale(y_s)
    slope_scale = robust_scale(slope)

    x_span_total = float(x[-1] - x[0]) if n > 1 else 1.0

    # stationary points
    stat_idx = stationary_points_from_slope(
        slope=slope,
        slope_scale=slope_scale,
        edge_margin_points=params.edge_margin_points,
        zero_mult=params.slope_zero_mad_mult,
        min_sep=params.stationary_min_sep_pts,
    )
    if len(stat_idx) < 3:
        return None, {"reason": "insufficient_stationary_points", "stationary_idx": stat_idx}

    min_x_span = params.min_x_span_frac * x_span_total
    candidates: List[PlateauSegment] = []
    details: List[Dict[str, Any]] = []

    for k in range(len(stat_idx) - 2):
        i1, i2, i3 = stat_idx[k], stat_idx[k + 1], stat_idx[k + 2]

        # hard padding from both ends
        if (i1 < params.plateau_pad_points) or ((n - 1 - i3) < params.plateau_pad_points):
            continue
        if (x[i1] - x[0]) < (params.plateau_pad_x_frac * x_span_total) or \
           (x[-1] - x[i3]) < (params.plateau_pad_x_frac * x_span_total):
            continue

        if i3 - i1 + 1 < params.min_points:
            continue

        x_span = float(x[i3] - x[i1])
        if x_span < min_x_span:
            continue

        inside_y = y_s[i1 : i3 + 1]
        inside_med = float(np.median(inside_y))

        # shoulders
        halfw = max(params.min_points, (i3 - i1) // 2)
        Ls, Le = max(params.edge_margin_points, i1 - halfw), i1
        Rs, Re = i3 + 1, min(n - params.edge_margin_points, i3 + 1 + halfw)
        if Le - Ls < params.min_points or Re - Rs < params.min_points:
            continue

        left_med = float(np.median(y_s[Ls:Le]))
        right_med = float(np.median(y_s[Rs:Re]))
        surround = min(left_med, right_med)
        contrast = (surround - inside_med) / (y_scale + 1e-12)
        if contrast <= params.contrast_min:
            continue

        seg_slope = slope[i1 : i3 + 1]
        seg_curv = curv[i1 : i3 + 1]
        flat_frac = float(np.mean(np.abs(seg_slope) <= params.slope_mad_mult * robust_scale(slope)))
        curv_flat_frac = float(np.mean(np.abs(seg_curv) <= params.curv_mad_mult * robust_scale(curv)))
        span_score = x_span / (x_span_total + 1e-12)
        score = contrast * (0.6 * flat_frac + 0.4 * curv_flat_frac) * span_score

        candidates.append(
            PlateauSegment(
                start_idx=i1,
                end_idx=i3,
                start_x=float(x[i1]),
                end_x=float(x[i3]),
                x_span=x_span,
                mean_y=float(np.mean(inside_y)),
                min_y=float(np.min(inside_y)),
                max_abs_slope=float(np.max(np.abs(seg_slope))),
                mean_abs_slope=float(np.mean(np.abs(seg_slope))),
                mean_abs_curv=float(np.mean(np.abs(seg_curv))),
                score=float(score),
            )
        )
        details.append(
            {
                "i1": int(i1),
                "i2": int(i2),
                "i3": int(i3),
                "left_med": left_med,
                "right_med": right_med,
                "inside_med": inside_med,
                "contrast": contrast,
                "span_score": span_score,
                "score": score,
            }
        )

    if not candidates:
        return None, {"reason": "no_triple_tangent_candidates", "stationary_idx": stat_idx}

    best = max(candidates, key=lambda seg: seg.score)

    # Before accepting, run FP-suppression filters and edge-monotone rejection
    dx = np.gradient(x)
    eps = 1e-12
    slope_arr = np.gradient(y_s) / (dx + eps)
    y_sc = robust_scale(y_s)

    ok, filt = _depth_shoulders_filter(
        x=x, y_s=y_s, s=best.start_idx, e=best.end_idx, slope=slope_arr, y_scale=y_sc, params=params
    )
    if not ok:
        return None, {"reason": "depth_filter_reject_fallback", "filters": filt}

    em_reject, em = _edge_monotone_reject(
        x=x, y_s=y_s, s=best.start_idx, e=best.end_idx, slope=slope_arr, y_scale=y_sc, params=params
    )
    if em_reject:
        return None, {"reason": "edge_monotone_reject_fallback", "edge_metrics": em}

    debug = {
        "y_s": y_s,
        "slope": slope,
        "curv": curv,
        "stationary_idx": stat_idx,
        "method": "triple_tangent",
        "best": asdict(best),
    }
    return best, debug


# ------------------------- Main detector (model) ----------------------- #

def detect_flat_bottom_modelselect(
    x: np.ndarray,
    y: np.ndarray,
    params: DetectionParams,
) -> Tuple[Optional[PlateauSegment], Dict[str, Any]]:
    n = len(x)
    if n < max(7, params.window):
        return None, {"reason": "too_few_points"}

    # smoothing
    y_s = moving_average_centered(y, params.window)

    # quadratic
    quad_sse, quad_beta, quad_yhat = _fit_quadratic(x, y_s)

    # plateau piecewise
    best_pl = _best_plateau_piecewise(
        x=x,
        y=y_s,
        min_points=params.min_points,
        edge_margin=params.edge_margin_points,
        min_x_frac=params.min_x_span_frac,
        width_pref=params.width_pref,
        depth_pref=params.depth_pref,
        widen_rel_tol=params.widen_rel_tol,
        pad_pts=params.plateau_pad_points,
        pad_x_frac=params.plateau_pad_x_frac,
    )
    if best_pl is None:
        return None, {"reason": "no_piecewise_candidate", "y_s": y_s, "quad_sse": quad_sse}

    improve = (quad_sse - best_pl["sse"]) / (quad_sse + 1e-12)
    if not ((improve >= params.model_improve_min) and (best_pl["contrast"] >= params.model_contrast_min)):
        return None, {
            "reason": "model_selection_reject",
            "y_s": y_s,
            "quad_sse": quad_sse,
            "piecewise": best_pl,
            "improve_ratio": improve,
            "threshold_improve": params.model_improve_min,
            "threshold_contrast": params.model_contrast_min,
        }

    # extra FP suppression filters
    dx = np.gradient(x)
    eps = 1e-12
    slope = np.gradient(y_s) / (dx + eps)
    y_scale = robust_scale(y_s)

    ok, filt = _depth_shoulders_filter(
        x=x, y_s=y_s, s=best_pl["s"], e=best_pl["e"], slope=slope, y_scale=y_scale, params=params
    )
    if not ok:
        return None, {
            "reason": "depth_filter_reject",
            "y_s": y_s,
            "quad_sse": quad_sse,
            "piecewise": best_pl,
            "improve_ratio": improve,
            "filters": filt,
        }

    # Edge-monotone rejection
    em_reject, em = _edge_monotone_reject(
        x=x, y_s=y_s, s=best_pl["s"], e=best_pl["e"], slope=slope, y_scale=y_scale, params=params
    )
    if em_reject:
        return None, {
            "reason": "edge_monotone_reject",
            "y_s": y_s,
            "quad_sse": quad_sse,
            "piecewise": best_pl,
            "improve_ratio": improve,
            "edge_metrics": em,
        }

    # build segment
    s = best_pl["s"]
    e = best_pl["e"]
    seg_y = y_s[s : e + 1]
    curv = np.gradient(slope) / (dx + eps)
    seg_slope = slope[s : e + 1]
    seg_curv = curv[s : e + 1]

    seg = PlateauSegment(
        start_idx=s,
        end_idx=e,
        start_x=float(x[s]),
        end_x=float(x[e]),
        x_span=float(x[e] - x[s]),
        mean_y=float(np.mean(seg_y)),
        min_y=float(np.min(seg_y)),
        max_abs_slope=float(np.max(np.abs(seg_slope))),
        mean_abs_slope=float(np.mean(np.abs(seg_slope))),
        mean_abs_curv=float(np.mean(np.abs(seg_curv))),
        score=float(improve),
    )
    debug = {
        "y_s": y_s,
        "quad_sse": quad_sse,
        "piecewise": best_pl,
        "improve_ratio": improve,
        "method": "model_selection",
        "filters": filt,
    }
    return seg, debug


# ---------------------------- Plot helper ------------------------------ #

def plot_with_plateau(
    x: np.ndarray,
    y: np.ndarray,
    y_s: np.ndarray,
    best: Optional[PlateauSegment],
    out_path: Path,
    title: str,
    y_raw: Optional[np.ndarray] = None,
    right_meta_lines: Optional[List[str]] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9, 5))

    # Main detrended scatter
    ax.scatter(x, y, s=6, alpha=0.60, label="Detrended (scatter)", color=COLOR_MAIN)
    # Trendline (moving average on detrended)
    ax.plot(x, y_s, linewidth=1.2, alpha=0.95, label="moving avg (detrended)", color=COLOR_TREND)

    # Optional raw overlay (light smoothing for readability)
    if y_raw is not None and len(y_raw) == len(y):
        w = max(3, min(51, (len(x)//50)*2 + 1))
        ax.scatter(x, moving_average_centered(y_raw, w), s=12, alpha=0.8,
                label="raw", color=COLOR_RAW)

    # Detected plateau region
    if best is not None:
        ax.axvspan(best.start_x, best.end_x, alpha=0.20, label="detected flat-bottom", color=COLOR_REGION)
        ax.axvline(best.start_x, linewidth=1.2, linestyle="--", alpha=0.85, color=COLOR_REGION)
        ax.axvline(best.end_x, linewidth=1.2, linestyle="--", alpha=0.85, color=COLOR_REGION)

    print(title)
    ax.set_title(title)
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.legend(loc="best")

    # RIGHT_LINES metadata as an inside-plot helper (still keep; banner will add headline)
    if right_meta_lines:
        text = "\n".join(right_meta_lines)
        bbox_props = dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9, edgecolor="0.8")
        ax.text(0.99, 0.01, text, transform=ax.transAxes, va="bottom", ha="right", fontsize=8, bbox=bbox_props)

    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


# ---------------------------- Banner helpers --------------------------- #

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _banner_clear_out(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # safety: require folder name endswith "plots_bannered"
    if not str(out_dir).lower().endswith("plots_bannered"):
        raise RuntimeError(f"Safety: unexpected banner output folder: {out_dir}")
    for child in out_dir.rglob("*"):
        try:
            if child.is_file() or child.is_symlink():
                child.unlink()
        except Exception:
            pass
    for child in sorted([d for d in out_dir.rglob("*") if d.is_dir()], reverse=True):
        try:
            dlist = list(child.iterdir())
            if not dlist:
                child.rmdir()
        except Exception:
            pass

def _add_header_banner(
    input_path: str,
    output_path: str,
    title: str,
    right_meta_lines: Optional[List[str]] = None,
    logo_path: Optional[str] = None,
    banner_height: int = 180,
    padding: int = 28,
) -> str:
    """PIL로 상단 배너 합성 (좌상 로고, 중앙 제목, 우상 메타)."""
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception as e:
        raise RuntimeError("Pillow(PIL)가 필요합니다. pip install pillow") from e

    def _load_font(name: str, size: int):
        try:
            return ImageFont.truetype(name, size=size)
        except Exception:
            try:
                return ImageFont.truetype("Arial.ttf", size=size)
            except Exception:
                return ImageFont.load_default()

    base = Image.open(input_path).convert("RGBA")
    W, H = base.size

    # 배너+본문 캔버스
    canvas = Image.new("RGBA", (W, H + banner_height), (255, 255, 255, 255))
    canvas.paste(base, (0, banner_height))
    draw = ImageDraw.Draw(canvas)

    title_font = _load_font("arialbd.ttf", 60)
    meta_font  = _load_font("DejaVuSans.ttf", 28)

    # 좌상단 로고
    if logo_path and os.path.exists(logo_path):
        try:
            logo = Image.open(logo_path).convert("RGBA")
            target_h = int((banner_height - 2 * padding) * 1.15)
            logo = ImageOps.contain(logo, (target_h, target_h))
            logo_x = padding + 40
            logo_y = (banner_height - logo.height) // 2
            canvas.paste(logo, (logo_x, logo_y), logo)
        except Exception:
            pass

    # 중앙 제목 (가로·세로 중앙 정렬)
    x0, y0, x1, y1 = draw.textbbox((0, 0), title, font=title_font)
    title_w, title_h = (x1 - x0), (y1 - y0)
    title_x = (W - title_w) // 2
    title_y = (banner_height - title_h) // 2
    draw.text((title_x, title_y), title, fill=(20, 20, 20), font=title_font)

    # 우상단 메타 (오른쪽 정렬)
    if right_meta_lines:
        line_h = meta_font.getbbox("Ag")[3] + 6
        right_y = padding + 6
        for idx, ln in enumerate(right_meta_lines):
            wx0, wy0, wx1, wy1 = draw.textbbox((0, 0), ln, font=meta_font)
            line_w = wx1 - wx0
            line_x = W - padding - line_w
            draw.text((line_x, right_y + idx * line_h), ln, fill=(20,20,20), font=meta_font)

    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.suffix.lower() in {".jpg", ".jpeg"}:
        canvas.convert("RGB").save(out_path, quality=95)
    else:
        canvas.save(out_path)
    return str(out_path)


# ------------------------------ IO utils ------------------------------- #

def find_csv_files(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.csv") if p.is_file()]

def sanitize_rel_path(base: Path, p: Path) -> Path:
    try:
        return p.relative_to(base)
    except Exception:
        return Path(p.name)


# ---------------------------- Main pipeline ---------------------------- #

def process_file(
    path: Path,
    rel_root: Path,
    out_dir: Path,
    y_col: Optional[str],
    y_raw_col: Optional[str],
    fits_dir: Optional[Path],
    params: DetectionParams,
    export_debug_csv: bool,
    make_plot: bool,
) -> Dict[str, Any]:
    df = pd.read_csv(path)
    if df.shape[1] < 2:
        return {"file": str(path), "status": "error", "error": "<2 columns"}

    x_col = df.columns[0]
    y_col_use = y_col if (y_col is not None and y_col in df.columns) else df.columns[1]

    x = df[x_col].to_numpy(dtype=float)
    y = df[y_col_use].to_numpy(dtype=float)

    # Optional raw Y for plotting only
    y_raw = None
    if y_raw_col is not None and y_raw_col in df.columns:
        try:
            y_raw = df[y_raw_col].to_numpy(dtype=float)
        except Exception:
            y_raw = None

    # clean: finite & sort by x
    mask_fin = np.isfinite(x) & np.isfinite(y)
    if y_raw is not None:
        mask_fin = mask_fin & np.isfinite(y_raw)
    x = x[mask_fin]
    y = y[mask_fin]
    y_raw = (y_raw[mask_fin] if (y_raw is not None and len(y_raw) == len(mask_fin)) else y_raw)

    order = np.argsort(x)
    if not np.all(order == np.arange(len(x))):
        x = x[order]
        y = y[order]
        if y_raw is not None and len(y_raw) == len(order):
            y_raw = y_raw[order]

    # run detection
    best: Optional[PlateauSegment]
    dbg: Dict[str, Any]

    if params.use_model_selection:
        best, dbg = detect_flat_bottom_modelselect(x, y, params)
        if best is None:
            best, dbg = detect_flat_bottom_triple(x, y, params)
    else:
        best, dbg = detect_flat_bottom_triple(x, y, params)

    # outputs
    rel = sanitize_rel_path(rel_root, path)
    title = "Star %04d lightcurve fitting" % (int(''.join([ch for ch in str(path) if ch.isdigit()])),)
    y_s = moving_average_centered(y, params.window)

    plot_path: Optional[Path] = None
    right_lines = None
    if make_plot:
        plot_path = out_dir / "plots" / rel.with_suffix(".png")
        # Build RIGHT_LINES from FITS (if available)
        if fits_dir is not None:
            fits_path = _find_matching_fits(fits_dir, path)
            hdr = _read_fits_header(fits_path) if fits_path else None
            right_lines = _build_right_lines_from_header(hdr)
        plot_with_plateau(x, y, y_s, best, plot_path, title, y_raw=y_raw, right_meta_lines=right_lines)

    debug_path: Optional[Path] = None
    if export_debug_csv:
        debug_path = out_dir / "debug_csv" / rel.with_suffix(".csv")
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        dx = np.gradient(x)
        eps = 1e-12
        slope = np.gradient(y_s) / (dx + eps)
        curv = np.gradient(slope) / (dx + eps)
        data_dict: Dict[str, Any] = {
            "x": x,
            "y": y,
            "y_smooth": y_s,
            "slope": slope,
            "curv": curv,
        }
        if y_raw is not None and len(y_raw) == len(y):
            data_dict["y_raw"] = y_raw
        debug_df = pd.DataFrame(data_dict)
        if best is not None:
            mask_seg = np.zeros(len(x), dtype=bool)
            mask_seg[best.start_idx : best.end_idx + 1] = True
            debug_df["in_plateau"] = mask_seg
        debug_df.to_csv(debug_path, index=False)

    row: Dict[str, Any] = {
        "file": str(path),
        "rel_file": rel.as_posix(),
        "x_col": x_col,
        "y_col": y_col_use,
        "y_raw_col": (y_raw_col if (y_raw_col is not None and y_raw is not None) else None),
        "status": "ok",
        "detected": bool(best is not None),
        "plot": (plot_path.as_posix() if plot_path else None),
        "debug_csv": (debug_path.as_posix() if debug_path else None),
        "window": params.window,
        "model_improve_min": params.model_improve_min,
        "model_contrast_min": params.model_contrast_min,
        "depth_min": params.depth_min,
        "depth_bilateral_min": params.depth_bilateral_min,
        "plateau_pad_points": params.plateau_pad_points,
        "plateau_pad_x_frac": params.plateau_pad_x_frac,
    }
    if best is not None:
        row.update(
            {
                "start_idx": best.start_idx,
                "end_idx": best.end_idx,
                "start_x": best.start_x,
                "end_x": best.end_x,
                "x_span": best.x_span,
                "mean_y": best.mean_y,
                "min_y": best.min_y,
                "score": best.score,
            }
        )
    else:
        row.update(
            {
                "start_idx": None,
                "end_idx": None,
                "start_x": None,
                "end_x": None,
                "x_span": None,
                "mean_y": None,
                "min_y": None,
                "score": None,
            }
        )
    return row

def run(
    input_dir: Path,
    output_dir: Path,
    y_col: Optional[str],
    y_raw_col: Optional[str],
    fits_dir: Optional[Path],
    params: DetectionParams,
    export_debug_csv: bool,
    make_plot: bool,
    banner: bool,
    logo_path: Optional[str],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    files = find_csv_files(input_dir)
    results: List[Dict[str, Any]] = []
    for p in files:
        try:
            row = process_file(
                path=p,
                rel_root=input_dir,
                out_dir=output_dir,
                y_col=y_col,
                y_raw_col=y_raw_col,
                fits_dir=fits_dir,
                params=params,
                export_debug_csv=export_debug_csv,
                make_plot=make_plot,
            )
        except Exception as e:
            row = {
                "file": str(p),
                "rel_file": sanitize_rel_path(input_dir, p).as_posix(),
                "status": "error",
                "error": str(e),
            }
        results.append(row)

    summary = pd.DataFrame(results)
    summary_path = output_dir / "summary.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_path, index=False)
    print(f"Saved summary: {summary_path}")

    # Banner pipeline
    if banner and make_plot:
        try:
            # clear banner folder
            banner_dir = output_dir / "plots_bannered"
            _banner_clear_out(banner_dir)
            plots_dir = output_dir / "plots"
            if plots_dir.exists():
                for img in plots_dir.rglob("*"):
                    if img.is_file() and img.suffix.lower() in EXTS:
                        # 제목: 원본 이미지 파일명(확장자 제외)
                        title = img.stem
                        # FITS 메타
                        right_lines = None
                        if fits_dir is not None:
                            fits_path = _find_matching_fits(fits_dir, img)
                            hdr = _read_fits_header(fits_path) if fits_path else None
                            right_lines = _build_right_lines_from_header(hdr)
                        out_name = f"{img.stem}_bannered{img.suffix}"
                        out_path = banner_dir / img.relative_to(plots_dir).with_name(out_name)
                        _add_header_banner(
                            input_path=str(img),
                            output_path=str(out_path),
                            title=title,
                            right_meta_lines=right_lines,
                            logo_path=logo_path,
                            banner_height=180,
                            padding=28,
                        )
                print(f"Bannered images saved to: {banner_dir}")
            else:
                print("No plots directory found; skip banner.")
        except Exception as e:
            print(f"[WARN] Banner pipeline failed: {e}")

# ------------------------------- CLI ---------------------------------- #

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Flat-bottom plateau detector (polished plotting + FITS banner)")
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder to search recursively for CSV files")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output directory for plots and summary.csv")
    ap.add_argument("--y-col", type=str, default=None, help="Corrected Y column (defaults to 2nd column if omitted)")
    ap.add_argument("--y-raw-col", type=str, default=None, help="Raw Y column (overlay for plotting only)")
    ap.add_argument("--fits-dir", type=Path, default=None, help="Directory containing original FITS files")
    ap.add_argument("--logo-path", type=str, default=None, help="Path to EXOHUNT_LOGO.png (for banner)")

    ap.add_argument("--window", type=int, default=20, help="Moving average window (odd enforced)")
    ap.add_argument("--slope-mad-mult", type=float, default=1.0, help="MAD multiplier for small |slope|")
    ap.add_argument("--curv-mad-mult", type=float, default=1.0, help="MAD multiplier for small |curv|")

    ap.add_argument("--y-thresh-mode", type=str, default="both", choices=["min+mad", "percentile", "both"])
    ap.add_argument("--y-quantile", type=float, default=0.10)
    ap.add_argument("--y-mad-mult", type=float, default=0.5)

    ap.add_argument("--min-points", type=int, default=5, help="Minimum number of points in a plateau")
    ap.add_argument("--min-x-span-frac", type=float, default=0.05, help="Minimum plateau width as fraction of x-span")
    ap.add_argument("--edge-margin-points", type=int, default=3, help="Exclude segments within this many points of either edge")

    # triple-tangent fallback
    ap.add_argument("--slope-zero-mad-mult", type=float, default=0.5, help="Tolerance for |slope|≈0 when finding stationary points")
    ap.add_argument("--stationary-min-sep-pts", type=int, default=3, help="Min separation (indices) between stationary candidates")
    ap.add_argument("--legacy-mask", action="store_true", help="Use legacy low-Y mask method in fallback")
    ap.add_argument("--contrast-min", type=float, default=0.0, help="Fallback: minimal contrast for acceptance")

    # model selection
    ap.add_argument("--no-model-selection", action="store_true", help="Disable plateau-vs-quadratic model selection")
    ap.add_argument("--model-improve-min", type=float, default=0.10, help="Require SSE improvement over quadratic (e.g., 0.10 = 10%)")
    ap.add_argument("--model-contrast-min", type=float, default=0.00, help="Require contrast >= this (y-scale MAD units)")

    # prefer wider & deeper
    ap.add_argument("--width-pref", type=float, default=0.5, help="Weight for width in candidate scoring (0..1)")
    ap.add_argument("--depth-pref", type=float, default=0.5, help="Weight for depth/contrast in candidate scoring (0..1)")
    ap.add_argument("--widen-rel-tol", type=float, default=0.02, help="Allow relative SSE increase when greedily widening")

    # FP suppression filters
    ap.add_argument("--depth-min", type=float, default=0.15, help="Require min(left_gap,right_gap) >= this (MAD units)")
    ap.add_argument("--depth-bilateral-min", type=float, default=0.08, help="Require both left/right gaps >= this (MAD units)")
    ap.add_argument("--shoulder-win-frac", type=float, default=0.5, help="Shoulder window length as fraction of plateau width")
    ap.add_argument("--slope-sign-frac-min", type=float, default=0.6, help="Min fraction of negative slopes (left) / positive (right)")
    ap.add_argument("--inside-flat-frac-min", type=float, default=0.6, help="Min fraction inside plateau with small |slope| (MAD-thresh)")
    ap.add_argument("--baseline-line-depth-min", type=float, default=0.0, help="Min depth vs linear shoulder extrapolation at mid (MAD)")

    # edge-monotone rejection
    ap.add_argument("--edge-monotone-frac-min", type=float, default=0.75, help="Min fraction of monotone slopes from edges to reject")
    ap.add_argument("--edge-drop-min", type=float, default=0.50, help="Min drop (MAD units) from edge median to plateau to reject")
    ap.add_argument("--edge-monotone-scope-pts", type=int, default=12, help="Only apply edge-monotone if plateau boundary is within this many points from the edge")
    ap.add_argument("--edge-monotone-scope-x-frac", type=float, default=0.06, help="... or within this x-fraction of total span from the edge")

    # hard padding from x-ends
    ap.add_argument("--plateau-pad-points", type=int, default=8, help="At least this many points before/after plateau")
    ap.add_argument("--plateau-pad-x-frac", type=float, default=0.02, help="At least this fraction of x-span away from each end")

    # outputs
    ap.add_argument("--plot", action="store_true", help="Save annotated plots")
    ap.add_argument("--export-debug-csv", action="store_true", help="Export per-file debug CSV with internals")
    ap.add_argument("--banner", action="store_true", help="After plotting, create bannered images in plots_bannered (deletes folder first)")

    return ap

def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()

    params = DetectionParams(
        window=args.window,
        slope_mad_mult=args.slope_mad_mult,
        curv_mad_mult=args.curv_mad_mult,
        y_thresh_mode=args.y_thresh_mode,
        y_quantile=args.y_quantile,
        y_mad_mult=args.y_mad_mult,
        min_points=args.min_points,
        min_x_span_frac=args.min_x_span_frac,
        edge_margin_points=args.edge_margin_points,
        slope_zero_mad_mult=args.slope_zero_mad_mult,
        stationary_min_sep_pts=args.stationary_min_sep_pts,
        use_legacy_mask=args.legacy_mask,
        contrast_min=args.contrast_min,
        use_model_selection=not args.no_model_selection,
        model_improve_min=args.model_improve_min,
        model_contrast_min=args.model_contrast_min,
        width_pref=args.width_pref,
        depth_pref=args.depth_pref,
        widen_rel_tol=args.widen_rel_tol,
        depth_min=args.depth_min,
        depth_bilateral_min=args.depth_bilateral_min,
        shoulder_win_frac=args.shoulder_win_frac,
        slope_sign_frac_min=args.slope_sign_frac_min,
        inside_flat_frac_min=args.inside_flat_frac_min,
        baseline_line_depth_min=args.baseline_line_depth_min,
        edge_monotone_frac_min=args.edge_monotone_frac_min,
        edge_drop_min=args.edge_drop_min,
        edge_monotone_scope_pts=args.edge_monotone_scope_pts,
        edge_monotone_scope_x_frac=args.edge_monotone_scope_x_frac,
        plateau_pad_points=args.plateau_pad_points,
        plateau_pad_x_frac=args.plateau_pad_x_frac,
    )

    run(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        y_col=args.y_col,
        y_raw_col=args.y_raw_col,
        fits_dir=args.fits_dir,
        params=params,
        export_debug_csv=args.export_debug_csv,
        make_plot=args.plot,
        banner=args.banner,
        logo_path=args.logo_path,
    )

if __name__ == "__main__":
    main()
