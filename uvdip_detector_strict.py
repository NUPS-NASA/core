
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
U/V dip detector (clean re-write)
- Coarse grid search + optional JAX refine
- Robust to edge noise (weighted Huber loss)
- Rectangle/bottom-fill rejection + curvature/slope/symmetry/coverage checks
- Coarse fallback if refine collapses a true dip
- Generates per-file PNGs + summary CSV
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------- Utils -----------------------------

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x)
    mad = np.median(np.abs(x - med))
    return 1.4826 * (mad + 1e-12)

def find_columns(df: pd.DataFrame, time_col: Optional[str], y_col: Optional[str]) -> Tuple[str, str]:
    if time_col is not None and y_col is not None:
        return time_col, y_col
    cand = {c.lower(): c for c in df.columns}
    time_aliases = ["time since first frame", "time", "t", "x", "timestamp", "frame", "hour", "jd"]
    y_aliases = ["relative flux", "flux", "y", "value", "intensity", "detrended"]
    def guess(aliases):
        for a in aliases:
            if a in cand: return cand[a]
            for k,v in cand.items():
                if k.startswith(a): return v
        return None
    tc = time_col or guess(time_aliases)
    yc = y_col or guess(y_aliases)
    if tc is None or yc is None:
        nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(nums) >= 2:
            tc = tc or nums[0]
            yc = yc or nums[1]
    if tc is None or yc is None:
        raise ValueError("Could not infer time/y columns. Pass --time-col and --y-col.")
    return tc, yc

def rectangular_y_distribution(y: np.ndarray, bins: int = 50) -> float:
    if len(y) < 10: return 0.0
    hist,_ = np.histogram(y, bins=bins)
    occ = (hist > 0).mean()
    spread = np.percentile(y,97.5) - np.percentile(y,2.5)
    sigma = robust_mad(y)
    wide = np.clip(spread/(sigma+1e-12), 0, 20)/20.0
    return float(0.6*occ + 0.4*wide)

def soft_box_np(t: np.ndarray, c: float, wv: float, tau: float) -> np.ndarray:
    taus = max(float(tau), 1e-9*(float(t.max())-float(t.min())+1))
    z1 = (t-(c-0.5*wv))/taus
    z2 = (t-(c+0.5*wv))/taus
    z1 = np.clip(z1, -60, 60); z2 = np.clip(z2, -60, 60)
    s1 = 1.0/(1.0+np.exp(-z1)); s2 = 1.0/(1.0+np.exp(-z2))
    return s1 - s2

def huber_np(r, dlt):
    ar=np.abs(r); quad=0.5*(ar**2); lin=dlt*(ar-0.5*dlt)
    return np.where(ar<=dlt, quad, lin)

# ----------------------------- Coarse grid -----------------------------

def coarse_grid_search(t: np.ndarray, y: np.ndarray, centers:int=80, widths:int=25, depths:int=40)->Dict[str,float]:
    sigma = robust_mad(y); delta = 1.345*sigma
    n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w=0.25+0.75*w_edge
    a0=float(np.median(y)); loss_base=float(np.sum(huber_np((y-a0)*w, delta)))
    tspan=(float(t.max())-float(t.min())+1e-12); tau=0.01*tspan
    C=np.linspace(float(t.min()), float(t.max()), centers)
    W=np.linspace(0.05*tspan, 0.70*tspan, widths)
    dmax=max(1e-6, a0 - float(np.min(y)))
    D=np.linspace(0.0, dmax*2.0, depths)
    best={"loss": np.inf}
    for wv in W:
        for c in C:
            box=soft_box_np(t,c,wv,tau)
            yhat_all=a0 - np.outer(D, box)
            res=(y - yhat_all)*w
            L=np.sum(huber_np(res, delta), axis=1)
            idx=int(np.argmin(L)); Lmin=float(L[idx])
            if Lmin < best["loss"]:
                best={"loss": Lmin, "center": float(c), "width": float(wv), "depth": float(D[idx]), "a": a0}
    improvement=max(0.0, (loss_base-best["loss"])/(loss_base+1e-12)); snr=best["depth"]/(sigma+1e-12)
    return {"a": best["a"], "center": best["center"], "width": best["width"], "depth": best["depth"],
            "improvement": float(improvement), "snr": float(snr), "loss_base": float(loss_base), "loss_model": float(best["loss"]), "tau": float(tau)}

# ----------------------------- Optional JAX refine -----------------------------

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    import optax
    HAVE_JAX = True
except Exception:
    HAVE_JAX = False

if HAVE_JAX:
    def soft_box_jax(t, c, w, tau):
        left=c-0.5*w; right=c+0.5*w
        s1=jax.nn.sigmoid((t-left)/tau); s2=jax.nn.sigmoid((t-right)/tau)
        return jnp.clip(s1-s2, 0.0, 1.0)

    def huber_jax(residuals, delta):
        abs_r=jnp.abs(residuals); quad=0.5*(abs_r**2); lin=delta*(abs_r-0.5*delta)
        return jnp.where(abs_r<=delta, quad, lin)

    @jit
    def objective(params: Dict[str,jnp.ndarray], t:jnp.ndarray, y:jnp.ndarray, w_weights:jnp.ndarray,
                  tau:jnp.ndarray, delta:jnp.ndarray, w_min:jnp.ndarray, w_max:jnp.ndarray,
                  lam_width:jnp.ndarray, lam_amp:jnp.ndarray)->jnp.ndarray:
        tmin=t.min(); tmax=t.max()
        a=params["a"]; d=jax.nn.softplus(params["d_raw"])
        c=tmin + (tmax-tmin)*jax.nn.sigmoid(params["c_sig"])
        w=w_min + (w_max-w_min)*jax.nn.sigmoid(params["w_sig"])
        box=soft_box_jax(t,c,w,tau); yhat=a - d*box
        res=(y - yhat)*w_weights; loss=jnp.sum(huber_jax(res, delta))
        width_reg=lam_width*jnp.exp(-(w/(w_min+1e-6)))
        amp_reg=lam_amp*(d**2)  # keep small to avoid killing depth
        return loss + width_reg + amp_reg

    def jax_refine(t: np.ndarray, y: np.ndarray, seed:int, coarse: Dict[str,float])->Dict[str,Any]:
        key=jax.random.PRNGKey(seed); t_j=jnp.asarray(t); y_j=jnp.asarray(y)
        sigma=float(robust_mad(y)); delta=np.float32(1.345*sigma)
        n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w_weights=jnp.asarray(0.25+0.75*w_edge)
        tmin=float(np.min(t)); tmax=float(np.max(t)); span=float(max(tmax-tmin,1e-9))
        tau=float(max(1e-9, 0.01*(span+1e-12))); w_min=0.05*(span+1e-12); w_max=0.80*(span+1e-12)
        lam_width=1.0; lam_amp=1e-6  # softened
        steps=800; lr=0.02

        def safe_logit(p:float, eps:float=1e-6)->float: p=np.clip(p, eps, 1-eps); return float(np.log(p/(1-p)))
        def inv_softplus(d:float, eps:float=1e-12)->float: d=max(0.0, d); return float(np.log(np.expm1(d)+eps))
        def pack(a:float,d:float,c:float,w:float):
            frac_c=(c-tmin)/span; alpha=(w-w_min)/max(w_max-w_min,1e-9)
            c_sig=safe_logit(frac_c); w_sig=safe_logit(alpha); d_raw=inv_softplus(d)
            return {"a": np.float32(a), "d_raw": np.float32(d_raw), "c_sig": np.float32(c_sig), "w_sig": np.float32(w_sig)}

        a0=float(np.median(y)); inits=[pack(coarse["a"], coarse["depth"], coarse["center"], coarse["width"])]
        for _ in range(5):
            key,k1,k2,k3,k4=jax.random.split(key, 5)
            a_i=a0 + 0.1*sigma*float(jax.random.normal(k1))
            d_i=abs(0.5*sigma*float(jax.random.normal(k2))) + 0.1*sigma
            c_i=float(tmin + span*jax.random.uniform(k3)); w_i=float(w_min + (w_max-w_min)*jax.random.uniform(k4))
            inits.append(pack(a_i,d_i,c_i,w_i))

        opt=optax.adam(lr)
        obj=lambda p: objective(p,t_j,y_j,w_weights,jnp.asarray(tau),jnp.asarray(delta),jnp.asarray(w_min),jnp.asarray(w_max),jnp.asarray(lam_width),jnp.asarray(lam_amp))
        best_params=None; best_val=np.inf

        @jit
        def step(p,s):
            val,grads=value_and_grad(obj)(p)
            updates,s=opt.update(grads,s,p); p=optax.apply_updates(p, updates)
            return p,s,val

        for init in inits:
            params={k:jnp.asarray(v) for k, v in init.items()}; opt_state=opt.init(params)
            val=None
            for _ in range(steps):
                params,opt_state,val=step(params,opt_state)
            val_np=float(val)
            if val_np < best_val:
                best_val=val_np; best_params={k: float(v) for k, v in params.items()}

        a=best_params["a"]; d=float(np.log1p(np.exp(best_params["d_raw"])))
        c=tmin + span*(1/(1+np.exp(-best_params["c_sig"])))
        w=w_min + (w_max-w_min)*(1/(1+np.exp(-best_params["w_sig"])))
        sigma=robust_mad(y); snr=float(d/(sigma+1e-12))
        return {"a": float(a), "center": float(c), "width": float(w), "depth": float(d), "snr": snr, "improvement": coarse["improvement"], "refined": True}
else:
    def jax_refine(t: np.ndarray, y: np.ndarray, seed:int, coarse: Dict[str,float])->Dict[str,Any]:
        return {**coarse, "refined": False}

# ----------------------------- Shape checks -----------------------------

def uv_shape_checks(t: np.ndarray, y: np.ndarray, c: float, w: float,
                    rect_thr: float, z_drop_thr: float, slope_sigma_factor: float,
                    q3_near_baseline_sigmas: float, local_rect_occ_thr: float,
                    slope_gap_factor: float, shoulder_sym_sigmas: float,
                    hcov_bins: int, hcov_min_occ: float,
                    min_edge_frac: float, edge_override_imp: float,
                    curv_sigma_factor: float,
                    lower_band_k: float, lower_band_frac_thr: float,
                    inside_mask: Optional[np.ndarray]=None)->Dict[str,Any]:

    rect_score = rectangular_y_distribution(y, bins=50); is_rect = rect_score >= rect_thr
    if inside_mask is None: inside_mask = (t >= (c-0.5*w)) & (t <= (c+0.5*w))

    # Shoulders (outside region near interval)
    L1=max(c-1.0*w,float(t.min())); L0=max(c-0.6*w,float(t.min()))
    R0=min(c+0.6*w,float(t.max())); R1=min(c+1.0*w,float(t.max()))
    left_sh=y[(t>=L1)&(t<L0)]; right_sh=y[(t>R0)&(t<=R1)]
    outside = ~inside_mask
    shoulders = np.concatenate([left_sh,right_sh]) if (len(left_sh)+len(right_sh))>0 else y[outside]
    if len(shoulders) < 6: shoulders = y[outside]
    base_level = np.median(shoulders); sigma_sh = robust_mad(shoulders)

    # Vertex/inside stats
    V0=c-0.25*w; V1=c+0.25*w; inside_V = (t>=V0)&(t<=V1); yV=y[inside_V]; yI=y[inside_mask]
    if len(yV)==0:
        return {"shape_ok": True, "is_rect": is_rect, "z_drop": 0.0, "q3_ok": True, "lower_rect_like": False, "slope_ok": True,
                "rect_score": rect_score, "sym_ok": True, "hcov_ok": True, "gap_ok": True, "edge_ok": True, "curv_ok": True, "lower_band_dom": False}
    y_bot=np.percentile(yV,10.0); drop=base_level - y_bot; z_drop=drop/(sigma_sh+1e-12)

    q3_in=np.percentile(yI,75.0) if len(yI) else np.percentile(yV,75.0)
    q3_ok=(q3_in >= base_level - q3_near_baseline_sigmas*sigma_sh)

    if len(yI)>=20:
        hist,_=np.histogram(yI,bins=30); occ_in=(hist>0).mean()
    else:
        occ_in=0.0
    lower_rect_like = (occ_in>=local_rect_occ_thr)

    # Slope L/R inside halves
    left_mask=(t>=c-0.5*w)&(t<c); right_mask=(t>c)&(t<=c+0.5*w); slope_ok=True; sl=sr=np.nan
    if (np.sum(left_mask)>=5) and (np.sum(right_mask)>=5):
        def winsorize(a,p=2.5):
            lo,hi=np.percentile(a,[p,100-p]); return np.clip(a,lo,hi)
        tl,yl=winsorize(t[left_mask]),winsorize(y[left_mask]); tr,yr=winsorize(t[right_mask]),winsorize(y[right_mask])
        sl=np.polyfit(tl,yl,1)[0]; sr=np.polyfit(tr,yr,1)[0]
        slope_min=slope_sigma_factor*(sigma_sh/(w+1e-12)); slope_ok=(sl<=-slope_min) and (sr>=slope_min)

    # Symmetry of shoulders
    sym_ok=True
    if len(left_sh)>=4 and len(right_sh)>=4:
        ml=np.median(left_sh); mr=np.median(right_sh); sym_ok=(abs(ml-mr) <= shoulder_sym_sigmas*sigma_sh)

    # Horizontal coverage inside
    hcov_ok=True
    if np.sum(inside_mask) >= max(10, hcov_bins):
        tb=t[inside_mask]
        if tb.size>0:
            bins=np.linspace(tb.min(), tb.max(), hcov_bins+1); hist,_=np.histogram(tb, bins=bins)
            hcov=(hist>0).mean(); hcov_ok=(hcov >= hcov_min_occ)
        else:
            hcov_ok=False

    # Edge distance
    span=float(t.max()-t.min()+1e-12); left_gap=(c-0.5*w)-float(t.min()); right_gap=float(t.max())-(c+0.5*w)
    edge_ok=(left_gap >= min_edge_frac*span) and (right_gap >= min_edge_frac*span)

    # Gap & curvature
    gap_ok=True
    if np.isfinite(sl) and np.isfinite(sr):
        gap_ok = ((sr - (-sl)) >= slope_gap_factor * (sigma_sh / (w + 1e-12)))

    curv_ok=True
    try:
        xin = (t[inside_mask] - c) / max(w, 1e-9); yin = y[inside_mask]
        if xin.size >= 6:
            def wz(a, p=2.5):
                lo, hi = np.percentile(a, [p, 100-p]); return np.clip(a, lo, hi)
            xx = wz(xin); yy = wz(yin)
            M = np.vstack([np.ones_like(xx), xx, xx**2]).T
            coef, *_ = np.linalg.lstsq(M, yy, rcond=None); k = float(coef[2])
            k_unit = sigma_sh / (max(w, 1e-9)**2); curv_ok = (k >= curv_sigma_factor * k_unit)
        else:
            curv_ok = False
    except Exception:
        curv_ok = True

    # Lower-band dominance (reject bottom-filled rectangles) – conditional to avoid hiding real dips
    lower_band_dom=False
    if len(yI) >= 12:
        y_thr = base_level - lower_band_k * sigma_sh
        frac_low = float(np.mean(yI <= y_thr))
        lower_band_dom = (frac_low >= lower_band_frac_thr)
    apply_lower_band = (not slope_ok) and (gap_ok or (not curv_ok)) and (z_drop >= (z_drop_thr + 1.2))

    shape_basic = (z_drop >= z_drop_thr) and (q3_ok or slope_ok or (z_drop >= (z_drop_thr+1.0))) \
                  and (not lower_rect_like) and (not (lower_band_dom and apply_lower_band))

    # Borderline allowance: near-threshold z, good Q3+sym -> allow even if gap/curv fail
    borderline = (z_drop < (z_drop_thr + 0.6)) and q3_ok and sym_ok
    shape_ok = shape_basic and hcov_ok and ( (sym_ok and (gap_ok or curv_ok)) or (z_drop >= (z_drop_thr+1.0)) or slope_ok or borderline )

    return {"shape_ok": bool(shape_ok), "is_rect": bool(is_rect), "z_drop": float(z_drop), "q3_ok": bool(q3_ok),
            "lower_rect_like": bool(lower_rect_like), "slope_ok": bool(slope_ok), "rect_score": float(rect_score),
            "sym_ok": bool(sym_ok), "hcov_ok": bool(hcov_ok), "gap_ok": bool(gap_ok), "edge_ok": bool(edge_ok),
            "curv_ok": bool(curv_ok), "lower_band_dom": bool(lower_band_dom)}

# ----------------------------- Detector -----------------------------

def detect_on_xy(t: np.ndarray, y: np.ndarray,
                 prefer_sensitivity: bool=True, imp_override: float=0.08, base_snr: float=0.8, base_imp: float=0.02,
                 grid_centers:int=80, grid_widths:int=25, grid_depths:int=40, seed:int=0,
                 z_drop_thr: float=1.8, slope_sigma_factor: float=0.8, q3_near_baseline_sigmas: float=1.0,
                 local_rect_occ_thr: float=0.94, rect_thr: float=0.88,
                 slope_gap_factor: float=1.0, shoulder_sym_sigmas: float=1.8,
                 hcov_bins: int=6, hcov_min_occ: float=0.60,
                 min_edge_frac: float=0.06, edge_override_imp: float=0.24,
                 curv_sigma_factor: float=0.6,
                 lower_band_k: float=2.2, lower_band_frac: float=0.92,
                 min_support_abs: int=12, support_frac: float=0.05, guard_min_support: int=24,
                 grid_only: bool=False, debug: bool=False)->Dict[str,Any]:

    mask=np.isfinite(t)&np.isfinite(y); t=t[mask]; y=y[mask]; o=np.argsort(t); t=t[o]; y=y[o]

    # Coarse fit
    coarse=coarse_grid_search(t,y,centers=grid_centers,widths=grid_widths,depths=grid_depths)
    inside_c=(t>=(coarse["center"]-0.5*coarse["width"])) & (t<=(coarse["center"]+0.5*coarse["width"]))
    support_c=int(np.sum(inside_c)); min_points=max(int(min_support_abs), int(support_frac*len(t)))
    shape_c=uv_shape_checks(t,y,coarse["center"],coarse["width"], rect_thr, z_drop_thr, slope_sigma_factor,
                            q3_near_baseline_sigmas, local_rect_occ_thr, slope_gap_factor, shoulder_sym_sigmas,
                            hcov_bins, hcov_min_occ, min_edge_frac, edge_override_imp, curv_sigma_factor,
                            lower_band_k, lower_band_frac, inside_c)
    rule_or_c = prefer_sensitivity and (coarse["improvement"]>=imp_override) and (support_c>=min_points)

    if debug:
        print(f"[COARSE] imp={coarse['improvement']:.3f} snr={coarse['snr']:.3f} support={support_c} "
              f"z={shape_c['z_drop']:.2f} q3={shape_c['q3_ok']} slope={shape_c['slope_ok']} "
              f"lower_rect={shape_c['lower_rect_like']} rect_score={shape_c['rect_score']:.2f} "
              f"sym={shape_c['sym_ok']} hcov={shape_c['hcov_ok']} gap={shape_c['gap_ok']} curv={shape_c['curv_ok']} edge={shape_c['edge_ok']}")

    if rule_or_c and shape_c["shape_ok"] and (not shape_c["is_rect"]) and (shape_c["edge_ok"] or (coarse["improvement"]>=edge_override_imp)):
        return {"has_dip": True, "reason": None, "a":coarse["a"],"depth":coarse["depth"],"center":coarse["center"],"width":coarse["width"],
                "snr":coarse["snr"],"improvement":coarse["improvement"],"rect_score":shape_c["rect_score"],
                "support_points":support_c,"source":"coarse","z_drop":shape_c["z_drop"]}

    # Refine
    refined = jax_refine(t,y,seed=seed,coarse=coarse) if (not grid_only and HAVE_JAX) else (dict(coarse)|{"refined":False})

    # Recompute improvement/SNR using refined params
    a0=float(np.median(y)); sigma=robust_mad(y); delta=1.345*sigma
    n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w=0.25+0.75*w_edge
    tau=0.01*(float(t.max())-float(t.min())); box=soft_box_np(t,refined["center"],refined["width"],tau)
    yhat=refined["a"] - refined["depth"]*box
    loss_base=float(np.sum(huber_np((y-a0)*w, delta))); loss_model=float(np.sum(huber_np((y-yhat)*w, delta)))
    refined["improvement"]=max(0.0,(loss_base-loss_model)/(loss_base+1e-12)); refined["snr"]=float(refined["depth"]/(sigma+1e-12))

    inside=(t>=(refined["center"]-0.5*refined["width"])) & (t<=(refined["center"]+0.5*refined["width"])); support_r=int(np.sum(inside))
    shape_r=uv_shape_checks(t,y,refined["center"],refined["width"], rect_thr, z_drop_thr, slope_sigma_factor,
                            q3_near_baseline_sigmas, local_rect_occ_thr, slope_gap_factor, shoulder_sym_sigmas,
                            hcov_bins, hcov_min_occ, min_edge_frac, edge_override_imp, curv_sigma_factor,
                            lower_band_k, lower_band_frac, inside)

    # Decision thresholds
    snr_thr=1.1 if (not prefer_sensitivity) else base_snr; imp_thr=0.05 if (not prefer_sensitivity) else base_imp
    if prefer_sensitivity: snr_thr=float(np.interp(refined["improvement"], [0.00,0.08,0.20], [snr_thr,0.55,0.35]))
    rule_and=(refined["snr"]>=snr_thr) and (refined["improvement"]>=imp_thr) and (support_r>=min_points)
    rule_or=prefer_sensitivity and (refined["improvement"]>=imp_override) and (support_r>=min_points)

    has_dip=(not shape_r["is_rect"]) and shape_r["shape_ok"] and (rule_and or rule_or) and (shape_r["edge_ok"] or (refined["improvement"]>=edge_override_imp))

    # Guard for FN: strong z-drop + good coverage/symmetry can pass with OR rule and enough points
    if (not has_dip) and (not shape_r.get("is_rect", False)) and shape_r.get("q3_ok", False) \
       and shape_r.get("hcov_ok", False) and shape_r.get("sym_ok", False) \
       and (shape_r.get("z_drop", 0.0) >= z_drop_thr) and rule_or and (support_r >= guard_min_support):
        has_dip = True

    # Borderline leniency support floor: if lenient pass but weak support, drop
    borderline_case = shape_r["shape_ok"] and (not shape_r["slope_ok"]) and (not shape_r["gap_ok"]) and (not shape_r["curv_ok"]) \
                      and (shape_r["z_drop"] < (z_drop_thr + 0.6)) and shape_r["q3_ok"] and shape_r["sym_ok"]
    if has_dip and borderline_case and (support_r < 20):
        has_dip = False

    # Fallback to coarse if refinement degraded while coarse was acceptable
    if (not has_dip):
        coarse_support_ok = (support_c >= min_points)
        coarse_uv_ok = (not shape_c["is_rect"]) and shape_c["shape_ok"] and (shape_c["edge_ok"] or (coarse["improvement"] >= edge_override_imp))
        coarse_or = prefer_sensitivity and (coarse["improvement"] >= imp_override) and coarse_support_ok
        if coarse_or and coarse_uv_ok and (coarse["improvement"] >= refined["improvement"]):
            has_dip = True
            refined = dict(coarse)|{"refined": False}
            shape_r = shape_c
            support_r = support_c

    reason=None
    if has_dip:
        reason=None
    elif shape_r["is_rect"]:
        reason=f"Global vertical distribution looks rectangular (score={shape_r['rect_score']:.2f})."
    elif not shape_r["shape_ok"]:
        reason=f"Rejected by U/V shape (z={shape_r['z_drop']:.2f}, q3={shape_r['q3_ok']}, slope={shape_r['slope_ok']}, lower_rect={shape_r['lower_rect_like']})."
    else:
        reason="Insufficient SNR/improvement/support or edge proximity."

    if debug:
        print(f"[REFINE={'True' if (not grid_only and HAVE_JAX) else 'False'}] imp={refined['improvement']:.3f} snr={refined['snr']:.3f} support={support_r} "
              f"z={shape_r['z_drop']:.2f} q3={shape_r['q3_ok']} slope={shape_r['slope_ok']} "
              f"lower_rect={shape_r['lower_rect_like']} rect_score={shape_r['rect_score']:.2f} "
              f"sym={shape_r['sym_ok']} hcov={shape_r['hcov_ok']} gap={shape_r['gap_ok']} curv={shape_r['curv_ok']} edge={shape_r['edge_ok']} "
              f"AND={rule_and} OR={rule_or} has_dip={has_dip}")

    return {"has_dip": bool(has_dip), "reason": reason, "a": refined["a"], "depth": refined["depth"], "center": refined["center"],
            "width": refined["width"], "snr": refined["snr"], "improvement": refined["improvement"], "rect_score": shape_r["rect_score"],
            "support_points": support_r, "source": "refined" if ((not grid_only) and HAVE_JAX) else "coarse", "z_drop": shape_r["z_drop"]}

# ----------------------------- I/O & CLI -----------------------------

def plot_and_save(t: np.ndarray, y: np.ndarray, res: Dict[str,Any], out_png: Path, title: str="")->None:
    plt.figure(figsize=(9,5.2))
    plt.scatter(t, y, s=18)
    plt.xlabel("Time"); plt.ylabel("Y")
    if title: plt.title(title)
    if "a" in res and np.isfinite(res["a"]):
        plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")
    if res.get("has_dip", False) and np.isfinite(res.get("center", np.nan)) and np.isfinite(res.get("width", np.nan)):
        s = res["center"] - 0.5 * res["width"]
        e = res["center"] + 0.5 * res["width"]
        plt.axvspan(s, e, alpha=0.2, label=f"Detected interval ({res.get('source','')})")
        try:
            plt.text(s, res.get("a", np.nan), f"depth≈{res.get('depth',float('nan')):.4g}, SNR≈{res.get('snr',float('nan')):.2f}, imp≈{100*res.get('improvement',0.0):.1f}%", va="bottom", ha="left")
        except Exception:
            pass
    else:
        reason = res.get("reason")
        if reason:
            try:
                xmin = float(np.min(t)); ymax = float(np.max(y))
                plt.text(xmin, ymax, reason, va="top", ha="left")
            except Exception:
                pass
    plt.legend(frameon=False, loc="best")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, dpi=150); plt.close()

def load_csv(path: Path, time_col: Optional[str], y_col: Optional[str])->Tuple[np.ndarray,np.ndarray,str,str]:
    df=pd.read_csv(path); tc,yc=find_columns(df,time_col,y_col)
    t=df[tc].to_numpy(float); y=df[yc].to_numpy(float); return t,y,tc,yc

def process_folder(root: Path, out_dir: Path, time_col: Optional[str], y_col: Optional[str],
                   prefer_sensitivity: bool=True, imp_override: float=0.08, base_snr: float=0.8, base_imp: float=0.02,
                   grid_centers:int=80, grid_widths:int=25, grid_depths:int=40, seed:int=0,
                   z_drop_thr: float=1.8, slope_sigma_factor: float=0.8, q3_near_baseline_sigmas: float=1.0,
                   local_rect_occ_thr: float=0.94, rect_thr: float=0.88,
                   slope_gap_factor: float=1.0, shoulder_sym_sigmas: float=1.8, hcov_bins: int=6, hcov_min_occ: float=0.60,
                   min_edge_frac: float=0.06, edge_override_imp: float=0.24, curv_sigma_factor: float=0.6,
                   lower_band_k: float=2.2, lower_band_frac: float=0.92,
                   min_support_abs: int=12, support_frac: float=0.05, guard_min_support: int=24,
                   grid_only: bool=False, debug: bool=False)->Path:
    rows: List[Dict[str,Any]]=[]
    csv_files=sorted(root.rglob("*.csv"))
    if not csv_files: print(f"No CSV files found under {root}")
    for p in csv_files:
        try:
            t,y,tc,yc=load_csv(p,time_col,y_col)
            res=detect_on_xy(t,y,prefer_sensitivity=prefer_sensitivity,imp_override=imp_override,base_snr=base_snr,base_imp=base_imp,
                             grid_centers=grid_centers,grid_widths=grid_widths,grid_depths=grid_depths,seed=seed,
                             z_drop_thr=z_drop_thr,slope_sigma_factor=slope_sigma_factor,q3_near_baseline_sigmas=q3_near_baseline_sigmas,
                             local_rect_occ_thr=local_rect_occ_thr,rect_thr=rect_thr,
                             slope_gap_factor=slope_gap_factor, shoulder_sym_sigmas=shoulder_sym_sigmas, hcov_bins=hcov_bins, hcov_min_occ=hcov_min_occ,
                             min_edge_frac=min_edge_frac, edge_override_imp=edge_override_imp, curv_sigma_factor=curv_sigma_factor,
                             lower_band_k=lower_band_k, lower_band_frac=lower_band_frac,
                             min_support_abs=min_support_abs, support_frac=support_frac, guard_min_support=guard_min_support,
                             grid_only=grid_only,debug=debug)
            png_path=out_dir / p.with_suffix(".png").name
            plot_and_save(t,y,res,png_path,title=f"{p.name} [has_dip={res['has_dip']}]")
            rows.append({"file":str(p),"has_dip":res["has_dip"],"baseline_a":res["a"],"center":res["center"],"width":res["width"],"depth":res["depth"],
                         "snr":res["snr"],"improvement":res["improvement"],"support_points":res["support_points"],"rect_score":res["rect_score"],
                         "z_drop":res.get("z_drop",0.0),"note":res.get("reason",""),"image":str(png_path),"time_col":tc,"y_col":yc,"source":res.get("source","")})
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
    out_dir.mkdir(parents=True, exist_ok=True); summary_path=out_dir/"dip_summary.csv"; pd.DataFrame(rows).to_csv(summary_path,index=False); return summary_path

def main():
    ap=argparse.ArgumentParser(description="Detect U/V-shaped dips in scatter charts (grid + optional JAX refine).")
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
    ap.add_argument("--z-drop", type=float, default=1.8, dest="z_drop", help="Vertex drop z-score threshold vs shoulders.")
    ap.add_argument("--slope-sigma", type=float, default=0.8, dest="slope_sigma", help="Slope threshold factor (sigma_shoulder / w).")
    ap.add_argument("--q3-sigmas", type=float, default=1.0, dest="q3_sigmas", help="Q3 inside must be within this many sigmas of baseline.")
    ap.add_argument("--local-rect-occ", type=float, default=0.94, dest="local_rect_occ", help="Inside-interval vertical occupancy threshold to reject rectangle-like intervals.")
    ap.add_argument("--rect-thr", type=float, default=0.88, dest="rect_thr", help="Global rectangularness score threshold to treat as y≈a.")
    ap.add_argument("--slope-gap", type=float, default=1.0, dest="slope_gap", help="Require (sr - (-sl)) >= slope_gap*(sigma_shoulder/w).")
    ap.add_argument("--shoulder-sym", type=float, default=1.8, dest="shoulder_sym", help="|median_left - median_right| <= shoulder_sym * sigma_shoulder.")
    ap.add_argument("--hcov-bins", type=int, default=6, dest="hcov_bins", help="Number of bins for horizontal coverage inside interval.")
    ap.add_argument("--hcov-min", type=float, default=0.60, dest="hcov_min", help="Min fraction of non-empty bins inside interval.")
    ap.add_argument("--min-edge-frac", type=float, default=0.06, dest="min_edge_frac", help="Min fraction of span the interval must be away from both edges to avoid edge noise.")
    ap.add_argument("--edge-imp", type=float, default=0.24, dest="edge_imp", help="If improvement >= this, allow edge proximity.")
    ap.add_argument("--curv-sigma", type=float, default=0.6, dest="curv_sigma", help="Min quadratic curvature (k) in units of sigma_shoulder/w^2.")
    ap.add_argument("--lower-band-k", type=float, default=2.2, dest="lower_band_k", help="How far below baseline (in sigma_shoulder) counts as lower band.")
    ap.add_argument("--lower-band-frac", type=float, default=0.92, dest="lower_band_frac", help="If this fraction of inside points falls in the lower band, reject as bottom-fill (conditional).")
    ap.add_argument("--min-support", type=int, default=12, dest="min_support", help="Absolute minimum number of points required inside the interval.")
    ap.add_argument("--support-frac", type=float, default=0.05, dest="support_frac", help="Minimum fraction of points inside the interval.")
    ap.add_argument("--guard-min-support", type=int, default=24, dest="guard_min_support", help="Guarded-pass requires at least this many inside points.")
    ap.add_argument("--grid-only", action="store_true", help="Disable JAX refine; decide on coarse fit only.")
    ap.add_argument("--debug", action="store_true", help="Print detailed diagnostics per file.")
    args=ap.parse_args()

    root=Path(args.folder).expanduser().resolve()
    out_dir=Path(args.out_dir).expanduser().resolve() if args.out_dir else (root/"_dip_outputs")
    summary_path=process_folder(root,out_dir,args.time_col,args.y_col, prefer_sensitivity=(not args.strict),
                                imp_override=args.imp_override, base_snr=args.snr, base_imp=args.imp,
                                grid_centers=args.grid_centers, grid_widths=args.grid_widths, grid_depths=args.grid_depths, seed=args.seed,
                                z_drop_thr=args.z_drop, slope_sigma_factor=args.slope_sigma, q3_near_baseline_sigmas=args.q3_sigmas,
                                local_rect_occ_thr=args.local_rect_occ, rect_thr=args.rect_thr,
                                slope_gap_factor=args.slope_gap, shoulder_sym_sigmas=args.shoulder_sym,
                                hcov_bins=args.hcov_bins, hcov_min_occ=args.hcov_min, min_edge_frac=args.min_edge_frac, edge_override_imp=args.edge_imp,
                                curv_sigma_factor=args.curv_sigma, lower_band_k=args.lower_band_k, lower_band_frac=args.lower_band_frac,
                                min_support_abs=args.min_support, support_frac=args.support_frac, guard_min_support=args.guard_min_support,
                                grid_only=args.grid_only, debug=args.debug)
    print(f"\nSummary written to: {summary_path}\nImages saved to: {out_dir}")

if __name__=="__main__":
    main()
