
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
uvdip_detector_debug.py
-----------------------
Hybrid detector (coarse NumPy grid + optional JAX refine) for U/V-shaped dips in noisy scatter plots.
This version adds explicit debugging and control flags.

Install
  pip install -U "jax[cpu]" optax pandas matplotlib
(If JAX is missing, the script still works using the coarse grid only.)
"""

import argparse
from pathlib import Path
from typing import Tuple, Optional, Dict, Any, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- Optional JAX / Optax (for refinement) ----
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, value_and_grad
    import optax
    HAVE_JAX = True
except Exception:
    print("[WARN] JAX not available; running in coarse-grid mode only. Install with: pip install -U 'jax[cpu]' optax")
    HAVE_JAX = False

def robust_mad(x: np.ndarray) -> float:
    med = np.median(x); mad = np.median(np.abs(x - med))
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
            tc = tc or nums[0]; yc = yc or nums[1]
    if tc is None or yc is None:
        raise ValueError("Could not infer time/y columns. Pass --time-col and --y-col.")
    return tc, yc

def rectangular_y_distribution(y: np.ndarray, bins: int = 50) -> float:
    if len(y) < 10: return 0.0
    hist,_ = np.histogram(y, bins=bins)
    occ = (hist>0).mean()
    spread = np.percentile(y,97.5) - np.percentile(y,2.5)
    sigma = robust_mad(y)
    wide = np.clip(spread/(sigma+1e-12),0,20)/20.0
    return float(0.6*occ + 0.4*wide)

def soft_box_np(t: np.ndarray, c: float, wv: float, tau: float) -> np.ndarray:
    taus = max(tau, 1e-9*(t.max()-t.min()+1))
    z1=(t-(c-0.5*wv))/taus; z2=(t-(c+0.5*wv))/taus
    z1=np.clip(z1,-60,60); z2=np.clip(z2,-60,60)
    s1=1/(1+np.exp(-z1)); s2=1/(1+np.exp(-z2)); return s1-s2

def huber_np(r, dlt):
    ar=np.abs(r); quad=0.5*(ar**2); lin=dlt*(ar-0.5*dlt); return np.where(ar<=dlt,quad,lin)

def coarse_grid_search(t: np.ndarray, y: np.ndarray, centers: int=80, widths: int=25, depths: int=40) -> Dict[str,float]:
    sigma=robust_mad(y); delta=1.345*sigma
    n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w=0.25+0.75*w_edge
    a0=float(np.median(y)); loss_base=float(np.sum(huber_np((y-a0)*w, delta)))
    tspan=(t.max()-t.min()); tau=0.01*tspan
    C=np.linspace(t.min(), t.max(), centers); W=np.linspace(0.05*tspan, 0.70*tspan, widths)
    dmax=max(1e-6, a0-np.min(y)); D=np.linspace(0.0, dmax*2.0, depths)
    best={"loss":np.inf}
    for wv in W:
        for c in C:
            box=soft_box_np(t,c,wv,tau)
            yhat_all=a0 - np.outer(D, box)
            res=(y - yhat_all)*w
            L=np.sum(huber_np(res, delta), axis=1)
            idx=int(np.argmin(L)); Lmin=float(L[idx])
            if Lmin<best["loss"]:
                best={"loss":Lmin,"center":float(c),"width":float(wv),"depth":float(D[idx]),"a":a0}
    improvement=max(0.0,(loss_base-best["loss"]) / (loss_base+1e-12)); snr=best["depth"]/(sigma+1e-12)
    return {"a":best["a"],"center":best["center"],"width":best["width"],"depth":best["depth"],"improvement":float(improvement),"snr":float(snr),"loss_base":float(loss_base),"loss_model":float(best["loss"]),"tau":float(tau)}

if HAVE_JAX:
    def soft_box_jax(t, c, w, tau):
        left=c-0.5*w; right=c+0.5*w
        s1=jax.nn.sigmoid((t-left)/tau); s2=jax.nn.sigmoid((t-right)/tau)
        return jnp.clip(s1-s2,0.0,1.0)
    def huber_jax(residuals, delta):
        abs_r=jnp.abs(residuals); quad=0.5*(abs_r**2); lin=delta*(abs_r-0.5*delta); return jnp.where(abs_r<=delta,quad,lin)
    @jit
    def objective(params: Dict[str,jnp.ndarray], t:jnp.ndarray, y:jnp.ndarray, w_weights:jnp.ndarray,
                  tau:jnp.ndarray, delta:jnp.ndarray, w_min:jnp.ndarray, w_max:jnp.ndarray,
                  lam_width:jnp.ndarray, lam_amp:jnp.ndarray)->jnp.ndarray:
        tmin=t.min(); tmax=t.max()
        a=params["a"]; d=jax.nn.softplus(params["d_raw"])
        c=tmin+(tmax-tmin)*jax.nn.sigmoid(params["c_sig"])
        w=w_min+(w_max-w_min)*jax.nn.sigmoid(params["w_sig"])
        box=soft_box_jax(t,c,w,tau); yhat=a - d*box
        res=(y - yhat)*w_weights; loss=jnp.sum(huber_jax(res, delta))
        width_reg=lam_width*jnp.exp(-(w/(w_min+1e-6))); amp_reg=lam_amp*(d**2)
        return loss + width_reg + amp_reg
    def jax_refine(t: np.ndarray, y: np.ndarray, seed: int, coarse: Dict[str,float])->Dict[str,Any]:
        key=jax.random.PRNGKey(seed); t_j=jnp.asarray(t); y_j=jnp.asarray(y)
        sigma=float(robust_mad(y)); delta=np.float32(1.345*sigma)
        n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w_weights=jnp.asarray(0.25+0.75*w_edge)
        tmin=float(np.min(t)); tmax=float(np.max(t)); span=float(max(tmax-tmin,1e-9))
        tau=float(max(1e-9,0.01*(span+1e-12))); w_min=0.05*(span+1e-12); w_max=0.80*(span+1e-12)
        lam_width=1.0; lam_amp=1e-4; steps=800; lr=0.02
        def safe_logit(p: float, eps:float=1e-6)->float: p=np.clip(p,eps,1-eps); return float(np.log(p/(1-p)))
        def inv_softplus(d: float, eps:float=1e-12)->float: d=max(0.0,d); return float(np.log(np.expm1(d)+eps))
        def pack(a:float,d:float,c:float,w:float):
            frac_c=(c-tmin)/span; alpha=(w-w_min)/max(w_max-w_min,1e-9)
            c_sig=safe_logit(frac_c); w_sig=safe_logit(alpha); d_raw=inv_softplus(d)
            return {"a":np.float32(a),"d_raw":np.float32(d_raw),"c_sig":np.float32(c_sig),"w_sig":np.float32(w_sig)}
        a0=float(np.median(y)); inits:[Dict[str,float]]=[pack(coarse["a"],coarse["depth"],coarse["center"],coarse["width"])]
        for _ in range(5):
            key,k1,k2,k3,k4=jax.random.split(key,5)
            a_i=a0+0.1*sigma*float(jax.random.normal(k1))
            d_i=abs(0.5*sigma*float(jax.random.normal(k2)))+0.1*sigma
            c_i=float(tmin+span*jax.random.uniform(k3)); w_i=float(w_min+(w_max-w_min)*jax.random.uniform(k4))
            inits.append(pack(a_i,d_i,c_i,w_i))
        opt=optax.adam(lr); 
        obj=lambda p: objective(p, t_j, y_j, w_weights, jnp.asarray(tau), jnp.asarray(delta), jnp.asarray(w_min), jnp.asarray(w_max), jnp.asarray(lam_width), jnp.asarray(lam_amp))
        best_params=None; best_val=np.inf
        for init in inits:
            params={k:jnp.asarray(v) for k,v in init.items()}; opt_state=opt.init(params)
            @jit
            def step(p,s):
                val,grads=value_and_grad(obj)(p)
                updates,s=opt.update(grads,s,p); p=optax.apply_updates(p,updates); return p,s,val
            val=None
            for _ in range(steps): params,opt_state,val=step(params,opt_state)
            val_np=float(val)
            if val_np<best_val: best_val=val_np; best_params={k:float(v) for k,v in params.items()}
        a=best_params["a"]; d=float(np.log1p(np.exp(best_params["d_raw"])))
        c=tmin+span*(1/(1+np.exp(-best_params["c_sig"])))
        w=w_min+(w_max-w_min)*(1/(1+np.exp(-best_params["w_sig"])))
        sigma=robust_mad(y); snr=float(d/(sigma+1e-12))
        return {"a":float(a),"center":float(c),"width":float(w),"depth":float(d),"snr":snr,"improvement":coarse["improvement"],"refined":True}
else:
    def jax_refine(t: np.ndarray, y: np.ndarray, seed: int, coarse: Dict[str,float])->Dict[str,Any]:
        return {**coarse, "refined":False}

def uv_shape_checks(t: np.ndarray, y: np.ndarray, c: float, w: float,
                    rect_thr: float, z_drop_thr: float, slope_sigma_factor: float,
                    q3_near_baseline_sigmas: float, local_rect_occ_thr: float,
                    inside_mask: Optional[np.ndarray]=None)->Dict[str,Any]:
    rect_score=rectangular_y_distribution(y, bins=50); is_rect=rect_score>=rect_thr
    if inside_mask is None: inside_mask=(t>=(c-0.5*w)) & (t<=(c+0.5*w))
    L1=max(c-1.0*w,float(t.min())); L0=max(c-0.6*w,float(t.min()))
    R0=min(c+0.6*w,float(t.max())); R1=min(c+1.0*w,float(t.max()))
    left_sh=y[(t>=L1)&(t<L0)]; right_sh=y[(t>R0)&(t<=R1)]
    outside=~inside_mask
    shoulders=np.concatenate([left_sh,right_sh]) if (len(left_sh)+len(right_sh))>0 else y[outside]
    if len(shoulders)<6: shoulders=y[outside]
    base_level=np.median(shoulders); sigma_sh=robust_mad(shoulders)
    V0=c-0.25*w; V1=c+0.25*w; inside_V=(t>=V0)&(t<=V1); yV=y[inside_V]; yI=y[inside_mask]
    if len(yV)==0:
        return {"shape_ok":True,"is_rect":is_rect,"z_drop":0.0,"q3_ok":True,"lower_rect_like":False,"slope_ok":True,"rect_score":rect_score}
    y_bot=np.percentile(yV,10.0); drop=base_level - y_bot; z_drop=drop/(sigma_sh+1e-12)
    q3_in=np.percentile(yI,75.0) if len(yI) else np.percentile(yV,75.0)
    q3_ok=(q3_in >= base_level - q3_near_baseline_sigmas * sigma_sh)
    if len(yI)>=20:
        hist,_=np.histogram(yI,bins=30); occ_in=(hist>0).mean()
    else:
        occ_in=0.0
    lower_rect_like=(occ_in>=local_rect_occ_thr) and (not q3_ok) and (z_drop < (z_drop_thr+1.0))
    left_mask=(t>=c-0.5*w)&(t<c); right_mask=(t>c)&(t<=c+0.5*w); slope_ok=True
    if (np.sum(left_mask)>=5) and (np.sum(right_mask)>=5):
        def winsorize(a,p=2.5):
            lo,hi=np.percentile(a,[p,100-p]); return np.clip(a,lo,hi)
        tl,yl=winsorize(t[left_mask]),winsorize(y[left_mask])
        tr,yr=winsorize(t[right_mask]),winsorize(y[right_mask])
        sl=np.polyfit(tl,yl,1)[0]; sr=np.polyfit(tr,yr,1)[0]
        slope_min=slope_sigma_factor * (sigma_sh/(w+1e-12)); slope_ok=(sl<=-slope_min) and (sr>=slope_min)
    shape_ok=(z_drop>=z_drop_thr) and (q3_ok or slope_ok or (z_drop>=(z_drop_thr+1.0))) and (not lower_rect_like)
    return {"shape_ok":bool(shape_ok),"is_rect":bool(is_rect),"z_drop":float(z_drop),"q3_ok":bool(q3_ok),"lower_rect_like":bool(lower_rect_like),"slope_ok":bool(slope_ok),"rect_score":float(rect_score)}

def detect_on_xy(t: np.ndarray, y: np.ndarray,
                 prefer_sensitivity: bool=True, imp_override: float=0.08,
                 base_snr: float=0.8, base_imp: float=0.02,
                 grid_centers:int=80, grid_widths:int=25, grid_depths:int=40, seed:int=0,
                 z_drop_thr: float=1.8, slope_sigma_factor: float=0.8, q3_near_baseline_sigmas: float=1.0,
                 local_rect_occ_thr: float=0.92, rect_thr: float=0.80,
                 grid_only: bool=False, debug: bool=False)->Dict[str,Any]:
    mask=np.isfinite(t)&np.isfinite(y); t=t[mask]; y=y[mask]; o=np.argsort(t); t=t[o]; y=y[o]
    coarse=coarse_grid_search(t,y,centers=grid_centers,widths=grid_widths,depths=grid_depths)
    inside_c=(t>=(coarse["center"]-0.5*coarse["width"])) & (t<=(coarse["center"]+0.5*coarse["width"]))
    support_c=int(np.sum(inside_c)); min_points=max(4,int(0.05*len(t)))
    shape_c=uv_shape_checks(t,y,coarse["center"],coarse["width"],rect_thr,z_drop_thr,slope_sigma_factor,q3_near_baseline_sigmas,local_rect_occ_thr,inside_c)
    rule_or_c = prefer_sensitivity and (coarse["improvement"]>=imp_override) and (support_c>=min_points)
    if debug:
        print(f"[COARSE] imp={coarse['improvement']:.3f} snr={coarse['snr']:.3f} support={support_c} z={shape_c['z_drop']:.2f} q3={shape_c['q3_ok']} slope={shape_c['slope_ok']} lower_rect={shape_c['lower_rect_like']} rect_score={shape_c['rect_score']:.2f}")
    if rule_or_c and shape_c["shape_ok"] and (not shape_c["is_rect"]):
        return {"has_dip":True,"reason":None,"a":coarse["a"],"depth":coarse["depth"],"center":coarse["center"],"width":coarse["width"],"snr":coarse["snr"],"improvement":coarse["improvement"],"rect_score":shape_c["rect_score"],"support_points":support_c,"source":"coarse","z_drop":shape_c["z_drop"]}
    if grid_only or (not HAVE_JAX):
        refined=coarse | {"refined":False}
    else:
        refined=jax_refine(t,y,seed=seed,coarse=coarse)
    a0=float(np.median(y)); sigma=robust_mad(y); delta=1.345*sigma
    n=len(t); edge=np.linspace(0,1,n); w_edge=(1-np.exp(-5*np.minimum(edge,1-edge))); w=0.25+0.75*w_edge
    tau=0.01*(t.max()-t.min()); box=soft_box_np(t,refined["center"],refined["width"],tau)
    yhat=refined["a"] - refined["depth"]*box
    loss_base=float(np.sum(huber_np((y-a0)*w, delta))); loss_model=float(np.sum(huber_np((y-yhat)*w, delta)))
    refined["improvement"]=max(0.0,(loss_base-loss_model)/(loss_base+1e-12)); refined["snr"]=float(refined["depth"]/(sigma+1e-12))
    inside=(t>=(refined["center"]-0.5*refined["width"])) & (t<=(refined["center"]+0.5*refined["width"])); support_r=int(np.sum(inside))
    shape_r=uv_shape_checks(t,y,refined["center"],refined["width"],rect_thr,z_drop_thr,slope_sigma_factor,q3_near_baseline_sigmas,local_rect_occ_thr,inside)
    snr_thr=1.1 if (not prefer_sensitivity) else base_snr; imp_thr=0.05 if (not prefer_sensitivity) else base_imp
    if prefer_sensitivity: snr_thr=float(np.interp(refined["improvement"],[0.00,0.08,0.20],[snr_thr,0.55,0.35]))
    rule_and=(refined["snr"]>=snr_thr) and (refined["improvement"]>=imp_thr) and (support_r>=min_points)
    rule_or=prefer_sensitivity and (refined["improvement"]>=imp_override) and (support_r>=min_points)
    has_dip=(not shape_r["is_rect"]) and shape_r["shape_ok"] and (rule_and or rule_or)
    reason=None
    if shape_r["is_rect"]: reason=f"Global vertical distribution looks rectangular (score={shape_r['rect_score']:.2f})."
    elif not shape_r["shape_ok"]: reason=f"Rejected by U/V shape (z={shape_r['z_drop']:.2f}, q3={shape_r['q3_ok']}, slope={shape_r['slope_ok']}, lower_rect={shape_r['lower_rect_like']})."
    elif not has_dip: reason="Insufficient SNR/improvement/support."
    if debug:
        print(f"[REFINE={not grid_only and HAVE_JAX}] imp={refined['improvement']:.3f} snr={refined['snr']:.3f} support={support_r} z={shape_r['z_drop']:.2f} q3={shape_r['q3_ok']} slope={shape_r['slope_ok']} lower_rect={shape_r['lower_rect_like']} rect_score={shape_r['rect_score']:.2f} AND={rule_and} OR={rule_or} has_dip={has_dip}")
    return {"has_dip":bool(has_dip),"reason":reason,"a":refined["a"],"depth":refined["depth"],"center":refined["center"],"width":refined["width"],"snr":refined["snr"],"improvement":refined["improvement"],"rect_score":shape_r["rect_score"],"support_points":support_r,"source":"refined" if (HAVE_JAX and (not grid_only)) else "coarse","z_drop":shape_r["z_drop"]}

def plot_and_save(t: np.ndarray, y: np.ndarray, res: Dict[str,Any], out_png: Path, title: str="")->None:
    plt.figure(figsize=(9,5.2)); plt.scatter(t,y,s=18); plt.xlabel("Time"); plt.ylabel("Y")
    if title: plt.title(title)
    plt.axhline(res["a"], linestyle="--", linewidth=1.0, label="Baseline a")
    if res["has_dip"] and np.isfinite(res["center"]):
        s=res["center"]-0.5*res["width"]; e=res["center"]+0.5*res["width"]
        plt.axvspan(s,e,alpha=0.2,label=f"Detected interval ({res.get('source','')})")
        plt.text(s,res["a"],f"depth≈{res['depth']:.4g}, SNR≈{res['snr']:.2f}, imp≈{100*res['improvement']:.1f}%",va="bottom",ha="left")
    else:
        if res.get("reason"): xmin,xmax=np.min(t),np.max(t); ymax=np.max(y); plt.text(xmin,ymax,res["reason"],va="top",ha="left")
    plt.legend(frameon=False, loc="best"); out_png.parent.mkdir(parents=True, exist_ok=True); plt.tight_layout(); plt.savefig(out_png,dpi=150); plt.close()

def load_csv(path: Path, time_col: Optional[str], y_col: Optional[str])->Tuple[np.ndarray,np.ndarray,str,str]:
    df=pd.read_csv(path); tc,yc=find_columns(df,time_col,y_col)
    t=df[tc].to_numpy(float); y=df[yc].to_numpy(float); return t,y,tc,yc

def process_folder(root: Path, out_dir: Path, time_col: Optional[str], y_col: Optional[str],
                   prefer_sensitivity: bool=True, imp_override: float=0.08, base_snr: float=0.8, base_imp: float=0.02,
                   grid_centers:int=80, grid_widths:int=25, grid_depths:int=40, seed:int=0,
                   z_drop_thr: float=1.8, slope_sigma_factor: float=0.8, q3_near_baseline_sigmas: float=1.0,
                   local_rect_occ_thr: float=0.92, rect_thr: float=0.80,
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
                             local_rect_occ_thr=local_rect_occ_thr,rect_thr=rect_thr,grid_only=grid_only,debug=debug)
            png_path=out_dir / p.with_suffix(".png").name
            plot_and_save(t,y,res,png_path,title=f"{p.name} [has_dip={res['has_dip']}]" )
            rows.append({"file":str(p),"has_dip":res["has_dip"],"baseline_a":res["a"],"center":res["center"],"width":res["width"],"depth":res["depth"],"snr":res["snr"],"improvement":res["improvement"],"support_points":res["support_points"],"rect_score":res["rect_score"],"z_drop":res.get("z_drop",0.0),"note":res.get("reason",""),"image":str(png_path),"time_col":tc,"y_col":yc,"source":res.get("source","")})
            if debug:
                print(f"[OK] {p.name} -> has_dip={res['has_dip']} c={res['center']:.6g} w={res['width']:.6g} d={res['depth']:.4g} SNR={res['snr']:.2f} imp={100*res['improvement']:.1f}% support={res['support_points']} rect={res['rect_score']:.2f} z={res.get('z_drop',0):.2f} src={res.get('source','')}")
        except Exception as e:
            print(f"[ERROR] {p}: {e}")
    out_dir.mkdir(parents=True, exist_ok=True); summary_path=out_dir/"dip_summary.csv"; pd.DataFrame(rows).to_csv(summary_path,index=False); return summary_path

def main():
    ap=argparse.ArgumentParser(description="Detect U/V-shaped dips in scatter charts (grid + optional JAX refine)." )
    ap.add_argument("folder", type=str, help="Root folder to search recursively for CSV files.")
    ap.add_argument("--time-col", type=str, default=None, help="Name of time/x column (optional)." )
    ap.add_argument("--y-col", type=str, default=None, help="Name of y/flux column (optional)." )
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory for images and summary CSV." )
    ap.add_argument("--strict", action="store_true", help="Be stricter (fewer FPs, maybe more FNs)." )
    ap.add_argument("--imp-override", type=float, default=0.08, help="Improvement threshold for FN-minimizing OR rule.")
    ap.add_argument("--snr", type=float, default=0.8, help="Base SNR threshold for AND rule (sensitivity mode)." )
    ap.add_argument("--imp", type=float, default=0.02, help="Base improvement threshold for AND rule (sensitivity mode)." )
    ap.add_argument("--grid-centers", type=int, default=80, help="Coarse grid: number of center samples." )
    ap.add_argument("--grid-widths", type=int, default=25, help="Coarse grid: number of width samples." )
    ap.add_argument("--grid-depths", type=int, default=40, help="Coarse grid: number of depth samples." )
    ap.add_argument("--seed", type=int, default=0, help="Random seed for JAX refinement.")
    ap.add_argument("--z-drop", type=float, default=1.8, dest="z_drop", help="Vertex drop z-score threshold vs shoulders." )
    ap.add_argument("--slope-sigma", type=float, default=0.8, dest="slope_sigma", help="Slope threshold factor (sigma_shoulder / w)." )
    ap.add_argument("--q3-sigmas", type=float, default=1.0, dest="q3_sigmas", help="Q3 inside must be within this many sigmas of baseline." )
    ap.add_argument("--local-rect-occ", type=float, default=0.92, dest="local_rect_occ", help="Inside-interval vertical occupancy threshold to reject lower-rectangle." )
    ap.add_argument("--rect-thr", type=float, default=0.80, dest="rect_thr", help="Global rectangularness score threshold to treat as y≈a." )
    ap.add_argument("--grid-only", action="store_true", help="Disable JAX refine; decide on coarse fit only." )
    ap.add_argument("--debug", action="store_true", help="Print detailed diagnostics per file." )
    args=ap.parse_args()
    root=Path(args.folder).expanduser().resolve(); out_dir=Path(args.out_dir).expanduser().resolve() if args.out_dir else (root/"_dip_outputs")
    summary_path=process_folder(root,out_dir,args.time_col,args.y_col, prefer_sensitivity=(not args.strict),
                                imp_override=args.imp_override, base_snr=args.snr, base_imp=args.imp,
                                grid_centers=args.grid_centers, grid_widths=args.grid_widths, grid_depths=args.grid_depths,
                                seed=args.seed, z_drop_thr=args.z_drop, slope_sigma_factor=args.slope_sigma,
                                q3_near_baseline_sigmas=args.q3_sigmas, local_rect_occ_thr=args.local_rect_occ,
                                rect_thr=args.rect_thr, grid_only=args.grid_only, debug=args.debug)
    print(f"\nSummary written to: {summary_path}\nImages saved to: {out_dir}")

if __name__=="__main__": main()
