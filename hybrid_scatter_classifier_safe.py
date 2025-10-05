"""
Hybrid Scatter Classifier (Safe Mode): FN-averse decision with conformal calibration

Key additions vs. the base version:
- Conformal-style calibration on a held-out set of POSITIVE samples:
  We compute the stage-1 score s for positives and set a negative-only
  threshold t_neg = quantile_alpha(s_pos). We will *only* output "negative"
  if s <= t_neg (and optionally ML proba <= p_neg_cap). With alpha=0 this
  means t_neg = min(s_pos) which eliminates FN on the calibration set.
- Optional ML backstop threshold: when ML is available, also require that
  p_ml <= p_neg_cap where p_neg_cap = quantile_alpha(p_pos) of the ML
  positive probabilities on the calibration set.

Effect:
- Strongly reduces/controls FN at the cost of higher FP. If you must avoid
  FN entirely, set alpha=0 and prefer "review"/"positive" otherwise.

This file embeds the base implementation and exposes a SafeHybridScatterClassifier
that wraps the decision rule with the FN guard.
"""
from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional
import math

# ---- Re-implement minimal base pieces to be self-contained ----
def median(x: np.ndarray) -> float:
    return float(np.median(x))

def mad(x: np.ndarray, c: float = 1.4826) -> float:
    m = median(x)
    return float(c * np.median(np.abs(x - m)) + 1e-12)

def iqr(x: np.ndarray) -> float:
    q1, q3 = np.percentile(x, [25, 75])
    return float(q3 - q1 + 1e-12)

def robust_scale(u: np.ndarray):
    m = median(u); s = mad(u)
    return (u - m) / s, m, s

def weights_by_quantile(x: np.ndarray, q_low=0.1, q_high=0.9) -> np.ndarray:
    lo = np.quantile(x, q_low); hi = np.quantile(x, q_high)
    x_min, x_max = np.min(x), np.max(x)
    w = np.ones_like(x, float)
    left = x < lo; right = x > hi
    w[left] = np.clip((x[left] - x_min) / max(lo - x_min, 1e-9), 0.0, 1.0)
    w[right] = np.clip((x_max - x[right]) / max(x_max - hi, 1e-9), 0.0, 1.0)
    return w

def running_median(y: np.ndarray, k: int = 3) -> np.ndarray:
    k = int(max(1, k));  k += (k % 2 == 0)
    pad = k // 2; ypad = np.pad(y, (pad, pad), mode='edge')
    out = np.empty_like(y, float)
    for i in range(len(y)):
        out[i] = np.median(ypad[i:i+k])
    return out

def bin_stats(x, y, n_bins=40, weights=None, min_count=8):
    x = np.asarray(x).ravel(); y = np.asarray(y).ravel()
    if weights is None: weights = np.ones_like(x, float)
    x_min, x_max = np.min(x), np.max(x)
    if x_min == x_max: x_max = x_min + 1.0
    edges = np.linspace(x_min, x_max, n_bins + 1)
    centers = 0.5*(edges[:-1] + edges[1:])
    y_med=[]; y_iqr=[]; w_bin=[]; cts=[]
    for i in range(n_bins):
        m = (x >= edges[i]) & (x < edges[i+1]) if i < n_bins-1 else (x >= edges[i]) & (x <= edges[i+1])
        if not np.any(m):
            y_med.append(np.nan); y_iqr.append(np.nan); w_bin.append(0.0); cts.append(0); continue
        yy = y[m]; ww = weights[m]
        y_med.append(np.median(yy)); q1, q3 = np.percentile(yy,[25,75]); y_iqr.append(q3-q1+1e-12)
        w_bin.append(float(np.mean(ww)*len(ww))); cts.append(int(np.sum(m)))
    centers=np.asarray(centers,float); y_med=np.asarray(y_med,float); y_iqr=np.asarray(y_iqr,float); w_bin=np.asarray(w_bin,float); cts=np.asarray(cts,int)
    good=(~np.isnan(y_med)) & (cts>=min_count)
    return centers[good], y_med[good], y_iqr[good], w_bin[good]

def huber_weights(r, delta):
    a=np.abs(r); w=np.ones_like(r,float); m=a>delta; w[m]=delta/(a[m]+1e-12); return w

def robust_wls(X, y, base_w=None, delta=1.0, n_iter=30, eps=1e-9):
    X=np.asarray(X,float); y=np.asarray(y,float).ravel(); n,d=X.shape
    if base_w is None: base_w=np.ones(n,float)
    base_w=base_w.ravel(); W=np.diag(base_w+eps)
    beta=np.linalg.lstsq(X.T@W@X+1e-6*np.eye(d), X.T@W@y, rcond=None)[0]
    for _ in range(n_iter):
        r=y-X@beta; w=huber_weights(r,delta); W=np.diag(base_w*w+eps)
        beta_new=np.linalg.lstsq(X.T@W@X+1e-6*np.eye(d), X.T@W@y, rcond=None)[0]
        if np.max(np.abs(beta_new-beta))<1e-6: beta=beta_new; break
        beta=beta_new
    r=y-X@beta
    loss=np.sum(base_w*(np.where(np.abs(r)<=delta,0.5*r**2, delta*(np.abs(r)-0.5*delta))))
    return beta, float(loss)

class ModelAFit:
    def __init__(self, beta, loss): self.beta=np.asarray(beta,float); self.loss=float(loss)
class ModelBFit:
    def __init__(self, beta, c, loss): self.beta=np.asarray(beta,float); self.c=float(c); self.loss=float(loss)

def fit_model_A(xb,yb,wb,delta):
    X=np.column_stack([np.ones_like(xb), xb])
    iqr_b=np.maximum((np.median(yb)-np.median(yb)) + np.std(yb), 1e-6)
    base_w=wb*(1.0/iqr_b)
    beta,loss=robust_wls(X,yb,base_w=base_w,delta=delta,n_iter=50)
    return ModelAFit(beta,loss)

def fit_model_B_hinge(xb,yb,wb,delta,c_grid=None):
    if c_grid is None:
        lo,hi=np.quantile(xb,[0.35,0.65]); c_grid=np.linspace(lo,hi,11)
    best=None; iqr_b=np.maximum((np.median(yb)-np.median(yb))+np.std(yb),1e-6); base_w=wb*(1.0/iqr_b)
    for c in c_grid:
        h1=np.maximum(c-xb,0.0); h2=np.maximum(xb-c,0.0)
        X=np.column_stack([np.ones_like(xb), h1,h2])
        beta,loss=robust_wls(X,yb,base_w=base_w,delta=delta,n_iter=60)
        if (best is None) or (loss<best.loss): best=ModelBFit(beta,c,loss)
    return best

def model_selection(x,y,n_bins=40,q_low=0.1,q_high=0.9,delta=1.2,penalty_lambda=0.1,smooth_k=3):
    xs,xm,xscl=robust_scale(np.asarray(x).ravel()); ys,ym,yscl=robust_scale(np.asarray(y).ravel())
    w=weights_by_quantile(xs,q_low=q_low,q_high=q_high)
    xb,y_med,y_iqr,w_bin = bin_stats(xs,ys,n_bins=n_bins,weights=w,min_count=6)
    if len(xb)<max(10,n_bins//5):
        xb=xs; y_med=ys; y_iqr=np.ones_like(ys)*np.std(ys); w_bin=w
    y_med_s=running_median(y_med,k=smooth_k)
    A=fit_model_A(xb,y_med_s,w_bin,delta=delta)
    B=fit_model_B_hinge(xb,y_med_s,w_bin,delta=delta)
    L_A=A.loss; L_B=B.loss*(1.0+penalty_lambda)
    s=(L_A-L_B)/max(L_A,1e-9)
    metrics=dict(L_A=L_A,L_B=L_B,score=s,xm=xm,xs=xscl,ym=ym,ys=yscl)
    inter=dict(xb=xb,yb=y_med_s,y_iqr=y_iqr,w_bin=w_bin,xs=xs,ys=ys,w=w,c_hat=B.c,betaA=A.beta,betaB=B.beta)
    return A,B,metrics,inter

def engineered_features(A,B,metrics,inter)->np.ndarray:
    xb=inter['xb']; yb=inter['yb']; y_iqr=inter['y_iqr']; w_bin=inter['w_bin']
    xs=inter['xs']; ys=inter['ys']
    s=metrics['score']; L_A=metrics['L_A']; L_B=metrics['L_B']
    a,b1,b2=B.beta; sL=-b1; sR=b2
    depth=np.median(yb)-np.min(yb); depth_ratio=depth/(mad(yb)+1e-9)
    X=np.column_stack([np.ones_like(xb), xb, xb**2])
    W=np.diag(w_bin); q=np.linalg.lstsq(X.T@W@X+1e-6*np.eye(3), X.T@W@yb, rcond=None)[0]
    curvature=float(q[2])
    corr=float(np.corrcoef(xs,ys)[0,1]) if len(xs)>2 else 0.0
    w_core=weights_by_quantile(xs,0.2,0.8); core=float(np.mean(w_core>0.99)); edge=1.0-core+1e-6; density_ratio=core/edge
    med_iqr=float(np.median(y_iqr)) if len(y_iqr) else float(np.std(ys))
    return np.array([s, math.log(L_A+1e-9), math.log(L_B+1e-9), sL, sR, depth_ratio, curvature, corr, density_ratio, med_iqr], float)

# Optional JAX logistic
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad
    _JAX_AVAILABLE=True
except Exception:
    _JAX_AVAILABLE=False
    jax=None; jnp=None

class JAXLogReg:
    def __init__(self, lr=0.05, l2=1e-3, steps=1500, seed=0):
        if not _JAX_AVAILABLE: raise ImportError("JAX not available")
        self.lr=lr; self.l2=l2; self.steps=steps; self.w=None; self.b=None
    @staticmethod
    @jit
    def _sigmoid(z): return 1.0/(1.0+jnp.exp(-z))
    def _loss(self, params, X, y):
        w,b=params; z=X@w+b; p=self._sigmoid(z)
        ce=-jnp.mean(y*jnp.log(p+1e-9)+(1-y)*jnp.log(1-p+1e-9))
        reg=0.5*self.l2*jnp.sum(w*w); return ce+reg
    def fit(self,X,y):
        X=jnp.array(X,dtype=jnp.float32); y=jnp.array(y,dtype=jnp.float32).reshape(-1,1)
        n,d=X.shape; self.w=jnp.zeros((d,1),dtype=jnp.float32); self.b=jnp.array(0.0,dtype=jnp.float32)
        params=(self.w,self.b); loss_grad=jit(grad(self._loss, argnums=0))
        for _ in range(self.steps):
            g_w,g_b=loss_grad(params,X,y); self.w=self.w-self.lr*g_w; self.b=self.b-self.lr*g_b; params=(self.w,self.b)
        return self
    def predict_proba(self,X):
        X=jnp.array(X,dtype=jnp.float32); z=X@self.w+self.b; p=self._sigmoid(z); return np.asarray(p).reshape(-1)

class SafeHybridScatterClassifier:
    def __init__(self, n_bins=40, q_low=0.1, q_high=0.9, huber_delta=1.2, penalty_lambda=0.1, base_s_thresh=0.15, jax_params=None):
        self.n_bins=n_bins; self.q_low=q_low; self.q_high=q_high
        self.huber_delta=huber_delta; self.penalty_lambda=penalty_lambda; self.base_s_thresh=base_s_thresh
        self._ml=None; self._feat_names=["score","log_LA","log_LB","slope_left","slope_right","depth_ratio","curvature","corr","density_ratio","median_iqr"]
        self._jax_params=jax_params or dict(lr=0.05, l2=1e-3, steps=1500, seed=0)
        self.t_neg=None; self.p_neg_cap=None; self.alpha=None

    def _first_stage(self,x,y):
        A,B,metrics,inter = model_selection(x,y,n_bins=self.n_bins,q_low=self.q_low,q_high=self.q_high,delta=self.huber_delta,penalty_lambda=self.penalty_lambda,smooth_k=3)
        s=metrics['score']
        stage1_label = 'positive' if s> self.base_s_thresh else ('negative' if s< -self.base_s_thresh else 'ambiguous')
        return A,B,metrics,inter,stage1_label
    def _featurize(self,A,B,metrics,inter): return engineered_features(A,B,metrics,inter)

    def fit_ml(self, samples: List[Tuple[np.ndarray,np.ndarray]], labels: List[int], use_only_ambiguous=True):
        feats=[]; ys=[]
        for (x,y),lab in zip(samples, labels):
            A,B,metrics,inter,stage = self._first_stage(x,y)
            if (not use_only_ambiguous) or stage=='ambiguous':
                feats.append(self._featurize(A,B,metrics,inter)); ys.append(int(lab))
        if len(feats)==0:
            for (x,y),lab in zip(samples,labels):
                A,B,metrics,inter,_=self._first_stage(x,y); feats.append(self._featurize(A,B,metrics,inter)); ys.append(int(lab))
        X=np.vstack(feats); y=np.array(ys,int)
        if _JAX_AVAILABLE:
            self._ml=JAXLogReg(**self._jax_params).fit(X,y.reshape(-1,1))
        else:
            self._ml=None
        return self

    def conformal_calibrate(self, calib_samples: List[Tuple[np.ndarray,np.ndarray]], calib_labels: List[int], alpha: float = 0.0):
        s_pos=[]; p_pos=[]
        for (x,y),lab in zip(calib_samples, calib_labels):
            if lab!=1: continue
            A,B,metrics,inter,_=self._first_stage(x,y); s_pos.append(metrics['score'])
            if self._ml is not None:
                f=self._featurize(A,B,metrics,inter).reshape(1,-1); p=float(self._ml.predict_proba(f)[0]); p_pos.append(p)
        if len(s_pos)==0: raise ValueError("Calibration set must include positives.")
        s_pos=np.array(s_pos,float); q=np.clip(alpha,0.0,1.0-1e-9)
        self.t_neg=float(np.quantile(s_pos,q))   # with alpha=0 => min(s_pos)
        if len(p_pos)>0: self.p_neg_cap=float(np.quantile(np.array(p_pos,float), q))
        else: self.p_neg_cap=None
        self.alpha=alpha
        return dict(t_neg=self.t_neg, p_neg_cap=self.p_neg_cap, alpha=self.alpha, n_calib_pos=len(s_pos))

    def predict(self, x: np.ndarray, y: np.ndarray, allow_review: bool = False) -> Dict[str,object]:
        A,B,metrics,inter,stage = self._first_stage(x,y)
        s=metrics['score']; c_hat=float(inter['c_hat'])
        t_neg = self.t_neg if self.t_neg is not None else -1.0
        p_ml=None
        if self._ml is not None:
            f=self._featurize(A,B,metrics,inter).reshape(1,-1); p_ml=float(self._ml.predict_proba(f)[0])
        neg_ok = (s <= t_neg) and ((self.p_neg_cap is None) or (p_ml is not None and p_ml <= self.p_neg_cap))
        if neg_ok:
            label='negative'; rule='NEG_safe'
        else:
            label='review' if (allow_review and stage=='ambiguous') else 'positive'; rule='FN_guard'
        return dict(label=label, stage1_label=stage, score=s, c_hat=c_hat, ml_proba=p_ml, rule=rule)

# ---- Simple synthetic for quick check ----
def synth_flat(n=800, noise_y=0.4, rect_height=2.0, edge_noise=0.8, seed=None):
    rng=np.random.default_rng(seed); x=np.sort(rng.uniform(-3,3,size=n))
    y=rng.uniform(-rect_height/2, rect_height/2, size=n); y+=rng.normal(0,noise_y,size=n)
    k=int(0.1*n); y[:k]+=rng.normal(0,edge_noise,size=k); y[-k:]+=rng.normal(0,edge_noise,size=k)
    return x,y
def synth_v(n=800, depth=2.0, noise_y=0.35, edge_noise=0.9, c=0.0, asym=0.0, seed=None):
    rng=np.random.default_rng(seed); x=np.sort(rng.uniform(-3,3,size=n))
    left_slope=depth*(1.0+max(0.0,-asym)); right_slope=depth*(1.0+max(0.0,asym))
    y=np.where(x<c, left_slope*(c-x), right_slope*(x-c)); y+=rng.normal(0,noise_y,size=n)
    k=int(0.1*n); y[:k]+=rng.normal(0,edge_noise,size=k); y[-k:]+=rng.normal(0,edge_noise,size=k)
    return x,y

def _quick_demo():
    # tiny train
    train=[]; ytrain=[]
    for i in range(40):
        if i%2==0: train.append(synth_flat(seed=i)); ytrain.append(0)
        else: train.append(synth_v(seed=i)); ytrain.append(1)
    clf=SafeHybridScatterClassifier()
    clf.fit_ml(train,ytrain,use_only_ambiguous=True)
    # calib positives
    calib=[]; ycal=[]
    for i in range(20):
        calib.append(synth_v(seed=100+i)); ycal.append(1)
    info=clf.conformal_calibrate(calib,ycal,alpha=0.0)
    print("Calibrated:", info)
    # test
    test=[]; ytest=[]
    for i in range(40):
        if i%2==0: test.append(synth_flat(seed=1000+i)); ytest.append(0)
        else: test.append(synth_v(seed=1000+i)); ytest.append(1)
    preds=[clf.predict(x,y) for (x,y) in test]
    yhat=[1 if p['label']=='positive' else 0 for p in preds]
    ytest=np.array(ytest); yhat=np.array(yhat)
    tp=int(np.sum((ytest==1)&(yhat==1))); fn=int(np.sum((ytest==1)&(yhat==0)))
    tn=int(np.sum((ytest==0)&(yhat==0))); fp=int(np.sum((ytest==0)&(yhat==1)))
    print(f"TP={tp} FP={fp} TN={tn} FN={fn}  Recall={tp/(tp+fn+1e-9):.3f}")

if __name__=="__main__":
    _quick_demo()
