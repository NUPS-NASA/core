
from __future__ import annotations

# --- Standard library ---
import os
import glob
import math
import warnings
import argparse
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

# --- Third-party ---
import numpy as np
import pandas as pd

from astropy.io import fits
from astropy.time import Time
from astropy.stats import sigma_clipped_stats

from photutils.detection import DAOStarFinder
from photutils.aperture import CircularAperture, CircularAnnulus

import astroalign as aa
from tqdm import tqdm

# Machine learning / GP imports moved to top as requested
import jax
import jax.numpy as jnp
from jax import vmap
import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import tinygp
from tinygp import kernels

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

@dataclass
class FilePrefixes:
    # Default prefixes per your request (single folder with different prefixes)
    light: str = "object"
    bias: str = "bias"
    dark: str = "dark"
    flat: str = "flat"

@dataclass
class InstrumentConfig:
    gain_e_per_adu: Optional[float] = None  # electrons / ADU; add Poisson term if given
    # read_noise_e: Optional[float] = None  # If you also want to model read noise, add here


@dataclass
class PhotometryConfig:
    fwhm_pix: float = 3.5
    thresh_sigma: float = 5.0
    max_stars_detect: int = 2000
    edge_margin: int = 12
    r_ap: Optional[float] = None  # if None → 3×FWHM
    r_in: Optional[float] = None  # if None → 6×FWHM
    r_out: Optional[float] = None # if None → 10×FWHM

    def radii(self) -> Tuple[float, float, float]:
        rap = self.r_ap if self.r_ap is not None else 3.0 * self.fwhm_pix
        rin = self.r_in if self.r_in is not None else 6.0 * self.fwhm_pix
        rout = self.r_out if self.r_out is not None else 10.0 * self.fwhm_pix
        return rap, rin, rout

@dataclass
class NormalizationConfig:
    enabled: bool = True
    # 1) Fast MCMC defaults
    samples: int = 300
    warmup: int = 300
    chains: int = 1
    mean_const: bool = True
    center_flux: bool = True
    seed: int = 42
    # If None → auto-suggest per star
    transit_duration_hours: Optional[float] = None
    rho_mult: Optional[float] = None
    unit: str = "days"  # 'days', 'hours', 'minutes', 'seconds'
    output_csv_name: str = "allstars_flux_norm_wide.csv"
    return_residual: bool = True  # True → flux - trend (≈ zero-mean); False → trend-only

    # 2) Mode & MAP options
    mode: str = "mcmc"          # "mcmc" | "map"
    map_max_iter: int = 300
    map_lr: float = 0.02

    # 3) Time downsampling (applied to training set; predictions on full times in MAP mode)
    downsample_frac: float = 1.0  # (0,1] e.g. 0.5 keeps ~50% of points

    # Per-star CSV options
    per_star_csv: bool = False                 # write one CSV per star
    per_star_dir: str = "norm_per_star"       # subdirectory under output_dir
    per_star_prefix: str = "star"             # filename prefix
    per_star_digits: int = 4                   # zero padding width
    per_star_include_gpmean: bool = False      # also dump GP trend if available
    per_star_include_error: bool = True        # include error column in per-star CSV


@dataclass
class Covariates:
    airmass: np.ndarray
    fwhm_px: np.ndarray
    sky_med: np.ndarray
    times_jd: np.ndarray

@dataclass
class RunConfig:
    # I/O
    root_dir: str
    output_dir: str = "./output"
    prefixes: FilePrefixes = field(default_factory=FilePrefixes)

    # Which calibrations to apply
    use_bias: bool = True
    use_dark: bool = True
    use_flat: bool = True

    # Alignment
    do_alignment: bool = True
    save_aligned_fits: bool = False

    # Output options
    csv_wide_path: str = field(default_factory=lambda: "allstars_flux_err_wide.csv")

    # Submodules
    phot: PhotometryConfig = field(default_factory=PhotometryConfig)
    inst: InstrumentConfig = field(default_factory=InstrumentConfig)
    norm: NormalizationConfig = field(default_factory=NormalizationConfig)

    # Misc
    random_seed: int = 42

    def prepare(self) -> None:
        os.makedirs(self.output_dir, exist_ok=True)
        if self.save_aligned_fits:
            os.makedirs(self.aligned_dir, exist_ok=True)

    @property
    def aligned_dir(self) -> str:
        return os.path.join(self.output_dir, "aligned_fits")

# -----------------------------------------------------------------------------
# FITS helpers
# -----------------------------------------------------------------------------

def list_prefixed_fits(dirpath: str, prefix: str) -> List[str]:
    pats = [f"{prefix}*.fits", f"{prefix}*.fit"]
    files: List[str] = []
    for p in pats:
        files.extend(glob.glob(os.path.join(dirpath, p)))
    return sorted(files)


def load_fits(path: str) -> Tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = hdul[0].data.astype(float)
        hdr = hdul[0].header
    return data, hdr


def read_time_from_header(hdr: fits.Header) -> float:
    # Priority: JD/BJD/HJD → MJD → DATE-OBS
    for key in ("JD", "BJD", "HJD"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    if "MJD" in hdr:
        try:
            v = float(hdr["MJD"])  # MJD
            if np.isfinite(v):
                return v + 2400000.5
        except Exception:
            pass
    if "DATE-OBS" in hdr:
        for fmt in ("isot", None):
            try:
                if fmt == "isot":
                    return Time(hdr["DATE-OBS"], format="isot", scale="utc").jd
                else:
                    return Time(hdr["DATE-OBS"], scale="utc").jd
            except Exception:
                continue
    return np.nan


def extract_exptime(hdr: fits.Header) -> Optional[float]:
    for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
        if key in hdr:
            try:
                v = float(hdr[key])
                if np.isfinite(v):
                    return v
            except Exception:
                pass
    return None

# -----------------------------------------------------------------------------
# Master calibrations
# -----------------------------------------------------------------------------

def median_combine(files: Sequence[str]) -> Tuple[Optional[np.ndarray], Optional[fits.Header]]:
    if not files:
        return None, None
    stack = []
    hdr0 = None
    for p in files:
        dat, hdr = load_fits(p)
        if hdr0 is None:
            hdr0 = hdr
        stack.append(dat.astype(float))
    master = np.nanmedian(np.stack(stack, axis=0), axis=0)
    return master, hdr0


def build_master_bias(files: Sequence[str]) -> Optional[np.ndarray]:
    if not files:
        return None
    mbias, _ = median_combine(files)
    return mbias


def build_master_dark_by_exptime(files: Sequence[str]) -> Dict[float, np.ndarray]:
    if not files:
        return {}
    by_exp: Dict[float, List[str]] = {}
    for p in files:
        _, hdr = load_fits(p)
        expt = extract_exptime(hdr)
        if expt is None:
            continue
        by_exp.setdefault(expt, []).append(p)

    out: Dict[float, np.ndarray] = {}
    for expt, flist in by_exp.items():
        mdark, _ = median_combine(flist)
        if mdark is not None:
            out[expt] = mdark
    return out


def build_master_flat(files: Sequence[str], master_bias: Optional[np.ndarray], dark_dict: Dict[float, np.ndarray]) -> Optional[np.ndarray]:
    if not files:
        return None
    cal_stack = []
    for p in files:
        dat, hdr = load_fits(p)
        if master_bias is not None:
            dat = dat - master_bias
        if dark_dict:
            expt = extract_exptime(hdr)
            if expt is not None and len(dark_dict) > 0:
                nearest = min(dark_dict.keys(), key=lambda k: abs(k - expt))
                scale = expt / nearest if nearest else 1.0
                dat = dat - dark_dict[nearest] * scale
        cal_stack.append(dat)
    mflat = np.nanmedian(np.stack(cal_stack, axis=0), axis=0)
    # Normalize by median
    finite = np.isfinite(mflat)
    med = np.nanmedian(mflat[finite]) if np.any(finite) else np.nan
    if np.isfinite(med) and med != 0:
        mflat = mflat / med
    return mflat


def calibrate_frame(data: np.ndarray, hdr: fits.Header,
                    master_bias: Optional[np.ndarray],
                    dark_dict: Dict[float, np.ndarray],
                    flat_norm: Optional[np.ndarray]) -> np.ndarray:
    out = data.astype(float).copy()
    if master_bias is not None:
        out = out - master_bias
    if dark_dict:
        expt = extract_exptime(hdr)
        if expt is not None and len(dark_dict) > 0:
            nearest = min(dark_dict.keys(), key=lambda k: abs(k - expt))
            scale = expt / nearest if nearest else 1.0
            out = out - dark_dict[nearest] * scale
    if flat_norm is not None:
        with np.errstate(divide="ignore", invalid="ignore"):
            out = out / flat_norm
    return out

# -----------------------------------------------------------------------------
# Alignment & detection
# -----------------------------------------------------------------------------

def align_to_reference(src_img: np.ndarray, ref_img: np.ndarray) -> np.ndarray:
    aligned, _ = aa.register(src_img, ref_img, detection_sigma=3.0, max_control_points=50)
    return aligned.astype(float)


def detect_stars(ref_img: np.ndarray, pcfg: PhotometryConfig) -> np.ndarray:
    mean, med, std = sigma_clipped_stats(ref_img, sigma=3.0, maxiters=5)
    dao = DAOStarFinder(fwhm=pcfg.fwhm_pix, threshold=pcfg.thresh_sigma * std)
    tbl = dao(ref_img - med)
    if tbl is None or len(tbl) == 0:
        raise RuntimeError("No stars detected. Adjust FWHM/thresh.")
    tbl.sort("flux"); tbl = tbl[::-1]
    if len(tbl) > pcfg.max_stars_detect:
        tbl = tbl[:pcfg.max_stars_detect]
    xyf = np.vstack([tbl["xcentroid"].data, tbl["ycentroid"].data, tbl["flux"].data]).T
    H, W = ref_img.shape
    m = (xyf[:, 0] > pcfg.edge_margin) & (xyf[:, 0] < W - pcfg.edge_margin) \
        & (xyf[:, 1] > pcfg.edge_margin) & (xyf[:, 1] < H - pcfg.edge_margin)
    return xyf[m, :2]

# -----------------------------------------------------------------------------
# Photometry with per-aperture uncertainty
# -----------------------------------------------------------------------------

def measure_aperture_photometry_with_error(img: np.ndarray, xy: np.ndarray,
                                            pcfg: PhotometryConfig,
                                            inst: InstrumentConfig) -> Tuple[np.ndarray, np.ndarray]:
    rap, rin, rout = pcfg.radii()
    apert = CircularAperture(xy, r=rap)
    ann = CircularAnnulus(xy, r_in=rin, r_out=rout)

    ap_masks = apert.to_mask(method="exact")
    ann_masks = ann.to_mask(method="exact")

    fluxes: List[float] = []
    sigmas: List[float] = []

    for m_ap, m_an in zip(ap_masks, ann_masks):
        # Background annulus stats (robust)
        ann_data = m_an.multiply(img)
        ann_mask = m_an.data > 0
        ann_valid = ann_mask & np.isfinite(ann_data)
        if not np.any(ann_valid):
            fluxes.append(np.nan)
            sigmas.append(np.nan)
            continue
        bkg_vals = ann_data[ann_valid]
        bkg_med = np.nanmedian(bkg_vals)
        bkg_std = np.nanstd(bkg_vals)  # robust enough for large annulus
        N_bkg = float(np.sum(ann_valid))

        # Aperture sum
        ap_data = m_ap.multiply(img)
        ap_mask = m_ap.data > 0
        ap_valid = ap_mask & np.isfinite(ap_data)
        if not np.any(ap_valid):
            fluxes.append(np.nan)
            sigmas.append(np.nan)
            continue

        ap_vals = ap_data[ap_valid]
        N_ap = float(np.sum(ap_valid))
        sum_ap = float(np.nansum(ap_vals))
        flux = sum_ap - bkg_med * N_ap  # background-subtracted total (ADU)
        fluxes.append(flux)

        # Uncertainty (see notes above)
        var_sky = N_ap * (bkg_std ** 2)
        var_bkg_model = (N_ap ** 2 / max(N_bkg, 1)) * (bkg_std ** 2)

        if inst.gain_e_per_adu and np.isfinite(inst.gain_e_per_adu) and inst.gain_e_per_adu > 0:
            g = float(inst.gain_e_per_adu)
            signal_e = max(flux, 0.0) * g  # electrons
            var_shot_e = signal_e  # Poisson
            # Convert background terms to electrons
            var_bg_e = (var_sky + var_bkg_model) * (g ** 2)
            sigma_e = math.sqrt(var_bg_e + var_shot_e)
            sigma_adu = sigma_e / g
            sigmas.append(sigma_adu)
        else:
            # No gain: approximate error from background terms only (ADU)
            sigma_adu = math.sqrt(var_sky + var_bkg_model)
            sigmas.append(sigma_adu)

    return np.asarray(fluxes, float), np.asarray(sigmas, float)

# -----------------------------------------------------------------------------
# End-to-end pipeline
# -----------------------------------------------------------------------------

def discover_files(cfg: RunConfig) -> Tuple[List[str], List[str], List[str], List[str]]:
    L = list_prefixed_fits(cfg.root_dir, cfg.prefixes.light)
    B = list_prefixed_fits(cfg.root_dir, cfg.prefixes.bias)
    D = list_prefixed_fits(cfg.root_dir, cfg.prefixes.dark)
    F = list_prefixed_fits(cfg.root_dir, cfg.prefixes.flat)
    if not L:
        raise FileNotFoundError(f"No light frames found with prefix '{cfg.prefixes.light}' in {cfg.root_dir}")
    return L, B, D, F


def build_masters(cfg: RunConfig, bias_files: List[str], dark_files: List[str], flat_files: List[str]) -> Tuple[Optional[np.ndarray], Dict[float, np.ndarray], Optional[np.ndarray]]:
    mbias = build_master_bias(bias_files) if (cfg.use_bias and bias_files) else None
    dark_dict = build_master_dark_by_exptime(dark_files) if (cfg.use_dark and dark_files) else {}
    mflat = build_master_flat(flat_files, mbias, dark_dict) if (cfg.use_flat and flat_files) else None
    return mbias, dark_dict, mflat


def process_all_frames(cfg: RunConfig) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    List[str],
    Optional[np.ndarray],
    Dict[float, np.ndarray],
    Optional[np.ndarray],
]:
    """Run calibration, alignment, detection, and photometry for all frames.

    Returns
    -------
    times_jd : (N_frames,) float
    flux_mat : (N_frames, N_stars) float (ADU)
    err_mat  : (N_frames, N_stars) float (ADU)
    xy       : (N_stars, 2) float, positions in reference frame (x, y)
    light_files : list of processed light-frame paths (sorted)
    master_bias : calibrated master bias (or None)
    master_dark_dict : dict mapping exposure time → master dark (possibly empty)
    master_flat : normalized master flat (or None)
    """
    light_files, bias_files, dark_files, flat_files = discover_files(cfg)
    mbias, dark_dict, mflat = build_masters(cfg, bias_files, dark_files, flat_files)

    # Reference: first light frame (calibrated)
    ref_raw, ref_hdr = load_fits(light_files[0])
    ref_img = calibrate_frame(ref_raw, ref_hdr, mbias, dark_dict, mflat)

    # Detect sources on reference
    xy = detect_stars(ref_img, cfg.phot)

    # Iterate over frames
    times: List[float] = []
    rows_flux: List[np.ndarray] = []
    rows_err: List[np.ndarray] = []

    for i, path in enumerate(tqdm(light_files, desc="Frames", unit="frm")):
        data, hdr = load_fits(path)
        cal = calibrate_frame(data, hdr, mbias, dark_dict, mflat)
        img = align_to_reference(cal, ref_img) if (cfg.do_alignment and i > 0) else cal

        t_jd = read_time_from_header(hdr)
        times.append(float(t_jd) if np.isfinite(t_jd) else np.nan)

        fluxes, sigmas = measure_aperture_photometry_with_error(img, xy, cfg.phot, cfg.inst)
        rows_flux.append(fluxes)
        rows_err.append(sigmas)

        if cfg.save_aligned_fits:
            h = hdr.copy(); h["HISTORY"] = "calibrated & aligned"
            fits.writeto(os.path.join(cfg.aligned_dir, os.path.basename(path)),
                         img.astype(np.float32), h, overwrite=True)

    times_arr = np.asarray(times, float)
    if np.any(~np.isfinite(times_arr)):
        # Fall back to frame index if no time in headers
        times_arr = np.arange(len(times_arr), dtype=float)

    flux_mat = np.vstack(rows_flux)
    err_mat = np.vstack(rows_err)
    return times_arr, flux_mat, err_mat, xy, light_files, mbias, dark_dict, mflat

# -----------------------------------------------------------------------------
# Normalization (GP-based) — optional
# -----------------------------------------------------------------------------

# We import heavy ML deps lazily so the pipeline still runs when normalization is disabled.

def _to_days(t, unit="days"):
    factors = dict(days=1.0, hours=1/24.0, minutes=1/(24*60.0), seconds=1/(24*3600.0))
    if unit not in factors:
        raise ValueError(f"Unsupported unit: {unit}")
    return np.asarray(t, dtype=float) * factors[unit]


def _build_gp(t, yerr, log_sigma, log_rho, mu=None):
    """Matern-3/2 GP; observational errors in the diagonal."""
    sigma = jnp.exp(log_sigma)          # process std
    rho   = jnp.exp(log_rho)            # time scale (in days)
    k = sigma**2 * kernels.stationary.Matern32(scale=rho)
    mean = 0.0 if mu is None else mu    # constant mean option
    return tinygp.GaussianProcess(k, t, diag=yerr**2, mean=mean)



def _model(t, y=None, yerr=None, use_const_mean=True,
           transit_duration_hours=2.0, rho_mult=5.0):
    # Guard time scale range using estimated transit duration
    dur_days = transit_duration_hours / 24.0
    rho_min = jnp.maximum(rho_mult * dur_days, 1e-3)
    rho_max = jnp.maximum(t.max() - t.min(), rho_min + 1e-3)

    log_sigma = numpyro.sample("log_sigma", dist.Uniform(jnp.log(1e-6), jnp.log(1.0)))
    log_rho   = numpyro.sample("log_rho",   dist.Uniform(jnp.log(rho_min), jnp.log(rho_max)))
    mu = numpyro.sample("mu", dist.Normal(0., 1.)) if use_const_mean else None

    gp = _build_gp(t, yerr, log_sigma, log_rho, mu=mu)
    numpyro.factor("gp_loglike", gp.log_probability(y))



def _posterior_noise_mean_via_predict(t, y, yerr, samples, use_const_mean=True):
    """Posterior mean at training points using GP.predict; returns shape (N,)."""
    def one_predict(ls, lr, mu):
        gp = _build_gp(t, yerr, ls, lr, mu if use_const_mean else None)
        return gp.predict(y, t)

    log_sigma_s = samples["log_sigma"]
    log_rho_s   = samples["log_rho"]
    mu_s        = samples.get("mu", jnp.zeros_like(log_sigma_s))

    mean_stack = vmap(one_predict)(log_sigma_s, log_rho_s, mu_s)
    return jnp.mean(mean_stack, axis=0)



def _run_mcmc(t, y, yerr, use_const_mean, num_warmup, num_samples, chains, seed,
              transit_duration_hours=2.0, rho_mult=5.0):
    nuts = NUTS(_model, target_accept_prob=0.9)
    mcmc = MCMC(nuts, num_warmup=num_warmup, num_samples=num_samples,
                num_chains=chains, progress_bar=False)
    mcmc.run(
        jax.random.key(seed),
        t=t, y=y, yerr=yerr, use_const_mean=use_const_mean,
        transit_duration_hours=transit_duration_hours, rho_mult=rho_mult,
    )
    return mcmc

# ----- MAP utilities (JIT-accelerated, with static argnums via closure) -----

def _make_nll(use_const_mean: bool):
    def nll(params, t, y, yerr):
        gp = _build_gp(
            t, yerr,
            params["log_sigma"], params["log_rho"],
            params["mu"] if use_const_mean else None,
        )
        return -gp.log_probability(y)
    # JIT the value_and_grad of NLL; use_const_mean is a closure (acts like static)
    return jax.value_and_grad(nll)

def _gp_predict(
    params,
    t_train,
    y_train,
    yerr_train,
    use_const_mean: bool,
    t_pred = None,
):
    """Posterior mean prediction from MAP parameters."""
    gp = _build_gp(
        t_train,
        yerr_train,
        params["log_sigma"],
        params["log_rho"],
        params["mu"] if use_const_mean else None,
    )
    if t_pred is None:
        t_pred = t_train
    return gp.predict(y_train, t_pred)


def _map_fit_params(t, y, yerr, use_const_mean=True, max_iter=300, lr=0.02, rho_min: Optional[float]=None, rho_max: Optional[float]=None):
    # time span & default rho bounds
    span = float(jnp.max(t) - jnp.min(t))
    if rho_min is None:
        rho_min = max(1e-3, 0.05 * span)
    if rho_max is None:
        rho_max = max(rho_min + 1e-3, span)

    # reasonable starting point
    init_rho = max(min(span/5.0, rho_max), rho_min)
    params = {
        "log_sigma": jnp.array(jnp.log(0.05)),
        "log_rho":   jnp.array(jnp.log(init_rho)),
        "mu":        jnp.array(0.0),
    }
    nll_and_grad = _make_nll(use_const_mean)
    for _ in range(int(max_iter)):
        val, g = nll_and_grad(params, t, y, yerr)
        params = {
            "log_sigma": params["log_sigma"] - lr * g["log_sigma"],
            "log_rho":   params["log_rho"]   - lr * g["log_rho"],
            "mu":        params["mu"]        - (lr * g["mu"]) if use_const_mean else params["mu"],
        }
        # clamp to bounds
        params["log_rho"] = jnp.clip(params["log_rho"], jnp.log(rho_min), jnp.log(rho_max))
    return params



def _moving_average(y: np.ndarray, window: int) -> np.ndarray:
    window = int(max(3, window));  window += (window % 2 == 0)
    pad = window // 2
    ypad = np.pad(y, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(ypad, kernel, mode="valid")


def _local_minima(y: np.ndarray) -> np.ndarray:
    return np.where((y[1:-1] < y[:-2]) & (y[1:-1] < y[2:]))[0] + 1


def _half_depth_width(time: np.ndarray, flux: np.ndarray, i: int,
                      w_left: int = 50, w_right: int = 50) -> float:
    n = len(flux);  L = max(0, i - w_left);  R = min(n, i + w_right)
    baseline = np.percentile(flux[L:R], 75) if R > L else np.median(flux)
    depth = baseline - flux[i]
    if not np.isfinite(depth) or depth <= 0:
        return np.nan
    half = baseline - 0.5 * depth
    li = i
    while li > 0 and flux[li] < half:
        li -= 1
    if li == 0:
        t_left = time[0]
    else:
        f0, f1 = flux[li], flux[li + 1]
        t0, t1 = time[li], time[li + 1]
        t_left = t0 if f1 == f0 else t0 + (half - f0) * (t1 - t0) / (f1 - f0)
    ri = i
    while ri < n - 1 and flux[ri] < half:
        ri += 1
    if ri == n - 1:
        t_right = time[-1]
    else:
        f0, f1 = flux[ri - 1], flux[ri]
        t0, t1 = time[ri - 1], time[ri]
        t_right = t1 if f1 == f0 else t0 + (half - f0) * (t1 - t0) / (f1 - f0)
    return max(0.0, t_right - t_left)


def estimate_transit_duration_hours(time, flux, error=None,
                                    smooth_frac: float = 0.02,
                                    max_dips: int = 7) -> float:
    time = np.asarray(time, float); flux = np.asarray(flux, float)
    order = np.argsort(time); time = time[order]; flux = flux[order]
    n = len(time)
    if n < 20:
        return 2.5
    window = int(max(11, round(n * smooth_frac))); window = min(window, 301)
    sm = _moving_average(flux, window)
    minima = _local_minima(sm); widths = []
    if len(minima) == 0:
        i = int(np.argmin(sm))
        w = _half_depth_width(time, sm, i, w_left=window, w_right=window)
        if np.isfinite(w) and w > 0: widths.append(w)
    else:
        idx_sorted = minima[np.argsort(sm[minima])][:max_dips]
        for i in idx_sorted:
            w = _half_depth_width(time, sm, i, w_left=window, w_right=window)
            if np.isfinite(w) and w > 0: widths.append(w)
    width_hours = 2.5 if len(widths) == 0 else float(np.median(widths) * 24.0)
    dt = np.median(np.diff(time)); min_hours = max(0.25, 5.0 * dt * 24.0)
    return float(max(width_hours, min_hours))


def estimate_rho_mult(time, flux, duration_hours: float) -> float:
    time = np.asarray(time, float); flux = np.asarray(flux, float)
    order = np.argsort(time); time = time[order]; flux = flux[order]
    n = len(time)
    if n < 50:
        return 5.0
    flux0 = flux - np.nanmedian(flux)
    window = int(max(31, round(n * 0.10))); window = min(window, 1001)
    trend = _moving_average(flux0, window)
    total_std = np.nanstd(flux0); trend_std = np.nanstd(trend)
    ratio = 0.0 if total_std == 0 else trend_std / total_std
    if ratio > 0.6:  return 3.5
    elif ratio > 0.3: return 5.0
    else:            return 7.0


def suggest_from_df(df: "pd.DataFrame",
                    time_col: str = "time",
                    flux_col: str = "flux",
                    error_col: Optional[str] = None) -> Dict[str, float]:
    time = df[time_col].to_numpy(); flux = df[flux_col].to_numpy()
    error = df[error_col].to_numpy() if error_col else None
    duration_hours = estimate_transit_duration_hours(time, flux, error)
    rho_mult = estimate_rho_mult(time, flux, duration_hours)
    return {"transit_duration_hours": float(duration_hours),
            "rho_mult": float(rho_mult)}


def detrend_df(df: "pd.DataFrame", time="time", flux="flux", err="error",
               unit="days", center_flux=True, mean_const=True,
               samples=800, warmup=800, chains=2, seed=42,
               transit_duration_hours=2.0, rho_mult=5.0):
    """Run GP on one star and return corrected series (flux - GP_mean).
    Returns (out_df, gp_mean, mcmc). out_df has columns: time, flux, error, flux_corrected.
    Note: we intentionally define flux_corrected = flux - gp_mean (+ undo-centering),
    aligning with the specification that constant stars ≈ 0 after correction.
    """
    import jax.numpy as jnp
    t_days = _to_days(df[time], unit); t0 = float(np.median(t_days)); t_cent = t_days - t0
    y = df[flux].to_numpy(float); yerr = df[err].to_numpy(float)
    shift = float(np.median(y)) if center_flux else 0.0
    y0 = y - shift
    mcmc = _run_mcmc(
        t=jnp.array(t_cent), y=jnp.array(y0), yerr=jnp.array(yerr),
        use_const_mean=mean_const, num_warmup=warmup, num_samples=samples,
        chains=chains, seed=seed, transit_duration_hours=transit_duration_hours,
        rho_mult=rho_mult,
    )
    samples_post = mcmc.get_samples(group_by_chain=False)
    gp_mean = np.asarray(_posterior_noise_mean_via_predict(
        jnp.array(t_cent), jnp.array(y0), jnp.array(yerr), samples_post,
        use_const_mean=mean_const,
    ))
    # Corrected residual (≈ 0 for constant): (y0 - gp_mean)
    corrected = (y0 - gp_mean)  # zero-centered residual in ADU
    out = df.copy()
    out["flux_corrected"] = corrected
    return out, gp_mean + shift, mcmc


# ---- Airmass / FWHM / Sky helpers ----

def get_header_airmass(hdr) -> float:
    for k in ("AIRMASS", "SECZ"):
        if k in hdr:
            try:
                v = float(hdr[k])
                if np.isfinite(v) and v > 0:
                    return v
            except Exception:
                pass
    return np.nan


def estimate_frame_fwhm(img: np.ndarray, xy_ref: np.ndarray, n: int = 50, box: int = 11) -> float:
    idx = np.arange(len(xy_ref))
    use = idx[:min(n, len(idx))]
    h, w = img.shape
    r = box // 2
    sigmas = []
    yy, xx = np.mgrid[0:box, 0:box]
    for i in use:
        x0, y0 = xy_ref[i]
        xi, yi = int(round(x0)), int(round(y0))
        if xi - r < 0 or yi - r < 0 or xi + r >= w or yi + r >= h:
            continue
        cut = img[yi - r:yi + r + 1, xi - r:xi + r + 1]
        if not np.all(np.isfinite(cut)):
            continue
        s = cut.sum()
        if s <= 0:
            continue
        xbar = (cut * xx).sum() / s
        ybar = (cut * yy).sum() / s
        varx = (cut * ((xx - xbar) ** 2)).sum() / s
        vary = (cut * ((yy - ybar) ** 2)).sum() / s
        if varx > 0 and vary > 0:
            sigmas.append(float(np.sqrt(0.5 * (varx + vary))))
    if not sigmas:
        return np.nan
    return 2.3548 * np.median(sigmas)


def _open_img_and_hdr(path: str) -> Tuple[np.ndarray, fits.Header]:
    try:
        data = fits.getdata(path).astype(float)
        hdr = fits.getheader(path)
    except Exception:
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            hdr = hdul[0].header
    return data, hdr


def compute_covariates(light_paths: Sequence[str], aligned_dir: Optional[str], xy_ref: np.ndarray,
                        mbias=None, dark_dict=None, mflat=None) -> Covariates:
    # 1) airmass from original headers
    airmass = []
    times = []
    for p in light_paths:
        with fits.open(p) as hdul:
            hdr = hdul[0].header
        airmass.append(get_header_airmass(hdr))
        # time
        t = np.nan
        # Prefer JD/BJD/HJD/MJD/DATE-OBS similar to original
        for key in ("JD", "BJD", "HJD"):
            if key in hdr:
                try:
                    t = float(hdr[key])
                    if np.isfinite(t):
                        break
                except Exception:
                    pass
        if not np.isfinite(t):
            if "MJD" in hdr:
                try:
                    t = float(hdr["MJD"]) + 2400000.5
                except Exception:
                    t = np.nan
        if not np.isfinite(t) and "DATE-OBS" in hdr:
            for fmt in ("isot", None):
                try:
                    t = Time(hdr["DATE-OBS"], format=fmt or None, scale="utc").jd
                    break
                except Exception:
                    continue
        times.append(t if np.isfinite(t) else np.nan)

    airmass_arr = np.array(airmass, float)
    times_arr = np.array(times, float)

    # 2) choose image set for FWHM/sky
    src_paths: Sequence[str]
    if aligned_dir and os.path.isdir(aligned_dir):
        cand = sorted(glob.glob(os.path.join(aligned_dir, "*.fit*")))
        src_paths = cand if len(cand) == len(light_paths) else light_paths
    else:
        src_paths = light_paths

    fwhm, sky = [], []
    for p in src_paths:
        img, hdr = _open_img_and_hdr(p)
        # if this isn't calibrated, roughly calibrate if masters provided
        if (aligned_dir is None or not os.path.isdir(aligned_dir)) and (mbias is not None or (dark_dict and len(dark_dict)>0) or (mflat is not None)):
            img = calibrate_frame(img, hdr, master_bias=mbias, dark_dict=dark_dict, flat_norm=mflat)
        fwhm.append(estimate_frame_fwhm(img, xy_ref, n=50, box=11))
        finite = np.isfinite(img)
        sky.append(np.nanmedian(img[finite]) if np.any(finite) else np.nan)

    return Covariates(
        airmass=np.array(airmass, float),
        fwhm_px=np.array(fwhm, float),
        sky_med=np.array(sky, float),
        times_jd=times_arr if np.any(np.isfinite(times_arr)) else np.arange(len(light_paths), dtype=float),
    )


def normalize_lightcurves(times_jd: np.ndarray, flux_mat: np.ndarray, err_mat: np.ndarray,
                          ncfg: NormalizationConfig, out_dir: Optional[str] = None) -> np.ndarray:
    """Apply GP detrending per star; return matrix of residuals (≈ zero mean).
    If `ncfg.per_star_csv` and `out_dir` are provided, write one CSV per star to
    `${out_dir}/${ncfg.per_star_dir}/${ncfg.per_star_prefix}{index}.csv`.

    Downsampling: in MAP mode, training uses a fraction of points (`downsample_frac`),
    but predictions are evaluated on **all available** times.
    In MCMC mode, downsampling is ignored (kept at 1.0).
    """
    t_all = np.asarray(times_jd, float)
    n_frames, n_stars = flux_mat.shape
    norm = np.full_like(flux_mat, np.nan, dtype=float)

    per_dir = None
    if ncfg.per_star_csv and out_dir:
        per_dir = os.path.join(out_dir, ncfg.per_star_dir)
        os.makedirs(per_dir, exist_ok=True)

    for si in tqdm(range(n_stars), desc="Normalize", unit="star"):
        y = flux_mat[:, si]
        s = err_mat[:, si] if err_mat is not None and err_mat.shape == flux_mat.shape else np.full_like(y, np.nan)
        good = np.isfinite(t_all) & np.isfinite(y) & np.isfinite(s)
        if good.sum() < 5:
            continue

        tg = t_all[good]; yg = y[good]; sg = s[good]

        if ncfg.mode == "mcmc":
            # (1) MCMC path — quick settings already trimmed by defaults
            df = pd.DataFrame({"time": tg, "flux": yg, "error": sg})
            td = ncfg.transit_duration_hours; rm = ncfg.rho_mult
            if (td is None) or (rm is None):
                sug = suggest_from_df(df, "time", "flux", "error")
                if td is None: td = sug["transit_duration_hours"]
                if rm is None: rm = sug["rho_mult"]
            out_df, gp_mean, _ = detrend_df(
                df, time="time", flux="flux", err="error",
                unit=ncfg.unit, center_flux=ncfg.center_flux, mean_const=ncfg.mean_const,
                samples=ncfg.samples, warmup=ncfg.warmup, chains=ncfg.chains, seed=ncfg.seed,
                transit_duration_hours=float(td), rho_mult=float(rm),
            )
            res = out_df["flux_corrected"].to_numpy(float)
            full = np.full_like(y, np.nan); full[good] = res
            norm[:, si] = full

            if per_dir is not None:
                fname = f"{ncfg.per_star_prefix}{si:0{ncfg.per_star_digits}d}_norm.csv"
                cols = {"JD": tg, "flux": yg, **({"error": sg} if ncfg.per_star_include_error else {}), "flux_norm": res}
                if ncfg.per_star_include_gpmean:
                    cols["trend_gpmean"] = gp_mean
                pd.DataFrame(cols).to_csv(os.path.join(per_dir, fname), index=False)
            continue

        # (2) MAP path — optionally downsample training, predict on full tg
        tg_days = _to_days(tg, unit=ncfg.unit)
        t0 = float(np.median(tg_days))
        tg_cent = tg_days - t0

        yy = yg.copy()
        ee = sg.copy()
        tt_cent = tg_cent
        if (0.0 < ncfg.downsample_frac < 1.0) and (len(tg) > 10):
            m = max(5, int(np.floor(len(tg) * ncfg.downsample_frac)))
            if m < len(tg):
                idx = np.linspace(0, len(tg) - 1, m)
                idx = np.unique(idx.astype(int))
                tt_cent = tg_cent[idx]
                yy = yg[idx]
                ee = sg[idx]

        min_err = np.nanmedian(ee)
        min_err = (0.02 * min_err) if np.isfinite(min_err) else 1e-6
        ee = np.clip(ee, min_err, None)

        tt = jnp.array(tt_cent)
        yy_arr = jnp.array(yy)
        ee_arr = jnp.array(ee)
        shift = float(jnp.median(yy_arr)) if ncfg.center_flux else 0.0
        y0 = yy_arr - shift

        # Suggest bounds from duration & rho_mult (or auto)
        td = ncfg.transit_duration_hours; rm = ncfg.rho_mult
        if (td is None) or (rm is None):
            sug = suggest_from_df(pd.DataFrame({"time": tg, "flux": yg, "error": sg}), "time", "flux", "error")
            if td is None: td = sug["transit_duration_hours"]
            if rm is None: rm = sug["rho_mult"]
        span_days = float(np.max(tg_days) - np.min(tg_days))
        dur_days = float(td) / 24.0
        rho_min = max(1e-3, rm * dur_days, 0.05 * span_days)
        rho_max = max(rho_min + 1e-3, span_days)

        params = _map_fit_params(tt, y0, ee_arr, use_const_mean=ncfg.mean_const,
                                 max_iter=ncfg.map_max_iter, lr=ncfg.map_lr,
                                 rho_min=rho_min, rho_max=rho_max)
        gp_mean_full = np.array(
            _gp_predict(
                params,
                tt,
                y0,
                ee_arr,
                ncfg.mean_const,
                t_pred=jnp.array(tg_cent),
            )
        )
        res_full = ( (yg - shift) - gp_mean_full ).astype(float)

        full = np.full_like(y, np.nan); full[good] = res_full
        norm[:, si] = full

        if per_dir is not None:
            fname = f"{ncfg.per_star_prefix}{si:0{ncfg.per_star_digits}d}_norm.csv"
            cols = {"JD": tg, "flux": yg, **({"error": sg} if ncfg.per_star_include_error else {}), "flux_norm": res_full}
            if ncfg.per_star_include_gpmean:
                cols["trend_gpmean"] = gp_mean_full
            pd.DataFrame(cols).to_csv(os.path.join(per_dir, fname), index=False)

    return norm


def save_norm_wide_csv(csv_path: str, times_jd: np.ndarray, norm_mat: np.ndarray) -> None:
    n_frames, n_stars = norm_mat.shape
    cols: Dict[str, np.ndarray] = {"JD": times_jd}
    for si in range(n_stars):
        cols[f"star{si:04d}_norm"] = norm_mat[:, si]
    pd.DataFrame(cols).to_csv(csv_path, index=False)

# -----------------------------------------------------------------------------
# CSV writer
# -----------------------------------------------------------------------------

def save_wide_csv(csv_path: str, times_jd: np.ndarray, flux_mat: np.ndarray, err_mat: np.ndarray) -> None:
    n_frames, n_stars = flux_mat.shape
    cols: Dict[str, np.ndarray] = {"JD": times_jd}

    for si in range(n_stars):
        cols[f"star{si:04d}"] = flux_mat[:, si]
        cols[f"star{si:04d}_err"] = err_mat[:, si]

    df = pd.DataFrame(cols)
    df.to_csv(csv_path, index=False)

# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def main() -> None:
    warnings.filterwarnings("ignore", category=UserWarning)

    parser = argparse.ArgumentParser(description="Photometry + GP normalization pipeline")
    parser.add_argument("root", help="Input directory containing FITS with type prefixes (e.g., object*.fits)")
    parser.add_argument("out", help="Output directory for results")
    args = parser.parse_args()

    cfg = RunConfig(
        root_dir=args.root,
        output_dir=args.out,
        prefixes=FilePrefixes(),
        use_bias=True,
        use_dark=True,
        use_flat=True,
        do_alignment=True,
        save_aligned_fits=False,
        csv_wide_path="allstars_flux_err_wide.csv",
        phot=PhotometryConfig(
            fwhm_pix=3.5,
            thresh_sigma=5.0,
            max_stars_detect=2000,
            edge_margin=12,
            r_ap=None,
            r_in=None,
            r_out=None,
        ),
        inst=InstrumentConfig(
            gain_e_per_adu=None,
        ),
        norm=NormalizationConfig(
            enabled=True,
            samples=300,
            warmup=300,
            chains=1,
            mean_const=True,
            center_flux=True,
            seed=42,
            transit_duration_hours=None,
            rho_mult=None,
            unit="days",
            output_csv_name="allstars_flux_norm_wide.csv",
            mode="mcmc",
            map_max_iter=300,
            map_lr=0.02,
            downsample_frac=1.0,
            per_star_csv=True,
            per_star_dir="norm_per_star",
            per_star_prefix="star",
            per_star_digits=4,
            per_star_include_gpmean=False,
            per_star_include_error=True,
        ),
    )

    cfg.prepare()

    times, flux_mat, err_mat, xy, light_files, mbias, dark_dict, mflat = process_all_frames(cfg)

    # Save wide flux+err
    csv_path = os.path.join(cfg.output_dir, cfg.csv_wide_path)
    save_wide_csv(csv_path, times, flux_mat, err_mat)

    # ---- Compute & save airmass/FWHM/sky covariates (from original logic) ----
    light_paths = light_files
    aligned_dir = cfg.aligned_dir if cfg.save_aligned_fits else None
    cov = compute_covariates(light_paths, aligned_dir, xy,
                             mbias=mbias, dark_dict=dark_dict, mflat=mflat)

    cov_df = pd.DataFrame({
        "JD": cov.times_jd,
        "airmass": cov.airmass,
        "fwhm_px": cov.fwhm_px,
        "sky_med": cov.sky_med,
    })
    cov_csv = os.path.join(cfg.output_dir, "covariates.csv")
    cov_df.to_csv(cov_csv, index=False)

    if cfg.norm.enabled:
        norm_mat = normalize_lightcurves(times, flux_mat, err_mat, cfg.norm, out_dir=cfg.output_dir)
        norm_csv = os.path.join(cfg.output_dir, cfg.norm.output_csv_name)
        save_norm_wide_csv(norm_csv, times, norm_mat)


if __name__ == "__main__":
    main()
    
