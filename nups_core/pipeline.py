"""End-to-end light-curve pipeline derived from the original notebook."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# Configure matplotlib before importing pyplot (headless-friendly)
import matplotlib

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from astropy.io import fits
from astropy.stats import sigma_clipped_stats
from astropy.time import Time
from photutils.aperture import CircularAnnulus, CircularAperture
from photutils.detection import DAOStarFinder

import astroalign as aa

from .config import (
    AlignmentConfig,
    CalibrationConfig,
    EnsembleConfig,
    OutputConfig,
    PathsConfig,
    PhotometryConfig,
    PipelineConfig,
)

warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams["axes.unicode_minus"] = False


@dataclass(slots=True)
class CalibrationFrames:
    bias: Optional[np.ndarray]
    darks: Dict[float, np.ndarray]
    flat: Optional[np.ndarray]


@dataclass(slots=True)
class PipelineResult:
    times: np.ndarray
    flux_matrix: np.ndarray
    star_positions: np.ndarray
    detrended_flux: np.ndarray
    comparison_map: Dict[int, List[int]]
    covariates: Dict[str, np.ndarray]
    output_paths: Dict[str, Path]


class LightCurvePipeline:
    """High-level orchestrator that mirrors the notebook workflow."""

    def __init__(self, config: PipelineConfig) -> None:
        self.cfg = config
        self.paths: PathsConfig = config.paths
        self.calib_cfg: CalibrationConfig = config.calibration
        self.align_cfg: AlignmentConfig = config.alignment
        self.phot_cfg: PhotometryConfig = config.photometry
        self.ensemble_cfg: EnsembleConfig = config.ensemble
        self.output_cfg: OutputConfig = config.output
        self.verbose = config.verbose
        self._calibrations: Optional[CalibrationFrames] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self) -> PipelineResult:
        self._log("Starting pipeline run")
        self._prepare_directories()

        light_files = self._list_fits(self.paths.light_dir)
        if not light_files:
            raise FileNotFoundError(f"No FITS light frames found in {self.paths.light_dir}")
        self._log(f"Found {len(light_files)} light frames")

        calibrations = self._build_calibration_frames()
        self._calibrations = calibrations
        ref_img, ref_hdr = self._load_calibrated_reference(light_files[0], calibrations)
        xyf = self._detect_stars(ref_img)
        positions = xyf[:, :2]
        self._log(f"Detected {len(positions)} stars in reference frame")

        output_paths: Dict[str, Path] = {}
        if self.output_cfg.save_detection_preview:
            preview_path = self._save_detection_preview(ref_img, positions)
            output_paths["detection_preview"] = preview_path

        flux_rows: List[np.ndarray] = []
        times: List[float] = []
        aligned_paths: List[Path] = []

        for idx, path in enumerate(light_files):
            data, hdr = self._load_fits(path)
            calibrated = self._calibrate_frame(data, hdr, calibrations)

            if idx == 0 or not self.align_cfg.enabled:
                aligned = calibrated
                transform = None
            else:
                try:
                    aligned, transform = self._align_to_reference(calibrated, ref_img)
                except RuntimeError as exc:
                    self._log(f"Alignment failed for {path.name}: {exc}; skipping frame", level="warn")
                    continue

            flux = self._measure_photometry(aligned, positions)
            flux_rows.append(flux)
            time_val = self._read_time_from_header(hdr)
            times.append(time_val if np.isfinite(time_val) else np.nan)

            if self.align_cfg.save_aligned_fits:
                aligned_path = self.paths.aligned_dir / path.name
                self._write_aligned_fits(aligned_path, aligned, hdr, transform)
                aligned_paths.append(aligned_path)
            else:
                aligned_paths.append(path)

            if (idx + 1) % 10 == 0:
                self._log(f"Processed {idx + 1}/{len(light_files)} frames")

        if not flux_rows:
            raise RuntimeError("No frames processed successfully")

        flux_matrix = np.vstack(flux_rows)
        times_arr = np.asarray(times, dtype=float)
        if np.any(~np.isfinite(times_arr)):
            self._log("Missing times in headers; substituting frame indices", level="warn")
            times_arr = np.arange(len(times_arr), dtype=float)

        flux_matrix, positions = self._filter_low_quality_stars(flux_matrix, positions)

        covariates = self._build_covariates(aligned_paths, positions)
        output_paths.update(self._write_intermediate_products(times_arr, flux_matrix))

        detrended, comp_map = self._detrend_all_stars(flux_matrix, positions, covariates, times_arr)
        output_paths.update(self._write_final_products(times_arr, detrended))

        self._log("Pipeline finished")
        return PipelineResult(
            times=times_arr,
            flux_matrix=flux_matrix,
            star_positions=positions,
            detrended_flux=detrended,
            comparison_map=comp_map,
            covariates=covariates,
            output_paths=output_paths,
        )

    # ------------------------------------------------------------------
    # File system helpers
    # ------------------------------------------------------------------
    def _prepare_directories(self) -> None:
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        self.paths.raw_plot_dir.mkdir(parents=True, exist_ok=True)
        if self.align_cfg.save_aligned_fits:
            self.paths.aligned_dir.mkdir(parents=True, exist_ok=True)
        if self.output_cfg.save_detrended_plots:
            self.paths.detrended_plot_dir.mkdir(parents=True, exist_ok=True)

    def _list_fits(self, directory: Path) -> List[Path]:
        if not directory.is_dir():
            return []
        fits_files = sorted(list(directory.glob("*.fits")) + list(directory.glob("*.fit")))
        return fits_files

    # ------------------------------------------------------------------
    # Calibration construction
    # ------------------------------------------------------------------
    def _build_calibration_frames(self) -> CalibrationFrames:
        bias = self._build_master_bias(self.paths.bias_dir) if self.calib_cfg.use_bias else None
        darks = (
            self._build_master_dark_by_exptime(self.paths.dark_dir)
            if self.calib_cfg.use_dark
            else {}
        )
        flat = (
            self._build_master_flat(self.paths.flat_dir, bias, darks)
            if self.calib_cfg.use_flat
            else None
        )

        if bias is not None:
            self._log(self._array_summary("Master bias", bias))
        if darks:
            for expt, mdark in sorted(darks.items(), key=lambda kv: kv[0]):
                self._log(self._array_summary(f"Master dark {expt:g}s", mdark))
        if flat is not None:
            self._log(self._array_summary("Master flat", flat))

        return CalibrationFrames(bias=bias, darks=darks, flat=flat)

    def _build_master_bias(self, directory: Optional[Path]) -> Optional[np.ndarray]:
        if directory is None:
            return None
        files = self._list_fits(directory)
        if not files:
            return None
        stack = [self._load_fits(path)[0].astype(float) for path in files]
        return np.nanmedian(np.stack(stack, axis=0), axis=0)

    def _build_master_dark_by_exptime(self, directory: Optional[Path]) -> Dict[float, np.ndarray]:
        if directory is None:
            return {}
        files = self._list_fits(directory)
        if not files:
            return {}
        grouped: Dict[float, List[Path]] = {}
        for path in files:
            _, hdr = self._load_fits(path)
            exptime = self._extract_exptime(hdr)
            if exptime is None:
                continue
            grouped.setdefault(exptime, []).append(path)
        out: Dict[float, np.ndarray] = {}
        for exptime, paths in grouped.items():
            stack = [self._load_fits(p)[0].astype(float) for p in paths]
            out[exptime] = np.nanmedian(np.stack(stack, axis=0), axis=0)
        return out

    def _build_master_flat(
        self,
        directory: Optional[Path],
        master_bias: Optional[np.ndarray],
        master_darks: Dict[float, np.ndarray],
    ) -> Optional[np.ndarray]:
        if directory is None:
            return None
        files = self._list_fits(directory)
        if not files:
            return None
        calibrated: List[np.ndarray] = []
        for path in files:
            data, hdr = self._load_fits(path)
            frame = data.astype(float)
            if master_bias is not None:
                frame -= master_bias
            if master_darks:
                exptime = self._extract_exptime(hdr)
                if exptime is not None:
                    nearest = min(master_darks.keys(), key=lambda t: abs(t - exptime))
                    scale = exptime / nearest if nearest else 1.0
                    frame -= master_darks[nearest] * scale
            calibrated.append(frame)
        flat = np.nanmedian(np.stack(calibrated, axis=0), axis=0)
        finite = np.isfinite(flat)
        if np.any(finite):
            med = np.nanmedian(flat[finite])
            if np.isfinite(med) and med != 0:
                flat = flat / med
        return flat

    def _array_summary(self, name: str, arr: np.ndarray) -> str:
        finite = np.isfinite(arr)
        if not np.any(finite):
            return f"{name}: shape={arr.shape}, all NaN"
        med = float(np.nanmedian(arr[finite]))
        mean = float(np.nanmean(arr[finite]))
        std = float(np.nanstd(arr[finite]))
        return f"{name}: shape={arr.shape}, med={med:.3f}, mean={mean:.3f}, std={std:.3f}"

    # ------------------------------------------------------------------
    # Frame utilities
    # ------------------------------------------------------------------
    def _load_fits(self, path: Path) -> Tuple[np.ndarray, fits.Header]:
        with fits.open(path) as hdul:
            data = hdul[0].data.astype(float)
            header = hdul[0].header
        return data, header

    def _load_calibrated_reference(
        self, path: Path, calibrations: CalibrationFrames
    ) -> Tuple[np.ndarray, fits.Header]:
        data, hdr = self._load_fits(path)
        calibrated = self._calibrate_frame(data, hdr, calibrations)
        return calibrated, hdr

    def _calibrate_frame(
        self,
        data: np.ndarray,
        hdr: fits.Header,
        calibrations: CalibrationFrames,
    ) -> np.ndarray:
        result = data.astype(float)
        if calibrations.bias is not None:
            result = result - calibrations.bias
        if calibrations.darks:
            exptime = self._extract_exptime(hdr)
            if exptime is not None:
                nearest = min(calibrations.darks.keys(), key=lambda t: abs(t - exptime))
                scale = exptime / nearest if nearest else 1.0
                result = result - calibrations.darks[nearest] * scale
        if calibrations.flat is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                result = result / calibrations.flat
        return result

    def _write_aligned_fits(
        self,
        path: Path,
        data: np.ndarray,
        hdr: fits.Header,
        transform: Optional[Any],
    ) -> None:
        header = hdr.copy()
        history = "calibrated"
        if self.align_cfg.enabled:
            history += " & aligned"
        if transform is not None and hasattr(transform, "translation"):
            dx, dy = transform.translation
            header["HISTORY"] = f"{history}; dx={dx:.3f}; dy={dy:.3f}"
        else:
            header["HISTORY"] = history
        fits.writeto(path, data.astype(np.float32), header, overwrite=True)

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def _extract_exptime(self, hdr: fits.Header) -> Optional[float]:
        for key in ("EXPTIME", "EXPOSURE", "EXP_TIME"):
            if key in hdr:
                try:
                    value = float(hdr[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        return None

    def _read_time_from_header(self, hdr: fits.Header) -> float:
        for key in ("JD", "BJD", "HJD"):
            if key in hdr:
                try:
                    value = float(hdr[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value):
                    return value
        if "MJD" in hdr:
            try:
                value = float(hdr["MJD"])
            except (TypeError, ValueError):
                value = math.nan
            if np.isfinite(value):
                return value + 2400000.5
        if "DATE-OBS" in hdr:
            for fmt in ("isot", None):
                try:
                    t = Time(hdr["DATE-OBS"], format=fmt or "fits", scale="utc").jd
                    if np.isfinite(t):
                        return float(t)
                except Exception:
                    continue
        return math.nan

    # ------------------------------------------------------------------
    # Alignment & detection
    # ------------------------------------------------------------------
    def _align_to_reference(self, src_img: np.ndarray, ref_img: np.ndarray) -> Tuple[np.ndarray, Any]:
        try:
            aligned, transform = aa.register(
                src_img,
                ref_img,
                detection_sigma=self.align_cfg.detection_sigma,
                max_control_points=self.align_cfg.max_control_points,
            )
            return aligned.astype(float), transform
        except aa.MaxIterError as exc:  # pragma: no cover - rare
            raise RuntimeError(f"Alignment failed (MaxIterError): {exc}") from exc
        except Exception as exc:
            raise RuntimeError(str(exc)) from exc

    def _detect_stars(self, ref_img: np.ndarray) -> np.ndarray:
        mean, median, std = sigma_clipped_stats(ref_img, sigma=3.0, maxiters=5)
        dao = DAOStarFinder(fwhm=self.phot_cfg.fwhm_pix, threshold=self.phot_cfg.thresh_sigma * std)
        table = dao(ref_img - median)
        if table is None or len(table) == 0:
            raise RuntimeError("No stars detected; adjust FWHM/thresh parameters")
        table.sort("flux")
        table = table[::-1]
        if len(table) > self.phot_cfg.max_stars_detect:
            table = table[: self.phot_cfg.max_stars_detect]
        xyf = np.vstack([
            table["xcentroid"].data,
            table["ycentroid"].data,
            table["flux"].data,
        ]).T
        height, width = ref_img.shape
        mask = (
            (xyf[:, 0] > self.phot_cfg.edge_margin)
            & (xyf[:, 0] < width - self.phot_cfg.edge_margin)
            & (xyf[:, 1] > self.phot_cfg.edge_margin)
            & (xyf[:, 1] < height - self.phot_cfg.edge_margin)
        )
        return xyf[mask]

    def _measure_photometry(self, img: np.ndarray, xy: np.ndarray) -> np.ndarray:
        r_ap = self.phot_cfg.aperture_scale * self.phot_cfg.fwhm_pix
        r_in = self.phot_cfg.annulus_in_scale * self.phot_cfg.fwhm_pix
        r_out = self.phot_cfg.annulus_out_scale * self.phot_cfg.fwhm_pix
        apert = CircularAperture(xy, r=r_ap)
        ann = CircularAnnulus(xy, r_in=r_in, r_out=r_out)
        ap_masks = apert.to_mask(method="exact")
        ann_masks = ann.to_mask(method="exact")

        sky_vals: List[float] = []
        for mask in ann_masks:
            annulus_data = mask.multiply(img)
            finite = np.isfinite(annulus_data)
            if not np.any(finite):
                sky_vals.append(0.0)
                continue
            sky_vals.append(float(np.nanmedian(annulus_data[finite])))
        sky_arr = np.asarray(sky_vals, dtype=float)

        fluxes: List[float] = []
        for mask, sky in zip(ap_masks, sky_arr):
            aperture_data = mask.multiply(img)
            finite = np.isfinite(aperture_data)
            pixels = aperture_data[finite]
            area = np.sum(finite)
            if area == 0:
                fluxes.append(math.nan)
            else:
                fluxes.append(float(np.nansum(pixels) - sky * area))
        return np.asarray(fluxes, dtype=float)

    # ------------------------------------------------------------------
    # Covariates & filtering
    # ------------------------------------------------------------------
    def _filter_low_quality_stars(
        self, flux_matrix: np.ndarray, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        valid_ratio = np.mean(np.isfinite(flux_matrix), axis=0)
        keep = valid_ratio >= self.phot_cfg.min_valid_ratio
        if not np.any(keep):
            raise RuntimeError(
                "All stars rejected by quality filter; relax min_valid_ratio or review data"
            )
        removed = np.count_nonzero(~keep)
        if removed:
            self._log(f"Dropped {removed} stars below valid-ratio threshold")
        return flux_matrix[:, keep], positions[keep]

    def _build_covariates(
        self,
        frame_paths: Sequence[Path],
        xy: np.ndarray,
    ) -> Dict[str, np.ndarray]:
        airmass: List[float] = []
        fwhm: List[float] = []
        sky: List[float] = []
        for path in frame_paths:
            data, hdr = self._safe_open_calibrated_frame(path)
            airmass.append(self._get_header_airmass(hdr))
            fwhm.append(self._estimate_frame_fwhm(data, xy, n=50, box=11))
            finite = np.isfinite(data)
            sky.append(float(np.nanmedian(data[finite])) if np.any(finite) else math.nan)
        covariates = {
            "airmass": np.asarray(airmass, dtype=float),
            "fwhm": np.asarray(fwhm, dtype=float),
            "sky": np.asarray(sky, dtype=float),
        }
        return covariates

    def _safe_open_calibrated_frame(self, path: Path) -> Tuple[np.ndarray, fits.Header]:
        try:
            data = fits.getdata(path).astype(float)
            hdr = fits.getheader(path)
        except Exception:
            data, hdr = self._load_fits(path)
        if self.align_cfg.save_aligned_fits and path.parent == self.paths.aligned_dir:
            return data, hdr
        if self._calibrations is not None:
            data = self._calibrate_frame(data, hdr, self._calibrations)
        return data, hdr

    def _get_header_airmass(self, hdr: fits.Header) -> float:
        for key in ("AIRMASS", "SECZ"):
            if key in hdr:
                try:
                    value = float(hdr[key])
                except (TypeError, ValueError):
                    continue
                if np.isfinite(value) and value > 0:
                    return value
        return math.nan

    def _estimate_frame_fwhm(
        self,
        img: np.ndarray,
        xy_ref: np.ndarray,
        n: int = 40,
        box: int = 11,
    ) -> float:
        idx = np.arange(len(xy_ref))
        use = idx[: min(n, len(idx))]
        half = box // 2
        height, width = img.shape
        sigmas: List[float] = []
        grid_y, grid_x = np.mgrid[0:box, 0:box]
        for i in use:
            x0, y0 = xy_ref[i]
            xi, yi = int(round(x0)), int(round(y0))
            if xi - half < 0 or yi - half < 0 or xi + half >= width or yi + half >= height:
                continue
            cut = img[yi - half : yi + half + 1, xi - half : xi + half + 1]
            if not np.all(np.isfinite(cut)):
                continue
            total = cut.sum()
            if total <= 0:
                continue
            xbar = float((cut * grid_x).sum() / total)
            ybar = float((cut * grid_y).sum() / total)
            varx = float((cut * (grid_x - xbar) ** 2).sum() / total)
            vary = float((cut * (grid_y - ybar) ** 2).sum() / total)
            if varx > 0 and vary > 0:
                sigmas.append(math.sqrt(0.5 * (varx + vary)))
        if not sigmas:
            return math.nan
        sigma_pix = float(np.median(sigmas))
        return 2.3548 * sigma_pix

    # ------------------------------------------------------------------
    # Saving helpers
    # ------------------------------------------------------------------
    def _write_intermediate_products(
        self,
        times: np.ndarray,
        flux_matrix: np.ndarray,
    ) -> Dict[str, Path]:
        output_paths: Dict[str, Path] = {}
        if self.output_cfg.save_raw_wide_csv:
            df = pd.DataFrame(
                flux_matrix,
                columns=[f"star{i:04d}" for i in range(flux_matrix.shape[1])],
            )
            df.insert(0, "time", times)
            df.to_csv(self.paths.raw_wide_csv_path, index=False)
            output_paths["raw_wide_csv"] = self.paths.raw_wide_csv_path
        if self.output_cfg.save_times_csv:
            np.savetxt(self.paths.times_csv_path, times, delimiter=",")
            output_paths["times_csv"] = self.paths.times_csv_path
        return output_paths

    def _write_final_products(
        self,
        times: np.ndarray,
        detrended: np.ndarray,
    ) -> Dict[str, Path]:
        output_paths: Dict[str, Path] = {}
        if self.output_cfg.save_detrended_csv:
            df_det = pd.DataFrame(
                detrended,
                columns=[f"star{i:04d}" for i in range(detrended.shape[1])],
            )
            df_det.insert(0, "time", times)
            df_det.to_csv(self.paths.detrended_wide_csv_path, index=False)
            output_paths["detrended_wide_csv"] = self.paths.detrended_wide_csv_path
        return output_paths

    def _save_detection_preview(self, ref_img: np.ndarray, positions: np.ndarray) -> Path:
        disp = self._stretch(ref_img)
        path = self.paths.detection_preview_path
        plt.figure(figsize=(8, 6))
        ax = plt.gca()
        im = ax.imshow(disp, cmap="gray", origin="lower")
        plt.colorbar(im, ax=ax, label="stretched intensity")
        limit = min(self.phot_cfg.detection_preview_limit, len(positions))
        apert = CircularAperture(positions[:limit], r=self.phot_cfg.aperture_scale * self.phot_cfg.fwhm_pix)
        ann = CircularAnnulus(
            positions[:limit],
            r_in=self.phot_cfg.annulus_in_scale * self.phot_cfg.fwhm_pix,
            r_out=self.phot_cfg.annulus_out_scale * self.phot_cfg.fwhm_pix,
        )
        apert.plot(ax=ax, lw=1.0, color="cyan")
        ann.plot(ax=ax, lw=0.8, color="lime")
        for idx in range(limit):
            x, y = positions[idx]
            ax.text(x + 5, y + 5, f"{idx}", color="yellow", fontsize=8, weight="bold")
        ax.set_title(f"Detected stars (N={len(positions)})")
        ax.set_xlim(0, ref_img.shape[1])
        ax.set_ylim(0, ref_img.shape[0])
        plt.tight_layout()
        plt.savefig(path, dpi=160)
        plt.close()
        return path

    def _stretch(self, img: np.ndarray, p_lo: float = 1.0, p_hi: float = 99.0) -> np.ndarray:
        finite = np.isfinite(img)
        if not np.any(finite):
            return img
        v1, v2 = np.percentile(img[finite], [p_lo, p_hi])
        v1 = float(v1)
        v2 = float(v2)
        return np.clip((img - v1) / max(v2 - v1, 1e-9), 0, 1)

    def _plot_detrended_curve(
        self,
        times: np.ndarray,
        rel_flux: np.ndarray,
        star_index: int,
        position: np.ndarray,
    ) -> None:
        if not self.output_cfg.save_detrended_plots:
            return
        x_axis = times - np.nanmin(times)
        plt.figure(figsize=(7.2, 4.2))
        plt.plot(x_axis * 24.0, rel_flux, ".", ms=3)
        plt.xlabel("Time since first frame [hr]")
        plt.ylabel("Relative flux (detrended)")
        plt.title(f"Star {star_index:04d} @ (x={position[0]:.1f}, y={position[1]:.1f})")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        out_path = self.paths.detrended_plot_dir / f"lc_star{star_index:04d}_det.png"
        plt.savefig(out_path, dpi=180)
        plt.close()

    # ------------------------------------------------------------------
    # Ensemble / detrending helpers
    # ------------------------------------------------------------------
    def _detrend_all_stars(
        self,
        flux_matrix: np.ndarray,
        positions: np.ndarray,
        covariates: Dict[str, np.ndarray],
        times: np.ndarray,
    ) -> Tuple[np.ndarray, Dict[int, List[int]]]:
        series_mat = flux_matrix
        bright_vec = np.nanmedian(series_mat, axis=0)
        cov_list = [arr for arr in covariates.values() if np.any(np.isfinite(arr))]
        cov_names = [name for name, arr in covariates.items() if np.any(np.isfinite(arr))]
        detrended = np.full_like(series_mat, np.nan, dtype=float)
        comp_map: Dict[int, List[int]] = {}
        for star_idx in range(series_mat.shape[1]):
            comp_ids = self._pick_comps_rms_aware(
                star_idx,
                series_mat,
                bright_vec,
                positions,
                bright_tol=self.ensemble_cfg.bright_tolerance,
                k=self.ensemble_cfg.rms_k,
            )
            if len(comp_ids) < self.ensemble_cfg.min_comps:
                continue
            ref_curve, _ = self._weighted_reference(series_mat[:, comp_ids])
            raw_rel = series_mat[:, star_idx] / ref_curve
            med = np.nanmedian(raw_rel)
            if np.isfinite(med) and med != 0:
                raw_rel = raw_rel / med
            if cov_list:
                baseline, rel_corr, good = self._detrend_by_covariates(raw_rel, cov_list)
                rel_series = rel_corr
            else:
                rel_series = raw_rel
            detrended[:, star_idx] = rel_series
            comp_map[star_idx] = comp_ids
            self._plot_detrended_curve(times, rel_series, star_idx, positions[star_idx])
            if self.verbose:
                self._log(
                    f"Star {star_idx:04d}: comps={len(comp_ids)}" + (
                        f" covariates={','.join(cov_names)}" if cov_names else ""
                    )
                )
        return detrended, comp_map

    def _pick_comps_rms_aware(
        self,
        target_idx: int,
        series_mat: np.ndarray,
        bright_vec: np.ndarray,
        xy: np.ndarray,
        bright_tol: float,
        k: int,
    ) -> List[int]:
        tflux = bright_vec[target_idx]
        if not np.isfinite(tflux):
            return []
        lower = (1.0 - bright_tol) * tflux
        upper = (1.0 + bright_tol) * tflux
        candidates: List[int] = []
        for idx in range(series_mat.shape[1]):
            if idx == target_idx:
                continue
            if not np.isfinite(bright_vec[idx]):
                continue
            if not (lower <= bright_vec[idx] <= upper):
                continue
            dx = xy[idx, 0] - xy[target_idx, 0]
            dy = xy[idx, 1] - xy[target_idx, 1]
            min_sep = self.phot_cfg.min_separation_scale * self.phot_cfg.fwhm_pix
            if math.hypot(dx, dy) < min_sep:
                continue
            candidates.append(idx)
        if not candidates:
            return []
        rms_scores: List[Tuple[int, float]] = []
        for idx in candidates:
            series = series_mat[:, idx]
            med = np.nanmedian(series)
            if not np.isfinite(med) or med == 0:
                continue
            series_norm = series / med
            rms_scores.append((idx, float(np.nanstd(series_norm))))
        if not rms_scores:
            return []
        rms_scores.sort(key=lambda item: item[1])
        return [idx for idx, _ in rms_scores[:k]]

    def _weighted_reference(self, series_mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        normalized = series_mat / np.nanmedian(series_mat, axis=0)
        variance = np.nanvar(normalized, axis=0)
        weights = 1.0 / np.clip(variance, 1e-8, None)
        weights /= np.nansum(weights)
        ref = np.nansum(series_mat * weights, axis=1)
        med = np.nanmedian(ref)
        if np.isfinite(med) and med != 0:
            ref = ref / med
        return ref, weights

    def _detrend_by_covariates(
        self,
        y: np.ndarray,
        covariates: Sequence[np.ndarray],
        max_iter: int = 4,
        clip: float = 3.0,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        X_cols = [np.ones_like(y)] + [arr for arr in covariates]
        X = np.column_stack(X_cols)
        good = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        beta = np.zeros(X.shape[1])
        for _ in range(max_iter):
            if not np.any(good):
                break
            beta, *_ = np.linalg.lstsq(X[good], y[good], rcond=None)
            model = X @ beta
            resid = y - model
            sigma = np.nanstd(resid[good])
            if not (np.isfinite(sigma) and sigma > 0):
                break
            good = good & (np.abs(resid) < clip * sigma)
        baseline = X @ beta
        with np.errstate(divide="ignore", invalid="ignore"):
            corrected = y / baseline
        med = np.nanmedian(corrected[good]) if np.any(good) else np.nanmedian(corrected)
        if np.isfinite(med) and med != 0:
            corrected = corrected / med
        return baseline, corrected, good

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    def _log(self, message: str, level: str = "info") -> None:
        if not self.verbose and level == "info":
            return
        prefix = {"info": "[INFO]", "warn": "[WARN]"}.get(level, "[INFO]")
        print(f"{prefix} {message}")
