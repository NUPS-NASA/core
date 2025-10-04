"""Simple light-curve fitting utilities for the NUPS CLI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np

from .photometry import LightCurve, PhotometryResult

__all__ = ["FittedLightCurve", "FittingResult", "run_fitting"]


@dataclass
class FittedLightCurve:
    """Best-fit model and residuals for a single star."""

    star_id: int
    times: List[float]
    fluxes: List[float]
    model_fluxes: List[float]
    residuals: List[float]
    mask: List[bool]
    degree: int
    coefficients: List[float]
    rms: float


@dataclass
class FittingResult:
    """Collection of fitted light curves keyed by star id."""

    curves: List[FittedLightCurve]
    by_id: Dict[int, FittedLightCurve]


def run_fitting(
    photometry: PhotometryResult,
    *,
    star_ids: Sequence[int] | None = None,
    degree: int = 1,
) -> FittingResult:
    """Fit low-order polynomials to photometric light curves."""

    if photometry is None or not photometry.light_curves:
        return FittingResult(curves=[], by_id={})

    if star_ids:
        selected_ids = [int(star_id) for star_id in star_ids]
    else:
        selected_ids = [curve.star_id for curve in photometry.light_curves]

    degree = int(max(0, degree))

    results: Dict[int, FittedLightCurve] = {}

    for star_id in selected_ids:
        base_curve: LightCurve | None = photometry.by_id.get(star_id)
        if base_curve is None:
            continue

        times = np.array(base_curve.times, dtype=float)
        fluxes = np.array(base_curve.fluxes, dtype=float)
        mask = np.array(base_curve.valid, dtype=bool)
        mask &= np.isfinite(times) & np.isfinite(fluxes)

        if mask.sum() <= degree:
            model = np.full_like(fluxes, np.nan, dtype=float)
            residuals = fluxes - model
            rms = float(np.nan)
            coeffs: np.ndarray | List[float] = np.array([], dtype=float)
        else:
            coeffs = np.polyfit(times[mask], fluxes[mask], deg=degree)
            model = np.polyval(coeffs, times)
            residuals = fluxes - model
            rms = float(np.sqrt(np.nanmean((residuals[mask]) ** 2))) if mask.any() else float(np.nan)

        fitted = FittedLightCurve(
            star_id=star_id,
            times=base_curve.times,
            fluxes=base_curve.fluxes,
            model_fluxes=model.tolist(),
            residuals=residuals.tolist(),
            mask=mask.tolist(),
            degree=degree,
            coefficients=np.array(coeffs, dtype=float).tolist(),
            rms=rms,
        )
        results[star_id] = fitted

    ordered = [results[star_id] for star_id in selected_ids if star_id in results]
    return FittingResult(curves=ordered, by_id=results)
