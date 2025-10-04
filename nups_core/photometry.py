"""Aperture photometry utilities for NUPS CLI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np

from .reduction import HOPS_TIME_KEY
from .tagging import TaggedStar
from .util import FitsFrame

__all__ = ["LightCurve", "PhotometryResult", "run_photometry"]


@dataclass
class LightCurve:
    """Photometric measurements for a single star."""

    star_id: int
    times: List[float]
    fluxes: List[float]
    backgrounds: List[float]
    valid: List[bool]


@dataclass
class PhotometryResult:
    """Container for photometry outputs keyed by star id."""

    light_curves: List[LightCurve]
    by_id: Dict[int, LightCurve]


def run_photometry(
    frames: Sequence[FitsFrame],
    star_catalogs: Sequence[Sequence[TaggedStar]],
    *,
    aperture_radius: float = 3.5,
    annulus_inner: float = 6.0,
    annulus_outer: float = 8.0,
    time_key: str = HOPS_TIME_KEY,
) -> PhotometryResult:
    if not frames or not star_catalogs or len(frames) != len(star_catalogs):
        return PhotometryResult(light_curves=[], by_id={})

    star_ids = sorted({star.id for catalog in star_catalogs for star in catalog if star.id is not None})
    if not star_ids:
        return PhotometryResult(light_curves=[], by_id={})

    curves: Dict[int, LightCurve] = {
        star_id: LightCurve(star_id=star_id, times=[], fluxes=[], backgrounds=[], valid=[])
        for star_id in star_ids
    }

    aperture_radius = float(max(1.0, aperture_radius))
    annulus_inner = float(max(aperture_radius + 0.5, annulus_inner))
    annulus_outer = float(max(annulus_inner + 0.5, annulus_outer))

    for index, (frame, catalog) in enumerate(zip(frames, star_catalogs)):
        arr = np.array(frame.data, dtype=float, copy=True)
        arr = np.nan_to_num(arr, copy=False)

        time_value = _extract_time(frame, time_key, index)
        star_map = {star.id: star for star in catalog if star.id is not None}

        for star_id, curve in curves.items():
            star = star_map.get(star_id)
            if star is None:
                curve.times.append(time_value)
                curve.fluxes.append(float("nan"))
                curve.backgrounds.append(float("nan"))
                curve.valid.append(False)
                continue

            flux, background = _aperture_photometry(
                arr,
                star.x,
                star.y,
                aperture_radius=aperture_radius,
                annulus_inner=annulus_inner,
                annulus_outer=annulus_outer,
            )

            curve.times.append(time_value)
            curve.fluxes.append(flux)
            curve.backgrounds.append(background)
            curve.valid.append(np.isfinite(flux))

    ordered_curves = [curves[star_id] for star_id in star_ids]
    return PhotometryResult(light_curves=ordered_curves, by_id=curves)


def _extract_time(frame: FitsFrame, key: str, index: int) -> float:
    value = frame.header.get(key)
    if value is None:
        return float(index)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(index)


def _aperture_photometry(
    data: np.ndarray,
    x: float,
    y: float,
    *,
    aperture_radius: float,
    annulus_inner: float,
    annulus_outer: float,
) -> tuple[float, float]:
    height, width = data.shape
    x0 = float(x)
    y0 = float(y)
    if not (0 <= x0 < width and 0 <= y0 < height):
        return float("nan"), float("nan")

    max_radius = annulus_outer
    x_min = max(0, int(np.floor(x0 - max_radius - 1)))
    x_max = min(width, int(np.ceil(x0 + max_radius + 1)))
    y_min = max(0, int(np.floor(y0 - max_radius - 1)))
    y_max = min(height, int(np.ceil(y0 + max_radius + 1)))

    patch = data[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return float("nan"), float("nan")

    yy, xx = np.indices(patch.shape, dtype=float)
    xx = xx + x_min + 0.5
    yy = yy + y_min + 0.5

    r = np.sqrt((xx - x0) ** 2 + (yy - y0) ** 2)

    aperture_mask = r <= aperture_radius
    annulus_mask = (r >= annulus_inner) & (r <= annulus_outer)

    aperture_values = patch[aperture_mask]
    if aperture_values.size == 0:
        return float("nan"), float("nan")

    annulus_values = patch[annulus_mask]
    if annulus_values.size >= 5:
        background = float(np.median(annulus_values))
    else:
        background = float(np.median(patch[~aperture_mask])) if (~aperture_mask).any() else float(np.median(patch))

    if not np.isfinite(background):
        background = 0.0

    flux = float(np.sum(aperture_values - background))
    return flux, background
