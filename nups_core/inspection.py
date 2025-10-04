"""Star detection routines for the NUPS CLI pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from astropy.stats import sigma_clipped_stats
from scipy.ndimage import find_objects, label, maximum_filter

from .util import FitsFrame


__all__ = ["DetectedStar", "detect_stars", "inspect_frames"]


@dataclass
class DetectedStar:
    """Description of a star detected on a frame."""

    x: float
    y: float
    flux: float
    peak: float


def detect_stars(
    data: np.ndarray,
    *,
    threshold_sigma: float = 5.0,
    min_separation: int = 7,
    max_stars: int = 150,
    margin: int = 5,
) -> List[DetectedStar]:
    """Detect bright stars in ``data`` using a sigma threshold."""

    if data.size == 0:
        return []

    arr = np.array(data, dtype=float, copy=False)
    if not np.isfinite(arr).any():
        return []

    arr = np.nan_to_num(arr, copy=False)

    _, median, std_dev = sigma_clipped_stats(arr, sigma=3.0)
    if not np.isfinite(std_dev) or std_dev <= 0:
        std_dev = float(np.std(arr)) or 1.0
    threshold = median + threshold_sigma * std_dev

    separation = max(3, int(round(min_separation)))
    size = separation if separation % 2 else separation + 1
    max_filtered = maximum_filter(arr, size=size, mode="nearest")
    peak_mask = (arr >= threshold) & (arr == max_filtered)

    if not np.any(peak_mask):
        return []

    structure = np.ones((3, 3), dtype=int)
    labels, num_features = label(peak_mask, structure=structure)
    if num_features == 0:
        return []

    objects = find_objects(labels)

    stars: List[DetectedStar] = []
    height, width = arr.shape
    for label_id, slc in enumerate(objects, start=1):
        if slc is None:
            continue

        region = arr[slc]
        region_mask = labels[slc] == label_id
        if not np.any(region_mask):
            continue

        masked_values = region[region_mask]
        peak_value = float(np.max(masked_values))

        # Sub-pixel centroid using intensity-weighted average inside region.
        weights = masked_values - median
        weights = np.where(weights > 0, weights, 0)
        yy, xx = np.indices(region.shape)
        total_weight = float(np.sum(weights))
        if total_weight == 0:
            coords = np.argwhere(region_mask)
            cy, cx = coords.mean(axis=0)
        else:
            cy = float(np.sum(weights * yy[region_mask]) / total_weight)
            cx = float(np.sum(weights * xx[region_mask]) / total_weight)

        y_pos = slc[0].start + cy
        x_pos = slc[1].start + cx

        refine_radius = max(2, separation // 2)
        x_pos, y_pos, refined_flux, refined_peak = _refine_centroid(
            arr, x_pos, y_pos, radius=refine_radius
        )
        if refined_peak > 0:
            peak_value = max(peak_value, refined_peak)

        if (
            x_pos <= margin
            or y_pos <= margin
            or x_pos >= (width - margin)
            or y_pos >= (height - margin)
        ):
            continue

        total_flux = refined_flux if refined_flux > 0 else float(np.sum(masked_values - median))
        if total_flux <= 0:
            continue

        stars.append(DetectedStar(x=x_pos, y=y_pos, flux=total_flux, peak=peak_value))

    stars.sort(key=lambda star: star.flux, reverse=True)
    return stars[:max_stars]


def inspect_frames(
    frames: Sequence[FitsFrame | np.ndarray],
    *,
    threshold_sigma: float = 5.0,
    min_separation: int = 7,
    max_stars: int = 150,
    margin: int = 5,
) -> List[List[DetectedStar]]:
    """Inspect ``frames`` and return detected star catalogs."""

    catalogs: List[List[DetectedStar]] = []
    for frame in frames:
        data = _extract_data(frame)
        stars = detect_stars(
            data,
            threshold_sigma=threshold_sigma,
            min_separation=min_separation,
            max_stars=max_stars,
            margin=margin,
        )
        catalogs.append(stars)
    return catalogs




def _refine_centroid(arr: np.ndarray, x: float, y: float, *, radius: int) -> tuple[float, float, float, float]:
    height, width = arr.shape
    radius = max(1, int(radius))
    xi = int(round(x))
    yi = int(round(y))

    y_min = max(0, yi - radius)
    y_max = min(height, yi + radius + 1)
    x_min = max(0, xi - radius)
    x_max = min(width, xi + radius + 1)

    patch = arr[y_min:y_max, x_min:x_max]
    if patch.size == 0:
        return x, y, 0.0, 0.0

    background = float(np.median(patch))
    weights = patch - background
    weights = np.where(weights > 0, weights, 0)
    weight_sum = float(weights.sum())
    peak_value = float(patch.max()) if patch.size else 0.0

    if weight_sum <= 0:
        return x, y, 0.0, peak_value

    yy, xx = np.indices(patch.shape)
    x_coords = x_min + xx
    y_coords = y_min + yy

    refined_x = float((weights * x_coords).sum() / weight_sum)
    refined_y = float((weights * y_coords).sum() / weight_sum)

    return refined_x, refined_y, weight_sum, peak_value


def _extract_data(frame: FitsFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame, FitsFrame):
        data = np.array(frame.data, dtype=float, copy=True)
    else:
        data = np.array(frame, dtype=float, copy=True)
    return np.nan_to_num(data, copy=False)
