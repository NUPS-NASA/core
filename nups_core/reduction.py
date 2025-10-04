"""Image reduction routines for NUPS Core."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy import units as u
from astropy.coordinates import Angle
from astropy.time import Time

from hops.hops_tools.image_analysis import (
    bin_frame,
    image_burn_limit,
    image_mean_std,
    image_psf,
)

from .option import NupsOption
from .util import FitsFrame


__all__ = [
    "FrameStatistics",
    "ReducedFrame",
    "ReductionContext",
    "ReductionResult",
    "reduction",
]


HOPS_OBSERVATORY_LAT_KEY = "SITELAT"
HOPS_OBSERVATORY_LONG_KEY = "SITELONG"
HOPS_TARGET_RA_KEY = "RA"
HOPS_TARGET_DEC_KEY = "DEC"
HOPS_DATETIME_KEY = "DATE-OBS"
HOPS_EXPOSURE_KEY = "EXPTIME"
HOPS_FILTER_KEY = "FILTER"
HOPS_TIME_KEY = "HOPSJD"
HOPS_AIRMASS_KEY = "AIRMASS"
HOPS_MEAN_KEY = "MEAN"
HOPS_STD_KEY = "STD"
HOPS_SATURATION_KEY = "SATUR"
HOPS_PSF_KEY = "PSF"
HOPS_SKIP_KEY = "SKIP"
HOPS_ALIGN_X0_KEY = "ALIGN_X0"
HOPS_ALIGN_Y0_KEY = "ALIGN_Y0"
HOPS_ALIGN_U0_KEY = "ALIGN_U0"


@dataclass
class FrameStatistics:
    """Summary statistics for a reduced frame."""

    mean: float
    std: float
    psf: float
    julian_date: float
    airmass: float
    exposure_time: float
    skip: bool
    saturation: float


@dataclass
class ReducedFrame:
    """Container for reduced data, header, and metadata."""

    name: str
    frame: FitsFrame
    stats: FrameStatistics


@dataclass
class ReductionContext:
    """Shared data required while reducing frames."""

    options: NupsOption
    location: str
    target_ra_dec: str
    master_bias: np.ndarray | float
    master_dark: np.ndarray | float
    master_dark_flat: np.ndarray | float
    master_flat: np.ndarray | float
    bias_exposure: float


@dataclass
class ReductionResult:
    """Final output of the reduction routine."""

    frames: List[ReducedFrame]
    statistics: Dict[str, FrameStatistics]


def reduction(
    bias_frames: Sequence[FitsFrame],
    dark_frames: Sequence[FitsFrame],
    flat_frames: Sequence[FitsFrame],
    science_frames: Sequence[FitsFrame],
    options: NupsOption,
    *,
    dark_flat_frames: Optional[Sequence[FitsFrame]] = None,
) -> ReductionResult:
    """Reduce science frames using the provided calibration frames and options."""

    dark_flat_frames = list(dark_flat_frames or [])

    if science_frames and not options.location_inited:
        options = options.update_location(science_frames[0])

    master_bias, bias_exposure = _calculate_master_bias(bias_frames, options)
    master_dark = _calculate_master_dark(dark_frames, master_bias, bias_exposure, options)
    master_dark_flat = _calculate_master_dark_flat(
        dark_flat_frames, master_bias, bias_exposure, master_dark, options
    )
    master_flat = _calculate_master_flat(flat_frames, master_bias, master_dark_flat, bias_exposure, options)

    context = ReductionContext(
        options=options,
        location=options.location,
        target_ra_dec=options.target_ra_dec,
        master_bias=master_bias,
        master_dark=master_dark,
        master_dark_flat=master_dark_flat,
        master_flat=master_flat,
        bias_exposure=bias_exposure,
    )

    reduced_frames: List[ReducedFrame] = []
    stats: Dict[str, FrameStatistics] = {}

    for index, frame in enumerate(science_frames):
        reduced = _reduce_science_frame(frame, index, context)
        reduced_frames.append(reduced)
        stats[reduced.name] = reduced.stats

    return ReductionResult(frames=reduced_frames, statistics=stats)

def _calculate_master_bias(
    frames: Sequence[FitsFrame], options: NupsOption
) -> Tuple[np.ndarray | float, float]:
    if not frames:
        return 0.0, 0.0

    exposures = _extract_exposure_times(frames, options.exposure_time_key)
    median_exposure = float(np.nanmedian(exposures)) if exposures.size else 0.0
    if exposures.size:
        keep_mask = np.isclose(exposures, median_exposure, atol=1e-6, rtol=0)
        selected_frames = [frame for frame, keep in zip(frames, keep_mask) if keep]
        if not selected_frames:
            selected_frames = list(frames)
    else:
        selected_frames = list(frames)

    stack = np.stack([np.array(frame.data, dtype=float) for frame in selected_frames])
    master = _reduce_stack(stack, options.master_bias_method)
    return master, median_exposure


def _calculate_master_dark(
    frames: Sequence[FitsFrame],
    master_bias: np.ndarray | float,
    bias_exposure: float,
    options: NupsOption,
) -> np.ndarray | float:
    if not frames:
        return 0.0

    corrected = []
    for frame in frames:
        exp_time = _safe_float(frame.header.get(options.exposure_time_key, 0.0))
        scale = exp_time - bias_exposure
        if abs(scale) < 1e-9:
            continue
        corrected.append((np.array(frame.data, dtype=float) - master_bias) / scale)

    if not corrected:
        return 0.0

    stack = np.stack(corrected)
    return _reduce_stack(stack, options.master_dark_method)


def _calculate_master_dark_flat(
    frames: Sequence[FitsFrame],
    master_bias: np.ndarray | float,
    bias_exposure: float,
    master_dark: np.ndarray | float,
    options: NupsOption,
) -> np.ndarray | float:
    if not frames:
        return master_dark

    corrected = []
    for frame in frames:
        exp_time = _safe_float(frame.header.get(options.exposure_time_key, 0.0))
        scale = exp_time - bias_exposure
        if abs(scale) < 1e-9:
            continue
        corrected.append((np.array(frame.data, dtype=float) - master_bias) / scale)

    if not corrected:
        return master_dark

    stack = np.stack(corrected)
    return _reduce_stack(stack, options.master_darkf_method)


def _calculate_master_flat(
    frames: Sequence[FitsFrame],
    master_bias: np.ndarray | float,
    master_dark_flat: np.ndarray | float,
    bias_exposure: float,
    options: NupsOption,
) -> np.ndarray | float:
    if not frames:
        return 1.0

    corrected = []
    for frame in frames:
        exp_time = _safe_float(frame.header.get(options.exposure_time_key, 0.0))
        scale = exp_time - bias_exposure
        corrected_frame = (
            np.array(frame.data, dtype=float)
            - master_bias
            - scale * master_dark_flat
        )
        corrected.append(corrected_frame)

    if not corrected:
        return 1.0

    normalized = []
    for item in corrected:
        reference = np.nanmedian(item) if options.master_flat_method != "mean" else np.nanmean(item)
        if reference == 0:
            reference = 1.0
        normalized.append(item / reference)

    stack = np.stack(normalized)
    master_flat = _reduce_stack(stack, options.master_flat_method)

    if options.colour_camera_mode:
        master_flat = _normalise_colour_master_flat(master_flat)
    else:
        median = np.nanmedian(master_flat)
        master_flat = master_flat / median if median else master_flat

    master_flat = np.where(master_flat == 0, 1, master_flat)
    return master_flat


def _reduce_science_frame(frame: FitsFrame, index: int, context: ReductionContext) -> ReducedFrame:
    options = context.options

    data_frame = np.array(frame.data, dtype=float)
    original_height, original_width = data_frame.shape
    dq_frame = np.zeros_like(data_frame)

    saturation = _safe_float(image_burn_limit(frame.header, key=HOPS_SATURATION_KEY))
    exp_time = _safe_float(frame.header.get(options.exposure_time_key))

    dq_frame = np.where(data_frame == saturation, 1.0, 0.0)

    data_frame = (
        data_frame
        - context.master_bias
        - (exp_time - context.bias_exposure) * context.master_dark
    ) / context.master_flat
    data_frame = np.where(np.isnan(data_frame), 0.0, data_frame)

    crop_x1 = int(max(0, options.crop_x1))
    crop_x2 = int(options.crop_x2) if options.crop_x2 else original_width
    crop_y1 = int(max(0, options.crop_y1))
    crop_y2 = int(options.crop_y2) if options.crop_y2 else original_height

    data_frame = data_frame[crop_y1:crop_y2, crop_x1:crop_x2]
    dq_frame = dq_frame[crop_y1:crop_y2, crop_x1:crop_x2]

    crop_edge = int(options.crop_edge_pixels)
    if crop_edge > 0 and min(data_frame.shape) > 2 * crop_edge:
        data_frame = data_frame[crop_edge:-crop_edge, crop_edge:-crop_edge]
        dq_frame = dq_frame[crop_edge:-crop_edge, crop_edge:-crop_edge]

    bin_factor = int(max(1, options.bin_fits))
    if bin_factor > 1:
        data_frame = bin_frame(data_frame, bin_factor)
        dq_frame = bin_frame(dq_frame, bin_factor)
        dq_frame = np.where(dq_frame > 0, 1.0, 0.0)
        saturation = saturation * bin_factor * bin_factor

    data_frame = np.where(dq_frame > 0, saturation, data_frame)

    mean, std = image_mean_std(data_frame, samples=10000, mad_filter=5.0)
    psf = image_psf(
        data_frame,
        frame.header,
        mean,
        std,
        0.8 * saturation,
        centroids_snr=options.centroids_snr,
        stars_snr=options.stars_snr,
        psf_guess=options.psf_guess,
    )
    skip = np.isnan(psf)
    if skip:
        psf = 10

    observation_time = _compute_observation_time(frame.header, exp_time, options)
    julian_date = observation_time.jd
    airmass = math.nan

    header = dict(frame.header)
    header.update(
        {
            "BITPIX": frame.header.get("BITPIX"),
            "NAXIS1": original_width,
            "NAXIS2": original_height,
            "XBINNING": bin_factor,
            "YBINNING": bin_factor,
            "BZERO": 0,
            "BSCALE": 1,
            HOPS_OBSERVATORY_LAT_KEY: _format_location_component(context.location, 0),
            HOPS_OBSERVATORY_LONG_KEY: _format_location_component(context.location, 1),
            HOPS_TARGET_RA_KEY: _format_target_component(context.target_ra_dec, 0),
            HOPS_TARGET_DEC_KEY: _format_target_component(context.target_ra_dec, 1),
            HOPS_DATETIME_KEY: observation_time.utc.isot,
            HOPS_EXPOSURE_KEY: exp_time,
            HOPS_FILTER_KEY: options.filter,
            HOPS_TIME_KEY: julian_date,
            HOPS_AIRMASS_KEY: airmass,
            HOPS_MEAN_KEY: mean,
            HOPS_STD_KEY: std,
            HOPS_SATURATION_KEY: saturation,
            HOPS_PSF_KEY: psf,
            HOPS_SKIP_KEY: skip,
            HOPS_ALIGN_X0_KEY: False,
            HOPS_ALIGN_Y0_KEY: False,
            HOPS_ALIGN_U0_KEY: False,
        }
    )

    data_to_store = np.array(data_frame, dtype=np.int32)
    reduced_frame = FitsFrame(data=data_to_store, header=header, source=_derive_source_path(frame.source))

    name = _build_output_name(frame, observation_time, options, index)

    stats = FrameStatistics(
        mean=mean,
        std=std,
        psf=psf,
        julian_date=julian_date,
        airmass=airmass,
        exposure_time=exp_time,
        skip=skip,
        saturation=saturation,
    )

    return ReducedFrame(name=name, frame=reduced_frame, stats=stats)

def _compute_observation_time(
    header: Dict[str, object],
    exposure_time: float,
    options: NupsOption,
) -> Time:
    date_value = header.get(options.observation_date_key)
    if date_value is None:
        raise KeyError(f"Header missing {options.observation_date_key}")

    if options.observation_date_key == options.observation_time_key:
        datetime_str = " ".join(str(date_value).split("T"))
    else:
        time_value = header.get(options.observation_time_key)
        if time_value is None:
            raise KeyError(f"Header missing {options.observation_time_key}")
        datetime_str = " ".join([str(date_value).split("T")[0], str(time_value)])

    datetime_str = datetime_str.strip().replace(" ", "T")
    try:
        observation_time = Time(datetime_str, format="isot", scale="utc")
    except ValueError:
        observation_time = Time(datetime_str, format="iso", scale="utc")

    exposure_seconds = float(exposure_time) if exposure_time else 0.0
    delta = exposure_seconds * u.second

    if options.time_stamp == "mid-exposure":
        observation_time = observation_time - 0.5 * delta
    elif options.time_stamp == "exposure end":
        observation_time = observation_time - delta
    elif options.time_stamp != "exposure start":
        raise ValueError(f"Unsupported time_stamp: {options.time_stamp}")

    return observation_time


def _build_output_name(
    frame: FitsFrame,
    observation_time: Time,
    options: NupsOption,
    index: int,
) -> str:
    base_name = frame.source.name if frame.source else f"frame_{index:05d}.fits"
    time_stamp = observation_time.utc.isot.split(".")[0]
    time_stamp = time_stamp.replace("-", "_").replace(":", "_").replace("T", "_")
    return f"{options.reduction_prefix}{time_stamp}_{base_name}"


def _format_location_component(location: str, index: int) -> object:
    components = _split_components(location)
    if index < len(components):
        value = components[index]
        parsed = _parse_angle(value, u.deg)
        return float(parsed) if parsed is not None else value
    return ""


def _format_target_component(target: str, index: int) -> object:
    components = _split_components(target)
    if index < len(components):
        value = components[index]
        if index == 0:
            parsed = _parse_angle(value, u.hourangle)
        else:
            parsed = _parse_angle(value, u.deg)
        return float(parsed) if parsed is not None else value
    return ""


def _split_components(value: str) -> List[str]:
    if not value:
        return []
    return [component for component in value.split() if component]


def _parse_angle(value: str, unit: u.UnitBase) -> Optional[float]:
    try:
        angle = Angle(value, unit=unit)
        return angle.degree
    except (ValueError, TypeError):
        return None


def _reduce_stack(stack: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        return np.nanmean(stack, axis=0)
    return np.nanmedian(stack, axis=0)


def _normalise_colour_master_flat(master_flat: np.ndarray) -> np.ndarray:
    r = master_flat[::2, ::2]
    g = master_flat[::2, 1::2]
    b = master_flat[1::2, ::2]
    y = master_flat[1::2, 1::2]

    r /= np.nanmedian(r) if np.nanmedian(r) else 1
    g /= np.nanmedian(g) if np.nanmedian(g) else 1
    b /= np.nanmedian(b) if np.nanmedian(b) else 1
    y /= np.nanmedian(y) if np.nanmedian(y) else 1

    master_flat[::2, ::2] = r
    master_flat[::2, 1::2] = g
    master_flat[1::2, ::2] = b
    master_flat[1::2, 1::2] = y

    return master_flat


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_exposure_times(frames: Sequence[FitsFrame], key: str) -> np.ndarray:
    values = [
        _safe_float(frame.header.get(key))
        for frame in frames
    ]
    return np.array(values, dtype=float)


def _derive_source_path(path: Optional[Path]) -> Optional[Path]:
    return Path(path) if path else None
