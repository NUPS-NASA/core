"""Utility helpers for NUPS library workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from astropy.io import fits as pf


__all__ = [
    "FitsFrame",
    "ReductionFiles",
    "find_fits_paths",
    "load_fits_frame",
    "load_fits_frames",
    "load_reduction_files",
]


@dataclass
class FitsFrame:
    """Container for FITS image data."""

    data: np.ndarray
    header: Dict[str, Any]
    source: Optional[Path] = None


@dataclass
class ReductionFiles:
    """Collections of frames used in the reduction step."""

    bias: List[FitsFrame]
    dark: List[FitsFrame]
    flat: List[FitsFrame]
    science: List[FitsFrame]
    dark_flat: List[FitsFrame]


def find_fits_paths(name_identifier: str, base_dir: Path) -> List[Path]:
    """Return sorted FITS paths matching the given identifier under ``base_dir``."""

    if not name_identifier:
        return []

    candidates: List[Path] = []
    patterns = [f"*{name_identifier}*.f*t*", f"*{name_identifier}*.F*T*"]
    for pattern in patterns:
        candidates.extend(path for path in base_dir.glob(pattern) if path.is_file())

    unique_paths = sorted({path.resolve() for path in candidates})
    return unique_paths


def load_fits_frame(path: Path) -> FitsFrame:
    """Load a FITS frame from ``path`` into memory."""

    with pf.open(path, memmap=False) as hdul:
        hdul.verify("fix")

        data: Optional[np.ndarray] = None
        header: Dict[str, Any] = {}

        if "SCI" in hdul:
            data = np.array(hdul["SCI"].data, dtype=float)
            header = {key: hdul["SCI"].header[key] for key in hdul["SCI"].header}
        else:
            for hdu in hdul:
                if getattr(hdu, "data", None) is None:
                    continue
                data = np.array(hdu.data, dtype=float)
                header = {key: hdu.header[key] for key in hdu.header if key not in {"HISTORY", "COMMENT", ""}}
                break

        if data is None:
            raise ValueError(f"No image data found in {path}")

        data = np.where(np.isnan(data), 1.0, data)
        data = np.where(data == 0, 1.0, data)

    return FitsFrame(data=data, header=header, source=path)


def load_fits_frames(paths: Sequence[Path]) -> List[FitsFrame]:
    """Load each FITS file in ``paths``."""

    return [load_fits_frame(path) for path in paths]


def load_reduction_files(base_dir: Path, options: "NupsOption") -> ReductionFiles:
    """Load all reduction input files described by ``options``."""

    from .option import NupsOption  # local import to avoid cycles

    if not isinstance(options, NupsOption):
        raise TypeError("options must be an instance of NupsOption")

    bias_paths = find_fits_paths(options.bias_files, base_dir)
    dark_paths = find_fits_paths(options.dark_files, base_dir)
    dark_flat_paths = find_fits_paths(options.darkf_files, base_dir)
    flat_paths = find_fits_paths(options.flat_files, base_dir)
    science_paths = find_fits_paths(options.observation_files, base_dir)

    bias_frames = load_fits_frames(bias_paths)
    dark_frames = load_fits_frames(dark_paths)
    flat_frames = load_fits_frames(flat_paths)
    science_frames = load_fits_frames(science_paths)
    dark_flat_frames = load_fits_frames(dark_flat_paths)

    science_frames = _sort_by_time(science_frames, options, allow_fallback=True)

    return ReductionFiles(
        bias=bias_frames,
        dark=dark_frames,
        flat=flat_frames,
        science=science_frames,
        dark_flat=dark_flat_frames,
    )


def _sort_by_time(
    frames: List[FitsFrame],
    options: "NupsOption",
    *,
    allow_fallback: bool = False,
) -> List[FitsFrame]:
    if not frames:
        return frames

    def _key(frame: FitsFrame) -> Tuple[int, Any]:
        header = frame.header
        try:
            date_value = header.get(options.observation_date_key)
            time_value = header.get(options.observation_time_key)
            if date_value is None:
                raise KeyError
            if options.observation_date_key == options.observation_time_key:
                timestamp = str(date_value)
            else:
                if time_value is None:
                    raise KeyError
                timestamp = f"{str(date_value).split('T')[0]} {time_value}"
            return (0, timestamp)
        except KeyError:
            return (1 if allow_fallback else 0, frame.source.name if frame.source else "")

    return sorted(frames, key=_key)
