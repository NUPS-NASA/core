"""Configuration dataclasses for the nups_core light-curve pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class PathsConfig:
    """Describe the input/output layout for a single observing run."""

    root_dir: Path
    output_dir: Path
    light_prefix: str = "object"
    bias_prefix: Optional[str] = "bias"
    dark_prefix: Optional[str] = "dark"
    flat_prefix: Optional[str] = "flat"
    aligned_dirname: str = "aligned_fits"
    raw_plot_dirname: str = "plots_allstars_lc"
    detrended_plot_dirname: str = "plots_allstars_lc_detrended"
    raw_wide_csv_name: str = "allstars_relflux_wide.csv"
    detrended_wide_csv_name: str = "allstars_relflux_detrended_wide.csv"
    times_csv_name: str = "times_jd.csv"
    detection_preview_name: str = "detected_stars_preview.png"

    def __post_init__(self) -> None:
        self.root_dir = Path(self.root_dir)
        self.output_dir = Path(self.output_dir)

    def _resolve(self, prefix: Optional[str]) -> Optional[Path]:
        if prefix is None:
            return None
        candidate = Path(prefix)
        if not candidate.is_absolute():
            candidate = self.root_dir / candidate
        return candidate

    @property
    def light_dir(self) -> Path:
        path = self._resolve(self.light_prefix)
        if path is None:
            raise ValueError("light_prefix cannot be None")
        return path

    @property
    def bias_dir(self) -> Optional[Path]:
        return self._resolve(self.bias_prefix)

    @property
    def dark_dir(self) -> Optional[Path]:
        return self._resolve(self.dark_prefix)

    @property
    def flat_dir(self) -> Optional[Path]:
        return self._resolve(self.flat_prefix)

    @property
    def aligned_dir(self) -> Path:
        return self.output_dir / self.aligned_dirname

    @property
    def raw_plot_dir(self) -> Path:
        return self.output_dir / self.raw_plot_dirname

    @property
    def detrended_plot_dir(self) -> Path:
        return self.output_dir / self.detrended_plot_dirname

    @property
    def raw_wide_csv_path(self) -> Path:
        return self.output_dir / self.raw_wide_csv_name

    @property
    def detrended_wide_csv_path(self) -> Path:
        return self.output_dir / self.detrended_wide_csv_name

    @property
    def times_csv_path(self) -> Path:
        return self.output_dir / self.times_csv_name

    @property
    def detection_preview_path(self) -> Path:
        return self.output_dir / self.detection_preview_name


@dataclass(slots=True)
class CalibrationConfig:
    use_bias: bool = True
    use_dark: bool = True
    use_flat: bool = True


@dataclass(slots=True)
class AlignmentConfig:
    enabled: bool = True
    save_aligned_fits: bool = True
    detection_sigma: float = 3.0
    max_control_points: int = 50


@dataclass(slots=True)
class PhotometryConfig:
    fwhm_pix: float = 3.5
    thresh_sigma: float = 5.0
    max_stars_detect: int = 2000
    edge_margin: int = 12
    aperture_scale: float = 3.0
    annulus_in_scale: float = 6.0
    annulus_out_scale: float = 10.0
    min_separation_scale: float = 3.0
    detection_preview_limit: int = 100
    min_valid_ratio: float = 0.5

    def __post_init__(self) -> None:
        if self.min_valid_ratio <= 0.0 or self.min_valid_ratio > 1.0:
            raise ValueError("min_valid_ratio must be in (0, 1]")


@dataclass(slots=True)
class EnsembleConfig:
    bright_tolerance: float = 0.30
    rms_k: int = 20
    min_comps: int = 5
    clip_sigma: float = 4.0

    def __post_init__(self) -> None:
        if self.bright_tolerance <= 0:
            raise ValueError("bright_tolerance must be positive")
        if self.rms_k < 1:
            raise ValueError("rms_k must be >= 1")
        if self.min_comps < 1:
            raise ValueError("min_comps must be >= 1")
        if self.clip_sigma <= 0:
            raise ValueError("clip_sigma must be positive")


@dataclass(slots=True)
class ErrorConfig:
    gain_e_per_adu: Optional[float] = None
    error_floor: float = 1e-6


@dataclass(slots=True)
class GPConfig:
    enabled: bool = False
    samples: int = 500
    warmup: int = 300
    chains: int = 1
    seed: Optional[int] = None
    star_limit: Optional[int] = None
    time_unit: str = "days"


@dataclass(slots=True)
class OutputConfig:
    save_detection_preview: bool = True
    save_raw_wide_csv: bool = True
    save_times_csv: bool = True
    save_detrended_plots: bool = True
    save_detrended_csv: bool = True


@dataclass(slots=True)
class PipelineConfig:
    paths: PathsConfig
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    photometry: PhotometryConfig = field(default_factory=PhotometryConfig)
    ensemble: EnsembleConfig = field(default_factory=EnsembleConfig)
    errors: ErrorConfig = field(default_factory=ErrorConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    verbose: bool = True

    def copy_with_output_dir(self, output_dir: Path) -> "PipelineConfig":
        cfg = PipelineConfig(
            paths=PathsConfig(
                root_dir=self.paths.root_dir,
                output_dir=output_dir,
                light_prefix=self.paths.light_prefix,
                bias_prefix=self.paths.bias_prefix,
                dark_prefix=self.paths.dark_prefix,
                flat_prefix=self.paths.flat_prefix,
                aligned_dirname=self.paths.aligned_dirname,
                raw_plot_dirname=self.paths.raw_plot_dirname,
                detrended_plot_dirname=self.paths.detrended_plot_dirname,
                raw_wide_csv_name=self.paths.raw_wide_csv_name,
                detrended_wide_csv_name=self.paths.detrended_wide_csv_name,
                times_csv_name=self.paths.times_csv_name,
                detection_preview_name=self.paths.detection_preview_name,
            ),
            calibration=self.calibration,
            alignment=self.alignment,
            photometry=self.photometry,
            ensemble=self.ensemble,
            errors=self.errors,
            gp=self.gp,
            output=self.output,
            verbose=self.verbose,
        )
        return cfg
