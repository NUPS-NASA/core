"""nups_core package: calibrated photometry and light-curve post-processing."""

from .config import (
    AlignmentConfig,
    CalibrationConfig,
    EnsembleConfig,
    ErrorConfig,
    GPConfig,
    OutputConfig,
    PathsConfig,
    PhotometryConfig,
    PipelineConfig,
)
from .pipeline import LightCurvePipeline, PipelineResult

__all__ = [
    "AlignmentConfig",
    "CalibrationConfig",
    "EnsembleConfig",
    "ErrorConfig",
    "GPConfig",
    "OutputConfig",
    "PathsConfig",
    "PhotometryConfig",
    "PipelineConfig",
    "LightCurvePipeline",
    "PipelineResult",
]
