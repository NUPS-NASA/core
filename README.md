# core light-curve pipeline

This repository packages the photometric light-curve workflow that previously lived in the `Almostly_final (2).ipynb` notebook.  The code is now structured as the importable package **`nups_core`**, exposing a configurable `LightCurvePipeline` class and a set of dataclasses that describe the processing parameters.

## Installation

The pipeline depends on the scientific Python stack used in the original notebook:

- `numpy`
- `pandas`
- `astropy`
- `photutils`
- `astroalign`
- `matplotlib`

Create a virtual environment and install the requirements with `pip`:

```bash
python -m venv .venv
source .venv/bin/activate
pip install numpy pandas astropy photutils astroalign matplotlib
```

## Running the pipeline

Edit `Almostly_final.py` (or create your own runner) to point to the desired dataset.  Then execute:

```bash
python Almostly_final.py
```

The script instantiates `LightCurvePipeline` with the supplied configuration, executes the end-to-end calibration → alignment → photometry → ensemble detrending workflow, and writes the outputs to the configured `output_dir`.

### Programmatic use

```python
from nups_core import (
    PathsConfig,
    CalibrationConfig,
    AlignmentConfig,
    PhotometryConfig,
    EnsembleConfig,
    OutputConfig,
    PipelineConfig,
    LightCurvePipeline,
)

config = PipelineConfig(
    paths=PathsConfig(
        root_dir="/path/to/observation",
        output_dir="./pipeline_output",
    ),
)

result = LightCurvePipeline(config).run()
print(result.flux_matrix.shape, result.detrended_flux.shape)
```

The `PipelineResult` object captures the flux matrices, detrended series, star positions, covariates, and any artefacts saved to disk.

## Outputs

The pipeline mirrors the notebook artefacts:

- Calibrated and (optionally) aligned FITS frames
- Detection preview image with apertures/annuli overlays
- Raw flux matrix CSV + time stamps
- Detrended relative-flux CSV
- Per-star detrended plots

Adjust the `OutputConfig` flags to enable or disable specific artefacts.
