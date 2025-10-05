#!/usr/bin/env python
"""CLI wrapper around the :mod:`nups_core` light-curve pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from nups_core import (
    AlignmentConfig,
    CalibrationConfig,
    EnsembleConfig,
    ErrorConfig,
    GPConfig,
    LightCurvePipeline,
    OutputConfig,
    PathsConfig,
    PhotometryConfig,
    PipelineConfig,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the NUPS light-curve reduction pipeline on a FITS dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root-dir",
        type=Path,
        default=Path("./WASP-12b data"),
        help="Root directory that holds light/bias/dark/flat subdirectories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./WASP-12b_Basic_out5"),
        help="Directory to write pipeline artefacts (created if missing).",
    )
    parser.add_argument("--light-prefix", default="object", help="Subdirectory for science frames (relative to root-dir if not absolute).")
    parser.add_argument("--bias-prefix", default="bias", help="Subdirectory for bias frames; use '--no-bias' to skip bias calibration.")
    parser.add_argument("--dark-prefix", default="dark", help="Subdirectory for dark frames; use '--no-dark' to skip dark calibration.")
    parser.add_argument("--flat-prefix", default="flat", help="Subdirectory for flat frames; use '--no-flat' to skip flat calibration.")

    parser.add_argument("--no-bias", action="store_true", help="Disable bias calibration regardless of prefix.")
    parser.add_argument("--no-dark", action="store_true", help="Disable dark calibration regardless of prefix.")
    parser.add_argument("--no-flat", action="store_true", help="Disable flat calibration regardless of prefix.")

    parser.add_argument("--no-alignment", action="store_true", help="Skip astroalign frame registration.")
    parser.add_argument("--no-save-aligned", action="store_true", help="Do not persist aligned FITS frames.")
    parser.add_argument("--skip-detection-preview", action="store_true", help="Do not generate the detection preview PNG.")
    parser.add_argument("--skip-raw-csv", action="store_true", help="Do not save raw flux matrix CSV.")
    parser.add_argument("--skip-times-csv", action="store_true", help="Do not save extracted time stamps CSV.")
    parser.add_argument("--skip-detrended-plots", action="store_true", help="Do not save per-star detrended plots.")
    parser.add_argument("--skip-detrended-csv", action="store_true", help="Do not save detrended relative-flux CSV.")

    parser.add_argument("--gp-samples", type=int, default=400, help="Number of draw samples if GP modelling is enabled.")
    parser.add_argument("--gp-warmup", type=int, default=400, help="Number of warmup steps for the GP sampler.")
    parser.add_argument("--gp-chains", type=int, default=1, help="Number of GP chains.")
    parser.add_argument("--gp-seed", type=int, default=42, help="Random seed for the GP sampler.")
    parser.add_argument("--gp-star-limit", type=int, default=None, help="Optional cap on stars to run through GP modelling.")
    parser.add_argument("--no-gp", action="store_true", help="Disable GP modelling stage.")

    parser.add_argument("--verbose", action="store_true", help="Increase logging verbosity from the pipeline.")

    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> PipelineConfig:
    root_dir = args.root_dir.expanduser()
    output_dir = args.output_dir.expanduser()

    if not root_dir.exists():
        raise FileNotFoundError(f"Root directory not found: {root_dir}")

    paths = PathsConfig(
        root_dir=root_dir,
        output_dir=output_dir,
        light_prefix=args.light_prefix,
        bias_prefix=None if args.no_bias else args.bias_prefix,
        dark_prefix=None if args.no_dark else args.dark_prefix,
        flat_prefix=None if args.no_flat else args.flat_prefix,
    )

    calibration = CalibrationConfig(
        use_bias=not args.no_bias and args.bias_prefix is not None,
        use_dark=not args.no_dark and args.dark_prefix is not None,
        use_flat=not args.no_flat and args.flat_prefix is not None,
    )

    alignment = AlignmentConfig(
        enabled=not args.no_alignment,
        save_aligned_fits=not args.no_save_aligned,
    )

    photometry = PhotometryConfig(
        fwhm_pix=3.5,
        thresh_sigma=5.0,
        max_stars_detect=2000,
        edge_margin=12,
        aperture_scale=3.0,
        annulus_in_scale=6.0,
        annulus_out_scale=10.0,
        min_separation_scale=3.0,
        detection_preview_limit=1000,
        min_valid_ratio=0.5,
    )

    ensemble = EnsembleConfig(
        bright_tolerance=0.25,
        rms_k=20,
        min_comps=5,
    )

    errors_cfg = ErrorConfig(
        gain_e_per_adu=None,
        error_floor=1e-6,
    )

    gp_cfg = GPConfig(
        enabled=not args.no_gp,
        samples=args.gp_samples,
        warmup=args.gp_warmup,
        chains=args.gp_chains,
        seed=args.gp_seed,
        star_limit=args.gp_star_limit,
        time_unit="days",
    )

    output_cfg = OutputConfig(
        save_detection_preview=not args.skip_detection_preview,
        save_raw_wide_csv=not args.skip_raw_csv,
        save_times_csv=not args.skip_times_csv,
        save_detrended_plots=not args.skip_detrended_plots,
        save_detrended_csv=not args.skip_detrended_csv,
    )

    return PipelineConfig(
        paths=paths,
        calibration=calibration,
        alignment=alignment,
        photometry=photometry,
        ensemble=ensemble,
        errors=errors_cfg,
        gp=gp_cfg,
        output=output_cfg,
        verbose=args.verbose,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = build_config(args)
    pipeline = LightCurvePipeline(config)
    result = pipeline.run()
    print(
        "Pipeline finished: "
        f"frames={result.flux_matrix.shape[0]}, "
        f"stars={result.flux_matrix.shape[1]}"
    )


if __name__ == "__main__":
    main()
