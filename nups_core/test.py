"""Minimal executable harness for the NUPS reduction pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits as pf

from .alignment import AlignmentResult, align_frames, align_frames_tagged
from .inspection import inspect_frames
from .photometry import PhotometryResult, run_photometry
from .fitting import FittingResult, run_fitting
from .tagging import tag_star_catalogs
from .option import NupsOption
from .reduction import ReductionResult, reduction
from .util import ReductionFiles, load_reduction_files


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the NUPS reduction pipeline on a directory of FITS files.")
    parser.add_argument("directory", type=Path, help="Base directory containing the raw FITS files.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional destination directory for the reduced FITS files. Defaults to no file output.",
    )
    parser.add_argument(
        "--preview-dir",
        type=Path,
        default=None,
        help="If set, write stage preview images into this directory.",
    )
    parser.add_argument(
        "--preview-sample",
        type=int,
        default=1,
        help="Preview every Nth image (default: 1, i.e., all images).",
    )
    parser.add_argument(
        "--enable-logging",
        action="store_true",
        help="Reserved flag (logging routed through main process in earlier versions).",
    )
    parser.add_argument(
        "--detection-sigma",
        type=float,
        default=5.0,
        help="Sigma threshold for star detection (default: 5.0).",
    )
    parser.add_argument(
        "--detection-min-distance",
        type=int,
        default=7,
        help="Minimum pixel separation between detected stars (default: 7).",
    )
    parser.add_argument(
        "--detection-max-stars",
        type=int,
        default=150,
        help="Maximum number of stars to keep per frame (default: 150).",
    )
    parser.add_argument(
        "--tagging-min-presence",
        type=float,
        default=0.9,
        help="Minimum fraction of frames a star must appear in to receive an ID (default: 0.9).",
    )
    parser.add_argument(
        "--tagging-match-distance",
        type=float,
        default=30.0,
        help="Maximum pixel separation when linking detections into persistent stars (default: 30.0).",
    )
    parser.add_argument(
        "--tagging-max-frame-shift",
        type=float,
        default=10.0,
        help="Maximum per-frame shift (pixels) allowed when estimating global telescope motion.",
    )
    parser.add_argument(
        "--alignment-reference",
        type=int,
        default=None,
        help="Optional index of the reference frame for alignment.",
    )
    parser.add_argument(
        "--alignment-max-pairs",
        type=int,
        default=25,
        help="Maximum number of star pairs to match during alignment (default: 25).",
    )
    parser.add_argument(
        "--alignment-residual-threshold",
        type=float,
        default=5.0,
        help="Reject alignment if any matched star drifts more than this many pixels (default: 5.0).",
    )
    parser.add_argument(
        "--photometry-aperture",
        type=float,
        default=3.5,
        help="Radius of the photometry aperture in pixels (default: 3.5).",
    )
    parser.add_argument(
        "--photometry-annulus-inner",
        type=float,
        default=6.0,
        help="Inner radius of the background annulus in pixels (default: 6.0).",
    )
    parser.add_argument(
        "--photometry-annulus-outer",
        type=float,
        default=8.0,
        help="Outer radius of the background annulus in pixels (default: 8.0).",
    )
    parser.add_argument(
        "--fitting-degree",
        type=int,
        default=1,
        help="Polynomial degree for light-curve fitting (default: 1, linear).",
    )
    parser.add_argument(
        "--fitting-star-ids",
        type=int,
        nargs="*",
        default=None,
        help="Optional star IDs to fit (default: all tagged stars).",
    )
    parser.add_argument(
        "--use-legacy-alignment",
        action="store_true",
        help="Use the legacy alignment routine instead of the tag-based version.",
    )
    args = parser.parse_args()

    options = NupsOption()
    files = load_reduction_files(args.directory, options)

    if not files.science:
        raise RuntimeError("No science frames found for reduction.")

    try:
        options = options.update_location(files.science[0])
    except Exception as exc:
        print(f"Warning: could not extract location metadata from header: {exc}")

    print(f"Using location: {options.location}")
    print(f"Target RA/DEC: {options.target_ra_dec}")

    if args.enable_logging:
        print("Info: logging is always local in the current single-process pipeline.")

    result = reduction(
        bias_frames=files.bias,
        dark_frames=files.dark,
        flat_frames=files.flat,
        science_frames=files.science,
        options=options,
        dark_flat_frames=files.dark_flat,
    )

    # Detect stars in the raw and reduced frames
    raw_detected_catalogs = inspect_frames(
        files.science,
        threshold_sigma=args.detection_sigma,
        min_separation=args.detection_min_distance,
        max_stars=args.detection_max_stars,
    )
    reduced_frames = [frame.frame for frame in result.frames]
    reduced_detected_catalogs = inspect_frames(
        reduced_frames,
        threshold_sigma=args.detection_sigma,
        min_separation=args.detection_min_distance,
        max_stars=args.detection_max_stars,
    )

    _print_detection_summary("Raw (detected)", raw_detected_catalogs)
    _print_detection_summary("Reduced (detected)", reduced_detected_catalogs)

    raw_star_catalogs = tag_star_catalogs(
        raw_detected_catalogs,
        min_presence_fraction=args.tagging_min_presence,
        max_merge_distance=args.tagging_match_distance,
        max_frame_shift=args.tagging_max_frame_shift,
    )
    reduced_star_catalogs = tag_star_catalogs(
        reduced_detected_catalogs,
        min_presence_fraction=args.tagging_min_presence,
        max_merge_distance=args.tagging_match_distance,
        max_frame_shift=args.tagging_max_frame_shift,
    )

    _print_detection_summary("Raw (tagged)", raw_star_catalogs)
    _print_detection_summary("Reduced (tagged)", reduced_star_catalogs)

    align_kwargs = {
        "max_pairs": args.alignment_max_pairs,
        "residual_threshold": args.alignment_residual_threshold,
    }
    if args.alignment_reference is not None:
        align_kwargs["reference_index"] = args.alignment_reference

    alignment_function = align_frames if args.use_legacy_alignment else align_frames_tagged

    try:
        alignment_result = alignment_function(reduced_frames, reduced_star_catalogs, **align_kwargs)
    except ImportError as exc:  # pragma: no cover - optional dependency
        print(f"Warning: {exc}")
        alignment_result = None

    if alignment_result is not None and alignment_result.frames:
        photometry_frames = alignment_result.frames
        photometry_catalogs = alignment_result.star_catalogs
    else:
        photometry_frames = reduced_frames
        photometry_catalogs = reduced_star_catalogs

    photometry_result = run_photometry(
        photometry_frames,
        photometry_catalogs,
        aperture_radius=args.photometry_aperture,
        annulus_inner=args.photometry_annulus_inner,
        annulus_outer=args.photometry_annulus_outer,
    )

    fitting_result = run_fitting(
        photometry_result,
        degree=args.fitting_degree,
        star_ids=args.fitting_star_ids,
    )

    _print_summary(result)
    _print_alignment_summary(alignment_result)
    if alignment_result is not None:
        _print_detection_summary("Aligned (tagged)", alignment_result.star_catalogs)
    _print_photometry_summary(photometry_result)
    _print_fitting_summary(fitting_result)

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        _write_result(result, args.output)

    if args.preview_dir:
        args.preview_dir.mkdir(parents=True, exist_ok=True)
        _write_previews(
            files,
            result,
            args.preview_dir,
            raw_star_catalogs=raw_star_catalogs,
            reduced_star_catalogs=reduced_star_catalogs,
            alignment_result=alignment_result,
            sample=args.preview_sample,
        )
        _write_photometry_plots(photometry_result, args.preview_dir)
        _write_fitting_plots(fitting_result, args.preview_dir)


def _print_summary(result: ReductionResult) -> None:
    print(f"Reduced {len(result.frames)} science frames.")
    for frame in result.frames:
        stats = frame.stats
        print(
            f"- {frame.name}: mean={stats.mean:.3f}, std={stats.std:.3f}, psf={stats.psf:.3f}, "
            f"airmass={stats.airmass:.3f}, skip={stats.skip}"
        )


def _print_detection_summary(label: str, catalogs) -> None:
    counts = [len(catalog) for catalog in catalogs]
    if not counts:
        print(f"{label}: no stars found.")
        return

    has_ids = any(getattr(star, "id", None) is not None for catalog in catalogs for star in catalog)
    median_count = int(np.median(counts)) if counts else 0

    if has_ids:
        unique_ids = {star.id for catalog in catalogs for star in catalog if getattr(star, "id", None) is not None}
        print(
            f"{label}: {len(unique_ids)} persistent stars "
            f"(per-frame count median={median_count}, min={min(counts)}, max={max(counts)})."
        )
    else:
        print(
            f"{label}: per-frame detections median={median_count}, "
            f"min={min(counts)}, max={max(counts)}."
        )


def _print_alignment_summary(result: AlignmentResult | None) -> None:
    if result is None or not result.transforms:
        print("Alignment skipped (no transforms available).")
        return

    print("Alignment transforms:")
    for idx, transform in enumerate(result.transforms):
        print(
            f"- frame {idx:03d}: rotation={transform.rotation_degrees:.3f} deg, "
            f"shift=({transform.translation_x:.2f}, {transform.translation_y:.2f}) px, "
            f"rms={transform.rms_error:.3f}"
        )



def _print_photometry_summary(result: PhotometryResult) -> None:
    if not result.light_curves:
        print("Photometry skipped (no tagged stars).")
        return

    print("Photometry light curves:")
    for curve in result.light_curves:
        total = len(curve.times)
        valid = sum(1 for flag in curve.valid if flag)
        flux_values = np.array(curve.fluxes, dtype=float)
        finite_flux = flux_values[np.isfinite(flux_values)]
        median_flux = float(np.median(finite_flux)) if finite_flux.size else float("nan")
        print(
            f"- star {curve.star_id:04d}: valid={valid}/{total}, "
            f"median_flux={median_flux:.3f}"
        )



def _print_fitting_summary(result: FittingResult) -> None:
    if not result.curves:
        print("Fitting skipped (no usable light curves).")
        return

    print("Fitting results:")
    for curve in result.curves:
        dof = max(0, sum(curve.mask) - len(curve.coefficients))
        print(
            f"- star {curve.star_id:04d}: rms={curve.rms:.4f}, "
            f"degree={curve.degree}, dof={dof}"
        )


def _write_result(result: ReductionResult, output_dir: Path) -> None:
    for reduced in result.frames:
        primary = pf.PrimaryHDU()
        image = pf.CompImageHDU()
        image.data = reduced.frame.data

        header = pf.Header()
        for key, value in reduced.frame.header.items():
            header[key] = value
        image.header = header

        hdul = pf.HDUList([primary, image])
        hdul.writeto(output_dir / reduced.name, overwrite=True)


def _write_photometry_plots(result: PhotometryResult, output_dir: Path) -> None:
    if not result.light_curves:
        return

    for curve in result.light_curves:
        if not curve.fluxes:
            continue
        times = np.array(curve.times, dtype=float)
        fluxes = np.array(curve.fluxes, dtype=float)
        valid_mask = np.array(curve.valid, dtype=bool)

        mask = np.isfinite(times) & np.isfinite(fluxes) & valid_mask
        if not mask.any():
            mask = np.isfinite(fluxes)
        if not mask.any():
            continue

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(times[mask], fluxes[mask], marker='o', linestyle='-', color='tab:blue')
        ax.set_xlabel('Time')
        ax.set_ylabel('Flux (arb)')
        ax.set_title(f'Star {curve.star_id:04d}')
        ax.grid(True, alpha=0.3)

        filename = output_dir / f'photometry_star_{curve.star_id:04d}.jpg'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)


def _write_fitting_plots(result: FittingResult, output_dir: Path) -> None:
    if not result.curves:
        return

    for curve in result.curves:
        times = np.array(curve.times, dtype=float)
        fluxes = np.array(curve.fluxes, dtype=float)
        model = np.array(curve.model_fluxes, dtype=float)
        residuals = np.array(curve.residuals, dtype=float)
        mask = np.array(curve.mask, dtype=bool)

        valid_mask = np.isfinite(times) & np.isfinite(fluxes) & mask
        if not valid_mask.any():
            valid_mask = np.isfinite(model)
        if not valid_mask.any():
            continue

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
        ax1.plot(times[valid_mask], fluxes[valid_mask], 'o', color='tab:blue', label='Flux')
        if np.isfinite(model[valid_mask]).any():
            ax1.plot(times[valid_mask], model[valid_mask], '-', color='tab:orange', label='Model')
        ax1.set_ylabel('Flux (arb)')
        ax1.set_title(f'Star {curve.star_id:04d} fit (deg={curve.degree})')
        ax1.grid(True, alpha=0.3)
        if np.isfinite(model[valid_mask]).any():
            ax1.legend(loc='best', fontsize=8)

        if np.isfinite(residuals[valid_mask]).any():
            ax2.plot(times[valid_mask], residuals[valid_mask], 'o', color='tab:purple')
        ax2.axhline(0.0, color='k', linestyle='--', linewidth=1)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Residuals')
        ax2.grid(True, alpha=0.3)

        filename = output_dir / f'fitting_star_{curve.star_id:04d}.jpg'
        fig.savefig(filename, bbox_inches='tight', dpi=150)
        plt.close(fig)


def _write_previews(
    files: ReductionFiles,
    result: ReductionResult,
    output_dir: Path,
    *,
    raw_star_catalogs=None,
    reduced_star_catalogs=None,
    alignment_result: AlignmentResult | None = None,
    sample: int = 1,
) -> None:
    stages = {
        "bias": files.bias,
        "dark": files.dark,
        "dark_flat": files.dark_flat,
        "flat": files.flat,
        "science_raw": files.science,
        "science_reduced": [frame.frame for frame in result.frames],
    }

    if alignment_result is not None and alignment_result.frames:
        stages["science_aligned"] = alignment_result.frames

    star_map = {
        "science_raw": raw_star_catalogs or [],
        "science_reduced": reduced_star_catalogs or [],
    }

    if alignment_result is not None and alignment_result.star_catalogs:
        star_map["science_aligned"] = alignment_result.star_catalogs

    for stage, frames in stages.items():
        if not frames:
            continue
        for idx, frame in enumerate(frames):
            if sample > 1 and (idx % sample) != 0:
                continue
            data = np.array(frame.data, dtype=float)
            filename = f"{stage}_{idx:04d}.png"
            markers = None
            catalog_list = star_map.get(stage)
            if catalog_list and idx < len(catalog_list):
                markers = catalog_list[idx]
            _save_preview_image(data, output_dir / filename, markers=markers)


def _save_preview_image(data: np.ndarray, path: Path, *, markers=None) -> None:
    arr = np.array(data, dtype=float)
    if not np.isfinite(arr).any():
        arr = np.zeros((10, 10))
    arr = np.nan_to_num(arr, copy=False)
    arr -= arr.min()
    max_val = arr.max()
    if max_val > 0:
        arr /= max_val
    if markers:
        height, width = arr.shape
        fig, ax = plt.subplots(figsize=(max(width / 200.0, 2), max(height / 200.0, 2)))
        ax.imshow(arr, cmap="gray", origin="lower")
        xs = [star.x for star in markers]
        ys = [star.y for star in markers]
        if xs and ys:
            ax.scatter(xs, ys, s=30, facecolors="none", edgecolors="red", linewidths=1.2)
            for star in markers:
                star_id = getattr(star, "id", None)
                if star_id is not None:
                    ax.text(star.x + 2, star.y + 2, str(star_id), color="yellow", fontsize=6, ha="left", va="bottom")
        ax.set_axis_off()
        fig.savefig(path, bbox_inches="tight", pad_inches=0, dpi=200)
        plt.close(fig)
    else:
        plt.imsave(path, arr, cmap="gray", origin="lower")


if __name__ == "__main__":
    main()
