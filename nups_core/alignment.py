"""Frame alignment routines based on detected stars."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, degrees
from typing import Dict, List, Sequence, Tuple

import numpy as np
from scipy.ndimage import affine_transform

from .tagging import TaggedStar
from .reduction import HOPS_ALIGN_U0_KEY, HOPS_ALIGN_X0_KEY, HOPS_ALIGN_Y0_KEY
from .util import FitsFrame


__all__ = ["AlignmentResult", "AlignmentTransform", "align_frames", "align_frames_tagged"]


@dataclass
class AlignmentTransform:
    """Description of the geometric transform applied to a frame."""

    rotation_degrees: float
    translation_x: float
    translation_y: float
    matrix: np.ndarray
    offset: np.ndarray
    rms_error: float


@dataclass
class AlignmentResult:
    """Aligned frames together with their updated star catalogs."""

    frames: List[FitsFrame]
    star_catalogs: List[List[TaggedStar]]
    transforms: List[AlignmentTransform]
    reference_index: int


def align_frames(
    frames: Sequence[FitsFrame],
    star_catalogs: Sequence[Sequence[TaggedStar]],
    *,
    reference_index: int | None = None,
    max_pairs: int = 25,
    residual_threshold: float = 5.0,
) -> AlignmentResult:
    """Align ``frames`` using detected star catalogs with persistent IDs."""

    if not frames:
        return AlignmentResult(frames=[], star_catalogs=[], transforms=[], reference_index=0)

    frames_list = list(frames)
    catalogs_list = [list(catalog) for catalog in star_catalogs]

    if len(catalogs_list) < len(frames_list):
        catalogs_list.extend([] for _ in range(len(frames_list) - len(catalogs_list)))

    if reference_index is None:
        reference_index = _find_reference_index(catalogs_list)

    reference_index = max(0, min(reference_index, len(frames_list) - 1))

    reference_catalog = _filter_catalog_with_ids(catalogs_list[reference_index])
    if len(reference_catalog) < 2:
        return _identity_alignment(frames_list, catalogs_list, reference_index)

    reference_lookup = {star.id: np.array([star.x, star.y], dtype=float) for star in reference_catalog}

    aligned_frames: List[FitsFrame] = []
    aligned_catalogs: List[List[TaggedStar]] = []
    transforms: List[AlignmentTransform] = []

    for idx, frame in enumerate(frames_list):
        catalog = _filter_catalog_with_ids(catalogs_list[idx])
        data = np.array(frame.data, dtype=float, copy=True)

        transform = _identity_transform()
        aligned_data = data
        aligned_catalog = _clone_catalog(catalog)

        if idx != reference_index and catalog:
            shared_ids = sorted(set(reference_lookup.keys()).intersection(star.id for star in catalog))
            if len(shared_ids) >= 2:
                if max_pairs > 0:
                    shared_ids = shared_ids[:max_pairs]
                ref_points = np.array([reference_lookup[star_id] for star_id in shared_ids], dtype=float)
                target_lookup = {star.id: np.array([star.x, star.y], dtype=float) for star in catalog}
                target_points = np.array([target_lookup[star_id] for star_id in shared_ids], dtype=float)

                rotation, translation = _estimate_rigid_transform(ref_points, target_points)

                if rotation is not None:
                    residuals = _alignment_residuals(ref_points, target_points, rotation, translation)

                    trimmed = _trim_correspondences(ref_points, target_points, residuals, trim_fraction=0.1)
                    if trimmed is not None:
                        ref_trimmed, target_trimmed = trimmed
                        candidate_rotation, candidate_translation = _estimate_rigid_transform(ref_trimmed, target_trimmed)
                        if candidate_rotation is not None:
                            rotation = candidate_rotation
                            translation = candidate_translation
                            residuals = _alignment_residuals(ref_trimmed, target_trimmed, rotation, translation)
                        else:
                            residuals = _alignment_residuals(ref_points, target_points, rotation, translation)
                    else:
                        residuals = _alignment_residuals(ref_points, target_points, rotation, translation)

                    rms_error = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0

                    if not residuals.size or np.max(residuals) <= residual_threshold:
                        matrix, offset = _rotation_to_affine(rotation, translation)
                        background = float(np.median(data)) if np.isfinite(data).any() else 0.0
                        aligned_data = affine_transform(
                            data,
                            matrix=matrix,
                            offset=offset,
                            output_shape=data.shape,
                            order=3,
                            mode="constant",
                            cval=background,
                        )
                        aligned_catalog = _transform_catalog(catalog, rotation, translation)
                        transform = _build_transform(rotation, translation, matrix, offset, rms_error)
                    else:
                        transform = _identity_transform(rms_error=rms_error)

        header = dict(frame.header)
        header[HOPS_ALIGN_X0_KEY] = transform.translation_x
        header[HOPS_ALIGN_Y0_KEY] = transform.translation_y
        header[HOPS_ALIGN_U0_KEY] = transform.rotation_degrees

        aligned_frames.append(FitsFrame(data=aligned_data, header=header, source=frame.source))
        aligned_catalogs.append(aligned_catalog)
        transforms.append(transform)

    return AlignmentResult(
        frames=aligned_frames,
        star_catalogs=aligned_catalogs,
        transforms=transforms,
        reference_index=reference_index,
    )






def _compute_reference_positions(
    catalogs: Sequence[Sequence[TaggedStar]], *, reference_index: int | None = None
) -> Dict[int, np.ndarray]:
    positions: Dict[int, List[Tuple[float, float]]] = {}
    for catalog in catalogs:
        for star in catalog:
            if star.id is None:
                continue
            positions.setdefault(star.id, []).append((star.x, star.y))

    reference_positions: Dict[int, np.ndarray] = {}

    if reference_index is not None and 0 <= reference_index < len(catalogs):
        for star in catalogs[reference_index]:
            if star.id is None:
                continue
            reference_positions[star.id] = np.array([star.x, star.y], dtype=float)

    for star_id, coords in positions.items():
        if star_id not in reference_positions:
            arr = np.array(coords, dtype=float)
            reference_positions[star_id] = np.median(arr, axis=0)

    return reference_positions

def align_frames_tagged(
    frames: Sequence[FitsFrame],
    star_catalogs: Sequence[Sequence[TaggedStar]],
    *,
    reference_index: int | None = None,
    max_pairs: int = 25,
    trim_fraction: float = 0.1,
    residual_threshold: float = 5.0,
) -> AlignmentResult:
    """Align frames using persistent star IDs and global reference positions."""

    if not frames or not star_catalogs:
        return AlignmentResult(frames=[], star_catalogs=[], transforms=[], reference_index=0)

    frames_list = list(frames)
    catalogs_list = [list(catalog) for catalog in star_catalogs]
    if len(catalogs_list) < len(frames_list):
        catalogs_list.extend([] for _ in range(len(frames_list) - len(catalogs_list)))

    reference_positions = _compute_reference_positions(catalogs_list, reference_index=reference_index)
    if not reference_positions:
        return _identity_alignment(frames_list, catalogs_list, reference_index or 0)

    transforms: List[AlignmentTransform] = []
    aligned_frames: List[FitsFrame] = []
    aligned_catalogs: List[List[TaggedStar]] = []

    for index, (frame, catalog) in enumerate(zip(frames_list, catalogs_list)):
        catalog_map = {star.id: star for star in catalog if star.id is not None}
        common_ids = [star_id for star_id in reference_positions.keys() if star_id in catalog_map]
        if max_pairs and len(common_ids) > max_pairs:
            common_ids = sorted(common_ids, key=lambda sid: catalog_map[sid].flux, reverse=True)[:max_pairs]

        if len(common_ids) < 2:
            transforms.append(_identity_transform())
            aligned_frames.append(_clone_frame(frame))
            aligned_catalogs.append(_clone_catalog(catalog))
            continue

        ref_points = np.array([reference_positions[star_id] for star_id in common_ids], dtype=float)
        frame_points = np.array([[catalog_map[star_id].x, catalog_map[star_id].y] for star_id in common_ids], dtype=float)

        rotation, translation = _estimate_rigid_transform(ref_points, frame_points)
        if rotation is None:
            transforms.append(_identity_transform())
            aligned_frames.append(_clone_frame(frame))
            aligned_catalogs.append(_clone_catalog(catalog))
            continue

        residuals = _alignment_residuals(ref_points, frame_points, rotation, translation)

        trimmed = _trim_correspondences(ref_points, frame_points, residuals, trim_fraction=trim_fraction)
        if trimmed is not None:
            ref_trimmed, frame_trimmed = trimmed
            candidate_rotation, candidate_translation = _estimate_rigid_transform(ref_trimmed, frame_trimmed)
            if candidate_rotation is not None:
                rotation = candidate_rotation
                translation = candidate_translation
                residuals = _alignment_residuals(ref_trimmed, frame_trimmed, rotation, translation)

        rms_error = float(np.sqrt(np.mean(residuals ** 2))) if residuals.size else 0.0

        if residuals.size and np.max(residuals) > residual_threshold:
            transforms.append(_identity_transform(rms_error=rms_error))
            aligned_frames.append(_clone_frame(frame))
            aligned_catalogs.append(_clone_catalog(catalog))
            continue

        matrix, offset = _rotation_to_affine(rotation, translation)
        data = np.array(frame.data, dtype=float, copy=True)
        background = float(np.median(data)) if np.isfinite(data).any() else 0.0
        aligned_data = affine_transform(
            data,
            matrix=matrix,
            offset=offset,
            output_shape=data.shape,
            order=3,
            mode="constant",
            cval=background,
        )

        aligned_catalog = []
        for star in catalog:
            if star.id is None:
                continue
            if star.id in reference_positions:
                ref_x, ref_y = reference_positions[star.id]
                aligned_catalog.append(
                    TaggedStar(x=float(ref_x), y=float(ref_y), flux=star.flux, peak=star.peak, id=star.id)
                )
            else:
                aligned_catalog.append(
                    TaggedStar(
                        x=float((np.array([star.x, star.y]) @ rotation + translation)[0]),
                        y=float((np.array([star.x, star.y]) @ rotation + translation)[1]),
                        flux=star.flux,
                        peak=star.peak,
                        id=star.id,
                    )
                )

        aligned_catalog.sort(key=lambda s: s.id)
        transform = _build_transform(rotation, translation, matrix, offset, rms_error)

        header = dict(frame.header)
        header[HOPS_ALIGN_X0_KEY] = transform.translation_x
        header[HOPS_ALIGN_Y0_KEY] = transform.translation_y
        header[HOPS_ALIGN_U0_KEY] = transform.rotation_degrees

        aligned_frames.append(FitsFrame(data=aligned_data, header=header, source=frame.source))
        aligned_catalogs.append(aligned_catalog)
        transforms.append(transform)

    result_reference_index = reference_index if reference_index is not None else 0
    return AlignmentResult(
        frames=aligned_frames,
        star_catalogs=aligned_catalogs,
        transforms=transforms,
        reference_index=result_reference_index,
    )


def _identity_alignment(
    frames: Sequence[FitsFrame], catalogs: Sequence[Sequence[TaggedStar]], reference_index: int
) -> AlignmentResult:
    aligned_frames: List[FitsFrame] = []
    aligned_catalogs: List[List[TaggedStar]] = []
    transforms: List[AlignmentTransform] = []

    for frame, catalog in zip(frames, catalogs):
        transform = _identity_transform()
        header = dict(frame.header)
        header[HOPS_ALIGN_X0_KEY] = transform.translation_x
        header[HOPS_ALIGN_Y0_KEY] = transform.translation_y
        header[HOPS_ALIGN_U0_KEY] = transform.rotation_degrees
        aligned_frames.append(FitsFrame(data=np.array(frame.data, dtype=float, copy=True), header=header, source=frame.source))
        aligned_catalogs.append(_clone_catalog(_filter_catalog_with_ids(catalog)))
        transforms.append(transform)

    return AlignmentResult(
        frames=aligned_frames,
        star_catalogs=aligned_catalogs,
        transforms=transforms,
        reference_index=reference_index,
    )


def _filter_catalog_with_ids(catalog: Sequence[TaggedStar]) -> List[TaggedStar]:
    filtered = [TaggedStar(x=star.x, y=star.y, flux=star.flux, peak=star.peak, id=star.id) for star in catalog if star.id is not None]
    filtered.sort(key=lambda star: star.id)
    return filtered


def _find_reference_index(catalogs: Sequence[Sequence[TaggedStar]]) -> int:
    best_index = 0
    best_count = 0
    for idx, catalog in enumerate(catalogs):
        count = sum(1 for star in catalog if star.id is not None)
        if count > best_count:
            best_index = idx
            best_count = count
    return best_index


def _identity_transform(*, rms_error: float = 0.0) -> AlignmentTransform:
    matrix = np.eye(2, dtype=float)
    offset = np.zeros(2, dtype=float)
    return AlignmentTransform(
        rotation_degrees=0.0,
        translation_x=0.0,
        translation_y=0.0,
        matrix=matrix,
        offset=offset,
        rms_error=rms_error,
    )


def _estimate_rigid_transform(
    reference: np.ndarray, target: np.ndarray
) -> tuple[np.ndarray | None, np.ndarray | None]:
    if reference.shape != target.shape or reference.shape[0] < 2:
        return None, None

    src = target
    dst = reference

    centroid_src = src.mean(axis=0)
    centroid_dst = dst.mean(axis=0)

    src_centered = src - centroid_src
    dst_centered = dst - centroid_dst

    covariance = src_centered.T @ dst_centered
    U, _, Vt = np.linalg.svd(covariance)
    rotation = Vt.T @ U.T

    if np.linalg.det(rotation) < 0:
        Vt[-1, :] *= -1
        rotation = Vt.T @ U.T

    translation = centroid_dst - centroid_src @ rotation
    return rotation, translation


def _rotation_to_affine(rotation: np.ndarray, translation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    perm = np.array([[0.0, 1.0], [1.0, 0.0]])
    rot_axes = perm @ rotation @ perm
    trans_axes = perm @ translation
    matrix = rot_axes.T
    offset = -matrix @ trans_axes
    return matrix, offset




def _trim_correspondences(
    reference: np.ndarray, target: np.ndarray, residuals: np.ndarray, *, trim_fraction: float
) -> tuple[np.ndarray, np.ndarray] | None:
    if residuals.size < 3 or trim_fraction <= 0.0:
        return None

    keep = int(np.floor(residuals.size * (1.0 - trim_fraction)))
    keep = max(2, keep)
    if keep >= residuals.size:
        return None

    order = np.argsort(residuals)[:keep]
    return reference[order], target[order]

def _alignment_residuals(
    reference: np.ndarray, target: np.ndarray, rotation: np.ndarray, translation: np.ndarray
) -> np.ndarray:
    aligned = target @ rotation + translation
    return np.linalg.norm(aligned - reference, axis=1)


def _build_transform(
    rotation: np.ndarray,
    translation: np.ndarray,
    matrix: np.ndarray,
    offset: np.ndarray,
    rms_error: float,
) -> AlignmentTransform:
    angle = degrees(atan2(rotation[1, 0], rotation[0, 0]))
    return AlignmentTransform(
        rotation_degrees=float(angle),
        translation_x=float(translation[0]),
        translation_y=float(translation[1]),
        matrix=matrix,
        offset=offset,
        rms_error=rms_error,
    )



def _clone_frame(frame: FitsFrame) -> FitsFrame:
    return FitsFrame(data=np.array(frame.data, dtype=float, copy=True), header=dict(frame.header), source=frame.source)

def _transform_catalog(
    catalog: Sequence[TaggedStar], rotation: np.ndarray, translation: np.ndarray
) -> List[TaggedStar]:
    transformed: List[TaggedStar] = []
    for star in catalog:
        point = np.array([star.x, star.y], dtype=float)
        x_new, y_new = point @ rotation + translation
        transformed.append(
            TaggedStar(x=float(x_new), y=float(y_new), flux=star.flux, peak=star.peak, id=star.id)
        )
    transformed.sort(key=lambda star: star.id)
    return transformed


def _clone_catalog(catalog: Sequence[TaggedStar]) -> List[TaggedStar]:
    cloned = [TaggedStar(x=star.x, y=star.y, flux=star.flux, peak=star.peak, id=star.id) for star in catalog]
    cloned.sort(key=lambda star: star.id)
    return cloned
