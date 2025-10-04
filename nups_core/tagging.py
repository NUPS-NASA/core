"""Tagging utilities for linking detections across frames."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil, hypot
from typing import Dict, Iterable, List, Sequence, Tuple

from .inspection import DetectedStar


__all__ = ["TaggedStar", "tag_star_catalogs"]


@dataclass
class TaggedStar:
    """Detected star annotated with a persistent identifier."""

    x: float
    y: float
    flux: float
    peak: float
    id: int


def tag_star_catalogs(
    catalogs: Sequence[Sequence[DetectedStar]],
    *,
    min_presence_fraction: float = 0.9,
    max_merge_distance: float = 6.0,
    max_frame_shift: float = 10.0,
) -> List[List[TaggedStar]]:
    """Assign persistent IDs to stars that persist across frames."""

    total_frames = len(catalogs)
    if total_frames == 0:
        return []

    min_presence = min(1.0, max(0.0, float(min_presence_fraction)))
    distance_limit = max(0.5, float(max_merge_distance))
    frame_shift_limit = max(0.0, float(max_frame_shift))
    required_count = max(1, ceil(min_presence * total_frames))

    clusters: List[_StarCluster] = []
    for frame_index, stars in enumerate(catalogs):
        if not stars:
            continue
        shift = _estimate_frame_shift(stars, clusters, frame_shift_limit) if clusters and frame_shift_limit > 0 else None
        dx, dy = shift if shift is not None else (0.0, 0.0)

        for star in stars:
            cluster = _find_matching_cluster(clusters, frame_index, star, distance_limit, dx=dx, dy=dy)
            if cluster is None:
                clusters.append(_StarCluster(frame_index, star))
            else:
                cluster.add(frame_index, star)

    if not clusters:
        return [[] for _ in catalogs]

    stable_clusters = [cluster for cluster in clusters if cluster.frame_count >= required_count]

    if not stable_clusters:
        clusters = sorted(clusters, key=lambda cl: (cl.frame_count, cl.average_flux), reverse=True)
        stable_clusters = [cl for cl in clusters if cl.frame_count >= 2]
        if not stable_clusters and clusters:
            stable_clusters = [clusters[0]]

    stable_clusters.sort(key=lambda cl: cl.average_flux, reverse=True)

    tagged_catalogs: List[List[TaggedStar]] = [[] for _ in catalogs]
    for star_id, cluster in enumerate(stable_clusters, start=1):
        for frame_index, star in cluster.entries.items():
            tagged_catalogs[frame_index].append(
                TaggedStar(x=star.x, y=star.y, flux=star.flux, peak=star.peak, id=star_id)
            )

    for catalog in tagged_catalogs:
        catalog.sort(key=lambda star: star.id)

    return tagged_catalogs


def _estimate_frame_shift(
    stars: Sequence[DetectedStar], clusters: Sequence["_StarCluster"], limit: float
) -> Tuple[float, float] | None:
    offsets: List[Tuple[float, float]] = []
    for star in stars:
        nearest = _nearest_cluster(clusters, star)
        if nearest is None:
            continue
        cx, cy = nearest.centroid
        dx = cx - star.x
        dy = cy - star.y
        if hypot(dx, dy) <= limit:
            offsets.append((dx, dy))
    if not offsets:
        return None

    dx = _median(offset[0] for offset in offsets)
    dy = _median(offset[1] for offset in offsets)
    if hypot(dx, dy) > limit:
        return None
    return dx, dy


def _nearest_cluster(clusters: Sequence["_StarCluster"], star: DetectedStar) -> "_StarCluster" | None:
    closest_cluster: _StarCluster | None = None
    best_distance = float("inf")
    for cluster in clusters:
        distance = cluster.distance_to(star)
        if distance < best_distance:
            best_distance = distance
            closest_cluster = cluster
    return closest_cluster


def _median(values: Iterable[float]) -> float:
    data = sorted(values)
    if not data:
        return 0.0
    mid = len(data) // 2
    if len(data) % 2:
        return float(data[mid])
    return float((data[mid - 1] + data[mid]) / 2.0)


def _find_matching_cluster(
    clusters: Sequence["_StarCluster"], frame_index: int, star: DetectedStar, distance_limit: float, *, dx: float = 0.0, dy: float = 0.0
) -> "_StarCluster" | None:
    best_cluster: _StarCluster | None = None
    best_distance = distance_limit
    shifted_x = star.x + dx
    shifted_y = star.y + dy
    for cluster in clusters:
        if cluster.has_frame(frame_index):
            continue
        distance = cluster.distance_to_coords(shifted_x, shifted_y)
        if distance < best_distance:
            best_distance = distance
            best_cluster = cluster
    return best_cluster


class _StarCluster:
    """Internal helper to accumulate occurrences of the same star across frames."""

    def __init__(self, frame_index: int, star: DetectedStar) -> None:
        self.entries: Dict[int, DetectedStar] = {frame_index: star}
        self._sum_x = star.x
        self._sum_y = star.y
        self._sum_flux = star.flux

    @property
    def frame_count(self) -> int:
        return len(self.entries)

    @property
    def average_flux(self) -> float:
        return float(self._sum_flux) / max(1, self.frame_count)

    @property
    def centroid(self) -> Tuple[float, float]:
        count = max(1, self.frame_count)
        return self._sum_x / count, self._sum_y / count

    def has_frame(self, frame_index: int) -> bool:
        return frame_index in self.entries

    def add(self, frame_index: int, star: DetectedStar) -> None:
        self.entries[frame_index] = star
        self._sum_x += star.x
        self._sum_y += star.y
        self._sum_flux += star.flux

    def distance_to(self, star: DetectedStar) -> float:
        return self.distance_to_coords(star.x, star.y)

    def distance_to_coords(self, x: float, y: float) -> float:
        cx, cy = self.centroid
        return hypot(x - cx, y - cy)
