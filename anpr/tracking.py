"""Lightweight centroid tracking and OCR voting.

This deliberately has no extra tracking dependency, making it suitable for
webcams and phone streams. A stronger tracker can be plugged in later without
changing the detection/OCR interfaces.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
from math import hypot

from .types import Box, VehicleDetection


@dataclass
class _Track:
    track_id: int
    box: Box
    class_name: str
    missed: int = 0
    plate_votes: deque[str] = field(default_factory=lambda: deque(maxlen=12))


class CentroidTracker:
    """Assign stable local IDs based on nearest same-class bounding box."""

    def __init__(self, max_distance: int = 150, max_missed: int = 20):
        self.max_distance = max_distance
        self.max_missed = max_missed
        self._next_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[VehicleDetection]) -> list[VehicleDetection]:
        unmatched_tracks = set(self._tracks)
        unmatched_detections = set(range(len(detections)))
        candidates: list[tuple[float, int, int]] = []

        for track_id, track in self._tracks.items():
            tx, ty = track.box.center
            for index, detection in enumerate(detections):
                if detection.class_name != track.class_name:
                    continue
                dx, dy = detection.box.center
                candidates.append((hypot(tx - dx, ty - dy), track_id, index))

        for distance, track_id, index in sorted(candidates):
            if (
                distance > self.max_distance
                or track_id not in unmatched_tracks
                or index not in unmatched_detections
            ):
                continue
            track = self._tracks[track_id]
            track.box = detections[index].box
            track.missed = 0
            detections[index].track_id = track_id
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)

        for index in unmatched_detections:
            detection = detections[index]
            track_id = self._next_id
            self._next_id += 1
            self._tracks[track_id] = _Track(track_id, detection.box, detection.class_name)
            detection.track_id = track_id

        for track_id in unmatched_tracks:
            self._tracks[track_id].missed += 1
            if self._tracks[track_id].missed > self.max_missed:
                del self._tracks[track_id]

        return detections

    def add_plate_reading(self, track_id: int | None, plate_text: str) -> str:
        """Save a reading and return the most frequent non-empty text."""
        if not track_id or not plate_text or track_id not in self._tracks:
            return plate_text
        votes = self._tracks[track_id].plate_votes
        votes.append(plate_text)
        return Counter(votes).most_common(1)[0][0]

    def stable_plate(self, track_id: int | None) -> str:
        """Return the current consensus read for a tracked vehicle."""
        if not track_id or track_id not in self._tracks:
            return ""
        votes = self._tracks[track_id].plate_votes
        return Counter(votes).most_common(1)[0][0] if votes else ""
