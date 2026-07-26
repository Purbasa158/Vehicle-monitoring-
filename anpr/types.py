"""Small, framework-independent data objects used by the pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Box:
    """Pixel coordinates in xyxy form."""

    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self) -> int:
        return max(0, self.x2 - self.x1)

    @property
    def height(self) -> int:
        return max(0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[int, int]:
        return ((self.x1 + self.x2) // 2, (self.y1 + self.y2) // 2)

    @property
    def area(self) -> int:
        return self.width * self.height

    def clip(self, frame_width: int, frame_height: int) -> "Box":
        return Box(
            max(0, min(self.x1, frame_width - 1)),
            max(0, min(self.y1, frame_height - 1)),
            max(0, min(self.x2, frame_width)),
            max(0, min(self.y2, frame_height)),
        )


@dataclass
class VehicleDetection:
    box: Box
    confidence: float
    class_name: str
    track_id: Optional[int] = None


@dataclass
class PlateDetection:
    box: Box
    confidence: float
    text: str = ""
    ocr_confidence: float = 0.0
    vehicle_track_id: Optional[int] = None
    vehicle_class: str = ""


@dataclass
class FrameResult:
    frame_index: int
    vehicles: list[VehicleDetection] = field(default_factory=list)
    plates: list[PlateDetection] = field(default_factory=list)
    inference_ms: float = 0.0
