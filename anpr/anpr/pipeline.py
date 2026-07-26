"""Vehicle detection, plate detection, OCR, association and drawing."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .fast_alpr_adapter import FastALPRAdapter
from .ocr import PlateOCR
from .tracking import CentroidTracker
from .types import Box, FrameResult, PlateDetection, VehicleDetection


COCO_VEHICLE_CLASS_IDS = {1, 2, 3, 5, 6, 7}  # bicycle, car, motorcycle, bus, train, truck
VEHICLE_CLASS_NAMES = {
    "bicycle", "bus", "car", "motorcycle", "motorbike", "train", "truck", "van",
    "pickup", "suv", "taxi", "tractor", "trailer", "vehicle", "auto_rickshaw", "autorickshaw",
}


@dataclass
class PipelineConfig:
    vehicle_model: str = "yolo11n.pt"
    plate_model: str = "models/license_plate_detector.pt"
    vehicle_confidence: float = 0.20
    plate_confidence: float = 0.12
    image_size: int = 960
    plate_image_size: int = 1280
    ocr_languages: tuple[str, ...] = ("en",)
    use_gpu: bool = False
    crop_plate_search: bool = True
    classical_plate_fallback: bool = True
    fallback_min_ocr_confidence: float = 0.20
    use_fast_alpr: bool = True
    plate_profile: str = "india"
    ocr_every_n_frames: int = 3


class ANPRPipeline:
    """Two-stage ANPR pipeline that accepts OpenCV BGR frames."""

    def __init__(self, config: PipelineConfig | None = None):
        self.config = config or PipelineConfig()
        self._validate_plate_model()
        from ultralytics import YOLO

        self.vehicle_model = YOLO(self.config.vehicle_model)
        self.plate_model = YOLO(self.config.plate_model)
        self.fast_alpr = FastALPRAdapter(
            confidence=max(self.config.plate_confidence, 0.20),
            use_gpu=self.config.use_gpu,
            plate_profile=self.config.plate_profile,
        ) if self.config.use_fast_alpr else None
        self.ocr = PlateOCR(
            list(self.config.ocr_languages),
            gpu=self.config.use_gpu,
            plate_profile=self.config.plate_profile,
        )
        self.tracker = CentroidTracker()

    @property
    def fast_alpr_available(self) -> bool:
        return bool(self.fast_alpr and self.fast_alpr.available)

    def warmup(self) -> None:
        """Load/compile inference paths before live or video playback starts."""
        frame = np.zeros((384, 640, 3), dtype=np.uint8)
        self._predict_vehicles(frame)
        if self.fast_alpr_available:
            self.fast_alpr.predict(frame)
        else:
            self._predict_plates(frame)

    def _validate_plate_model(self) -> None:
        path = Path(self.config.plate_model)
        if not path.exists() and not str(self.config.plate_model).startswith(("http://", "https://")):
            raise FileNotFoundError(
                f"Plate model not found: {path}. Run `python scripts/download_models.py` "
                "or select your own YOLO plate weights."
            )

    @staticmethod
    def _boxes_from_result(result, allowed_classes: set[int] | None = None) -> list[tuple[Box, float, int, str]]:
        detections: list[tuple[Box, float, int, str]] = []
        names = result.names
        if result.boxes is None:
            return detections
        for values in result.boxes.data.cpu().tolist():
            x1, y1, x2, y2, confidence, class_id = values[:6]
            class_id = int(class_id)
            if allowed_classes is not None and class_id not in allowed_classes:
                continue
            name = str(names[class_id])
            detections.append((Box(int(x1), int(y1), int(x2), int(y2)), float(confidence), class_id, name))
        return detections

    def _predict_vehicles(self, frame: np.ndarray) -> list[VehicleDetection]:
        results = self.vehicle_model.predict(
            frame,
            conf=self.config.vehicle_confidence,
            imgsz=self.config.image_size,
            verbose=False,
        )
        # COCO IDs make the default model precise; names keep custom vehicle
        # models useful even when their class IDs are different.
        return [
            VehicleDetection(box, confidence, class_name)
            for box, confidence, class_id, class_name in self._boxes_from_result(results[0])
            if class_id in COCO_VEHICLE_CLASS_IDS or class_name.lower().replace(" ", "_") in VEHICLE_CLASS_NAMES
        ]

    def _predict_plates(self, image: np.ndarray) -> list[tuple[Box, float]]:
        results = self.plate_model.predict(
            image,
            conf=self.config.plate_confidence,
            imgsz=self.config.plate_image_size,
            verbose=False,
        )
        return [(box, confidence) for box, confidence, _, _ in self._boxes_from_result(results[0])]

    @staticmethod
    def _nms(
        plates: Iterable[tuple[Box, float, bool, str, float]],
        iou_threshold: float = 0.45,
    ) -> list[tuple[Box, float, bool, str, float]]:
        selected: list[tuple[Box, float, bool, str, float]] = []
        for box, confidence, is_fallback, recognized_text, recognized_confidence in sorted(plates, key=lambda item: item[1], reverse=True):
            keep = True
            for selected_box, _, _, _, _ in selected:
                inter_x1, inter_y1 = max(box.x1, selected_box.x1), max(box.y1, selected_box.y1)
                inter_x2, inter_y2 = min(box.x2, selected_box.x2), min(box.y2, selected_box.y2)
                intersection = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                union = box.area + selected_box.area - intersection
                if union and intersection / union >= iou_threshold:
                    keep = False
                    break
            if keep:
                selected.append((box, confidence, is_fallback, recognized_text, recognized_confidence))
        return selected

    @staticmethod
    def _classical_plate_candidates(image: np.ndarray) -> list[Box]:
        """Find plate-shaped regions when the neural plate model misses one.

        This is deliberately a *candidate generator*, not a final detector.
        The caller only keeps these boxes when OCR produces plausible text.
        """
        height, width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        edges = cv2.Canny(cv2.bilateralFilter(enhanced, 7, 60, 60), 45, 140)
        kernel_width = max(11, min(41, (width // 28) | 1))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, np.ones((5, kernel_width), np.uint8), iterations=1)
        contours, _ = cv2.findContours(closed, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: list[Box] = []
        min_width = max(24, int(width * 0.025))
        min_height = max(8, int(height * 0.012))
        for contour in contours:
            x, y, box_width, box_height = cv2.boundingRect(contour)
            if box_width < min_width or box_height < min_height:
                continue
            aspect_ratio = box_width / max(box_height, 1)
            area_ratio = (box_width * box_height) / max(width * height, 1)
            if not 1.6 <= aspect_ratio <= 8.5 or not 0.00012 <= area_ratio <= 0.20:
                continue
            padding_x = max(2, int(box_width * 0.04))
            padding_y = max(2, int(box_height * 0.16))
            candidates.append(Box(x - padding_x, y - padding_y, x + box_width + padding_x, y + box_height + padding_y).clip(width, height))
        return candidates

    @staticmethod
    def _associate(plate: Box, vehicles: list[VehicleDetection]) -> VehicleDetection | None:
        px, py = plate.center
        contained = [vehicle for vehicle in vehicles if vehicle.box.x1 <= px <= vehicle.box.x2 and vehicle.box.y1 <= py <= vehicle.box.y2]
        if contained:
            return min(contained, key=lambda vehicle: vehicle.box.area)
        # A small distance allowance helps plates detected just outside a vehicle box.
        return min(
            vehicles,
            key=lambda vehicle: (vehicle.box.center[0] - px) ** 2 + (vehicle.box.center[1] - py) ** 2,
            default=None,
        )

    def process(self, frame: np.ndarray, frame_index: int = 0) -> FrameResult:
        if frame is None or frame.size == 0:
            raise ValueError("Received an empty frame")
        started = time.perf_counter()
        height, width = frame.shape[:2]
        vehicles = self.tracker.update(self._predict_vehicles(frame))
        fast_reads = self.fast_alpr.predict(frame) if self.fast_alpr else []
        candidates: list[tuple[Box, float, bool, str, float]] = [
            (box, confidence, False, text, ocr_confidence)
            for box, confidence, text, ocr_confidence in fast_reads
        ]

        # FastALPR's detector + plate-specific recognizer is preferred when it
        # identifies any plates. It avoids duplicate false boxes around lamps
        # and grilles from generic detector weights. The existing high-res YOLO
        # path remains the fallback for frames FastALPR cannot read.
        if not candidates:
            candidates = [(box, confidence, False, "", 0.0) for box, confidence in self._predict_plates(frame)]
            vehicle_crops: list[Box] = []

            # Searching each vehicle crop catches smaller or oblique plates.
            if self.config.crop_plate_search:
                for vehicle in vehicles:
                    pad_x = int(vehicle.box.width * 0.08)
                    pad_y = int(vehicle.box.height * 0.08)
                    crop_box = Box(vehicle.box.x1 - pad_x, vehicle.box.y1 - pad_y, vehicle.box.x2 + pad_x, vehicle.box.y2 + pad_y).clip(width, height)
                    vehicle_crops.append(crop_box)
                    crop = frame[crop_box.y1:crop_box.y2, crop_box.x1:crop_box.x2]
                    if crop.size == 0:
                        continue
                    for local_box, confidence in self._predict_plates(crop):
                        candidates.append(
                            (Box(local_box.x1 + crop_box.x1, local_box.y1 + crop_box.y1, local_box.x2 + crop_box.x1, local_box.y2 + crop_box.y1), confidence, False, "", 0.0)
                        )

            if self.config.classical_plate_fallback:
                # If vehicle detection fails, inspect the complete frame;
                # otherwise restrict the classical search to vehicles.
                search_regions = vehicle_crops or [Box(0, 0, width, height)]
                for region in search_regions:
                    crop = frame[region.y1:region.y2, region.x1:region.x2]
                    if crop.size == 0:
                        continue
                    for local_box in self._classical_plate_candidates(crop):
                        candidates.append(
                            (Box(local_box.x1 + region.x1, local_box.y1 + region.y1, local_box.x2 + region.x1, local_box.y2 + region.y1), 0.10, True, "", 0.0)
                        )

        plates: list[PlateDetection] = []
        should_read_ocr = frame_index % max(1, self.config.ocr_every_n_frames) == 0
        for box, confidence, is_fallback, recognized_text, recognized_confidence in self._nms(candidates):
            box = box.clip(width, height)
            vehicle = self._associate(box, vehicles)
            crop = frame[box.y1:box.y2, box.x1:box.x2]
            track_id = vehicle.track_id if vehicle else None
            if recognized_text:
                text, ocr_confidence = recognized_text, recognized_confidence
            elif should_read_ocr:
                text, ocr_confidence = self.ocr.read(crop)
            else:
                text, ocr_confidence = self.tracker.stable_plate(track_id), 0.0
            # Classical candidates do not have neural-detector confidence.
            # Require actual plate-like OCR before showing/saving them.
            if is_fallback and (len(text) < 4 or ocr_confidence < self.config.fallback_min_ocr_confidence):
                continue
            if should_read_ocr and text:
                text = self.tracker.add_plate_reading(track_id, text)
            plates.append(
                PlateDetection(
                    box=box,
                    confidence=confidence,
                    text=text,
                    ocr_confidence=ocr_confidence,
                    vehicle_track_id=track_id,
                    vehicle_class=vehicle.class_name if vehicle else "",
                )
            )
        elapsed = (time.perf_counter() - started) * 1000
        return FrameResult(frame_index, vehicles, plates, elapsed)

    @staticmethod
    def draw(frame: np.ndarray, result: FrameResult) -> np.ndarray:
        output = frame.copy()
        vehicle_color = (0, 255, 0)  # bright green in OpenCV BGR
        plate_color = (0, 0, 255)  # bright red in OpenCV BGR
        for vehicle in result.vehicles:
            box = vehicle.box
            cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), vehicle_color, 2)
            label = f"{vehicle.class_name} #{vehicle.track_id or '?'} {vehicle.confidence:.0%}"
            cv2.putText(output, label, (box.x1, max(20, box.y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vehicle_color, 2, cv2.LINE_AA)
        for plate in result.plates:
            box = plate.box
            cv2.rectangle(output, (box.x1, box.y1), (box.x2, box.y2), plate_color, 2)
            label = f"{plate.text or 'plate'} {plate.confidence:.0%}"
            cv2.putText(output, label, (box.x1, min(output.shape[0] - 8, box.y2 + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, plate_color, 2, cv2.LINE_AA)
        cv2.putText(output, f"{result.inference_ms:.0f} ms", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
        return output
