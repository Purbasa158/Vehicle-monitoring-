"""Capture, reporting and batch-processing helpers used by CLI and UI."""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterator

import cv2
import numpy as np

from .pipeline import ANPRPipeline
from .types import FrameResult


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def create_run_directory(base: str | Path = "runs") -> Path:
    folder = Path(base) / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    folder.mkdir(parents=True, exist_ok=False)
    return folder


def open_capture(source: str | int) -> cv2.VideoCapture:
    parsed: str | int = int(source) if isinstance(source, str) and source.strip().isdigit() else source
    capture = cv2.VideoCapture(parsed)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open source: {source}")
    return capture


def result_rows(result: FrameResult, source: str, timestamp_seconds: float | None = None) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for plate in result.plates:
        rows.append(
            {
                "source": source,
                "frame": result.frame_index,
                "timestamp_seconds": round(timestamp_seconds or 0, 3),
                "track_id": plate.vehicle_track_id or "",
                "vehicle_class": plate.vehicle_class,
                "plate_text": plate.text,
                "plate_detector_confidence": round(plate.confidence, 4),
                "ocr_confidence": round(plate.ocr_confidence, 4),
                "plate_x1": plate.box.x1,
                "plate_y1": plate.box.y1,
                "plate_x2": plate.box.x2,
                "plate_y2": plate.box.y2,
            }
        )
    return rows


def write_reports(rows: list[dict[str, object]], output_dir: Path) -> tuple[Path, Path]:
    fields = [
        "source", "frame", "timestamp_seconds", "track_id", "vehicle_class", "plate_text",
        "plate_detector_confidence", "ocr_confidence", "plate_x1", "plate_y1", "plate_x2", "plate_y2",
    ]
    csv_path = output_dir / "detections.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    json_path = output_dir / "detections.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return csv_path, json_path


def save_evidence(frame: np.ndarray, result: FrameResult, evidence_dir: Path) -> None:
    for index, plate in enumerate(result.plates):
        if not plate.text:
            continue
        crop = frame[plate.box.y1:plate.box.y2, plate.box.x1:plate.box.x2]
        if crop.size:
            safe_text = "".join(char for char in plate.text if char.isalnum()) or "unread"
            cv2.imwrite(str(evidence_dir / f"f{result.frame_index:06d}_{index}_{safe_text}.jpg"), crop)


def process_capture(
    capture: cv2.VideoCapture,
    pipeline: ANPRPipeline,
    source_name: str,
    output_dir: Path,
    save_video: bool = False,
    save_evidence_crops: bool = False,
    view: bool = False,
    on_frame: Callable[[np.ndarray, FrameResult], bool | None] | None = None,
    max_frames: int | None = None,
    detection_stride: int = 1,
    realtime_playback: bool = False,
) -> list[dict[str, object]]:
    """Process one capture while optionally reusing detections between frames.

    A stride of 3 means that models run on frames 0, 3, 6, … but every input
    frame is rendered and written at the source FPS. This keeps normal video
    playback on CPU-only machines without changing the output video speed.
    """
    if detection_stride < 1:
        raise ValueError("detection_stride must be at least 1")
    rows: list[dict[str, object]] = []
    evidence_dir = output_dir / "evidence"
    if save_evidence_crops:
        evidence_dir.mkdir(exist_ok=True)
    writer = None
    frame_index = 0
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    latest_result: FrameResult | None = None
    playback_started = time.perf_counter()

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            run_detection = latest_result is None or frame_index % detection_stride == 0
            if run_detection:
                result = pipeline.process(frame, frame_index)
                latest_result = result
            else:
                # Boxes remain visually stable over a short gap, while the
                # next detection refreshes IDs, boxes, and plate consensus.
                result = FrameResult(
                    frame_index=frame_index,
                    vehicles=latest_result.vehicles,
                    plates=latest_result.plates,
                    inference_ms=latest_result.inference_ms,
                )
            annotated = pipeline.draw(frame, result)
            if save_video:
                if writer is None:
                    height, width = annotated.shape[:2]
                    writer = cv2.VideoWriter(str(output_dir / "annotated.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                writer.write(annotated)
            if run_detection:
                rows.extend(result_rows(result, source_name, frame_index / fps))
            if save_evidence_crops and run_detection:
                save_evidence(frame, result, evidence_dir)
            if on_frame is not None and on_frame(annotated, result) is False:
                break
            if realtime_playback:
                # Do not race through files when stride inference is faster
                # than the source video. If inference is slower, we continue
                # immediately and prioritize catching up.
                target_time = playback_started + (frame_index + 1) / fps
                remaining = target_time - time.perf_counter()
                if remaining > 0:
                    time.sleep(remaining)
            if view:
                cv2.imshow("ANPR - press q to exit", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            frame_index += 1
            if max_frames is not None and frame_index >= max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if view:
            cv2.destroyAllWindows()
    write_reports(rows, output_dir)
    return rows


def iter_media_files(dataset_dir: str | Path) -> Iterator[Path]:
    base = Path(dataset_dir)
    if not base.is_dir():
        raise NotADirectoryError(f"Dataset folder does not exist: {base}")
    for file in sorted(base.rglob("*")):
        if file.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
            yield file


def process_dataset(
    dataset_dir: str | Path,
    pipeline: ANPRPipeline,
    output_dir: Path,
    save_evidence_crops: bool = False,
    detection_stride: int = 1,
) -> list[dict[str, object]]:
    all_rows: list[dict[str, object]] = []
    evidence_dir = output_dir / "evidence"
    if save_evidence_crops:
        evidence_dir.mkdir(exist_ok=True)
    for media_file in iter_media_files(dataset_dir):
        if media_file.suffix.lower() in IMAGE_EXTENSIONS:
            frame = cv2.imread(str(media_file))
            if frame is None:
                continue
            result = pipeline.process(frame, 0)
            annotated = pipeline.draw(frame, result)
            relative_name = media_file.name
            cv2.imwrite(str(output_dir / f"annotated_{relative_name}"), annotated)
            all_rows.extend(result_rows(result, str(media_file), 0))
            if save_evidence_crops:
                save_evidence(frame, result, evidence_dir)
        else:
            child_output = output_dir / media_file.stem
            child_output.mkdir(exist_ok=True)
            capture = open_capture(str(media_file))
            all_rows.extend(
                process_capture(
                    capture,
                    pipeline,
                    str(media_file),
                    child_output,
                    save_video=True,
                    save_evidence_crops=save_evidence_crops,
                    detection_stride=detection_stride,
                )
            )
    write_reports(all_rows, output_dir)
    return all_rows
