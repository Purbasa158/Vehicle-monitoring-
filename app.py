"""Streamlit interface for real-time vehicle and licence-plate detection."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from anpr import ANPRPipeline, PipelineConfig
from anpr.runner import (
    create_run_directory,
    open_capture,
    process_capture,
    process_dataset,
    result_rows,
    save_evidence,
    write_reports,
)


st.set_page_config(page_title="ANPR Vehicle & Plate Detection", page_icon="🚘", layout="wide")


@st.cache_resource(show_spinner="Loading YOLO and OCR models…")
def load_pipeline(
    vehicle_model: str,
    plate_model: str,
    vehicle_confidence: float,
    plate_confidence: float,
    image_size: int,
    use_gpu: bool,
    crop_plate_search: bool,
    classical_plate_fallback: bool,
    use_fast_alpr: bool,
    plate_profile: str,
    ocr_every_n_frames: int,
) -> ANPRPipeline:
    pipeline = ANPRPipeline(
        PipelineConfig(
            vehicle_model=vehicle_model,
            plate_model=plate_model,
            vehicle_confidence=vehicle_confidence,
            plate_confidence=plate_confidence,
            image_size=image_size,
            use_gpu=use_gpu,
            crop_plate_search=crop_plate_search,
            classical_plate_fallback=classical_plate_fallback,
            use_fast_alpr=use_fast_alpr,
            plate_profile=plate_profile,
            ocr_every_n_frames=ocr_every_n_frames,
        )
    )
    pipeline.warmup()
    return pipeline


def show_results(rows: list[dict[str, object]], output_dir: Path) -> None:
    st.success(f"Finished. {len(rows)} plate observations saved in `{output_dir}`.")
    csv_path = output_dir / "detections.csv"
    if rows:
        frame = pd.DataFrame(rows)
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button("Download detections CSV", csv_path.read_bytes(), "detections.csv", "text/csv")
    else:
        st.info("No plates were detected. Try a clearer image, lower the confidence threshold, or use plate weights trained for your region.")


def process_source(
    pipeline: ANPRPipeline,
    source: str,
    label: str,
    save_video: bool,
    evidence: bool,
    limit: int,
    detection_stride: int,
    realtime_playback: bool = False,
) -> None:
    output_dir = create_run_directory("runs")
    frame_box = st.empty()
    status = st.empty()
    progress = st.progress(0, text="Opening source…")
    capture = open_capture(source)
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def on_frame(annotated, result):
        frame_box.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
        status.caption(
            f"Frame {result.frame_index} • {len(result.vehicles)} vehicles • {len(result.plates)} plates • {result.inference_ms:.0f} ms"
        )
        if total > 0:
            progress.progress(min(100, int((result.frame_index + 1) / total * 100)), text="Detecting…")
        return True

    try:
        rows = process_capture(
            capture,
            pipeline,
            label,
            output_dir,
            save_video=save_video,
            save_evidence_crops=evidence,
            on_frame=on_frame,
            max_frames=limit if limit > 0 else None,
            detection_stride=detection_stride,
            realtime_playback=realtime_playback,
        )
    except Exception as error:
        st.error(f"Processing failed: {error}")
        return
    progress.progress(100, text="Complete")
    show_results(rows, output_dir)
    annotated = output_dir / "annotated.mp4"
    if save_video and annotated.exists():
        st.video(annotated.read_bytes())


def main() -> None:
    st.title("🚘 Real-time Vehicle & Licence Plate Detection")
    st.caption("YOLO vehicle tracking + FastALPR plate reading, with local YOLO/EasyOCR fallback, phone/IP cameras, video imports, and datasets.")

    with st.sidebar:
        st.header("Detection settings")
        vehicle_model = st.text_input("Vehicle model", "yolo11n.pt", help="Real-time default. Use yolo11m.pt for higher vehicle recall on images, or yolo11l.pt with a strong GPU.")
        plate_model = st.text_input("Licence-plate model", "models/license_plate_detector.pt")
        vehicle_confidence = st.slider("Vehicle confidence", 0.05, 0.95, 0.20, 0.05)
        plate_confidence = st.slider("Plate confidence", 0.05, 0.95, 0.12, 0.05)
        image_size = st.select_slider("Vehicle inference size", options=[640, 768, 960, 1280], value=960)
        crop_search = st.checkbox("Search inside each vehicle crop", value=True, help="Better for small/angled plates; uses more compute.")
        classical_fallback = st.checkbox("Recover missed plate candidates", value=True, help="Uses a conservative image-processing fallback and OCR when YOLO misses a plate.")
        use_fast_alpr = st.checkbox("Use plate-specific FastALPR engine", value=True, help="Recommended: a fast ONNX detector + plate recognizer. Install the updated requirements first.")
        plate_profile = st.selectbox("Plate format profile", ("india", "auto"), format_func=lambda item: "India (TS09EG6531 style)" if item == "india" else "Automatic / global")
        ocr_every = st.slider("OCR every N frames", 1, 10, 3, help="Higher values improve frame rate; tracking stabilizes readings.")
        use_gpu = st.checkbox("Use EasyOCR GPU", value=False, help="Enable only with a CUDA PyTorch installation.")
        st.divider()
        st.caption("Before first run: `python scripts/download_models.py`")

    try:
        pipeline = load_pipeline(
            vehicle_model,
            plate_model,
            vehicle_confidence,
            plate_confidence,
            image_size,
            use_gpu,
            crop_search,
            classical_fallback,
            use_fast_alpr,
            plate_profile,
            ocr_every,
        )
    except Exception as error:
        st.error(f"Could not load models: {error}")
        st.stop()

    if use_fast_alpr and not pipeline.fast_alpr_available:
        st.info("FastALPR is not installed yet, so the app is using the older YOLO + EasyOCR fallback. Run `pip install -r requirements.txt`, then restart the app.")

    mode = st.radio("Input", ("Camera / IP stream", "Video file", "Image file", "Dataset folder"), horizontal=True)
    evidence = st.checkbox("Save readable plate crops as evidence", value=False)

    if mode == "Camera / IP stream":
        st.markdown("Use `0` for the PC webcam, or paste a phone-camera URL such as `http://192.168.1.25:8080/video` / RTSP URL.")
        source = st.text_input("Camera index or IP stream URL", "0")
        max_frames = st.number_input("Frames to process (0 = run until stream ends / app rerun)", min_value=0, value=500, step=50)
        detection_stride = st.slider("Detect every N frames", 1, 6, 3, help="3 is recommended for smoother CPU playback. Every frame is still displayed and saved.")
        if st.button("Start live detection", type="primary"):
            process_source(pipeline, source, source, save_video=False, evidence=evidence, limit=int(max_frames), detection_stride=detection_stride)

    elif mode == "Video file":
        uploaded = st.file_uploader("Upload a video", type=["mp4", "avi", "mov", "mkv", "webm"])
        save_video = st.checkbox("Save annotated output video", value=True)
        detection_stride = st.slider("Detect every N frames", 1, 6, 3, help="3 is recommended for smooth playback on CPU. The output video keeps its original FPS.")
        if uploaded and st.button("Detect in video", type="primary"):
            suffix = Path(uploaded.name).suffix or ".mp4"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temporary:
                temporary.write(uploaded.getbuffer())
                temporary_path = temporary.name
            process_source(pipeline, temporary_path, uploaded.name, save_video=save_video, evidence=evidence, limit=0, detection_stride=detection_stride, realtime_playback=True)

    elif mode == "Image file":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "bmp", "webp"])
        if uploaded and st.button("Detect in image", type="primary"):
            image = cv2.imdecode(np.frombuffer(uploaded.getvalue(), np.uint8), cv2.IMREAD_COLOR)
            if image is None:
                st.error("The image could not be read. Please choose a valid JPG, PNG, BMP, or WEBP image.")
            else:
                output_dir = create_run_directory("runs")
                with st.spinner("Detecting vehicles and plates…"):
                    result = pipeline.process(image, 0)
                    annotated = pipeline.draw(image, result)
                    output_image = output_dir / f"annotated_{Path(uploaded.name).stem}.jpg"
                    cv2.imwrite(str(output_image), annotated)
                    rows = result_rows(result, uploaded.name)
                    write_reports(rows, output_dir)
                    if evidence:
                        evidence_dir = output_dir / "evidence"
                        evidence_dir.mkdir(exist_ok=True)
                        save_evidence(image, result, evidence_dir)
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Detection result", use_container_width=True)
                st.download_button("Download annotated image", output_image.read_bytes(), output_image.name, "image/jpeg")
                show_results(rows, output_dir)

    else:
        st.markdown("Enter a local folder containing images and/or videos. Processing is recursive and produces CSV, JSON, and annotated media.")
        dataset_dir = st.text_input("Dataset folder path", "")
        detection_stride = st.slider("Detect every N frames in dataset videos", 1, 6, 3)
        if st.button("Process dataset", type="primary"):
            if not dataset_dir.strip():
                st.warning("Enter a dataset folder path.")
            else:
                output_dir = create_run_directory("runs")
                try:
                    with st.spinner("Processing dataset…"):
                        rows = process_dataset(dataset_dir, pipeline, output_dir, save_evidence_crops=evidence, detection_stride=detection_stride)
                    show_results(rows, output_dir)
                except Exception as error:
                    st.error(f"Dataset processing failed: {error}")

    st.divider()
    st.caption("Use only with authorization. Licence-plate data may be regulated personal data; secure the output and set a retention policy.")


if __name__ == "__main__":
    main()
