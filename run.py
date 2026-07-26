"""Command line entry point for ANPR."""

from __future__ import annotations

import argparse

from anpr import ANPRPipeline, PipelineConfig
from anpr.runner import create_run_directory, open_capture, process_capture, process_dataset


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Real-time vehicle and licence-plate detection")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--source", help="Webcam index, video path, RTSP URL, or phone IP-camera URL")
    source.add_argument("--dataset", help="Folder of images and videos to process")
    parser.add_argument("--vehicle-model", default="yolo11n.pt")
    parser.add_argument("--plate-model", default="models/license_plate_detector.pt")
    parser.add_argument("--vehicle-conf", type=float, default=0.20)
    parser.add_argument("--plate-conf", type=float, default=0.12)
    parser.add_argument("--imgsz", type=int, default=960, help="Vehicle inference image size")
    parser.add_argument("--plate-imgsz", type=int, default=1280, help="Plate detector image size; higher improves small plates")
    parser.add_argument("--disable-classical-fallback", action="store_true")
    parser.add_argument("--disable-fast-alpr", action="store_true", help="Use the local YOLO + EasyOCR plate path only")
    parser.add_argument("--plate-profile", choices=("india", "auto"), default="india")
    parser.add_argument("--detect-every", type=int, default=3, help="Run detection every N frames while preserving source-FPS output video")
    parser.add_argument("--gpu", action="store_true", help="Use EasyOCR GPU mode (PyTorch CUDA must be installed)")
    parser.add_argument("--view", action="store_true", help="Show an OpenCV preview; press q to stop")
    parser.add_argument("--save-video", action="store_true")
    parser.add_argument("--save-evidence", action="store_true")
    parser.add_argument("--output", default="runs", help="Parent output directory")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = PipelineConfig(
        vehicle_model=args.vehicle_model,
        plate_model=args.plate_model,
        vehicle_confidence=args.vehicle_conf,
        plate_confidence=args.plate_conf,
        image_size=args.imgsz,
        plate_image_size=args.plate_imgsz,
        use_gpu=args.gpu,
        classical_plate_fallback=not args.disable_classical_fallback,
        use_fast_alpr=not args.disable_fast_alpr,
        plate_profile=args.plate_profile,
    )
    pipeline = ANPRPipeline(config)
    pipeline.warmup()
    output_dir = create_run_directory(args.output)
    if args.dataset:
        rows = process_dataset(args.dataset, pipeline, output_dir, args.save_evidence, detection_stride=args.detect_every)
    else:
        capture = open_capture(args.source)
        rows = process_capture(
            capture,
            pipeline,
            args.source,
            output_dir,
            args.save_video,
            args.save_evidence,
            args.view,
            detection_stride=args.detect_every,
        )
    print(f"Finished. {len(rows)} plate observations saved to: {output_dir}")


if __name__ == "__main__":
    main()
