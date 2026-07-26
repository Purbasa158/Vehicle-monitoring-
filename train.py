"""Validate a YOLO plate dataset and launch fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def validate_dataset(data_path: str) -> dict:
    path = Path(data_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    for key in ("train", "val", "names"):
        if key not in data:
            raise ValueError(f"{path} is missing required '{key}' field")
    names = data["names"]
    if isinstance(names, dict):
        names = list(names.values())
    if not names or any(not str(name).strip() for name in names):
        raise ValueError("'names' must contain at least one non-empty class")
    base = Path(data.get("path", path.parent))
    if not base.is_absolute():
        base = path.parent / base
    for split in ("train", "val"):
        split_path = Path(data[split])
        if not split_path.is_absolute():
            split_path = base / split_path
        if not split_path.exists():
            raise FileNotFoundError(f"{split} images path does not exist: {split_path}")
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a custom YOLO licence-plate detector")
    parser.add_argument("--data", required=True, help="YOLO data.yaml")
    parser.add_argument("--model", default="yolo11s.pt", help="Starting Ultralytics weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=int, default=-1, help="-1 automatically sizes to GPU memory")
    parser.add_argument("--device", default=None, help="e.g. 0, 0,1, or cpu")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()

    data = validate_dataset(args.data)
    print(f"Dataset is valid. Classes: {data['names']}")
    if args.validate_only:
        return

    from ultralytics import YOLO

    model = YOLO(args.model)
    train_args = dict(data=args.data, epochs=args.epochs, imgsz=args.imgsz, batch=args.batch, project=args.project)
    if args.device:
        train_args["device"] = args.device
    model.train(**train_args)


if __name__ == "__main__":
    main()
