"""Download the default dedicated licence-plate YOLO model.

Vehicle weights are downloaded by Ultralytics on first use. Keeping the plate
model in a predictable local location makes the application work offline after
the initial setup.
"""

from __future__ import annotations

import argparse
import shutil
import urllib.request
from pathlib import Path


MODEL_PRESETS = {
    # Compact MIT model for a quick first run.
    "balanced": "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt?download=true",
    # Larger YOLO11-L detector. Its published model card is AGPL-3.0, so make
    # sure that licence is appropriate for your use before using it.
    "accurate": "https://huggingface.co/morsetechlab/yolov11-license-plate-detection/resolve/main/license-plate-finetune-v1l.pt?download=true",
}


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    print(f"Downloading plate model to {destination} ...")
    try:
        with urllib.request.urlopen(url) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=MODEL_PRESETS, default="balanced", help="balanced = compact MIT model; accurate = larger AGPL-3.0 YOLO11-L model")
    parser.add_argument("--url", help="Optional direct model URL; overrides --preset")
    parser.add_argument("--output", default="models/license_plate_detector.pt")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    destination = Path(args.output)
    if destination.exists() and not args.force:
        print(f"Already present: {destination} (use --force to re-download)")
        return
    url = args.url or MODEL_PRESETS[args.preset]
    if args.preset == "accurate" and not args.url:
        print("Selected accurate preset (larger YOLO11-L; review its AGPL-3.0 licence before deployment).")
    download(url, destination)
    print("Done.")


if __name__ == "__main__":
    main()
