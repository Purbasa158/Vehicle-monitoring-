"""OCR preparation, recognition and conservative plate-text cleanup."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .plate_text import apply_plate_profile, normalize_plate


def preprocess_plate_variants(image: np.ndarray) -> list[np.ndarray]:
    """Return complementary high-resolution plate views for OCR.

    Plate text is often only a few pixels high in road footage.  Enlarging to
    a useful character height, improving local contrast, and trying both
    natural and thresholded images is much more reliable than one OCR pass.
    """
    if image is None or image.size == 0:
        return []
    height, width = image.shape[:2]
    scale = min(8.0, max(1.0, 160.0 / max(height, 1)))
    image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)
    denoised = cv2.bilateralFilter(enhanced, 5, 50, 50)
    sharpened = cv2.addWeighted(denoised, 1.8, cv2.GaussianBlur(denoised, (0, 0), 2.0), -0.8, 0)
    _, otsu = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 7)
    return [sharpened, otsu, adaptive]


def preprocess_plate(image: np.ndarray) -> np.ndarray:
    """Compatibility helper returning the natural enhanced OCR view."""
    variants = preprocess_plate_variants(image)
    return variants[0] if variants else image


class PlateOCR:
    def __init__(self, languages: list[str], gpu: bool = False, plate_profile: str = "auto"):
        # Reader construction downloads/loads recognition weights. Keeping it
        # lazy avoids paying that cost when the plate-specific FastALPR engine
        # has already produced a valid text result.
        self.languages = languages
        self.gpu = gpu
        self.plate_profile = plate_profile
        self.reader: Any | None = None

    def _get_reader(self):
        if self.reader is None:
            import easyocr

            self.reader = easyocr.Reader(self.languages, gpu=self.gpu, verbose=False)
        return self.reader

    def read(self, plate_crop: np.ndarray) -> tuple[str, float]:
        variants = preprocess_plate_variants(plate_crop)
        if not variants:
            return "", 0.0
        candidates: list[tuple[str, float, float]] = []
        for prepared in variants:
            readings = self._get_reader().readtext(
                prepared,
                detail=1,
                paragraph=True,
                decoder="beamsearch",
                beamWidth=5,
                allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789",
                min_size=3,
                contrast_ths=0.03,
                adjust_contrast=0.7,
                text_threshold=0.35,
                low_text=0.20,
                link_threshold=0.20,
                canvas_size=4096,
                mag_ratio=2.0,
            )
            for reading in readings:
                # EasyOCR normally returns (box, text, confidence), but with
                # paragraph=True some versions return (text, confidence).
                if len(reading) == 3:
                    _, text, confidence = reading
                elif len(reading) == 2:
                    text, confidence = reading
                else:
                    continue
                if not isinstance(text, str):
                    continue
                text = apply_plate_profile(text, self.plate_profile)
                if not text:
                    continue
                # Typical plates are 5–10 characters. This only ranks OCR
                # alternatives; it never changes a character based on a
                # country-specific format assumption.
                shape_bonus = 0.20 if 5 <= len(text) <= 10 else (0.05 if len(text) >= 4 else -0.25)
                candidates.append((text, float(confidence), float(confidence) + shape_bonus))
        if not candidates:
            return "", 0.0
        text, confidence, _ = max(candidates, key=lambda candidate: candidate[2])
        return text, confidence
