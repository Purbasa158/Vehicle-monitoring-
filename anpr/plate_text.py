"""Plate text cleanup with an optional Indian registration-format profile."""

from __future__ import annotations

import re


_NON_ALNUM = re.compile(r"[^A-Z0-9]")
_TO_DIGIT = str.maketrans({"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2", "S": "5", "B": "8", "G": "6"})
_TO_LETTER = str.maketrans({"0": "O", "1": "I", "2": "Z", "5": "S", "8": "B", "6": "G"})

# State/UT prefixes make the correction conservative: it only fixes character
# confusion where the standard registration layout makes the position clear.
INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ", "HP", "HR",
    "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ", "NL", "OD", "PB",
    "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}


def normalize_plate(text: str) -> str:
    return _NON_ALNUM.sub("", text.upper())


def normalize_indian_plate(text: str) -> str:
    """Correct likely OCR confusion in a standard Indian registration plate.

    It supports ``TS09EG6531``-style registrations, including one/two digit
    district codes and up to three series letters. If no valid layout can be
    formed the original cleaned OCR text is returned unchanged.
    """
    raw = normalize_plate(text)
    if not 7 <= len(raw) <= 11:
        return raw

    candidates: list[tuple[int, str]] = []
    for district_length in (1, 2):
        series_length = len(raw) - 2 - district_length - 4
        if not 0 <= series_length <= 3:
            continue
        state = raw[:2].translate(_TO_LETTER)
        district = raw[2:2 + district_length].translate(_TO_DIGIT)
        series_start = 2 + district_length
        series = raw[series_start:series_start + series_length].translate(_TO_LETTER)
        number = raw[-4:].translate(_TO_DIGIT)
        if not (state.isalpha() and district.isdigit() and series.isalpha() and number.isdigit()):
            continue
        score = (10 if state in INDIAN_STATE_CODES else 0) + (2 if district_length == 2 else 0) + series_length
        candidates.append((score, state + district + series + number))
    return max(candidates, default=(0, raw), key=lambda candidate: candidate[0])[1]


def apply_plate_profile(text: str, profile: str) -> str:
    return normalize_indian_plate(text) if profile.lower() == "india" else normalize_plate(text)
