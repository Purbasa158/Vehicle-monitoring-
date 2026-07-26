from anpr.plate_text import normalize_indian_plate, normalize_plate
from anpr.tracking import CentroidTracker
from anpr.types import Box, VehicleDetection


def test_normalize_plate_keeps_only_uppercase_alphanumeric():
    assert normalize_plate("mh-12 ab 1234") == "MH12AB1234"


def test_tracker_reuses_id_for_nearby_same_class_detection():
    tracker = CentroidTracker(max_distance=100)
    first = tracker.update([VehicleDetection(Box(10, 10, 100, 100), 0.9, "car")])
    second = tracker.update([VehicleDetection(Box(20, 15, 110, 105), 0.9, "car")])
    assert first[0].track_id == second[0].track_id


def test_tracker_keeps_plate_vote():
    tracker = CentroidTracker()
    tracked = tracker.update([VehicleDetection(Box(0, 0, 50, 50), 0.9, "car")])[0]
    assert tracker.add_plate_reading(tracked.track_id, "AB12") == "AB12"
    assert tracker.add_plate_reading(tracked.track_id, "AB12") == "AB12"


def test_indian_profile_corrects_character_slot_confusion():
    assert normalize_indian_plate("T509EG653I") == "TS09EG6531"
