import pytest
import media.auto_reframe as mod

CASES = [
    ("frame_track", (0.5, 0.2)),
    ("smooth_path", (1.0, 2.0)),
    ("safe_area", (16, 9)),
    ("platform_consistent", (1, 2)),
    ("subtitle_reflow", (1, 2)),
    ("zoom_limit", (1.0, 2.0)),
    ("shot_detect", (1, 2)),
    ("audio_follow", (1, 2)),
    ("coord_unify", (1, 2)),
    ("rotation_once", (90, 2)),
    ("crop_window", (1920, 1080)),
    ("subject_score", (0.5, 0.2)),
    ("tracking_loss", (1, 2)),
    ("keyframe_interval", (30, 2)),
    ("export_preview", (1, 2)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
