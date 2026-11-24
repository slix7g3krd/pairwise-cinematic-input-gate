import pytest
import gameplay.cinematic_input_gate as mod

CASES = [
    ("relative_timeline", (1,)),
    ("skip_stop_audio", (1,)),
    ("asset_gate", (1,)),
    ("input_discard", (1,)),
    ("subtitle_bind", (1,)),
    ("fixed_seed", (1,)),
    ("determinism", (1,)),
    ("validation", (1,)),
    ("audit", (1,)),
    ("fallback", (1,)),
    ("timeline_state", (1,)),
    ("metrics", (1,)),
    ("camera_track", (1,)),
    ("skip", (1,)),
    ("gate_input", (1,)),
]

@pytest.mark.parametrize("fn_name,args", CASES)
def test_contract(fn_name, args):
    fn = getattr(mod, fn_name)
    assert fn(*args) is not None
