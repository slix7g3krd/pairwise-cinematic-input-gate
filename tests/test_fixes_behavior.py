"""六个缺陷修复的行为测试。"""

import pytest

import gameplay.cinematic_input_gate as mod
from gameplay.cinematic_input_gate import (
    AssetGate,
    AssetNotReadyError,
    AudioTrack,
    CameraSegment,
    CinematicController,
    InputGate,
    SubtitleBinding,
    Timeline,
    ValidationError,
)


SEGMENTS = [
    CameraSegment(
        duration=2.0,
        start_pos=(0.0, 0.0, 0.0),
        end_pos=(10.0, 0.0, 0.0),
        jitter_amplitude=0.5,
    ),
    CameraSegment(
        duration=3.0,
        start_pos=(10.0, 0.0, 0.0),
        end_pos=(10.0, 10.0, 0.0),
        jitter_amplitude=0.5,
    ),
]
SUBTITLES = [
    SubtitleBinding(segment=0, offset=0.0, text="开场"),
    SubtitleBinding(segment=1, offset=0.0, text="转场"),
]
ASSETS = ("camera_rig", "shot_a", "shot_b")
SEED = 20261125


def _controller():
    return CinematicController(
        SEGMENTS, seed=SEED, asset_names=ASSETS, subtitles=SUBTITLES
    )


def _start(controller):
    for asset in ASSETS:
        controller.mark_asset_ready(asset)
    controller.start()
    return controller


# 修复 1：时间轴按场景内相对时间，切场景归零，不看墙上时钟
def test_relative_timeline_resets_on_scene_switch():
    timeline = Timeline(duration=10.0)
    timeline.tick(4.0)
    assert timeline.elapsed == pytest.approx(4.0)

    timeline.reset()  # 切场景
    assert timeline.elapsed == 0.0
    assert not timeline.finished

    timeline.tick(1.0)
    assert timeline.elapsed == pytest.approx(1.0)


def test_relative_timeline_clamps_at_duration_and_ticks_ignore_wall_clock():
    timeline = Timeline(duration=5.0)
    for _ in range(10):
        timeline.tick(1.0)
    assert timeline.elapsed == 5.0
    assert timeline.finished
    assert timeline.remaining == 0.0


def test_controller_load_scene_resets_relative_time():
    controller = _start(_controller())
    controller.tick(3.0)
    assert controller.timeline.elapsed == pytest.approx(3.0)

    controller.load_scene()
    assert controller.timeline.elapsed == 0.0
    assert not controller.started


# 修复 2：跳过时同步停止音轨
def test_skip_stops_audio_in_same_call():
    controller = _start(_controller())
    assert controller.audio.playing

    state = controller.skip()

    assert state["finished"]
    assert state["elapsed"] == controller.timeline.duration
    assert not controller.audio.playing
    assert controller.metrics.audio_stops == 1
    assert controller.metrics.skips == 1


def test_skip_module_entry_stops_audio():
    state = mod.skip_stop_audio(1)
    assert state["finished"] is True
    assert state["audio_playing"] is False


# 修复 3：资源就绪后才启动
def test_start_blocked_until_assets_ready():
    controller = _controller()
    with pytest.raises(AssetNotReadyError):
        controller.start()
    assert not controller.started

    controller.mark_asset_ready("camera_rig")
    controller.mark_asset_ready("shot_a")
    with pytest.raises(AssetNotReadyError):
        controller.start()

    controller.mark_asset_ready("shot_b")
    controller.start()  # 全部就绪后放行
    assert controller.started


def test_asset_gate_reports_pending():
    gate = AssetGate(["a", "b"])
    gate.mark_ready("a")
    assert not gate.all_ready
    assert gate.pending == ["b"]
    with pytest.raises(ValidationError):
        gate.mark_ready("unknown")


# 修复 4：过场中输入丢弃，结束后不释放缓存
def test_input_discarded_during_cinematic_and_not_released_after():
    controller = _start(_controller())

    for payload in ("move_forward", "jump", "dash"):
        assert controller.input_gate.push(payload) is False

    assert controller.input_gate.discarded == 3
    assert controller.input_gate.drain() == []  # 没有任何缓存事件

    controller.skip()
    assert controller.input_gate.drain() == []  # 结束瞬间也没有一次性释放
    assert not controller.input_gate.active


def test_input_gate_normal_mode_queues_events():
    gate = InputGate()  # 默认未激活（正常游玩）
    assert gate.push("attack") is True
    assert gate.drain() == ["attack"]


# 修复 5：字幕绑定镜头段，同帧先切镜头后派字幕
def test_subtitles_bound_to_segments_and_dispatched_with_cuts():
    controller = _start(_controller())

    # 启动即派发第 0 段切入与该段起点字幕
    assert controller.metrics.cuts == 1
    assert controller.metrics.subtitles_shown == 1
    cut_entries = [m for m in controller.audit_log if "cut segment=0" in m]
    subtitle_entries = [m for m in controller.audit_log if "subtitle segment=0" in m]
    assert cut_entries and subtitle_entries
    assert controller.audit_log.index(cut_entries[0]) < controller.audit_log.index(
        subtitle_entries[0]
    )

    # 跨过段边界（2.0s）：同一帧内先切到段 1，再派段 1 的字幕
    controller.tick(2.0)
    assert controller.metrics.cuts == 2
    assert controller.metrics.subtitles_shown == 2
    cut1 = next(m for m in controller.audit_log if "cut segment=1" in m)
    sub1 = next(m for m in controller.audit_log if "subtitle segment=1" in m)
    assert controller.audit_log.index(cut1) < controller.audit_log.index(sub1)


def test_subtitle_rejected_when_segment_missing():
    with pytest.raises(ValidationError):
        CinematicController(
            SEGMENTS,
            seed=SEED,
            asset_names=ASSETS,
            subtitles=[SubtitleBinding(segment=9, offset=0.0, text="孤儿字幕")],
        )


# 修复 6：固定种子保证重播轨迹一致
def test_fixed_seed_replay_gives_identical_camera_track():
    track_a = mod.CameraTrack(SEGMENTS, seed=SEED)
    track_b = mod.CameraTrack(SEGMENTS, seed=SEED)
    samples = [0.0, 0.5, 1.5, 2.0, 3.3, 4.99]

    assert [track_a.sample(t) for t in samples] == [
        track_b.sample(t) for t in samples
    ]


def test_different_seed_can_change_trajectory():
    track_a = mod.CameraTrack(SEGMENTS, seed=1)
    track_b = mod.CameraTrack(SEGMENTS, seed=2)
    # 位姿插值相同，但抖动序列不同（极小概率撞值，幅度 0.5 下可忽略）
    assert track_a.sample(0.3)["jitter"] != track_b.sample(0.3)["jitter"]


def test_determinism_entry():
    report = mod.determinism(3, seed=SEED)
    assert report["deterministic"] is True
    assert report["sample_count"] > 0


def test_fixed_seed_entry_is_stable_across_instances():
    first = mod.fixed_seed(SEED, index=1, amplitude=0.5)
    second = mod.fixed_seed(SEED, index=1, amplitude=0.5)
    assert first["jitter"] == second["jitter"]
    assert first["stable"] is True


# 端到端：完整生命周期
def test_full_lifecycle_and_metrics():
    controller = _start(_controller())
    controller.input_gate.push("move")  # 被丢弃
    controller.tick(1.0)
    controller.tick(1.5)  # 跨入第二段
    state = controller.finish()

    assert state["finished"]
    assert not controller.audio.playing
    assert not controller.input_gate.active
    metrics = controller.metrics_snapshot()
    assert metrics["ticks"] == 2
    assert metrics["cuts"] == 2
    assert metrics["subtitles_shown"] == 2
    assert metrics["inputs_discarded"] == 1
    assert controller.input_gate.drain() == []


def test_fallback_makes_gate_pass():
    gate = AssetGate(["shot_a", "shot_b"])
    report = mod.fallback(gate)
    assert report["all_ready_after_fallback"] is True
    assert set(report["missing"]) == {"shot_a", "shot_b"}
    gate.ensure_ready()  # 不再抛异常


def test_validation_rejects_bad_config():
    with pytest.raises(ValidationError):
        mod.validation({"duration": 0, "segments": [], "assets": ()})
    ok = mod.validation(2)
    assert ok["valid"] is True
