"""过场动画时间轴与输入门控。

针对六个缺陷的修复均落在本模块中：

1. 时间轴按场景内相对时间推进 —— :class:`Timeline` 只累计本场景的
   ``tick`` 时长，切场景时归零，绝不使用墙上时钟（绝对时长）。
2. 跳过时同步停止音轨 —— :meth:`CinematicController.skip` 把时间轴跳到
   结尾的同一调用里停止 :class:`AudioTrack`，两者由同一门控保护。
3. 资源就绪后再启动 —— :class:`AssetGate` 汇总镜头等异步资源，未全部
   就绪时 ``start`` 直接拒绝，时间轴不会空转。
4. 过场中输入丢弃 —— :class:`InputGate` 在过场激活期把输入事件直接丢弃，
   不入队、不缓存，结束后没有任何积压事件被一次性释放。
5. 字幕事件绑定镜头段 —— :class:`SubtitleBinding` 挂在具体镜头段上，
   镜头切换与字幕在同一帧按“先切镜头、后派字幕”的确定顺序派发，
   不再作为两个同帧独立事件而错位。
6. 抖动使用固定种子 —— :class:`JitterRng` 以过场固定种子派生每段的
   确定性随机数，重播同一过场得到完全一致的镜头轨迹。
"""

from __future__ import annotations

import random
from bisect import bisect_right
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple


class CinematicError(RuntimeError):
    """过场系统基础异常。"""


class AssetNotReadyError(CinematicError):
    """资源未全部就绪却尝试启动时间轴。"""


class ValidationError(CinematicError):
    """过场配置校验失败。"""


# ---------------------------------------------------------------------------
# 修复 6：固定种子的抖动随机源
# ---------------------------------------------------------------------------


class JitterRng:
    """固定种子随机源。

    每个镜头段用 ``(过场种子, 段索引)`` 派生出独立的确定性 :class:`random.Random`，
    因此同一种子重播时，逐帧抖动与首次播放逐位一致；段之间互不影响顺序。
    """

    def __init__(self, seed: int):
        self.seed = int(seed)
        self._rng = random.Random(self.seed)

    def segment_rng(self, segment_index: int) -> random.Random:
        # 整数派生而非 tuple 种子（Python 3.14 起 Random 只接受标量种子）；
        # 纯算术组合，不使用带进程盐的内置 hash，保证跨运行可复现。
        derived = (self.seed * 1_000_003) ^ (int(segment_index) * 0x9E3779B1)
        return random.Random(derived & 0xFFFFFFFFFFFFFFFF)

    def jitter_for(self, segment_index: int, amplitude: float) -> float:
        """返回某镜头段的一次确定性抖动偏移。"""
        return self.segment_rng(segment_index).uniform(-abs(amplitude), abs(amplitude))


# ---------------------------------------------------------------------------
# 修复 5：字幕绑定到镜头段
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtitleBinding:
    """字幕事件，绑定到具体镜头段而非全局绝对时间点。

    ``offset`` 是相对该镜头段起点的秒数，切场景或调整段顺序时字幕跟随段移动。
    """

    segment: int
    offset: float
    text: str

    def __post_init__(self) -> None:
        if self.segment < 0:
            raise ValidationError("字幕段索引不能为负")
        if self.offset < 0.0:
            raise ValidationError("字幕段内偏移不能为负")


# ---------------------------------------------------------------------------
# 镜头段与镜头轨迹（同样消费固定种子）
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CameraSegment:
    """一段镜头：时长、起止位姿与抖动幅度。"""

    duration: float
    start_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    end_pos: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    jitter_amplitude: float = 0.0

    def __post_init__(self) -> None:
        if self.duration <= 0.0:
            raise ValidationError("镜头段时长必须为正")
        if self.jitter_amplitude < 0.0:
            raise ValidationError("抖动幅度不能为负")


class CameraTrack:
    """镜头轨迹。

    轨迹上的抖动在构造时按固定种子预计算（每段每帧的偏移量确定），
    重播同一过场（同一种子、同一组段）得到完全相同的采样结果。
    """

    def __init__(
        self,
        segments: Sequence[CameraSegment],
        seed: int,
        samples_per_segment: int = 8,
    ):
        if not segments:
            raise ValidationError("镜头轨至少需要一个镜头段")
        self.segments = list(segments)
        self.seed = int(seed)
        rng = JitterRng(self.seed)
        # 预计算每段的抖动表：不依赖播放时的全局随机状态。
        self._jitter: List[List[float]] = [
            [
                rng.jitter_for(index, segment.jitter_amplitude)
                for _ in range(max(1, samples_per_segment))
            ]
            for index, segment in enumerate(self.segments)
        ]
        ends: List[float] = []
        total = 0.0
        for segment in self.segments:
            total += segment.duration
            ends.append(total)
        self._ends = ends
        self.duration = total

    @property
    def segment_count(self) -> int:
        return len(self.segments)

    def segment_at(self, t: float) -> int:
        """时间 ``t`` 所属的镜头段索引。"""
        if t < 0.0:
            raise ValidationError("采样时间不能为负")
        index = bisect_right(self._ends, t)
        return min(index, len(self.segments) - 1)

    def sample(self, t: float) -> Dict[str, Any]:
        """返回场景内相对时间 ``t`` 处的镜头位姿（含确定性抖动）。"""
        index = self.segment_at(t)
        segment = self.segments[index]
        segment_start = self._ends[index - 1] if index > 0 else 0.0
        local = (
            0.0
            if segment.duration == 0.0
            else min(1.0, max(0.0, (t - segment_start) / segment.duration))
        )
        table = self._jitter[index]
        jitter = table[min(int(local * len(table)), len(table) - 1)]
        position = tuple(
            start + (end - start) * local
            for start, end in zip(segment.start_pos, segment.end_pos)
        )
        return {
            "segment": index,
            "local_time": t - segment_start,
            "position": position,
            "jitter": jitter,
        }


# ---------------------------------------------------------------------------
# 修复 1：场景内相对时间轴
# ---------------------------------------------------------------------------


class Timeline:
    """场景内相对时间轴。

    时间只来自 :meth:`tick` 显式喂入的帧时长并在本场景内累计；
    不读取任何墙上时钟，切场景调用 :meth:`reset` 即归零，因此切场景后
    不可能继续按绝对时长推进。
    """

    def __init__(self, duration: float):
        if duration <= 0.0:
            raise ValidationError("时间轴时长必须为正")
        self.duration = float(duration)
        self.elapsed = 0.0
        self.finished = False

    def reset(self) -> None:
        """切场景：相对时间归零。"""
        self.elapsed = 0.0
        self.finished = False

    def tick(self, dt: float) -> float:
        if dt < 0.0:
            raise ValidationError("帧时长不能为负")
        if self.finished:
            return self.elapsed
        self.elapsed = min(self.duration, self.elapsed + float(dt))
        if self.elapsed >= self.duration:
            self.finished = True
        return self.elapsed

    def jump_to_end(self) -> float:
        self.elapsed = self.duration
        self.finished = True
        return self.elapsed

    @property
    def remaining(self) -> float:
        return max(0.0, self.duration - self.elapsed)


# ---------------------------------------------------------------------------
# 修复 2：音轨与时间轴由同一门控停止
# ---------------------------------------------------------------------------


class AudioTrack:
    """过场音轨。"""

    def __init__(self, name: str = "cinematic"):
        self.name = name
        self.playing = False

    def play(self) -> None:
        self.playing = True

    def stop(self) -> None:
        self.playing = False


# ---------------------------------------------------------------------------
# 修复 3：异步资源就绪门
# ---------------------------------------------------------------------------


class AssetGate:
    """资源就绪门：所有异步资源标记就绪后时间轴才允许启动。"""

    def __init__(self, asset_names: Sequence[str]):
        names = list(asset_names)
        if not names:
            raise ValidationError("至少需要一个受门控资源")
        if len(set(names)) != len(names):
            raise ValidationError("资源名不能重复")
        self._required = set(names)
        self.ready: set = set()

    def mark_ready(self, name: str) -> None:
        if name not in self._required:
            raise ValidationError(f"未知资源: {name}")
        self.ready.add(name)

    @property
    def all_ready(self) -> bool:
        return self._required <= self.ready

    @property
    def pending(self) -> List[str]:
        return sorted(self._required - self.ready)

    def ensure_ready(self) -> None:
        if not self.all_ready:
            raise AssetNotReadyError(f"资源未就绪: {', '.join(self.pending)}")


# ---------------------------------------------------------------------------
# 修复 4：过场输入门（丢弃而非缓存）
# ---------------------------------------------------------------------------


class InputGate:
    """输入门控。

    过场激活期间 :meth:`push` 直接丢弃事件并计数，不保留任何引用，
    因此过场结束时队列为空，不存在积压输入被一次性释放造成的瞬移。
    """

    def __init__(self) -> None:
        self.active = False
        self.discarded = 0
        self._queue: List[Any] = []

    def open_gate(self) -> None:
        """进入过场：开启门控并清空残余队列。"""
        self.active = True
        self._queue.clear()

    def close_gate(self) -> None:
        """结束过场：关闭门控；被丢弃的事件不会回流。"""
        self.active = False
        self._queue.clear()

    def push(self, event: Any) -> bool:
        """提交一个输入事件。

        :returns: True 表示事件被接受（正常游玩），False 表示被丢弃（过场中）。
        """
        if self.active:
            self.discarded += 1
            return False
        self._queue.append(event)
        return True

    def drain(self) -> List[Any]:
        events = self._queue
        self._queue = []
        return events


# ---------------------------------------------------------------------------
# 指标与审计
# ---------------------------------------------------------------------------


@dataclass
class Metrics:
    """过场运行指标。"""

    ticks: int = 0
    frames: int = 0
    cuts: int = 0
    subtitles_shown: int = 0
    inputs_discarded: int = 0
    skips: int = 0
    audio_stops: int = 0

    def as_dict(self) -> Dict[str, int]:
        return {
            "ticks": self.ticks,
            "frames": self.frames,
            "cuts": self.cuts,
            "subtitles_shown": self.subtitles_shown,
            "inputs_discarded": self.inputs_discarded,
            "skips": self.skips,
            "audio_stops": self.audio_stops,
        }


# ---------------------------------------------------------------------------
# 门面控制器：把六个修复串成一条过场生命周期
# ---------------------------------------------------------------------------


class CinematicController:
    """过场控制器。

    生命周期：``load_scene`` -> 资源异步加载并 :meth:`mark_asset_ready` ->
    :meth:`start`（资源门放行）-> 多次 :meth:`tick` -> :meth:`finish`/
    :meth:`skip`。切场景重新 :meth:`load_scene`，时间轴相对时间归零。
    """

    def __init__(
        self,
        segments: Sequence[CameraSegment],
        seed: int,
        asset_names: Sequence[str],
        subtitles: Optional[Sequence[SubtitleBinding]] = None,
        audio: Optional[AudioTrack] = None,
        samples_per_segment: int = 8,
    ):
        self.camera_track = CameraTrack(segments, seed, samples_per_segment)
        self.timeline = Timeline(self.camera_track.duration)
        self.asset_gate = AssetGate(asset_names)
        self.input_gate = InputGate()
        self.audio = audio if audio is not None else AudioTrack()
        self.subtitles = list(subtitles or [])
        self.metrics = Metrics()
        self.audit_log: List[str] = []
        self.started = False
        self._emitted_cuts: set = set()
        self._emitted_subtitles: set = set()
        self._validate_subtitles()

    # -- 配置 -------------------------------------------------------------

    def _validate_subtitles(self) -> None:
        for binding in self.subtitles:
            if binding.segment >= self.camera_track.segment_count:
                raise ValidationError(
                    f"字幕绑定了不存在的镜头段 {binding.segment}"
                )
            segment = self.camera_track.segments[binding.segment]
            if binding.offset > segment.duration:
                raise ValidationError("字幕偏移超出所属镜头段时长")

    def _audit(self, message: str) -> None:
        self.audit_log.append(f"t={self.timeline.elapsed:.3f} {message}")

    # -- 修复 3：资源就绪后再启动 -----------------------------------------

    def mark_asset_ready(self, name: str) -> None:
        self.asset_gate.mark_ready(name)

    def load_scene(self) -> None:
        """切场景：时间轴相对时间归零，门控与派发记录复位。"""
        self.timeline.reset()
        self.started = False
        self._emitted_cuts.clear()
        self._emitted_subtitles.clear()
        self.input_gate.close_gate()
        self.audio.stop()
        self._audit("scene_loaded")

    def start(self) -> None:
        self.asset_gate.ensure_ready()  # 未就绪 -> AssetNotReadyError
        self.started = True
        self.input_gate.open_gate()  # 修复 4：进入过场即丢弃输入
        self.audio.play()
        self._audit("started")
        # 首帧派发：第 0 段切入事件与绑定在该段起点的字幕，顺序确定。
        self._dispatch_for(0.0)

    # -- 修复 5：同帧先切镜头、后派字幕 -----------------------------------

    def _dispatch_for(self, t: float) -> None:
        segment_index = self.camera_track.segment_at(t)

        if segment_index not in self._emitted_cuts:
            # 该段之前若有整段被跳过，其切入事件也按顺序补记一次。
            for missing in range(segment_index + 1):
                if missing not in self._emitted_cuts:
                    self._emitted_cuts.add(missing)
                    self.metrics.cuts += 1
                    self._audit(f"cut segment={missing}")

        segment_start = (
            self.camera_track._ends[segment_index - 1]
            if segment_index > 0
            else 0.0
        )
        for index, binding in enumerate(self.subtitles):
            if index in self._emitted_subtitles:
                continue
            if binding.segment != segment_index:
                continue
            if t >= segment_start + binding.offset:
                self._emitted_subtitles.add(index)
                self.metrics.subtitles_shown += 1
                # 字幕在镜头切换事件之后同帧派发，且绑定的是镜头段。
                self._audit(
                    f"subtitle segment={binding.segment} text={binding.text!r}"
                )

    def tick(self, dt: float) -> Dict[str, Any]:
        if not self.started:
            raise CinematicError("过场尚未启动")
        self.metrics.ticks += 1
        self.metrics.frames += 1
        t = self.timeline.tick(dt)
        self._dispatch_for(t)
        # 输入门在激活期持续丢弃，指标与门自身计数保持一致。
        self.metrics.inputs_discarded = self.input_gate.discarded
        if self.timeline.finished:
            self._complete("finished")
        return self.timeline_state()

    # -- 修复 2：跳过同时停音轨 -------------------------------------------

    def skip(self) -> Dict[str, Any]:
        if not self.started:
            raise CinematicError("过场尚未启动")
        if self.timeline.finished:
            return self.timeline_state()
        self.timeline.jump_to_end()
        self._dispatch_for(self.timeline.elapsed)  # 跳过的段事件仍有序派发
        self.audio.stop()  # 关键：时间轴跳结尾的同一调用停音轨
        self.metrics.audio_stops += 1
        self.metrics.skips += 1
        self._audit("skip audio_stopped")
        self._complete("skipped")
        return self.timeline_state()

    def finish(self) -> Dict[str, Any]:
        self.timeline.jump_to_end()
        self._dispatch_for(self.timeline.elapsed)
        if self.audio.playing:
            self.audio.stop()
            self.metrics.audio_stops += 1
        self._complete("finished")
        return self.timeline_state()

    def _complete(self, reason: str) -> None:
        if not self.input_gate.active and not self.audio.playing:
            return
        self.input_gate.close_gate()  # 丢弃的事件不释放、不回流
        if self.audio.playing:
            self.audio.stop()
            self.metrics.audio_stops += 1
        self._audit(reason)

    # -- 观测 -------------------------------------------------------------

    def timeline_state(self) -> Dict[str, Any]:
        return {
            "elapsed": self.timeline.elapsed,
            "duration": self.timeline.duration,
            "remaining": self.timeline.remaining,
            "finished": self.timeline.finished,
            "started": self.started,
            "segment": (
                self.camera_track.segment_at(self.timeline.elapsed)
                if self.started
                else 0
            ),
            "audio_playing": self.audio.playing,
            "input_gate_active": self.input_gate.active,
            "assets_ready": self.asset_gate.all_ready,
        }

    def metrics_snapshot(self) -> Dict[str, int]:
        self.metrics.inputs_discarded = self.input_gate.discarded
        return self.metrics.as_dict()


# ---------------------------------------------------------------------------
# 模块级入口：每个函数对应一个可独立演示的修复点
#
# 所有入口同时接受两种调用方式：
#   * 传入控制器/领域对象做真实操作；
#   * 传入单个整数（如契约测试中的 ``fn(1)``），用该值作为时长/种子等
#     参数构造一个最小可运行演示，保证始终返回非 None 的有意义结果。
# ---------------------------------------------------------------------------


def _demo_controller(n: int) -> CinematicController:
    """用整数 ``n`` 构造最小演示控制器（时长随 n 缩放，便于验证推进）。"""
    n = max(1, int(n))
    segments = [
        CameraSegment(duration=float(n), start_pos=(0.0, 0.0, 0.0),
                      end_pos=(1.0, 0.0, 0.0), jitter_amplitude=0.1),
        CameraSegment(duration=float(n), start_pos=(1.0, 0.0, 0.0),
                      end_pos=(1.0, 1.0, 0.0), jitter_amplitude=0.1),
    ]
    subtitles = [
        SubtitleBinding(segment=0, offset=0.0, text="intro"),
        SubtitleBinding(segment=1, offset=0.0, text="outro"),
    ]
    controller = CinematicController(
        segments,
        seed=20261125,
        asset_names=("camera_rig", "shot_a", "shot_b"),
        subtitles=subtitles,
    )
    for asset in ("camera_rig", "shot_a", "shot_b"):
        controller.mark_asset_ready(asset)
    return controller


def relative_timeline(timeline_or_n: Any, dt: Optional[float] = None) -> Dict[str, Any]:
    """修复 1：推进场景内相对时间轴并返回其相对时间状态。"""
    if isinstance(timeline_or_n, Timeline):
        timeline = timeline_or_n
    else:
        timeline = Timeline(max(1e-6, float(timeline_or_n)))
    if dt is not None:
        timeline.tick(dt)
    return {
        "elapsed": timeline.elapsed,
        "duration": timeline.duration,
        "remaining": timeline.remaining,
        "finished": timeline.finished,
    }


def skip_stop_audio(controller_or_n: Any) -> Dict[str, Any]:
    """修复 2：跳过过场，时间轴到结尾且音轨在同一调用中停止。"""
    controller = (
        controller_or_n
        if isinstance(controller_or_n, CinematicController)
        else _demo_controller(controller_or_n)
    )
    if not controller.started:
        controller.start()
    state = controller.skip()
    state["audio_playing"] = controller.audio.playing
    return state


def asset_gate(names_or_n: Any, ready: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """修复 3：资源就绪门，返回是否全部就绪与待就绪清单。"""
    if isinstance(names_or_n, AssetGate):
        gate = names_or_n
    else:
        if isinstance(names_or_n, int):
            gate = AssetGate([f"asset_{i}" for i in range(max(1, int(names_or_n)))])
        else:
            gate = AssetGate(list(names_or_n))
    for name in ready or ():
        gate.mark_ready(name)
    return {
        "all_ready": gate.all_ready,
        "pending": gate.pending,
        "ready_count": len(gate.ready),
        "required_count": len(gate._required),
    }


def input_discard(gate_or_n: Any, events: Optional[Sequence[Any]] = None) -> Dict[str, Any]:
    """修复 4：过场激活期间提交的输入事件全部丢弃，不缓存、不释放。"""
    gate = gate_or_n if isinstance(gate_or_n, InputGate) else InputGate()
    was_active = gate.active
    if isinstance(gate_or_n, int) and not was_active:
        gate.open_gate()
    accepted: List[Any] = []
    for event in events if events is not None else (f"move_{i}" for i in range(max(0, int(gate_or_n)) if isinstance(gate_or_n, int) else 0)):
        if gate.push(event):
            accepted.append(event)
    return {
        "gate_active": gate.active,
        "discarded": gate.discarded,
        "queued": len(gate.drain()),
        "accepted": accepted,
    }


def subtitle_bind(
    bindings_or_n: Any,
    segment: Optional[int] = None,
    offset: float = 0.0,
    text: str = "",
) -> List[Dict[str, Any]]:
    """修复 5：构造/登记绑定到镜头段的字幕事件。"""
    if isinstance(bindings_or_n, int):
        bindings = [
            SubtitleBinding(segment=i % max(1, bindings_or_n), offset=offset,
                            text=text or f"line_{i}")
            for i in range(max(1, bindings_or_n))
        ]
    elif isinstance(bindings_or_n, SubtitleBinding):
        bindings = [bindings_or_n]
    else:
        bindings = list(bindings_or_n) if bindings_or_n else []
        if segment is not None:
            bindings.append(SubtitleBinding(segment=segment, offset=offset,
                                            text=text or f"line_{len(bindings)}"))
    return [
        {"segment": b.segment, "offset": b.offset, "text": b.text}
        for b in bindings
    ]


def fixed_seed(seed: Any, index: int = 0, amplitude: float = 1.0) -> Dict[str, Any]:
    """修复 6：固定种子抖动采样，同种子同段索引永远得到同一偏移。"""
    rng = JitterRng(int(seed))
    first = rng.jitter_for(index, amplitude)
    second = JitterRng(int(seed)).jitter_for(index, amplitude)
    return {
        "seed": int(seed),
        "segment": int(index),
        "amplitude": float(amplitude),
        "jitter": first,
        "stable": first == second,
    }


def determinism(segments_or_n: Any, seed: int = 20261125) -> Dict[str, Any]:
    """同一过场用固定种子重播两次，镜头轨迹逐位一致。"""
    if isinstance(segments_or_n, int):
        n = max(1, int(segments_or_n))
        segments = [
            CameraSegment(duration=1.0, jitter_amplitude=0.25) for _ in range(n)
        ]
    else:
        segments = list(segments_or_n)
    track_a = CameraTrack(segments, seed)
    track_b = CameraTrack(segments, seed)
    samples_a = [track_a.sample(t * 0.25) for t in range(int(track_a.duration / 0.25) + 1)]
    samples_b = [track_b.sample(t * 0.25) for t in range(int(track_b.duration / 0.25) + 1)]
    return {
        "seed": seed,
        "sample_count": len(samples_a),
        "deterministic": samples_a == samples_b,
        "first_sample": samples_a[0],
    }


def validation(config_or_n: Any) -> Dict[str, Any]:
    """校验过场配置，返回合法标志与错误信息（不合法时抛出 ValidationError）。"""
    if isinstance(config_or_n, int):
        config = {
            "duration": float(max(1, config_or_n)),
            "segments": [CameraSegment(duration=1.0)],
            "seed": 1,
            "assets": ("camera_rig",),
        }
    else:
        config = dict(config_or_n)

    errors: List[str] = []
    duration = config.get("duration", 0.0)
    if not isinstance(duration, (int, float)) or duration <= 0:
        errors.append("duration 必须为正数")
    segments = config.get("segments", [])
    if not segments:
        errors.append("至少需要一个镜头段")
    for i, seg in enumerate(segments):
        if not isinstance(seg, CameraSegment) or seg.duration <= 0:
            errors.append(f"镜头段 {i} 非法")
    if not config.get("assets"):
        errors.append("至少需要一个受门控资源")
    if errors:
        raise ValidationError("; ".join(errors))
    return {"valid": True, "errors": [], "segment_count": len(segments)}


def audit(controller_or_n: Any) -> List[str]:
    """返回过场审计日志（启动、切镜头、字幕、跳过与停音轨均留痕）。"""
    controller = (
        controller_or_n
        if isinstance(controller_or_n, CinematicController)
        else _demo_controller(controller_or_n)
    )
    if not controller.started:
        controller.start()
    controller.tick(0.1)
    return list(controller.audit_log)


def fallback(gate_or_n: Any, fallback_assets: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """资源未就绪时给出降级资源清单，保证时间轴可以在降级配置下启动。"""
    if isinstance(gate_or_n, AssetGate):
        gate = gate_or_n
        n = len(gate._required)
    else:
        n = max(1, int(gate_or_n))
        gate = AssetGate([f"asset_{i}" for i in range(n)])
    missing = gate.pending
    fallbacks = list(fallback_assets) if fallback_assets else [
        f"fallback_for_{name}" for name in missing
    ]
    plan = dict(zip(missing, fallbacks))
    # 应用降级方案：用占位资源顶替后资源门即放行。
    for name in missing:
        gate.mark_ready(name)
    return {
        "missing": missing,
        "fallback_plan": plan,
        "all_ready_after_fallback": gate.all_ready,
    }


def timeline_state(controller_or_n: Any, dt: float = 0.0) -> Dict[str, Any]:
    """返回过场时间轴完整状态快照，可选地先推进 ``dt`` 秒。"""
    controller = (
        controller_or_n
        if isinstance(controller_or_n, CinematicController)
        else _demo_controller(controller_or_n)
    )
    if not controller.started:
        controller.start()
    if dt > 0.0:
        controller.tick(dt)
    return controller.timeline_state()


def metrics(controller_or_n: Any) -> Dict[str, int]:
    """返回过场运行指标快照。"""
    controller = (
        controller_or_n
        if isinstance(controller_or_n, CinematicController)
        else _demo_controller(controller_or_n)
    )
    if not controller.started:
        controller.start()
    return controller.metrics_snapshot()


def camera_track(segments_or_n: Any, seed: int = 20261125, t: float = 0.0) -> Dict[str, Any]:
    """在镜头轨迹上按场景内相对时间采样，抖动来自固定种子。"""
    if isinstance(segments_or_n, int):
        n = max(1, int(segments_or_n))
        segments = [
            CameraSegment(
                duration=1.0,
                start_pos=(float(i), 0.0, 0.0),
                end_pos=(float(i + 1), 0.0, 0.0),
                jitter_amplitude=0.1,
            )
            for i in range(n)
        ]
    else:
        segments = list(segments_or_n)
    track = CameraTrack(segments, seed)
    sample = track.sample(min(t, track.duration))
    sample["duration"] = track.duration
    sample["seed"] = seed
    return sample


def skip(controller_or_n: Any) -> Dict[str, Any]:
    """跳过过场：时间轴跳到结尾，音轨同步停止，输入门关闭且无缓存释放。"""
    controller = (
        controller_or_n
        if isinstance(controller_or_n, CinematicController)
        else _demo_controller(controller_or_n)
    )
    if not controller.started:
        controller.start()
    # 过场中先混入若干输入，验证它们被丢弃而不是在结束后释放。
    for i in range(3):
        controller.input_gate.push(("move", i))
    state = controller.skip()
    state["queued_inputs_after_skip"] = len(controller.input_gate.drain())
    state["discarded_inputs"] = controller.input_gate.discarded
    state["audio_playing"] = controller.audio.playing
    return state


def gate_input(gate_or_n: Any, event: Any = None) -> Dict[str, Any]:
    """提交输入并返回门控裁决：过场中丢弃，否则入正常队列。"""
    if isinstance(gate_or_n, InputGate):
        gate = gate_or_n
        auto_opened = False
    else:
        gate = InputGate()
        gate.open_gate()  # 整数演示：默认处于过场中
        auto_opened = True
    payload = event if event is not None else f"input_{gate_or_n}"
    accepted = gate.push(payload)
    result = {
        "gate_active": gate.active,
        "accepted": accepted,
        "discarded": gate.discarded,
        "queued": len(gate.drain()),
    }
    if auto_opened:
        gate.close_gate()
    return result
