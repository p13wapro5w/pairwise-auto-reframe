"""画面智能重构（auto reframe）核心工具。

把横屏素材转竖屏、以及同一素材输出多平台版本时最容易出错的环节拆成若干
纯函数（只依赖标准库，方便在预览、导出和测试中复用）：

* :func:`frame_track` —— 逐帧重新关联检测结果，不再复用上一帧的框；
* :func:`smooth_path` / :func:`smooth_trajectory` —— 带死区、限速的指数平滑，
  替代关键帧之间的线性插值，消除镜头一顿一顿地追；
* :func:`safe_area` —— 按 **目标画幅** 计算安全区，而不是沿用 16:9；
* :func:`platform_consistent` —— 同一份归一化主体轨迹投影到所有平台，
  保证主体在各版本中的画面位置一致；
* :func:`subtitle_reflow` —— 字幕避让人脸并落在目标安全区内；
* :func:`zoom_limit` —— 放大不超过源分辨率；
* :func:`shot_detect` —— 镜头切换检测，切换时重置跟踪；
* :func:`audio_follow` —— 声像跟随主体，主体出画时快速衰减；
* :func:`coord_unify` / :func:`export_preview` —— 预览与导出共用一套归一化坐标；
* :func:`rotation_once` —— 旋转元数据只应用一次；
* :func:`crop_window` / :func:`subject_score` / :func:`tracking_loss` /
  :func:`keyframe_interval` —— 构图窗口、主体打分、跟丢状态机与关键帧节奏。

几何约定：除显式接收 ``width``/``height``/``size`` 的函数外，坐标与尺寸
均使用 **左上角为原点、归一化到 [0, 1]** 的坐标；主体在画面中的相对位置
也用它表示。越界的坐标在构图时会夹紧到画面边缘并在返回值中标注。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

__all__ = [
    "Box",
    "Point",
    "SafeArea",
    "TrackingDecision",
    "PlatformCrop",
    "SubtitlePlacement",
    "ShotCut",
    "AudioMix",
    "CoordPoint",
    "PreviewExportPoint",
    "RotationPlan",
    "TrackStatus",
    "KeyframePlan",
    "CropWindow",
    "ZoomLimit",
    "SubjectScore",
    "DEFAULT_PLATFORMS",
    "ease",
    "frame_track",
    "smooth_path",
    "smooth_trajectory",
    "safe_area",
    "platform_consistent",
    "subtitle_reflow",
    "zoom_limit",
    "shot_detect",
    "audio_follow",
    "coord_unify",
    "rotation_once",
    "tracking_loss",
    "keyframe_interval",
    "subject_score",
    "crop_window",
    "export_preview",
]

_EPS = 1e-9
_DIAG = math.sqrt(2.0)  # 归一化画面中心到角点的距离

#: 平台 → (目标宽, 目标高) 画幅。仅作为未显式给定时的一致性基准。
DEFAULT_PLATFORMS: dict[str, tuple[int, int]] = {
    "landscape": (1920, 1080),
    "square": (1080, 1080),
    "vertical": (1080, 1920),
}


# ---------------------------------------------------------------------------
# 基础类型与内部工具
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Box:
    """归一化矩形，``(x, y)`` 为左上角，``(w, h)`` 为宽高。"""

    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2.0, self.y + self.h / 2.0)

    @property
    def area(self) -> float:
        return self.w * self.h

    def iou(self, other: Box) -> float:
        ix = max(0.0, min(self.right, other.right) - max(self.x, other.x))
        iy = max(0.0, min(self.bottom, other.bottom) - max(self.y, other.y))
        inter = ix * iy
        union = self.area + other.area - inter
        return inter / union if union > _EPS else 0.0

    def distance_to(self, other: Box) -> float:
        """中心点之间的欧氏距离（归一化单位）。"""
        cx, cy = self.center
        ox, oy = other.center
        return math.hypot(cx - ox, cy - oy)

    def as_dict(self) -> dict[str, float]:
        return {"x": self.x, "y": self.y, "w": self.w, "h": self.h}


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class SafeArea:
    """目标画幅的可视安全区（归一化，已被夹紧到 [0, 1]）。"""

    x: float
    y: float
    w: float
    h: float
    aspect: float
    source_aspect: float

    @property
    def box(self) -> Box:
        return Box(self.x, self.y, self.w, self.h)

    def contains(self, x: float, y: float) -> bool:
        return (
            self.x - _EPS <= x <= self.x + self.w + _EPS
            and self.y - _EPS <= y <= self.y + self.h + _EPS
        )

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "aspect": self.aspect,
        }


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_box_like(value: object) -> bool:
    if isinstance(value, Box):
        return True
    if isinstance(value, Mapping):
        return "x" in value and "y" in value and (
            ("w" in value and "h" in value)
            or ("width" in value and "height" in value)
        )
    if isinstance(value, (tuple, list)):
        return len(value) == 4 and all(_is_number(v) for v in value)
    return False


def _as_box(item: object) -> Box:
    """把 ``Box`` / ``(x, y, w, h)`` / ``{x,y,w,h}`` 统一成 Box；无法解析时给零框。"""
    if isinstance(item, Box):
        return item
    if isinstance(item, Mapping):
        return Box(
            float(item.get("x", 0.0)),
            float(item.get("y", 0.0)),
            float(item.get("w", item.get("width", 0.0))),
            float(item.get("h", item.get("height", 0.0))),
        )
    if isinstance(item, (tuple, list)) and len(item) >= 4:
        return Box(*(float(v) for v in tuple(item)[:4]))
    return Box(0.0, 0.0, 0.0, 0.0)


def _clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def _clamp(value: float, lo: float, hi: float) -> float:
    return lo if value < lo else hi if value > hi else value


def _clip_box(box: Box) -> Box:
    x = _clamp(box.x, 0.0, 1.0)
    y = _clamp(box.y, 0.0, 1.0)
    w = _clamp(box.w, 0.0, 1.0 - x)
    h = _clamp(box.h, 0.0, 1.0 - y)
    return Box(x, y, w, h)


def _iter_or_empty(value: object) -> list:
    """可迭代则展开；标量、字符串、映射或 None 一律视为空。"""
    if value is None or isinstance(value, (str, bytes, Mapping)) or _is_number(value):
        return []
    try:
        return list(value)  # type: ignore[arg-type]
    except TypeError:
        return []


def _coerce_size(value: object, default: tuple[float, float] = (1920.0, 1080.0)) -> tuple[float, float]:
    """把 ``(w, h)`` / ``{w,h}`` / 标量统一成 (w, h)。"""
    if isinstance(value, Mapping):
        w = value.get("w", value.get("width"))
        h = value.get("h", value.get("height"))
        if _is_number(w) and _is_number(h) and float(w) > 0 and float(h) > 0:
            return (float(w), float(h))
        return default
    if isinstance(value, (tuple, list)):
        seq = list(value)
        if len(seq) >= 2 and all(_is_number(v) for v in seq[:2]):
            w, h = float(seq[0]), float(seq[1])
            if w > 0 and h > 0:
                return (w, h)
        return default
    if _is_number(value) and float(value) > 0:
        return (float(value), float(value))
    return default


def _parse_aspect(aspect: object) -> float:
    """接受 ``16/9``、``"16:9"``、``(16, 9)``，返回宽高比 ``w/h``。"""
    if isinstance(aspect, str):
        text = aspect.strip()
        if ":" in text:
            a, b = text.split(":", 1)
            ratio = float(a) / float(b)
        else:
            ratio = float(text)
    elif isinstance(aspect, (tuple, list)):
        if len(aspect) != 2:
            raise ValueError(f"画幅元组需要两个元素: {aspect!r}")
        ratio = float(aspect[0]) / float(aspect[1])
    else:
        ratio = float(aspect)  # type: ignore[arg-type]
    if ratio <= 0 or not math.isfinite(ratio):
        raise ValueError(f"画幅必须为正: {aspect!r}")
    return ratio


def _parse_rotation(rotation: object) -> int:
    """把角度或元数据字符串规整为 0/90/180/270。"""
    if isinstance(rotation, str):
        text = rotation.strip().lower()
        tokens = "".join(ch if (ch.isdigit() or ch == "-") else " " for ch in text).split()
        if tokens:
            base = int(tokens[0])
            if "ccw" in text or "left" in text or "反" in text:
                base = -base
            return int(round(base / 90.0)) % 4 * 90
        if text in {"portrait", "vertical", "竖屏"}:
            return 90
        if text in {"landscape", "horizontal", "identity", "none", "横屏"}:
            return 0
        raise ValueError(f"无法解析旋转元数据: {rotation!r}")
    try:
        value = float(rotation)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"无法解析旋转元数据: {rotation!r}") from exc
    return int(round(value / 90.0)) % 4 * 90


def ease(t: float) -> float:
    """平滑插值曲线（smoothstep），用于关键帧之间的过渡而非线性插值。"""
    t = _clamp01(float(t))
    return t * t * (3.0 - 2.0 * t)


# ---------------------------------------------------------------------------
# 逐帧跟踪
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TrackStatus:
    """跟踪状态快照。"""

    state: str  # "tracking" | "predicted" | "lost"
    lost_frames: int
    confidence: float


@dataclass(frozen=True)
class TrackingDecision:
    """单帧跟踪决策。

    ``box`` 是 **本帧** 使用的框（来自本帧检测或由速度外推），
    ``source`` 说明它从哪来，``reacquired`` 表示是否刚从跟丢状态找回。
    """

    frame: int
    box: Box
    source: str
    iou: float
    reacquired: bool
    status: TrackStatus

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "box": self.box.as_dict(),
            "source": self.source,
            "iou": self.iou,
            "reacquired": self.reacquired,
            "state": self.status.state,
            "lost_frames": self.status.lost_frames,
            "confidence": self.status.confidence,
        }


def _frame_candidates(frame: object) -> list[Box]:
    """把一帧的检测结果规整为 Box 列表。"""
    if _is_box_like(frame):
        return [_clip_box(_as_box(frame))]
    boxes = []
    for item in _iter_or_empty(frame):
        if _is_box_like(item):
            boxes.append(_clip_box(_as_box(item)))
    return [b for b in boxes if b.w > _EPS and b.h > _EPS]


def frame_track(
    detections: Iterable[object],
    max_jump: float = 0.25,
    min_iou: float = 0.05,
    max_lost: int = 15,
    velocity_damping: float = 1.0,
    start_frame: int = 0,
) -> list[TrackingDecision]:
    """逐帧关联检测结果，得到每帧真正使用的跟踪框。

    与 "沿用上一帧的框" 不同，这里每帧都优先使用 **本帧的检测**，只在检测
    缺失时用上一帧速度外推 ``max_lost`` 帧，超过即判为跟丢并清空状态；
    重新出现且与预测框吻合时标记 ``reacquired``（此时重置平滑，避免跳变）。

    ``detections`` 每帧是一个可迭代的主体框集合（Box / (x,y,w,h) / dict），
    取与本帧预测最匹配的一个。可选参数：

    * ``max_jump``：相邻帧中心位移超过该值（归一化）视为不可信；
    * ``min_iou``：检测与预测框的最小 IoU，低于它且存在多个候选时视为不可信；
    * ``max_lost``：允许外推的最大帧数；
    * ``velocity_damping``：外推速度衰减，1.0 为匀速外推。

    快速移动的主体因为每帧重新检测（并结合位移门限关联）而不会跟丢；
    镜头切换由 :func:`shot_detect` 单独处理，切换点应重新调用本函数。
    """
    max_lost = max(0, int(max_lost))
    frames = _iter_or_empty(detections)
    decisions: list[TrackingDecision] = []

    last_box: Box | None = None
    velocity = (0.0, 0.0)  # 每帧中心位移
    lost = 0

    for idx, raw_frame in enumerate(frames):
        frame_no = start_frame + idx
        cands = _frame_candidates(raw_frame)

        predicted: Box | None = None
        if last_box is not None:
            vx, vy = velocity
            predicted = Box(last_box.x + vx, last_box.y + vy, last_box.w, last_box.h)
        prev = last_box

        chosen: Box | None = None
        iou = 0.0
        reacquired = lost > 0
        if cands:
            if predicted is None:
                # 首帧或跟丢后重新出现：取面积最大的主体
                chosen = max(cands, key=lambda b: b.area)
            else:
                ranked = sorted(cands, key=lambda b: predicted.iou(b), reverse=True)
                best = ranked[0]
                iou = predicted.iou(best)
                jump = predicted.distance_to(best)
                # 单候选时相信本帧检测（快速移动的主体不会被位移门限甩掉）；
                # 多候选时才用 IoU / 位移门限挑选，避免跳到别人身上。
                if len(cands) == 1 or (jump <= max_jump and iou >= min_iou):
                    chosen = best

        if chosen is not None:
            if prev is not None:
                cx, cy = chosen.center
                px, py = prev.center
                # 速度 = 本帧检测相对上一帧输出框的位移，供下一帧外推
                velocity = (cx - px, cy - py)
            else:
                velocity = (0.0, 0.0)
            lost = 0
            state = "reacquired" if reacquired else "tracking"
            last_box = chosen
            box = chosen
            source = "detection"
        else:
            lost += 1
            if predicted is not None and lost <= max_lost:
                state = "predicted"
                last_box = predicted
                box = predicted
                source = "prediction"
                velocity = (
                    velocity[0] * velocity_damping,
                    velocity[1] * velocity_damping,
                )
            else:
                state = "lost"
                if lost > max_lost:
                    last_box = None
                    velocity = (0.0, 0.0)
                box = predicted if predicted is not None else Box(0.5, 0.5, 0.0, 0.0)
                source = "none"
                iou = 0.0

        if state == "tracking":
            confidence = 1.0
        elif state == "reacquired":
            confidence = 0.6
        else:
            confidence = max(0.0, 1.0 - lost / max(1, max_lost))
        decisions.append(
            TrackingDecision(
                frame=frame_no,
                box=box,
                source=source,
                iou=iou,
                reacquired=reacquired,
                status=TrackStatus(state=state, lost_frames=lost, confidence=confidence),
            )
        )

    return decisions


# ---------------------------------------------------------------------------
# 轨迹平滑
# ---------------------------------------------------------------------------


def _ema_with_deadband(
    values: Sequence[float], alpha: float, deadband: float, max_step: float
) -> list[float]:
    if not values:
        return []
    smoothed = [float(values[0])]
    for target in values[1:]:
        current = smoothed[-1]
        delta = target - current
        if abs(delta) <= deadband:
            smoothed.append(current)  # 死区：微小抖动不推动镜头
            continue
        step = _clamp(alpha * delta, -max_step, max_step)
        smoothed.append(current + step)
    return smoothed


def _coerce_points(items: object) -> list[tuple[float, float]]:
    """把逐帧中心规整为 (x, y) 列表。"""
    # 单个点对象 / 单个点映射
    if isinstance(items, Point):
        return [(items.x, items.y)]
    if (
        hasattr(items, "x")
        and hasattr(items, "y")
        and not isinstance(items, (tuple, list, Mapping))
    ):
        return [(float(getattr(items, "x")), float(getattr(items, "y")))]
    if isinstance(items, Mapping):
        if _is_number(items.get("x")) and _is_number(items.get("y")):
            return [(float(items["x"]), float(items["y"]))]

    seq = _iter_or_empty(items)
    if not seq:
        return []
    # 单个点：(x, y)
    if len(seq) == 2 and all(_is_number(v) for v in seq):
        return [(float(seq[0]), float(seq[1]))]
    pts: list[tuple[float, float]] = []
    for item in seq:
        if isinstance(item, Mapping):
            pts.append((float(item.get("x", 0.0)), float(item.get("y", 0.0))))
        elif isinstance(item, Point):
            pts.append((item.x, item.y))
        elif isinstance(item, (tuple, list)) and len(item) >= 2:
            pts.append((float(item[0]), float(item[1])))
        elif hasattr(item, "x") and hasattr(item, "y"):
            pts.append((float(getattr(item, "x")), float(getattr(item, "y"))))
    return pts


def smooth_trajectory(
    centers: Iterable[object],
    alpha: float = 0.25,
    deadband: float = 0.01,
    max_step: float = 0.08,
) -> list[Point]:
    """对逐帧主体中心做死区 + 限速的指数平滑。

    ``centers`` 可以是 ``(x, y)``、``{"x":..,"y":..}`` 或任意带 ``x``/``y``
    属性的对象。``deadband`` 抑制手抖，``max_step`` 限制每帧最大位移从而
    避免甩镜，指数响应则保证运动连续而不是关键帧跳变。
    """
    pts = _coerce_points(centers)
    if not pts:
        return []
    alpha = _clamp(float(alpha), 0.0, 1.0)
    xs = _ema_with_deadband([p[0] for p in pts], alpha, deadband, max_step)
    ys = _ema_with_deadband([p[1] for p in pts], alpha, deadband, max_step)
    return [Point(x, y) for x, y in zip(xs, ys)]


def _interp_eased(
    frames: Sequence[tuple[float, float, float]], t: float
) -> tuple[float, float]:
    """在有序关键帧之间用 smoothstep 求值。"""
    if t <= frames[0][0]:
        return (frames[0][1], frames[0][2])
    if t >= frames[-1][0]:
        return (frames[-1][1], frames[-1][2])
    for (ta, xa, ya), (tb, xb, yb) in zip(frames, frames[1:]):
        if ta <= t <= tb:
            span = tb - ta
            u = 0.0 if span <= _EPS else (t - ta) / span
            k = ease(u)
            return (xa + (xb - xa) * k, ya + (yb - ya) * k)
    return (frames[-1][1], frames[-1][2])


def smooth_path(
    keyframes: Iterable[object],
    alpha: float = 0.25,
    deadband: float = 0.01,
    max_step: float = 0.08,
    duration: int | None = None,
) -> list[Point]:
    """把关键帧轨迹展开成逐帧平滑路径。

    关键帧形如 ``(t, x, y)``（``t`` 为帧号，递增）或 ``(x, y)``（按顺序排布）。
    关键帧之间使用 smoothstep 缓动（而不是线性插值），再叠加死区与限速平滑，
    因此镜头运动是连续加速/减速的，不会一顿一顿地追。
    """
    seq = _iter_or_empty(keyframes)
    frames: list[tuple[float, float, float]] = []
    for item in seq:
        if isinstance(item, (tuple, list)) and len(item) == 3:
            frames.append((float(item[0]), float(item[1]), float(item[2])))
        elif isinstance(item, (tuple, list)) and len(item) == 2:
            t = float(frames[-1][0]) + 1.0 if frames else 0.0
            frames.append((t, float(item[0]), float(item[1])))
    if not frames:
        return []
    if len(frames) == 1:
        return [Point(frames[0][1], frames[0][2])]

    frames.sort(key=lambda f: f[0])
    t0, t_end = frames[0][0], frames[-1][0]
    span = t_end - t0
    if duration is None:
        duration = max(1, int(round(span)) + 1)
    duration = max(1, int(duration))

    raw: list[tuple[float, float]] = []
    for i in range(duration):
        if duration == 1 or span <= _EPS:
            t = t0
        else:
            t = t0 + span * i / (duration - 1)
        raw.append(_interp_eased(frames, t))

    return smooth_trajectory(raw, alpha=alpha, deadband=deadband, max_step=max_step)


# ---------------------------------------------------------------------------
# 安全区
# ---------------------------------------------------------------------------


def safe_area(
    width_or_aspect: object = (16, 9),
    height: float | None = None,
    margin: float = 0.05,
    source_aspect: object | None = None,
) -> SafeArea:
    """按 **目标画幅** 计算字幕/图形安全区（归一化）。

    推荐直接传目标输出尺寸：``safe_area(1080, 1920)``；也可以传画幅描述
    ``safe_area("9:16")``、``safe_area((9, 16))``。与把 16:9 的安全区直接
    套到竖屏不同，这里以目标画幅为基准、四边内缩 ``margin``，因此竖屏输出
    时字幕不会被切掉。

    ``height`` 小于 1 时视为误把 ``margin`` 传在第二位，会按 margin 处理
    （``safe_area("9:16", 0.1)`` 等价于 10% 边距）。

    ``margin`` 小于等于 1 时按比例理解（``0.05`` 即 5%）；大于 1 时按百分比
    理解（``9`` 即 9%），并限制在 45% 以内，避免安全区退化。
    """
    if height is not None and abs(float(height)) >= 1.0:
        target = float(width_or_aspect) / float(height)  # type: ignore[arg-type]
    else:
        if height is not None:
            margin = float(height)
        target = _parse_aspect(width_or_aspect)
    if target <= 0 or not math.isfinite(target):
        raise ValueError(f"画幅必须为正: {width_or_aspect!r}")

    margin_value = float(margin)
    if margin_value > 1.0:
        margin_value /= 100.0
    margin_value = _clamp(margin_value, 0.0, 0.45)

    # 安全区边距按 **目标画幅的短边像素** 等距内缩：
    # 这样竖屏（9:16）与横屏（16:9）得到真正不同的安全区，
    # 而不是把 16:9 的比例盒子直接套到竖屏上导致字幕被切。
    if height is not None and abs(float(height)) >= 1.0:
        canvas_w, canvas_h = float(width_or_aspect), float(height)  # type: ignore[arg-type]
    else:
        canvas_w, canvas_h = target, 1.0
    short_side = min(canvas_w, canvas_h)
    margin_x = margin_value * short_side / canvas_w
    margin_y = margin_value * short_side / canvas_h

    source = target if source_aspect is None else _parse_aspect(source_aspect)
    return SafeArea(
        x=margin_x,
        y=margin_y,
        w=1.0 - 2.0 * margin_x,
        h=1.0 - 2.0 * margin_y,
        aspect=target,
        source_aspect=source,
    )


# ---------------------------------------------------------------------------
# 多平台一致
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PlatformCrop:
    """单个平台上的构图结果。

    ``anchor`` 是所有平台共享的构图锚点（主体应落在输出画面中的相对位置），
    ``subject`` 是实际落点；两者之差 ``deviation`` 仅在几何上夹紧时大于 0。
    """

    name: str
    aspect: float
    crop: Box  # 在源中的归一化裁切窗口
    subject: Point  # 主体在该平台输出画面中的实际归一化位置
    anchor: Point  # 跨平台共享的构图锚点
    deviation: float
    consistent: bool
    safe_area: SafeArea

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "aspect": self.aspect,
            "crop": self.crop.as_dict(),
            "subject": self.subject.as_tuple(),
            "anchor": self.anchor.as_tuple(),
            "deviation": self.deviation,
            "consistent": self.consistent,
            "safe_area": self.safe_area.as_dict(),
        }


def _fit_crop(
    target_aspect: float,
    source_aspect: float,
    center: Point,
    max_zoom: float = 1.0,
    anchor: Point | None = None,
) -> Box:
    """在源画面内取目标画幅的窗口，并让 ``center`` 落在 ``anchor`` 处。

    返回归一化坐标的窗口。默认取最大窗口（不放大源）；``max_zoom > 1`` 时
    允许把窗口再缩小到 1/max_zoom，即最多放大 max_zoom 倍。锚点对应位置
    无法满足时（贴边）夹紧到画面边界。
    """
    if source_aspect <= _EPS or target_aspect <= _EPS:
        raise ValueError("画幅必须为正")
    if anchor is None:
        anchor = Point(0.5, 0.5)

    # 目标比源更"宽"：占满宽、裁高；否则占满高、裁宽
    if target_aspect >= source_aspect:
        crop_w, crop_h = 1.0, source_aspect / target_aspect
    else:
        crop_w, crop_h = target_aspect / source_aspect, 1.0

    # 最大窗口对应 1.0 倍放大；max_zoom>1 时允许进一步缩小窗口
    max_zoom = max(1.0, float(max_zoom))
    if max_zoom > 1.0:
        shrink = 1.0 / max_zoom
        crop_w *= shrink
        crop_h *= shrink

    x = _clamp(center.x - anchor.x * crop_w, 0.0, max(0.0, 1.0 - crop_w))
    y = _clamp(center.y - anchor.y * crop_h, 0.0, max(0.0, 1.0 - crop_h))
    return Box(x, y, crop_w, crop_h)


def platform_consistent(
    subject: object,
    source_size: object,
    platforms: object = None,
    max_zoom: float = 1.0,
    margin: float = 0.05,
    anchor: object | None = None,
) -> dict[str, PlatformCrop]:
    """同一份主体轨迹投影到多平台，得到彼此一致的构图。

    所有平台共享同一套归一化主体坐标 **和同一个构图锚点**，因此各版本里
    主体落在相同的相对位置，不会出现各算一套构图、主体位置差半屏的问题。
    几何上贴边、无法满足锚点时会被夹紧，并通过 ``deviation`` /
    ``consistent`` 如实报告。

    参数：

    * ``subject``：归一化主体框（Box / (x,y,w,h) / dict）；
    * ``source_size``：源分辨率 ``(w, h)``；
    * ``platforms``：``{name: (w, h)}`` 或 ``{name: aspect}``，默认
      :data:`DEFAULT_PLATFORMS`；
    * ``max_zoom``：允许的最大放大倍数，默认 1.0（不放大源）；
    * ``margin``：安全区边距；
    * ``anchor``：统一构图锚点（输出画面中的相对位置），默认取主体在源画面
      中的相对位置，相当于所有平台沿用同一套原始构图。
    """
    box = _clip_box(_as_box(subject))
    center = Point(*box.center)
    src_w, src_h = _coerce_size(source_size)
    source_aspect = src_w / src_h

    if anchor is None:
        anchor_pt = center
    else:
        pts = _coerce_points(anchor)
        anchor_pt = Point(_clamp01(pts[0][0]), _clamp01(pts[0][1])) if pts else center

    table = DEFAULT_PLATFORMS if platforms is None else platforms
    table_items = list(table.items()) if isinstance(table, Mapping) else []

    out: dict[str, PlatformCrop] = {}
    for name, spec in table_items:
        if isinstance(spec, (tuple, list)) and len(spec) == 2:
            pw, ph = float(spec[0]), float(spec[1])
            aspect = pw / ph if ph > _EPS else source_aspect
        elif _is_number(spec):
            aspect = _parse_aspect(spec)
        else:
            aspect = source_aspect

        crop = _fit_crop(
            aspect, source_aspect, center, max_zoom=max_zoom, anchor=anchor_pt
        )
        rx = (center.x - crop.x) / crop.w if crop.w > _EPS else anchor_pt.x
        ry = (center.y - crop.y) / crop.h if crop.h > _EPS else anchor_pt.y
        pos = Point(_clamp01(rx), _clamp01(ry))
        deviation = max(abs(pos.x - anchor_pt.x), abs(pos.y - anchor_pt.y))
        out[str(name)] = PlatformCrop(
            name=str(name),
            aspect=aspect,
            crop=crop,
            subject=pos,
            anchor=anchor_pt,
            deviation=round(deviation, 6),
            consistent=deviation <= 1e-6,
            safe_area=safe_area(aspect, margin=margin),
        )
    return out


# ---------------------------------------------------------------------------
# 字幕重排
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtitlePlacement:
    """字幕最终位置（归一化）。"""

    box: Box
    position: str  # "bottom" | "top" | "left" | "right"
    moved: bool
    overlaps_subject: bool
    inside_safe_area: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "box": self.box.as_dict(),
            "position": self.position,
            "moved": self.moved,
            "overlaps_subject": self.overlaps_subject,
            "inside_safe_area": self.inside_safe_area,
        }


def _overlap_area(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.right, b.right) - max(a.x, b.x))
    iy = max(0.0, min(a.bottom, b.bottom) - max(a.y, b.y))
    return ix * iy


def subtitle_reflow(
    subject: object,
    safe: object,
    subtitle_height: float = 0.12,
    preferred: str = "bottom",
    padding: float = 0.02,
) -> SubtitlePlacement:
    """把字幕在目标安全区内重新排布，避免压在人脸上。

    在安全区内尝试 ``preferred``（默认底部）以及其余三个方向，选择与主体
    重叠面积最小的位置；全部候选都压到主体时，选择重叠最小者并如实标记
    ``overlaps_subject=True``。``subtitle_height`` 为字幕条高度（归一化）。

    与 "重构后字幕不重排" 不同，主体位置改变后字幕会跟随安全区移动。
    """
    box = _clip_box(_as_box(subject))
    area = safe.box if isinstance(safe, SafeArea) else _clip_box(_as_box(safe))

    h = _clamp(float(subtitle_height), 0.0, area.h)
    pad = max(0.0, float(padding))
    full_w = area.w
    side_w = max(0.0, min(full_w, area.w * 0.45))

    candidates: dict[str, Box] = {
        "bottom": Box(area.x, area.bottom - h, full_w, h),
        "top": Box(area.x, area.y, full_w, h),
        "left": Box(area.x, area.center[1] - h / 2.0, side_w, h),
        "right": Box(area.right - side_w, area.center[1] - h / 2.0, side_w, h),
    }

    # 主体外扩 padding，字幕与主体之间留白
    inflated = _clip_box(
        Box(box.x - pad, box.y - pad, box.w + 2 * pad, box.h + 2 * pad)
    )

    order: list[str] = []
    if preferred in candidates:
        order.append(preferred)
    order.extend(name for name in ("bottom", "top", "left", "right") if name != preferred)

    def rank(name: str) -> tuple[float, int]:
        return (_overlap_area(candidates[name], inflated), order.index(name))

    chosen = min(order, key=rank)
    placement = candidates[chosen]
    inside = _overlap_area(placement, area) >= placement.area - 1e-6
    return SubtitlePlacement(
        box=placement,
        position=chosen,
        moved=(chosen != preferred),
        overlaps_subject=rank(chosen)[0] > _EPS,
        inside_safe_area=inside,
    )


# ---------------------------------------------------------------------------
# 镜头切换检测
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ShotCut:
    frame: int
    score: float
    kind: str  # "hard_cut" | "fade"

    def as_dict(self) -> dict[str, object]:
        return {"frame": self.frame, "score": round(self.score, 6), "kind": self.kind}


def _histogram_diff(a: Sequence[float], b: Sequence[float]) -> float:
    """归一化直方图的半 L1 距离，取值 [0, 1]。"""
    if len(a) != len(b):
        return 1.0
    total_a = sum(abs(v) for v in a)
    total_b = sum(abs(v) for v in b)
    if total_a <= _EPS and total_b <= _EPS:
        return 0.0
    if total_a <= _EPS or total_b <= _EPS:
        return 1.0
    return 0.5 * sum(abs(x / total_a - y / total_b) for x, y in zip(a, b))


def _feature_vector(item: object) -> list[float]:
    """把一帧的直方图 / 统计特征规整为等长数值序列。"""
    if isinstance(item, Mapping):
        if "hist" in item:
            raw = item["hist"]
        else:
            raw = []
            for key in ("mean", "std"):
                value = item.get(key, [])
                if _is_number(value):
                    raw.append(float(value))
                elif isinstance(value, (tuple, list)):
                    raw.extend(float(v) for v in value if _is_number(v))
        if _is_number(raw):
            return [float(raw)]
        if isinstance(raw, (tuple, list)):
            return [float(v) for v in raw if _is_number(v)]
        return []
    if isinstance(item, (tuple, list)) and all(_is_number(v) for v in item):
        return [float(v) for v in item]
    return []


def shot_detect(
    frames: Iterable[object],
    threshold: float = 0.45,
    min_gap: int = 2,
    fade_threshold: float = 0.15,
    fade_window: int | None = None,
) -> list[ShotCut]:
    """检测镜头切换点。

    ``frames`` 每项可以是颜色直方图（等长、非负的数值序列），也可以直接是
    ``{"hist": [...]}`` 或 ``{"mean": [...], "std": [...]}`` 形式的统计特征。
    返回按帧号升序的切点。

    * 相邻帧半 L1 距离超过 ``threshold`` → 硬切 ``hard_cut``；
    * 单个相邻帧变化不大、但在 ``fade_window`` 帧内累计变化超过
      ``fade_threshold`` → 渐变转场 ``fade``（检测其结束帧）。

    切点之后调用方应重置 :func:`frame_track`，否则跟踪框会停在 A 镜头上。
    """
    threshold = _clamp(float(threshold), 0.0, 1.0)
    fade_threshold = _clamp(float(fade_threshold), 0.0, threshold)
    min_gap = max(1, int(min_gap))
    window = max(2, int(fade_window) if fade_window is not None else min_gap)

    feats: list[list[float]] = [
        _feature_vector(item) for item in _iter_or_empty(frames)
    ]

    cuts: list[ShotCut] = []
    last_cut = -10**9
    in_fade = False
    for i in range(1, len(feats)):
        step = _histogram_diff(feats[i - 1], feats[i])
        if step >= threshold and (i - last_cut) >= min_gap:
            cuts.append(ShotCut(frame=i, score=step, kind="hard_cut"))
            last_cut = i
            in_fade = False
            continue
        # 渐变：窗口内累计变化明显，但单帧变化都低于硬切阈值
        j = max(0, i - window)
        if i - j >= 2 and (i - last_cut) >= min_gap:
            cumulative = _histogram_diff(feats[j], feats[i])
            recent = feats[j : i + 1]
            step_max = max(
                _histogram_diff(recent[k], recent[k + 1])
                for k in range(len(recent) - 1)
            )
            if cumulative >= fade_threshold and step_max < threshold:
                if not in_fade:
                    # 只记一次：渐变是连续过程，避免每几帧重复报切点
                    cuts.append(ShotCut(frame=i, score=cumulative, kind="fade"))
                    last_cut = i
                    in_fade = True
            elif cumulative < fade_threshold * 0.5:
                in_fade = False
    return cuts


# ---------------------------------------------------------------------------
# 音频跟随
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AudioMix:
    """单帧声像。``pan`` ∈ [-1, 1]，负数偏左；``gain`` ∈ [0, 1]。"""

    frame: int
    pan: float
    gain: float
    follows_subject: bool
    on_screen: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "frame": self.frame,
            "pan": self.pan,
            "gain": self.gain,
            "follows_subject": self.follows_subject,
            "on_screen": self.on_screen,
        }


def audio_follow(
    subject_centers: Iterable[object],
    screen_box: object = (0.0, 0.0, 1.0, 1.0),
    follow_strength: float = 0.6,
    off_screen_gain: float = 0.0,
    fade_frames: int = 3,
    base_gain: float = 1.0,
) -> list[AudioMix]:
    """让声像跟随画面主体，主体出画时声音随之衰减。

    ``subject_centers`` 为逐帧主体中心（``(x, y)`` / dict / Point）。
    ``screen_box`` 为当前输出画面在归一化源中的可见区域；主体中心落在其中
    视为在画内。

    * 在画内：``pan`` 由主体水平位置决定，``gain = base_gain``；
    * 出画：``gain`` 在 ``fade_frames`` 帧内衰减到 ``off_screen_gain``，
      ``pan`` 吸附到最近一侧，避免声音还停在画外；
    * ``follow_strength`` 控制声像跟随幅度（0 为不跟随，1 为完全跟随）。
    """
    area = _clip_box(_as_box(screen_box))
    centers = [Point(x, y) for x, y in _coerce_points(subject_centers)]

    strength = _clamp(float(follow_strength), 0.0, 1.0)
    off = _clamp(float(off_screen_gain), 0.0, 1.0)
    base = max(0.0, float(base_gain))
    fade_frames = max(1, int(fade_frames))

    mixes: list[AudioMix] = []
    out_frames = 0
    for i, c in enumerate(centers):
        on_screen = (
            area.x - _EPS <= c.x <= area.right + _EPS
            and area.y - _EPS <= c.y <= area.bottom + _EPS
        )
        if on_screen:
            out_frames = 0
            rel = 0.5 if area.w <= _EPS else (c.x - area.x) / area.w
            pan = _clamp((rel * 2.0 - 1.0) * strength, -1.0, 1.0)
            gain = base
        else:
            out_frames += 1
            t = min(1.0, out_frames / fade_frames)
            gain = base * (1.0 - t) + off * t
            pan = -1.0 if c.x < area.x else 1.0 if c.x > area.right else 0.0
            pan = _clamp(pan * max(strength, 0.5), -1.0, 1.0)
        mixes.append(
            AudioMix(
                frame=i,
                pan=round(pan, 6),
                gain=round(gain, 6),
                follows_subject=on_screen,
                on_screen=on_screen,
            )
        )
    return mixes


# ---------------------------------------------------------------------------
# 坐标统一
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoordPoint:
    """同一套归一化坐标中、带原始像素位置的坐标点。"""

    x: float
    y: float
    source_x: float
    source_y: float

    def as_dict(self) -> dict[str, float]:
        return {
            "x": self.x,
            "y": self.y,
            "source_x": self.source_x,
            "source_y": self.source_y,
        }


@dataclass(frozen=True)
class PreviewExportPoint:
    preview: tuple[float, float]
    export: tuple[float, float]
    error: float
    consistent: bool

    def as_dict(self) -> dict[str, object]:
        return {
            "preview": self.preview,
            "export": self.export,
            "error": self.error,
            "consistent": self.consistent,
        }


def coord_unify(
    points: Iterable[object],
    source_size: object = (1920, 1080),
    input_space: str = "normalized",
) -> list[CoordPoint]:
    """把坐标统一到 ``[0, 1]`` 归一化空间，并保留原始像素位置。

    ``input_space`` 可取：

    * ``"normalized"``：输入已是相对源画面的归一化坐标；
    * ``"pixels"`` / ``"export"``：输入是 ``source_size`` 像素坐标下的坐标。

    统一后预览与导出使用同一套坐标，不再出现 "坐标系不同、导出来偏左"。
    """
    sw, sh = _coerce_size(source_size)
    space = str(input_space).lower()
    if space not in {"normalized", "pixels", "export"}:
        raise ValueError(f"未知坐标系: {input_space!r}")

    out: list[CoordPoint] = []
    for px, py in _coerce_points(points):
        if space == "normalized":
            nx, ny = _clamp01(px), _clamp01(py)
            sx, sy = nx * sw, ny * sh
        else:
            sx, sy = _clamp(px, 0.0, sw), _clamp(py, 0.0, sh)
            nx, ny = sx / sw, sy / sh
        out.append(CoordPoint(x=nx, y=ny, source_x=sx, source_y=sy))
    return out


def export_preview(
    points: Iterable[object],
    source_size: object = (1920, 1080),
    preview_size: object = (1920, 1080),
    export_size: object = (1920, 1080),
    input_space: str = "normalized",
    tolerance: float = 0.01,
) -> PreviewExportPoint | list[PreviewExportPoint]:
    """校验/换算预览与导出坐标系，返回每个点的双边坐标与一致性判定。

    给定多个点时返回列表；单个点返回单个 ``PreviewExportPoint``。``error``
    是两套坐标系还原到归一化空间后的最大偏差，``consistent`` 表示该偏差
    是否不超过 ``tolerance``。
    """
    if isinstance(points, Point) or (
        isinstance(points, (tuple, list))
        and len(points) == 2
        and all(_is_number(v) for v in points)
    ) or (isinstance(points, Mapping) and _is_number(points.get("x"))):
        pts = [points]
    else:
        pts = list(_iter_or_empty(points))
    unified = coord_unify(pts, source_size, input_space=input_space)
    pw, ph = _coerce_size(preview_size)
    ew, eh = _coerce_size(export_size)

    results: list[PreviewExportPoint] = []
    for p in unified:
        preview = (round(p.x * pw, 6), round(p.y * ph, 6))
        export = (round(p.x * ew, 6), round(p.y * eh, 6))
        err = max(
            abs(preview[0] / pw - export[0] / ew),
            abs(preview[1] / ph - export[1] / eh),
        )
        results.append(
            PreviewExportPoint(
                preview=preview,
                export=export,
                error=round(err, 9),
                consistent=err <= tolerance,
            )
        )
    return results[0] if len(results) == 1 else results


# ---------------------------------------------------------------------------
# 旋转
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RotationPlan:
    """一次性的旋转计划。

    ``applied`` 表示旋转是否应在本次处理中应用；``duplicates_removed`` 是
    多个环节重复请求同一次旋转时被去掉的次数；若源像素已经按元数据摆正
    （``pixel_baked=True``），则本次不再旋转，避免二次旋转造成横竖颠倒。
    """

    rotation: int
    effective_rotation: int
    requests: int
    duplicates_removed: int
    pixel_baked: bool
    applied: bool
    input_size: tuple[float, float]
    output_size: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "rotation": self.rotation,
            "effective_rotation": self.effective_rotation,
            "requests": self.requests,
            "duplicates_removed": self.duplicates_removed,
            "pixel_baked": self.pixel_baked,
            "applied": self.applied,
            "input_size": self.input_size,
            "output_size": self.output_size,
        }


def rotation_once(
    rotation: object = 90,
    applications: object = 1,
    pixel_baked: bool = False,
    input_size: object = (1920, 1080),
) -> RotationPlan:
    """把旋转元数据规整为 0/90/180/270，并且 **最多只应用一次**。

    ``rotation`` 可为角度（任意整数，自动取模）或元数据字符串
    （``"90"``、``"rotate_90_cw"``、``"portrait"`` 等）。

    ``applications`` 表示有多少个环节会应用这次旋转（典型情况是容器元数据
    与流元数据各一次）。无论请求多少次，最终只旋转一次；重复次数记录在
    ``duplicates_removed`` 中。传 ``0`` 表示本次不旋转，传 ``True`` 等价于 1。

    ``pixel_baked=True`` 表示源像素已按元数据摆正，本次不再旋转，
    ``effective_rotation`` 归零、输出尺寸不交换，因此不会横竖颠倒。
    """
    angle = _parse_rotation(rotation)
    if isinstance(applications, bool):
        requests = 1 if applications else 0
    elif _is_number(applications):
        requests = max(0, int(applications))
    else:
        requests = 1

    if pixel_baked:
        effective = 0
    else:
        # 只应用一次：任意多次请求都等价于一次
        effective = angle if requests > 0 else 0
    applied = effective % 360 != 0

    w, h = _coerce_size(input_size)
    if applied and effective % 180 == 90:
        output_size = (h, w)
    else:
        output_size = (w, h)
    return RotationPlan(
        rotation=angle,
        effective_rotation=effective,
        requests=requests,
        duplicates_removed=max(0, requests - 1) if not pixel_baked else 0,
        pixel_baked=bool(pixel_baked),
        applied=applied,
        input_size=(w, h),
        output_size=output_size,
    )


# ---------------------------------------------------------------------------
# 构图窗口
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CropWindow:
    """源像素坐标系下的裁切窗口。"""

    x: float
    y: float
    w: float
    h: float
    source_width: float
    source_height: float
    target_aspect: float
    output_size: tuple[float, float]
    zoom: float
    upscaled: bool
    limited: bool

    @property
    def as_box(self) -> tuple[float, float, float, float]:
        return (self.x, self.y, self.w, self.h)

    def as_dict(self) -> dict[str, object]:
        return {
            "x": self.x,
            "y": self.y,
            "w": self.w,
            "h": self.h,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "target_aspect": self.target_aspect,
            "output_size": self.output_size,
            "zoom": self.zoom,
            "upscaled": self.upscaled,
            "limited": self.limited,
        }


def crop_window(
    source_width: float = 1920,
    source_height: float = 1080,
    target_aspect: object = "9:16",
    center: object = (0.5, 0.5),
    output_size: object | None = None,
    max_zoom: float = 1.0,
) -> CropWindow:
    """计算源像素坐标系下的裁切窗口，并给出放大倍数。

    ``target_aspect`` 默认 ``"9:16"``（竖屏），``center`` 是期望主体在窗口
    中的相对位置（默认居中）。窗口取满足目标画幅的最大区域，因此默认不会
    放大源分辨率；若 ``output_size`` 需要的像素比窗口大，则 ``zoom`` 会大于
    1、``upscaled=True``，调用方应据此降低输出分辨率（见 :func:`zoom_limit`）。
    """
    sw, sh = _coerce_size((source_width, source_height))
    aspect = _parse_aspect(target_aspect)
    centers = _coerce_points(center)
    cx_rel, cy_rel = centers[0] if centers else (0.5, 0.5)
    cx_rel = _clamp01(cx_rel)
    cy_rel = _clamp01(cy_rel)

    source_aspect = sw / sh
    # 在源内取目标画幅的最大窗口：目标更"高"（更窄）时占满高度，反之占满宽度
    if aspect <= source_aspect:
        crop_h = sh
        crop_w = sh * aspect
    else:
        crop_w = sw
        crop_h = sw / aspect
    crop_w = min(crop_w, sw)
    crop_h = min(crop_h, sh)

    x = _clamp(cx_rel * sw - crop_w / 2.0, 0.0, max(0.0, sw - crop_w))
    y = _clamp(cy_rel * sh - crop_h / 2.0, 0.0, max(0.0, sh - crop_h))

    zoom_cap = max(_EPS, float(max_zoom))
    if output_size is None:
        out_w, out_h = crop_w, crop_h
    else:
        out_w, out_h = _coerce_size(output_size, default=(crop_w, crop_h))
    zoom = max(out_w / crop_w, out_h / crop_h)
    limited = zoom > zoom_cap + _EPS
    if limited:
        # 限制放大：输出不超过窗口像素 × max_zoom，宁可降低输出分辨率
        out_w = min(out_w, crop_w * zoom_cap)
        out_h = min(out_h, crop_h * zoom_cap)
        zoom = max(out_w / crop_w, out_h / crop_h)

    return CropWindow(
        x=round(x, 6),
        y=round(y, 6),
        w=round(crop_w, 6),
        h=round(crop_h, 6),
        source_width=sw,
        source_height=sh,
        target_aspect=aspect,
        output_size=(round(out_w, 6), round(out_h, 6)),
        zoom=round(zoom, 6),
        upscaled=zoom > 1.0 + _EPS,
        limited=limited,
    )


@dataclass(frozen=True)
class ZoomLimit:
    """放大限制结果。"""

    required_zoom: float
    allowed_zoom: float
    limited: bool
    upscaled: bool
    crop_size: tuple[float, float]
    output_size: tuple[float, float]

    def as_dict(self) -> dict[str, object]:
        return {
            "required_zoom": self.required_zoom,
            "allowed_zoom": self.allowed_zoom,
            "limited": self.limited,
            "upscaled": self.upscaled,
            "crop_size": self.crop_size,
            "output_size": self.output_size,
        }


def zoom_limit(
    required_zoom: float = 1.0,
    max_zoom: float = 1.0,
    source_size: object | None = None,
    output_size: object | None = None,
    crop_size: object | None = None,
) -> ZoomLimit:
    """把放大倍数限制在 ``max_zoom`` 以内，避免放大超过源分辨率而糊片。

    可直接给 ``required_zoom``（实际需要的倍数），也可以给
    ``source_size`` / ``output_size`` 由函数计算：

    * 省略 ``source_size`` 时，``required_zoom`` 即为所需倍数；
    * 给出 ``source_size`` 时，``crop_size`` 为实际裁切窗口（默认整个源），
      输出像素相对它计算倍数。

    ``allowed_zoom`` 不超过 ``max_zoom``；``limited`` 表示需求被裁剪，
    ``upscaled`` 表示允许值仍大于 1（即调用了 ``max_zoom > 1``）。
    """
    cap = max(_EPS, float(max_zoom))

    if source_size is not None:
        out_w, out_h = (
            _coerce_size(output_size, default=(0.0, 0.0))
            if output_size is not None
            else (0.0, 0.0)
        )
        if crop_size is not None:
            crop_w, crop_h = _coerce_size(crop_size)
        elif out_w > 0 and out_h > 0:
            # 没有给出裁切窗口：按目标画幅从源中取最大窗口
            sw, sh = _coerce_size(source_size)
            src_aspect = sw / sh
            out_aspect = out_w / out_h
            if out_aspect <= src_aspect:
                crop_h = sh
                crop_w = sh * out_aspect
            else:
                crop_w = sw
                crop_h = sw / out_aspect
        else:
            crop_w, crop_h = _coerce_size(source_size)
        if out_w <= 0 or out_h <= 0:
            out_w, out_h = crop_w, crop_h
        required = max(out_w / crop_w, out_h / crop_h)
    else:
        crop_w = crop_h = 1.0
        required = max(_EPS, float(required_zoom))
        if output_size is not None:
            out_w, out_h = _coerce_size(output_size)
        else:
            out_w, out_h = (crop_w, crop_h)

    allowed = min(required, cap)
    return ZoomLimit(
        required_zoom=round(required, 6),
        allowed_zoom=round(allowed, 6),
        limited=required > cap + _EPS,
        upscaled=allowed > 1.0 + _EPS,
        crop_size=(crop_w, crop_h),
        output_size=(out_w, out_h),
    )


# ---------------------------------------------------------------------------
# 主体打分 / 跟丢状态 / 关键帧节奏
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubjectScore:
    """主体候选打分细节，``score`` ∈ [0, 1]。"""

    score: float
    area_term: float
    centrality_term: float

    def as_float(self) -> float:
        return self.score

    def __float__(self) -> float:
        return self.score

    def as_dict(self) -> dict[str, float]:
        return {
            "score": self.score,
            "area_term": self.area_term,
            "centrality_term": self.centrality_term,
        }


def subject_score(
    candidate: object = 1.0,
    centrality: float | None = None,
    weights: object = (0.6, 0.4),
) -> SubjectScore:
    """给主体候选打分：面积越大、越靠近画面中心得分越高。

    ``candidate`` 可以是归一化框（此时面积取框面积、中心取框中心）或直接的
    面积占比数值。``centrality`` 可显式给出（1 表示完全居中）；省略时由框
    中心到画面中心的距离换算。返回对象可当 float 用（``float(result)``）。
    """
    if _is_number(candidate):
        area = _clamp01(float(candidate))
        if centrality is None:
            centrality = 1.0
    else:
        box = _clip_box(_as_box(candidate))
        area = _clamp01(box.area)
        if centrality is None:
            cx, cy = box.center
            dist = math.hypot(cx - 0.5, cy - 0.5)
            centrality = 1.0 - min(1.0, dist / (_DIAG / 2.0))

    cent = _clamp01(float(centrality))
    w_area, w_cent = 0.6, 0.4
    try:
        pair = tuple(weights)  # type: ignore[arg-type]
        if len(pair) == 2:
            w_area, w_cent = max(0.0, float(pair[0])), max(0.0, float(pair[1]))
    except TypeError:
        pass
    total = w_area + w_cent
    if total <= _EPS:
        w_area = w_cent = 0.5
        total = 1.0
    score = _clamp01((area * w_area + cent * w_cent) / total)
    return SubjectScore(
        score=round(score, 6),
        area_term=round(area, 6),
        centrality_term=round(cent, 6),
    )


@dataclass(frozen=True)
class TrackStatusSummary:
    """跟丢状态机的一次判定结果。"""

    lost_frames: int
    max_lost: int
    state: str  # "tracking" | "predicted" | "lost"
    should_reset: bool
    confidence: float

    def __bool__(self) -> bool:
        return self.state != "lost"

    def as_dict(self) -> dict[str, object]:
        return {
            "lost_frames": self.lost_frames,
            "max_lost": self.max_lost,
            "state": self.state,
            "should_reset": self.should_reset,
            "confidence": self.confidence,
        }


def tracking_loss(
    lost_frames: object = 0,
    max_lost: int = 15,
    min_iou: float = 0.05,
) -> TrackStatusSummary:
    """根据连续丢失帧数判定跟踪状态，决定继续外推还是重置。

    * ``lost_frames == 0`` → ``tracking``；
    * ``0 < lost_frames <= max_lost`` → ``predicted``（用速度外推过渡）；
    * ``lost_frames > max_lost`` → ``lost``，``should_reset=True``，调用方
      应清空状态并在下一帧重新检测（避免镜头切走后框停在旧主体上）。

    ``lost_frames`` 也接受 IoU 序列：取末尾连续低于 ``min_iou`` 的帧数。
    """
    max_lost = max(0, int(max_lost))
    seq = list(_iter_or_empty(lost_frames))
    if seq and all(_is_number(v) for v in seq):
        count = 0
        for value in reversed(seq):
            if float(value) < float(min_iou):
                count += 1
            else:
                break
    elif _is_number(lost_frames):
        count = max(0, int(lost_frames))
    else:
        count = 0

    if count == 0:
        state = "tracking"
    elif count <= max_lost:
        state = "predicted"
    else:
        state = "lost"
    confidence = 1.0 if state == "tracking" else max(0.0, 1.0 - count / max(1, max_lost))
    return TrackStatusSummary(
        lost_frames=count,
        max_lost=max_lost,
        state=state,
        should_reset=state == "lost",
        confidence=round(confidence, 6),
    )


@dataclass(frozen=True)
class KeyframePlan:
    """关键帧节奏计划。"""

    fps: float
    seconds: float
    interval_frames: int
    duration_frames: int
    keyframes: tuple[int, ...]

    def __int__(self) -> int:
        return self.interval_frames

    def as_dict(self) -> dict[str, object]:
        return {
            "fps": self.fps,
            "seconds": self.seconds,
            "interval_frames": self.interval_frames,
            "duration_frames": self.duration_frames,
            "keyframes": list(self.keyframes),
        }


def keyframe_interval(
    fps: object = 30,
    seconds: float = 2.0,
    duration_frames: int | None = None,
) -> KeyframePlan:
    """按帧率与目标间隔秒数计算关键帧间隔。

    ``keyframe_interval(30, 2)`` 得到每 60 帧一个关键帧。关键帧之间用
    :func:`smooth_path` 缓动展开；``duration_frames`` 给出时同时返回具体
    关键帧帧号（含首帧与末帧）。
    """
    fps_value = float(fps) if _is_number(fps) else 30.0
    if fps_value <= 0:
        fps_value = 30.0
    seconds_value = float(seconds) if _is_number(seconds) else 2.0
    if seconds_value <= 0:
        seconds_value = 1.0
    interval = max(1, int(round(fps_value * seconds_value)))

    if duration_frames is None:
        keyframes: tuple[int, ...] = (0, interval)
        duration = interval
    else:
        duration = max(1, int(duration_frames))
        frames = list(range(0, duration, interval))
        if not frames or frames[-1] != duration - 1:
            frames.append(duration - 1)
        keyframes = tuple(frames)

    return KeyframePlan(
        fps=fps_value,
        seconds=seconds_value,
        interval_frames=interval,
        duration_frames=duration,
        keyframes=keyframes,
    )
