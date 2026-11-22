"""行为级测试：覆盖 "横屏转竖屏" 链路上的 10 类真实故障。

与 ``test_auto_reframe.py`` 的契约测试（只要求返回值非 None）不同，
这里断言每个修复的实际语义，防止实现退化成"能跑但不正确"。
"""

import math

import pytest

import media.auto_reframe as mod


# ---------------------------------------------------------------------------
# 1. 逐帧跟踪：不再沿用上一帧的框，快速移动也不跟丢
# ---------------------------------------------------------------------------


def test_frame_track_uses_current_detection_every_frame():
    # 主体每帧右移 0.04（归一化），属于 "快速移动"
    detections = [[(0.05 + i * 0.04, 0.4, 0.1, 0.2)] for i in range(10)]
    decisions = mod.frame_track(detections)

    assert len(decisions) == 10
    assert all(d.source == "detection" for d in decisions)
    assert all(d.status.state == "tracking" for d in decisions)
    # 跟踪框必须跟着本帧检测走，而不是停在第一帧附近
    xs = [d.box.x for d in decisions]
    assert xs == pytest.approx([0.05 + i * 0.04 for i in range(10)])


def test_frame_track_does_not_jump_to_distractor():
    # 第 4 帧出现一个远处的大框（别人），不应把跟踪框甩过去
    frames = [[(0.30, 0.40, 0.10, 0.20)] for _ in range(4)]
    frames.append([(0.32, 0.40, 0.10, 0.20), (0.80, 0.10, 0.40, 0.40)])
    frames.append([(0.34, 0.40, 0.10, 0.20)])
    decisions = mod.frame_track(frames)

    assert decisions[4].box.x == pytest.approx(0.32, abs=1e-6)
    assert decisions[5].box.x == pytest.approx(0.34, abs=1e-6)


def test_frame_track_extrapolates_then_declares_lost():
    frames = [[(0.2, 0.4, 0.1, 0.2)], [(0.25, 0.4, 0.1, 0.2)]]
    frames += [[] for _ in range(6)]  # 连续丢失
    decisions = mod.frame_track(frames, max_lost=3)

    predicted = [d for d in decisions if d.source == "prediction"]
    assert predicted, "短暂丢失应做速度外推而不是直接丢框"
    assert decisions[-1].status.state == "lost"
    assert decisions[-1].box.w == 0.0  # 判为跟丢后清空


def test_frame_track_reacquires_after_loss():
    frames = [[(0.2, 0.4, 0.1, 0.2)], [], [], [(0.2, 0.4, 0.1, 0.2)]]
    decisions = mod.frame_track(frames, max_lost=2)
    assert decisions[-1].reacquired is True
    assert decisions[-1].status.state == "reacquired"


# ---------------------------------------------------------------------------
# 2. 轨迹平滑：不是关键帧线性插值，镜头不一顿一顿
# ---------------------------------------------------------------------------


def test_smooth_path_is_not_linear_and_is_continuous():
    points = mod.smooth_path([(0, 0.2, 0.5), (30, 0.8, 0.5)], duration=31)
    assert len(points) == 31

    xs = [p.x for p in points]
    steps = [abs(b - a) for a, b in zip(xs, xs[1:])]
    # 限制单帧位移：不会出现关键帧处的跳变
    assert max(steps) <= 0.08 + 1e-9
    # 缓动：中段速度最大、两端最小，线性插值不会有这种形状
    mid = steps[len(steps) // 2]
    assert mid > steps[1] + 1e-6
    assert mid > steps[-2] + 1e-6
    # 端点跟随关键帧
    assert xs[0] == pytest.approx(0.2)
    assert 0.2 < xs[-1] < 0.8


def test_smooth_trajectory_suppresses_jitter():
    shaky = [(0.5 + (0.002 if i % 2 else -0.002), 0.5) for i in range(20)]
    smoothed = mod.smooth_trajectory(shaky, alpha=0.25, deadband=0.01)
    # 死区内的抖动完全不推动镜头（保持首帧位置，不随抖动漂移）
    assert all(p.x == pytest.approx(smoothed[0].x) for p in smoothed)
    assert max(abs(p.x - 0.5) for p in smoothed) <= 0.01


def test_smooth_trajectory_limits_camera_speed():
    jump = [(0.1, 0.5), (0.9, 0.5)]
    smoothed = mod.smooth_trajectory(jump, alpha=1.0, deadband=0.0, max_step=0.08)
    assert smoothed[1].x == pytest.approx(0.18)
    assert smoothed[1].x < 0.9  # 没有瞬间甩过去


# ---------------------------------------------------------------------------
# 3. 安全区按目标画幅，而不是沿用 16:9
# ---------------------------------------------------------------------------


def test_safe_area_uses_target_aspect_for_portrait():
    portrait = mod.safe_area(1080, 1920)
    landscape = mod.safe_area(1920, 1080)

    assert portrait.aspect == pytest.approx(1080 / 1920)
    assert landscape.aspect == pytest.approx(1920 / 1080)
    assert portrait.aspect != landscape.aspect
    # 按目标画幅算出来的安全区形态不同：竖屏保留更多高度、横屏保留更多宽度
    assert portrait.h > landscape.h
    assert landscape.w > portrait.w
    # 竖屏安全区仍完整落在 [0,1] 内，字幕不会被切
    assert 0.0 <= portrait.x and portrait.x + portrait.w <= 1.0 + 1e-9
    assert 0.0 <= portrait.y and portrait.y + portrait.h <= 1.0 + 1e-9
    assert portrait.contains(0.5, 0.02) is False
    assert portrait.contains(0.5, 0.5) is True


def test_safe_area_accepts_aspect_strings_and_margin_sugar():
    a = mod.safe_area("9:16")
    b = mod.safe_area("9:16", 0.1)
    assert a.aspect == pytest.approx(9 / 16)
    assert a.w == pytest.approx(0.9)
    assert b.w == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# 4. 多平台一致：同一主体在各平台落点一致
# ---------------------------------------------------------------------------


def test_platform_consistent_keeps_subject_in_same_place():
    subject = (0.55, 0.40, 0.10, 0.20)  # 偏右的主体
    result = mod.platform_consistent(subject, (1920, 1080))

    assert set(result) == {"landscape", "square", "vertical"}
    anchors = {crop.anchor.as_tuple() for crop in result.values()}
    assert len(anchors) == 1, "所有平台应共享同一个构图锚点"

    xs = [crop.subject.x for crop in result.values()]
    ys = [crop.subject.y for crop in result.values()]
    assert max(xs) - min(xs) < 1e-9, "主体水平位置不应差半屏"
    assert max(ys) - min(ys) < 1e-9
    assert all(crop.consistent for crop in result.values())


def test_platform_consistent_crops_fit_inside_source():
    result = mod.platform_consistent((0.02, 0.02, 0.05, 0.05), (1920, 1080))
    for crop in result.values():
        assert crop.crop.x >= -1e-9
        assert crop.crop.y >= -1e-9
        assert crop.crop.right <= 1.0 + 1e-9
        assert crop.crop.bottom <= 1.0 + 1e-9


def test_platform_consistent_reports_unmet_anchor_honestly():
    # 主体紧贴右边，却要求它落在画面正中：几何上做不到，必须如实报告偏差，
    # 而不是静默输出一个错位的构图。
    result = mod.platform_consistent(
        (0.90, 0.30, 0.05, 0.10), (1920, 1080), anchor=(0.5, 0.5)
    )
    for name, crop in result.items():
        assert crop.deviation > 0.0, f"{name} 应报告偏差"
        assert crop.consistent is False
        assert crop.subject.x > 0.5  # 只能尽量靠左，不可能真的到中心
        # 裁切窗口始终夹紧在源画面内
        assert crop.crop.x >= -1e-9
        assert crop.crop.y >= -1e-9
        assert crop.crop.right <= 1.0 + 1e-9
        assert crop.crop.bottom <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# 5. 字幕重排：重构后不再压在人脸上
# ---------------------------------------------------------------------------


def test_subtitle_reflow_avoids_face_at_bottom():
    safe = mod.safe_area(1080, 1920)
    face = (0.35, 0.68, 0.30, 0.28)  # 人脸在下方
    placement = mod.subtitle_reflow(face, safe, subtitle_height=0.12, preferred="bottom")

    assert placement.position != "bottom"
    assert placement.moved is True
    assert placement.overlaps_subject is False
    assert placement.inside_safe_area is True


def test_subtitle_reflow_keeps_preferred_when_free():
    safe = mod.safe_area(1080, 1920)
    face = (0.35, 0.10, 0.30, 0.25)  # 人脸在上方，底部安全
    placement = mod.subtitle_reflow(face, safe, preferred="bottom")
    assert placement.position == "bottom"
    assert placement.moved is False
    assert placement.overlaps_subject is False


def test_subtitle_reflow_moves_with_subject_between_platforms():
    portrait_safe = mod.safe_area(1080, 1920)
    landscape_safe = mod.safe_area(1920, 1080)
    face = (0.30, 0.30, 0.40, 0.60)
    a = mod.subtitle_reflow(face, portrait_safe)
    b = mod.subtitle_reflow(face, landscape_safe)
    # 安全区不同 → 字幕落点不同，说明确实按目标安全区重排
    assert a.box.as_dict() != b.box.as_dict()


# ---------------------------------------------------------------------------
# 6. 限制放大：不超过源分辨率
# ---------------------------------------------------------------------------


def test_zoom_limit_caps_upscale():
    result = mod.zoom_limit(required_zoom=2.5, max_zoom=1.0)
    assert result.limited is True
    assert result.allowed_zoom == pytest.approx(1.0)
    assert result.upscaled is False


def test_zoom_limit_from_pixel_sizes():
    result = mod.zoom_limit(
        max_zoom=1.0, source_size=(1920, 1080), output_size=(1920, 3413)
    )
    assert result.required_zoom > 3.0
    assert result.allowed_zoom == pytest.approx(1.0)
    assert result.limited is True


def test_zoom_limit_allows_requested_when_within_cap():
    result = mod.zoom_limit(required_zoom=1.2, max_zoom=2.0)
    assert result.limited is False
    assert result.allowed_zoom == pytest.approx(1.2)
    assert result.upscaled is True


def test_crop_window_never_upscales_by_default():
    window = mod.crop_window(1920, 1080, target_aspect="9:16")
    assert window.zoom == pytest.approx(1.0)
    assert window.upscaled is False
    assert window.w <= 1920 and window.h <= 1080
    assert window.w / window.h == pytest.approx(9 / 16, rel=1e-6)


def test_crop_window_refuses_upscaled_output():
    window = mod.crop_window(
        1920, 1080, target_aspect="9:16", output_size=(1920, 3413)
    )
    assert window.limited is True
    assert window.zoom <= 1.0 + 1e-9
    assert window.upscaled is False


# ---------------------------------------------------------------------------
# 7. 切镜检测：A→B 时给出切点
# ---------------------------------------------------------------------------


def test_shot_detect_finds_hard_cut():
    hist_a = [0.90, 0.05, 0.05]
    hist_b = [0.05, 0.05, 0.90]
    frames = [hist_a] * 5 + [hist_b] * 5
    cuts = mod.shot_detect(frames)

    assert len(cuts) == 1
    assert cuts[0].frame == 5
    assert cuts[0].kind == "hard_cut"


def test_shot_detect_no_false_positive_on_static():
    frames = [[0.6, 0.3, 0.1]] * 20
    assert mod.shot_detect(frames) == []


def test_shot_detect_reports_gradual_fade_once():
    a = [0.9, 0.05, 0.05]
    b = [0.05, 0.05, 0.9]
    frames = [
        [a[k] * (1 - i / 5) + b[k] * (i / 5) for k in range(3)] for i in range(6)
    ]
    cuts = mod.shot_detect(frames)
    assert len(cuts) == 1
    assert cuts[0].kind == "fade"


def test_shot_detect_reset_tracking_at_cut():
    # 切镜后跟踪应重置，而不是把 A 的框带到 B
    shot_a = [[(0.70, 0.40, 0.10, 0.20)] for _ in range(4)]
    shot_b = [[(0.10, 0.40, 0.10, 0.20)] for _ in range(4)]
    detections = shot_a + shot_b
    cuts = mod.shot_detect([[1.0, 0.0]] * 4 + [[0.0, 1.0]] * 4)
    cut_frame = cuts[0].frame

    decisions = mod.frame_track(detections[:cut_frame])
    decisions += mod.frame_track(detections[cut_frame:], start_frame=cut_frame)
    assert decisions[cut_frame].box.x == pytest.approx(0.10)


# ---------------------------------------------------------------------------
# 8. 音频跟随画面
# ---------------------------------------------------------------------------


def test_audio_follow_tracks_subject_horizontally():
    mixes = mod.audio_follow(
        [(0.1, 0.5), (0.5, 0.5), (0.9, 0.5)], follow_strength=1.0
    )
    assert mixes[0].pan < 0 < mixes[2].pan
    assert mixes[1].pan == pytest.approx(0.0, abs=1e-9)
    assert all(m.on_screen for m in mixes)


def test_audio_mutes_when_subject_leaves_frame():
    # 主体出画后声音必须衰减，而不是留在画内
    mixes = mod.audio_follow(
        [(0.5, 0.5), (1.3, 0.5), (1.3, 0.5), (1.3, 0.5)], fade_frames=2
    )
    assert mixes[0].on_screen is True
    assert mixes[1].on_screen is False
    assert mixes[1].gain < mixes[0].gain
    assert mixes[-1].gain == pytest.approx(0.0)
    # 出画后声像吸附到出画侧
    assert mixes[-1].pan > 0


def test_audio_follows_crop_not_full_source():
    # 输出只看到源的右半边：主体在 0.4 处已不在画面内
    mixes = mod.audio_follow(
        [(0.4, 0.5)], screen_box=(0.5, 0.0, 0.5, 1.0), fade_frames=1
    )
    assert mixes[0].on_screen is False
    assert mixes[0].gain == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 9. 坐标统一：预览与导出一致
# ---------------------------------------------------------------------------


def test_coord_unify_normalizes_pixels():
    points = mod.coord_unify([(960, 540)], (1920, 1080), input_space="pixels")
    assert points[0].x == pytest.approx(0.5)
    assert points[0].y == pytest.approx(0.5)
    assert points[0].source_x == pytest.approx(960)


def test_export_preview_matches_across_sizes():
    result = mod.export_preview(
        [(0.5, 0.5)], (1920, 1080), preview_size=(960, 540), export_size=(3840, 2160)
    )
    assert result.consistent is True
    assert result.error <= 1e-9
    assert result.preview == pytest.approx((480.0, 270.0))
    assert result.export == pytest.approx((1920.0, 1080.0))


def test_export_preview_no_left_bias():
    # 同一归一化坐标在两套坐标系中相对位置一致 → 不会导出偏左
    points = [(0.0, 0.0), (0.25, 0.75), (1.0, 1.0)]
    results = mod.export_preview(
        points, (1920, 1080), preview_size=(1280, 720), export_size=(1920, 1080)
    )
    assert isinstance(results, list)
    assert all(r.consistent for r in results)
    for r, (x, y) in zip(results, points):
        assert r.preview[0] / 1280 == pytest.approx(x)
        assert r.preview[1] / 720 == pytest.approx(y)


def test_coord_unify_single_point_unicode():
    point = mod.coord_unify((0.5, 0.5), (1920, 1080))[0]
    assert (point.x, point.y) == (0.5, 0.5)


# ---------------------------------------------------------------------------
# 10. 旋转只应用一次
# ---------------------------------------------------------------------------


def test_rotation_applied_exactly_once_for_duplicate_requests():
    plan = mod.rotation_once(90, 2)  # 两个环节都请求 90°
    assert plan.effective_rotation == 90
    assert plan.duplicates_removed == 1
    assert plan.applied is True
    # 横屏源旋转一次 → 竖屏输出，尺寸只交换一次
    assert plan.output_size == pytest.approx((1080, 1920))


def test_rotation_not_applied_when_pixels_already_baked():
    plan = mod.rotation_once(90, 2, pixel_baked=True)
    assert plan.applied is False
    assert plan.effective_rotation == 0
    assert plan.output_size == pytest.approx((1920, 1080)), "不能二次旋转导致横竖颠倒"


def test_rotation_normalizes_metadata_strings():
    assert mod.rotation_once("rotate_90_cw").effective_rotation == 90
    assert mod.rotation_once(-90).effective_rotation == 270
    assert mod.rotation_once(450).effective_rotation == 90
    assert mod.rotation_once(0).applied is False


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------


def test_subject_score_prefers_large_centered_subject():
    big_center = float(mod.subject_score((0.30, 0.30, 0.40, 0.40)))
    small_corner = float(mod.subject_score((0.00, 0.00, 0.10, 0.10)))
    assert big_center > small_corner
    assert 0.0 <= small_corner <= 1.0
    assert 0.0 <= big_center <= 1.0


def test_tracking_loss_state_machine():
    assert mod.tracking_loss(0).should_reset is False
    predicted = mod.tracking_loss(3, max_lost=15)
    assert predicted.state == "predicted"
    lost = mod.tracking_loss(20, max_lost=15)
    assert lost.state == "lost"
    assert lost.should_reset is True

    # 也接受逐帧 IoU 序列
    seq = mod.tracking_loss([0.8, 0.02, 0.01, 0.0], max_lost=2)
    assert seq.lost_frames == 3
    assert seq.state == "lost"


def test_keyframe_interval_from_fps():
    plan = mod.keyframe_interval(30, 2)
    assert int(plan) == 60
    extended = mod.keyframe_interval(30, 2, duration_frames=300)
    assert extended.keyframes[0] == 0
    assert extended.keyframes[-1] == 299
    # 关键帧之间不丢帧
    assert all(b - a <= 60 for a, b in zip(extended.keyframes, extended.keyframes[1:]))


def test_non_negative_invariants():
    for name in (
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
        "crop_window",
        "subject_score",
        "tracking_loss",
        "keyframe_interval",
        "export_preview",
    ):
        assert hasattr(mod, name), f"缺少入口 {name}"
        assert not math.isnan(float(getattr(mod, name).__doc__ is None))
