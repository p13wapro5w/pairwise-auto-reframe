# pairwise-auto-reframe

## 领域

游戏、图形与多媒体

## 主题

画面智能重构

## 说明

逐帧跟踪、平滑轨迹与字幕重排。

`media/auto_reframe.py` 实现横屏素材转竖屏（及多平台输出）链路中的核心算法，
全部为只依赖标准库的纯函数：

| 入口 | 解决的问题 |
| --- | --- |
| `frame_track` | 逐帧使用本帧检测关联，而不是沿用上一帧的框；快速移动不跟丢 |
| `smooth_path` / `smooth_trajectory` | 关键帧之间用缓动 + 死区 + 限速平滑，而不是线性插值 |
| `safe_area` | 安全区按**目标画幅**计算，竖屏字幕不再被切 |
| `platform_consistent` | 多平台共享同一构图锚点，主体位置不差半屏 |
| `subtitle_reflow` | 重构后字幕避让人脸并落在目标安全区内 |
| `zoom_limit` / `crop_window` | 放大不超过源分辨率，拒绝糊片 |
| `shot_detect` | 检测硬切与渐变转场，切点处重置跟踪 |
| `audio_follow` | 声像跟随画面主体，出画时衰减 |
| `coord_unify` / `export_preview` | 预览与导出共用归一化坐标，导出不再偏左 |
| `rotation_once` | 旋转元数据只应用一次，避免横竖颠倒 |
| `subject_score` / `tracking_loss` / `keyframe_interval` | 主体打分、跟丢状态机与关键帧节奏 |

几何约定：除显式接收像素尺寸的函数外，坐标与尺寸均为左上角原点、归一化到
`[0, 1]`。

## 测试

```bash
python -m pytest -q
```

* `tests/test_auto_reframe.py`：15 个入口的契约测试；
* `tests/test_auto_reframe_behavior.py`：针对上述故障的行为级回归测试。
