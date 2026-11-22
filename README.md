# pairwise-auto-reframe

## 领域

游戏、图形与多媒体

## 主题

画面智能重构

## 说明

逐帧跟踪、平滑轨迹与字幕重排。

初始状态下 `media/auto_reframe.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
