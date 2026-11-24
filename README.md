# pairwise-cinematic-input-gate

## 领域

游戏、图形与多媒体

## 主题

过场时间轴与输入门控

## 说明

场景内相对时间、跳过停音轨与输入丢弃。

初始状态下 `gameplay/cinematic_input_gate.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
