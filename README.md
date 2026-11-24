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

## 修复对照

| 缺陷 | 修复实现 |
| --- | --- |
| 时间轴按绝对时长推进，切场景后继续走 | `Timeline` 只累计 `tick` 帧时长（场景内相对时间），不读墙上时钟；`load_scene`/`reset` 归零 |
| 跳过只跳时间轴，音轨继续播放 | `CinematicController.skip` 在同一调用内 `jump_to_end` 并 `audio.stop()`，审计日志与指标留痕 |
| 资源异步加载未完成时间轴已启动 | `AssetGate` 汇总资源就绪状态，`start` 调用 `ensure_ready`，未全部就绪抛 `AssetNotReadyError` |
| 过场输入被缓存、结束后一次性释放致瞬移 | `InputGate` 激活期直接丢弃事件（仅计数、不保留引用），结束时队列为空、无回流 |
| 字幕与镜头切换同帧派发导致错位 | `SubtitleBinding` 绑定镜头段与段内偏移；同帧按“先切镜头、后派字幕”的固定顺序派发 |
| 重播抖动种子随机、镜头轨迹不一致 | `JitterRng` 以过场固定种子按段索引派生确定性随机源，抖动表构造时预计算，同种子重播逐位一致 |

行为测试见 `tests/test_fixes_behavior.py`。
