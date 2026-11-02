# AMP 进度诊断序列化

训练 loss 有限时，AMP backward 仍可能溢出。GradScaler 跳过更新后，
`clip_grad_norm_` 返回的 `inf` 会使严格 JSON 写入失败。本工具仅处理该诊断字段：
记录 `null`、非有限类型及跳步标志；不放宽 loss、验证指标或全局 JSON 的有限性检查。
每一批都必须调用；非有限范数却没有跳过更新时立即报错。

```python
from progress import gradient_fields
loss, updated, norm = step(...)  # 已完成 scaler.step/update
fields = gradient_fields(norm, updated)
if should_log:
    write(progress_path, dict(last_loss=loss, **fields))
```

CPU 测试：`python -m unittest discover -s diagnostics/amp_progress -v`。
旧 PyTorch 没有 CPU GradScaler 时显式 skip，不能将 skip 记作 PASS；项目 GPU
共享锁与空闲检查通过后可显式设置 `AMP_TEST_DEVICE=cuda` 运行同一测试。
测试用有限 loss 和注入的
非有限梯度验证真实跳步、模型/AdamW状态不变、scale 减半及下一步恢复更新。
这是合成机制检查，不能证明某次实际训练故障步的状态，也不能代替故障回放。

该工具不启动训练、不重试队列、不迁移 checkpoint。若接入冻结 runner，应新建版本，
保留原始失败证据，核验完整模型/optimizer/scaler/scheduler/RNG 和 cursor，
再按项目恢复规则决定是否显式续跑。中断的实验不应当作收敛或负科学结果。
