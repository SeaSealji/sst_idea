# 历史到未来映射：Z0 诊断

关联 #7。在为直接预测模型增加时间推进或多路径头之前，分别检查末帧锚定与学习残差对历史输入的局部敏感性。工具不加载数据或权重，也不启动训练。

## 用法

依赖 Python 3.10+、PyTorch；只使用 CPU float64。调用前加载自己已审计的模型并设 `eval()`；输入为 `[1,T,C,H,W]`，输出为 `[1,L,C,H,W]`。

```python
from audit import audit_temporal_map

model = model.cpu().double().eval()
result = audit_temporal_map(
    model, synthetic_x.double(),
    anchor=lambda x: x[:, -1:].expand(-1, 7, -1, -1, -1),
    wet_mask=output_wet_mask, probes=[(0, row, col)],
)
```

这里的简易 anchor 仅适用于无额外填充的模型。若模型先填陆再掩码，anchor 必须复现相同操作，并显式扩展全部 lead。mask/probe 必须位于**输出**坐标系；带 halo 时不要把 core 坐标直接代入。

```bash
python -m unittest discover -s diagnostics/temporal_map -p 'test_*.py' -v
```

每个湿格 probe、每个 lead 分别返回总预测和去锚残差的逐历史日梯度 signed sum/L1、随机方向中央差分及仿射叠加缺陷。梯度 L1 保留被正负相消掩盖的敏感性。只对输入调用 `autograd.grad`，不执行 optimizer 或参数 backward。

## 解释边界

- 零初始化预测头的总梯度可能仅来自末帧锚定；不能据此诊断“训练后只看最后一天”。零残差梯度也可能来自局部饱和或断图，需要源码核查。
- 非零梯度仅说明这些输入点上的局部依赖，不证明条件预测增量、稳定动力学、物理输运或训练收益。probe 不是完整 Jacobian。
- 若中央差分与 autograd 不一致，先检查 `detach`、随机算子和数值尺度，不得发布梯度归因。例如 [NLinear 作者实现](https://github.com/cure-lab/LTSF-Linear/blob/main/models/NLinear.py)的末帧使用 `detach`；这可让前向函数的数值导数与 autograd 定义不同，不能自动判为训练实现错误。
- 非零仿射缺陷可否定测试点上的精确仿射映射；接近零不证明全局等价。不能仅凭带 GELU 的源码就断言某一组实际权重一定非线性。
- 工具不知道调用方的时间划分。首轮用合成输入；真实诊断仅传已审计的训练输入，不传测试值。`input_kind` 默认明确标为尚未验证划分，不构成防泄漏证明。结果不含 RMSE 或模型排名。
- CPU float64 诊断是数值验证，不宣称重现 AMP 训练或完整验证指标。调用方负责保证 model/anchor 没有自定义状态更新，并在前后核对状态。

## 当前任务与相关研究

当前自研默认模型已切为 T1-no-expert、1° 全球、7→7；主指标为统一湿格 pooled RMSE，面积加权另报。旧 30→7、T1-Fused 提案须先改为当前协议。

[PRO-DYN（ICML 2026）](https://proceedings.mlr.press/v306/brachet26a.html)分析历史到未来映射及模块位置；其[作者版 §4.3](https://arxiv.org/html/2507.15774v3#S4.SS3)也检查参数与输入长度的混淆。当前 L=H=7，时间轴上的 Linear(7→7) 无法仅凭形状区分 PRO/DYN，必须标明表示的时间语义。

[LTSF-Linear](https://arxiv.org/abs/2205.13504)已有直接线性预测；[PatchTST](https://arxiv.org/abs/2211.14730)已有编码器与预测头。不能把加一层 Linear、通道 reshape 或本工具称为新海洋动力学。当前模型已经在第一层学习混合七天历史，再从潜特征生成各 lead；它不保留独立时间 token，也没有可重复推进的状态接口。

后续可检验的差异是**显式保留时间轴的归纳偏置**是否优于已混合历史的直接头。先用同 decoder/同监督/同容量与计算预算的对照，再做配对种子实验；当前诊断不能替代这项预测证据。不改变现有最佳模型，不自动安排模型训练。
