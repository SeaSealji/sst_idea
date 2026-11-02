# #12：单层湿格分量变换 Z0

这是可逆局部算子 MVP，供海岸几何诊断与未来受控实验使用。它不改变当前 T1-no-expert，不是完整 MSWT、lifting 或 SST 预测模型；没有模型训练、RMSE 或测试集读取。

## API 与系数含义

依赖 Python 3.10+、PyTorch。静态 `mask` 为 Boolean `[H,W]`；动态特征仅允许已审计的历史输入或合成场。

```python
from transform import WetBlockTransform

op = WetBlockTransform(wet_mask)  # 默认从左上开始配对
coeff = op(x)                    # [B,C,H,W] -> [B,C,4,ceil(H/2),ceil(W/2)]
restored = op.inverse(coeff)      # 湿格恢复、干格为0
print(op.summary())
```

枚举全部 16 种 2×2 mask。对块内每个四邻接湿分量，用归一化常数向量作为 DC，再对坐标向量做固定 Gram–Schmidt 得到差分基。全湿块使用标准 Haar LL/LH/HL/HH。满足 `QᵀQ=diag(wet)`、`inverse(forward(x))=where(wet,x,0)`，保留湿格能量；不是面积加权或热量守恒。

**部分湿块的四个槽位没有固定 LL/LH/HL/HH 语义。** `component_support[slot,h,w]` 是该槽位所属分量的四位支持码；`dc_slots` 区分每个分量的均值/差分。对角断连时有两个独立 DC。固定槽位打包不压缩湿格自由度；不能只保留 slot0 并称它为整块的海域均值。

`torch.where` 在运算前屏蔽干格，干格 NaN 不影响结果；湿格 NaN/Inf 拒绝。仅支持 float32/64，算子与输入须在同一设备。底部/右侧奇数补点为干格，纬度不周期连接。

完整全球、偶数经度网格可设置 `longitude_offset=1`，通过循环平移将末列/首列放入同一个配对块，逆变换恢复原位置。该选项同时改变所有经度配对。**带 halo 或普通 tile 的外缘不是全球经度接缝**；这些域用 offset0，边界由上游既有拓扑处理。变换不提供跨 block、多层或海盆图连通保证。

```bash
python -m unittest discover -s diagnostics/wet_block -p 'test_*.py' -v
```

## 明确限制与反例

- 保留全部系数且不改动时，普通可逆变换也能精确恢复。精确重建仅证明接口正确，不证明能提高预测。
- 块内不相连的湿格可能经块外路径相连；“local component”不能叫独立海盆。
- 任意 1×1 系数混合、空间卷积或普通 attention 可立即破坏该隔离性质，测试包含反例。若未来预测器声称湿域隔离，**全部后续算子**都须遵守支持约束；当前主干在小波之前已经普通卷积/填陆，不能靠末端 mask 撤销已有混合。
- `interpolate(feature*valid)/interpolate(valid)` 加输出 mask 只处理二元有效性，不携带组件身份。两个有效低分辨率特征若来自不同组件，仍可能平均；测试包含反例。PixelShuffle 后乘 mask 同样不构成隔离证明。
- 保留全部系数加无约束线性层时，正交换基往往只是重参数化；须用同容量/同计算的普通基、随机正交基和支持约束开关对照。不能把固定变换零参数称作网络容量已匹配。

## 相关工作与可检验差异

| 工作 | 已有机制 | 本诊断的范围 |
| --- | --- | --- |
| [MSWT, ICML 2026](https://proceedings.mlr.press/v306/wang26bk.html) / [方法](https://arxiv.org/html/2602.01486v2) | token空间 Haar、多尺度下上采样、子带注意力 | 本工具只检验局部湿格支持，不实现 attention/多尺度模型；学习压缩后无无损保证 |
| [Gavish/Nadler/Coifman, ICML 2010](https://icml.cc/Conferences/2010/abstracts.html#) | 树/图上的多尺度正交基 | 图/自适应正交变换已有先例，不声称首次湿格图小波 |
| [Sweldens, 1996](https://doi.org/10.1006/acha.1996.0015) | lifting及定制小波构造 | 当前 Gram–Schmidt 单层基不是 lifting 新方法 |
| [exPreCast 作者源码](https://github.com/tony890048/exPreCast/blob/main/model.py) | 3D PixelShuffle+trilinear 双上采样和融合 | 干预在解码而非下采样；2D迁移不是新增专家或时间分支 |
| [SUNet 作者仓库](https://github.com/FanChiMao/SUNet) | 已有 dual upsample | 插值+重排不能单独声称创新 |

阅读边界：本轮核对 MSWT 官方页/作者方法、exPreCast 官方会议海报与可见源码 CDU、SUNet README；exPreCast 正式 PDF 本轮被浏览验证阻挡，未独立复核其表格数值。图小波/lifting 本轮核对官方摘要/作者文献页，不声称逐页复现理论。

当前任务仍为全球1°、7→7、纯SST、no-expert；主指标均匀湿格 pooled RMSE，面积权重另报。候选价值需证明**支持约束在实际误差集中的近岸位置**超越二元mask处理及普通上下采样；现有静态计数并不能提供这种证据。见 [VALIDATION.md](VALIDATION.md)。
