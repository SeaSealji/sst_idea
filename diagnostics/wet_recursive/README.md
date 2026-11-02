# #21：湿域多层递归共享上下文 Z0

CPU参考原型，验证连通粗粒化、面积归约与真正共享参数。不实现Mamba/SSM，不是DF-Mamba复现，也尚未替换当前SST预测器。没有训练或预测收益结论。

## 接口与支持语义

```python
import torch
from geometry import WetHierarchy, RecursiveContext

mask = torch.tensor([[True, False, True], [True, False, True]])
area = torch.ones(2, 3, dtype=torch.float64)
h = WetHierarchy(mask, area, depth=3, periodic_longitude=False)
context = RecursiveContext(h, width=32)  # 同一个实际block/gate跨层调用
x = torch.randn(1, 32, 2, 3, requires_grad=True)
y = context(x)
y.square().mean().backward()
print(h.summary())
```

依赖PyTorch，float32/64。输入/输出[B,C,H,W]，内部为[B,Nwet,C]。干格不参与pack，输入干格NaN安全，输出干格为0；湿格非有限值拒绝。此原型只验证CPU；Python建图/host检查无GPU加速或AMP主张。

初始图是湿格四邻接，只有明确的全球core可设经度周期；纬度不周期，不实现极点半周平移或halo反射。halo/tile不能把外缘当全球接缝。本轮真实集成仅180×360 core静态几何，并非当前244×424 halo主干的拓扑等价替换。

每层按坐标floor/2划分空间bin，但每个bin内**按子图诱导连通分量分别建节点**，允许一个bin有多个分量槽位。相邻粗格不是自动连边；仅由原有子边映射得到quotient edges。每个聚合节点对应连通子图，每条粗边有实际子边见证，因此不会把原本完全断开的湿分量接起来。不能只用全局海盆label：同一大洋的两个格点即使经块外绕路相通，在本块内仍可能不连通。递归继续携带分量节点、面积和真实边，不依赖粗格二元mask重建邻接。

D是湿面积加权均值，粗面积为子面积之和；U为parent映射的常数注入。纯D/U保常数、面积积分，且D(U(z))=z；U(D(x))一般不等于x，细节已压缩。没有严格可逆、热量守恒或有限体积物理求解器主张。

递归block：同一宽度C的self Linear + neighbor-mean Linear、tanh与残差；邻居仅取上述图边。每层先local→D→递归→U，再用逐节点标量gate融合；无跨节点BatchNorm/全局归一化。共享时每层调用同一个Parameter对象；非共享模式实例独立。学习后的消息/gate不保证常数或面积积分守恒。共享图消息不是原提案depthwise Conv的等价实现，也没有物理步长2^k缩放。宽度固定32，避开现有32→64→32的共享维度歧义。

## 验证

```bash
python -m unittest discover -s diagnostics/wet_recursive -p 'test_*.py' -v
```

12项CPU测试：15种非空2×2 mask、D/U常数/面积积分、粗边见证与聚合连通、同海盆局部断连、跨岸前向/梯度隔离、奇数尺寸经度接缝及纬度不周期、普通单槽池化反例、有损反例、解析pool梯度、共享对象/多次调用/有效梯度、double gradcheck与干格NaN/维度检查。执行结果见[VALIDATION.md](VALIDATION.md)。

## 相关工作与增量假设

| 工作 | 已有机制 | 本原型的区别与证据边界 |
|---|---|---|
| [DF-Mamba, ICML2026](https://proceedings.mlr.press/v306/fang26p.html) / [作者代码](https://github.com/yzlab1/Dynamic-Fractal-Mamba/blob/main/DFmamba.py) | 同一SSM跨尺度、配对学习压缩、2^k时间步、递归门控与重建/一致性 | 借用共享递归原则，改为二维湿图；无SSM/学习压缩/一致性loss或时间缩放 |
| [Graph U-Nets, ICML2019](https://proceedings.mlr.press/v97/gao19a.html) | 图池化/上采样、图编码解码；gPool选择节点 | 图多尺度已有先例；本原型是固定连通聚合的面积D/U，不是首次graph U-Net或新通用粗化算法 |
| 本仓库#12单层湿格变换 | 块内分量正交基、保留全部系数时重建 | 此处多层传播图边与面积、只保均值，有损；不把#12局部保证扩张成全网保证 |

直接核对DF-Mamba正式PDF第3–6页，渲染检查Figure1与Tables1/2；源码SHA256 `0e6d3c2094c2c39c1a006f31f7e1932f05df7f577e4d2af8e8efeec9473ede21`。正式PDF SHA256 `046a05b561796a4c74fa1561891ed012b452ecfc0b65b91acf48c3dee0b7143a`。源码使用单层Linear→LayerNorm→SiLU压缩，与正文两层GELU不同；源码扫描为Python循环；输入项deltaB_u使用ΔBu，不等于正文完整ZOH积分形式；正文说步长裁剪，所读核心源码未见clamp。不能将该最小源码当作论文所有实验的精确实现，也不由此否定其作者结果。

实测作者默认min_len8：L7只调用SSM一次；L8调用8→4；L16调用16→8→4。不能直接在当前7天时间轴套默认递归并期待多尺度收益。论文Table2的共享/非共享同时参数不同，不当作本项目等容量因果证据。

值得检验的增量是：**在相同数据、感受野与预算下，连通几何是否改善近岸长lead误差，以及跨层共享在几何修复之外是否提供额外作用**。局部支持诊断不证明增量；当前上游填陆和普通卷积、下游普通卷积仍可跨陆。只有全部相关运算受约束，才可谈端到端隔离，不能仅替换context就作此宣称。

后续2×2共享×湿约束及一致性独立开关见[ablation_plan.json](ablation_plan.json)。共享2145与非共享8515参数只是MVP有效容量清点，不是容量匹配。配置状态为blocked_before_training，缺少实际预测器集成、真实误差暴露、同参数与同计算对照，不自动启动训练。主指标沿用active均匀湿格RMSE，面积加权另报；不使用测试选模。
