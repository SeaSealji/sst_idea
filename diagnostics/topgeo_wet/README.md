# #21 补充：球面湿图粗化 Z0

独立CPU参考实现；仅验证静态图、面积D/U和合成前后向。不是CGFM生成模型或原论文全部实验复现，没有SST预测收益结论。

## 来源与实质重合

直接阅读CGFM的[正式23页PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/deng26c/deng26c.pdf)§3.2–3.3、Algorithm1、Table4和AppendixC/F.3，渲染查看第4页Figure2、第5页Algorithm1和第9页Table4。CGFM[正式页](https://proceedings.mlr.press/v306/deng26c.html)与公开实现[mesh.py](https://github.com/xcD6/cgfm-main/blob/4d603f3332740941f89702c65c985d75b83d26c7/cgfm/transforms/mesh.py)、multiscale.py已核对；未找到正式页/论文指向该仓库的链接，作者官方身份仍未独立确认。只执行审阅后的单个粗化函数和合成输入，未导入上游包或跑其训练。

CGFM正式Algorithm1按`degree × mean邻居距离`升序选代表节点，排除相邻节点；父节点先最少hop，再欧氏距离；粗边由子边见证。Table4确有相对Guillard的Wing分布指标改善，但不是SST或本项目匹配预算证据。

[DGN4CFD, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/bfc02f1f412e8a6e710b1e00ea9ca670-Abstract-Conference.html)已有多尺度图生成、Guillard粗化及可选flow matching。[作者仓库](https://github.com/tum-pbs/dgn4cfd)和[作者PDF](https://arxiv.org/pdf/2504.02843)AppendixB.3/Algorithm1（第17–18页）已核对。其按节点顺序选代表、最少hop/距离父子映射和子边商图与CGFM高度重合；CGFM排序中引入几何距离是可分离的变化。不能声称首次图生成、多尺度图或连通粗化。公开CGFM README也注明源自DGN4CFD。

与本仓库PR23的区别：PR23固定2×2bin内诱导连通聚合；这里从合法湿图选球面距离评分的独立集代表，允许聚合跨bin，但每个聚合是连通一跳星形。D/U与共享上下文复用我们自己的PR23设计，不复制第三方实现。不是学习几何MLP、锋面自适应分配或flow matching。

## 实现合同

- `SphereGraph`只接受湿节点、正有限面积、角度制经纬度与无向合法边列表；调用方负责证明边未穿陆。边规范化/去重，对边排列与方向表示不敏感，拒绝自边/越界。
- 距离是单位球面great-circle角`atan2(||u×v||, u·v)`，无经度接缝长距离；不拿经纬度直接做欧氏差。`geometry`评分为邻边角度之和，等于degree×mean距离；`degree`与`index`是机制控制。
- 按(score,node_id)贪心选极大独立集。无向图保证每个未保留节点都有保留的一跳邻居，选择最近保留邻居并按ID破同分；孤立点保留。无多轮父子传播或max_iter。节点重新编号可能改变同分决策，不声称重编号等变。
- 每个聚合有连通星形支持；粗边只由真实子边映射。保持连通分量分区，但不保证环洞、最短路、窄海峡分辨率或物理海盆身份不变。粗代表之间的球面线段可能穿陆：粗边表示有见证的湿路径，不是实际直线输运通道。
- D为面积均值、U为parent常数注入，纯D/U保常数、面积积分且D(U(z))=z；U(D(x))有损。学习消息/递归gate不保证面积均值，更不保证热量守恒。
- [B,C,H,W]网格输入只pack湿格，干格NaN不进入算子；仅float32/64。core只经度周期，无纬向wrap。halo反射/别名集成只验证计算接口，不证明物理极点拓扑等价；上/下游普通卷积可能跨陆。

## 上游排序依赖的正确解释

审阅函数以`row.split(indegree.tolist())`取邻居，要求边按接收节点排列。正常`cells_to_edge_index`与`pool_edges`使用`coalesce(..., sort_by_row=False)`，提供此合同。因此随机打乱原始函数输入能改变结果，是独立调用风险，不是已发表训练有bug的证据。合成五节点路径：规范输入保留[0,2,4]，一个打乱输入只保留[4]；本实现两种输入结果一致。公开上游缺少内部排序断言；本工具选择入口规范化。

## 使用与检验

```python
import torch
from context import WetHierarchy, RecursiveContext

mask = torch.ones(3, 5, dtype=torch.bool)
area = torch.ones(3, 5)
h = WetHierarchy(mask, area, [-40, 0, 40], [0, 1, 2, 3, 4], depth=2)
net = RecursiveContext(h, width=4)
x = torch.randn(1, 4, 3, 5, requires_grad=True)
net(x).square().mean().backward()
```

```bash
python -m unittest discover -s diagnostics/topgeo_wet -p 'test_*.py' -v
```

同width32、depth3、共享模式，三种排序均2145有效学习参数。跨排序复制学习参数，不复制形状不同的几何buffers；相同参数数不代表相同图节点、边数、FLOPs或感受野。当前T1替换原context模块的合成接口已通过，但总参数从138631降为46344，不是等容量对照。固定宽度共享不是CGFM原始HieraGraphNet。

静态geometry评分只依赖纬度、岸线度数与粗图形状，没有任何SST锋面信息。在规则经纬网格，低评分可能主要来自高纬东西间距缩小；不能把代表改变当作锋面保真改善。下一关卡是合规训练期残差/锋面暴露与同压缩/同实际预算控制。主指标仍active协议的均匀湿格RMSE，面积权重单列。详细结果见VALIDATION.md与ablation_plan.json。
