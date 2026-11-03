# #24 WET-OTR：湿面积切向更新 Z0

这是CPU/float32&64参考实现，检查加权几何、能量方向和二阶自动微分。没有SST训练、泛化或物理热量守恒结论，不是TANGO完整复现。

```python
import torch
from update import WetMetric, TangentUpdate

metric = WetMetric([.01, 1., .3], [[0, 1]])  # 湿节点；最后一个为独立分量
layer = TangentUpdate(metric, width=4, step=.05)
x = torch.randn(2, 3, 4, requires_grad=True)  # [B,Nwet,C]
y = layer(x)
y.square().mean().backward()
with torch.no_grad():
    inference = layer(x.detach())
```

只接收湿节点；从网格pack前须排除干格，不能将干格NaN送入。调用方必须保证边是真实合法湿邻接；本工具从边推导独立连通分量，不能验证输入边是否穿陆。边按无向支持去重。训练/推理均需对特征能量求梯度：支持外层no_grad并返回detached结果，明确拒绝inference_mode。训练不detach能量梯度，保留对能量参数和输入的高阶导数。未验证CUDA/AMP/图编译或吞吐。

## 明确的度量与原型限制

各连通分量分别归一化静态正面积，使该分量Σw=1，W=diag(w)，空间权重对所有通道广播。能量采用可学习正对角二次型：

V_bk(F)=0.5 Σ_(i属于k),c w_i a_c F_bic²，a_c=softplus(q_c)+0.1。

它是有严格可查Hessian的最小学习势能，不是原论文EnergyGNN、物理热含量或输出SST误差。欧氏autograd梯度g_E用于计算度量梯度g_W=W^-1 g_E。各样本/分量分别投影：

T=M−[⟨M,g_W⟩_W/⟨g_W,g_W⟩_W]g_W，g_W=0时T=M。

实现先将g_W除以该分量detached最大绝对值再计算投影，避免小梯度平方下溢/大梯度平方溢出；非零时代数等价，无分母加epsilon的系统正交偏差。精确零点的方向场一般不连续，不声称在零点可微。若输入或度量不可表示则拒绝，不提供任意幅度数值保证。测试覆盖float32尺度1e-30…1e30，double1e-150…1e150，M为正常范围；不将此范围推广到任意能量或M。

M由self/合法邻居均值Linear+tanh预测，α=sigmoid(可学习标量)，β为可学习有符号标量；首版不是样本条件gate。F'=F+step(−αg_W+βT)。连续链式法则为dV/dt=−α||g_W||²_W，但离散Euler有二阶项，纯切向Euler也可能增加V。这里不自动回溯步长、不强行修正负结果。

**投影系数是整个湿连通分量的归约，隐式引入分量内远程依赖**。局部邻接不会使整个模块成为局地算子。新增全局感受野本身可能带来增量，必须加入同参数/同全局归约的对照。输入已受上游跨陆普通卷积污染时，后接此层也不能撤销；下游普通卷积同样可能跨陆。不得称整个当前T1已连通隔离。

## 已实现机制对照

```python
projected = TangentUpdate(metric, 4, projection=True)
plain = TangentUpdate(metric, 4, projection=False)
global_control = TangentUpdate(metric, 4, projection='fixed_normal')
```

三者具有相同学习参数集合；对照时需复制同一state_dict初始化。plain仍计算并使用相同能量下降项，只把T换回M；fixed_normal把M投到固定seed0随机通道法向的正交补，使用同样分量归约，控制全局感受野/计算，但不保证对真实能量正交。固定buffer不计作学习容量。三臂width32均2114有效学习参数；不声称同参数就等训练FLOPs，真实预算仍需计量。energy-only/tangent-only臂若禁用参数，必须单独记录有效容量，不能以未参与参数充数。

## 终稿核对与相关工作

直接核对[TANGO正式页](https://proceedings.mlr.press/v306/eliasof26a.html)、[正式27页PDF](https://raw.githubusercontent.com/mlresearch/v306/main/assets/eliasof26a/eliasof26a.pdf)方法§3–4、附录C/D.1，以及[v1](https://arxiv.org/pdf/2508.05070v1)公式页。渲染比较v1第4页与终稿第3页：**终稿Eq.3已在两处使用归一化G，修正v1未归一化末项的记号问题**。不能把旧稿反例归给终稿。终稿D.1也已给Euler二阶余项和步长界；我们的数值反例是核验这一边界，不是发现终稿定理错误。终稿提到数值小常数，未取得官方实现，不能断言其具体epsilon策略。官方页无软件链接，本轮搜索未定位并核验官方代码，不称源码复现。

| 工作 | 已有机制 | 与本假设的关系 |
|---|---|---|
| TANGO, ICML2026 | 学习势能+学习切向投影、隐特征演化 | 核心思想高度继承；本原型仅改湿面积/分量度量与受控数值，不是新投影理论 |
| [SWAN, AAAI2025作者仓库](https://github.com/gravins/SWAN) | 空间/权重反对称、非耗散传播 | 非耗散传播已有先例；本方法以学习势能法向定义切向 |
| [PH-DGN, ICLR2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/abf731c2993f9b1ee417cc3734787d7a-Abstract-Conference.html) | port-Hamiltonian保守与耗散表征 | 分解并非首次；此处没有辛结构或真实状态模拟 |
| [IGNS, ICLR2026](https://proceedings.iclr.cc/paper_files/paper/2026/hash/5b6346a05a537d4cdb2f50323452a9fe-Abstract-Conference.html) | port-Hamiltonian图模拟器、几何与多步轨迹训练 | 原任务是模拟/滚动；本项目保持单头直接7天，不迁入未来状态回灌 |
| [LAMP, ICML2026](https://proceedings.mlr.press/v306/de-castelli26a.html) | 动态图反对称流及自适应耗散 | 传播/耗散平衡已有研究；此处只做一次特征更新 |

相关工作边界：TANGO精读上述部分并比较公式图页；其他工作核对官方摘要/作者仓库介绍，不声称全部源码/表格复现。TANGO图任务性能不能作为SST技能证据。

研究假设仍是：**同参数、同全局信息、同训练预算时，真实能量法向投影能否超出普通双映射残差/固定法向投影/仅下降方向，改善D5–D7和近岸锋面**。能量下降不等于预测loss下降，二次势能可能只学幅值收缩。下一关卡需要合规训练期残差/锋面暴露与成本计量，不因数学正确就启动大规模训练。

```bash
python -m unittest discover -s diagnostics/wet_tangent -p 'test_*.py' -v
```

实际结果见[VALIDATION.md](VALIDATION.md)，设计见[ablation_plan.json](ablation_plan.json)。现行ocean2_global_v1、SST-only7→7、T1-no-expert；主指标仍均匀湿格RMSE，面积权重另报，不按提案静默换选模口径。
