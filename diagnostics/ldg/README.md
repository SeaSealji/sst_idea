# 七天 LDG / 线性折叠 Z0

对应 #9 SiGMA 补充。此目录提供合成算子诊断，尚无 SST 预测器、训练入口或性能结论。

```sh
python -m unittest discover -s diagnostics/ldg -v
python diagnostics/ldg/audit.py
```

依赖 PyTorch、SciPy、NumPy。全部检查仅CPU。`ldg_kernel` 是64项 Bessel 幂级数，
保留原生 autograd，仅核验七点、float32/64、尺度 `[1e-4,20]`；不是无限范围
special-function 实现，未核验 CUDA/AMP/速度。输入范围检查本身含 host 同步，
不能据此宣称消除了训练时 CPU/GPU 同步。

## 核对的实现与论文边界

- [SiGMA 正式页](https://proceedings.mlr.press/v306/lee26al.html)，ICML2026/PMLR306。
- 本轮直接阅读[作者 v1 §4/§5.2/附录A.4](https://arxiv.org/html/2605.31129v1)；未逐页对照正式 PDF。
- [作者源码](https://github.com/cheonwoolee/SiGMA/blob/main/models/SiGMA.py) SHA256
  `409627a8005f39cef59bc78f4932fb4cca49e4b7a403285be4a4c904fe06e213`。
  `s=softplus(t_param)` 是每个距离一个参数，各样本/空间位置共享，推理固定；
  不是输入条件门控。源码前后向经 SciPy `.cpu().numpy()`；本目录不复制该源码。
- 作者实际先 embedding、LDG，再非线性 MLP；本审计只针对迁移时的**纯线性输入前处理**，
  没有证明完整 SiGMA 可折叠。作者实现对多层的列表处理也不能视为本项目端到端复现。

作者 v1 Definition4.1 的 consistency 要求共享尺度离散子族 `s*1` 满足缩放条件，
加上连续参数可微；**并未要求任意独立距离参数都 L2 非扩张**。附录A.4的共享尺度
证明使用无限整数格卷积、归一化与半群，不能直接赋给未做边界处理的有限七点矩阵。
这里的反例是迁移实现审计，不是对论文定理正确性的裁决。

## 算子结论

`K_ij=exp(-s_d) I_d(s_d), d=|i-j|`。默认 `s=softplus(0)=log(2)`，七点截断
行和端点约0.78094、中点0.99934：常数会受端点衰减。独立尺度
`[.01,1,1,1,1,1,1]` 的最大行和1.52208、谱范数1.45830。

行归一化 `P=diag(K1)^(-1)K` 保常数，但反例的 `||P||_2=1.000424>1`，
故不能同时宣称 L2 非扩张。它一般不再对称或 Toeplitz。本文约定列向量 `Px`；
作者代码为 `x_ct @ K`，原K对称掩盖了方向差异，直接换成 `x_ct @ P` 反而
不保常数（需 `x_ct @ P.T` 才对应此处算子）。必须单独审计。

`path_heat=exp(-s L_path)` 是**另一个控制算子**：对称组合路径图Laplacian，
端点度1，无通量/反射边界，保证常数、非负、谱范数1及半群；尺度系数与无限格
离散高斯的约定可不同。不是作者LDG等价实现，也不是海洋物理扩散模型。

## 线性折叠的适用范围

当同一个静态K在所有空间位置共享，先生成 `KX,(I-K)X`，再接同支持/stride/
padding/dilation的普通卷积、**融合前没有非线性**时：

`W_eff[o,j,u,v]=sum_i(W_s[o,i,u,v] K[i,j]+W_r[o,i,u,v](I-K)[i,j])`。

偏置直接相加。可折叠至普通七通道卷积，输入/权重/尺度的反向也一致（浮点误差内）。
K训练所得但推理固定仍成立；这不等价于用独立W_eff训练时有相同AdamW路径、正则或优化偏置。
共享静态land-fill与时间混合可交换，但逐日不同mask、样本/位置条件K、分支非线性不在结论范围。
保留原始最后历史日锚点，不能把滤波后的末帧偷偷替换为预测锚点。

分支先SiLU会破坏简单线性折叠；测试只证明函数非仿射，**并未证明时尺度有新增预测价值**。
`KX+(I-K)X=X`，分解本身不创造新观测。

## 最近相关工作与后续关卡

|方法|已存在的设计|此处必须隔离的差异|
|---|---|---|
|[DLinear](https://github.com/cure-lab/LTSF-Linear/blob/main/models/DLinear.py)|端点重复padding、固定moving average+残差、独立线性时间头|分解加线性头不是新颖性；固定线性版本可吸收为线性映射|
|[TimeMixer](https://github.com/kwuking/TimeMixer/blob/main/models/TimeMixer.py)|显式时间池化/stride、多尺度季节/趋势非线性混合|当前SST七通道Conv2d无时间stride，不称修复既有混叠|
|SiGMA|距离共享的连续可学习核+非线性MLP，通常历史96步|七天窗口/静态共享核的边界与容量控制，以及非线性路径独立增量|

本轮未读取真实SST、未加载训练权重、未更新优化器。实际历史差分/自相关及训练期残差
条件增量尚未完成，**完整Z0没有通过，不进入模型训练**。

若以后数据诊断支持继续，预先冻结等结构非线性支路：普通Conv、固定3/5日平均、
自由/低秩7×7混合、原LDG、行归一化LDG、单尺度path-heat；另保留可折叠线性负对照。
同原始锚点、窗口/采样/预算、有效参数与FLOPs记录，不能把无效填充参数当匹配容量。
配对seed先42/43/44，与已归档协议对齐，任何扩种子需另登记；逐日/D5–D7、春季、
近岸离岸与时间块不确定性都报告。active仍为均匀湿格RMSE主、面积权重另报。
未在这些强对照上有实质增量就停止复杂分支，不因为已有先例而直接否定所有非线性改进。
