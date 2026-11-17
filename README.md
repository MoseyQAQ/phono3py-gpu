# phono3py-gpu

phono3py 的 FP64 GPU 计算插件，提供普通 BTE-RTA、直接求解 LBTE、WTE-RTA、WTE-LBTE，支持单机多 GPU。结构、网格、参数、calculator 生命周期和 HDF5 读写沿用上游，替换重计算组件。公开结果仍是 NumPy 数组。

当前为 `0.1.0`。支持 NVIDIA CUDA 12 + CuPy 13.6；不包含跨节点 MPI、迭代 LBTE 或四声子散射。

## 安装与使用

建议在独立 Python 环境中安装，避免改变已有 phono3py 工作环境：

```bash
python -m pip install -e '.[test]'
```

需要 Python ≥ 3.11、可用的 CUDA 12 GPU/驱动、CUDA Toolkit 的运行时库（含 NVRTC），以及构建上游源码所需的 C/C++ 编译器。依赖锁定以下源码提交；PyPI 的 phono3py 4.10.0 尚不具有同一套批量 calculator 接口，不能直接替换：

- phono3py: `f0a2fdd005c03dcf2cc9734a52cbc6643406df86`
- phonopy: `cc8a6dd85912d8ae00f033ea9d38b1021d5b2dd2`
- WTE 公式与输出适配参考：`MSimoncelli/phono3py-wte@6463b233a350a842c5238a2efeade80428c99ecf`

安装后自动通过 `phono3py.conductivity` entry point 注册。原工作流只需选择 transport type：

```python
import phono3py

ph3 = phono3py.load("phono3py_params.yaml")
ph3.mesh_numbers = [9, 9, 9]
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(
    temperatures=[100, 300, 600],
    transport_type="GPU",
    is_LBTE=False,
    write_kappa=True,
)
kappa = ph3.thermal_conductivity.kappa
```

| 方法 | `transport_type` | `is_LBTE` |
|---|---|---|
| BTE-RTA | `"GPU"` | `False` |
| LBTE | `"GPU"` | `True` |
| WTE-RTA | `"WTE-GPU"` | `False` |
| WTE-LBTE | `"WTE-GPU"` | `True` |

命令行同样沿用上游参数：

```bash
phono3py-load --br --tt GPU --mesh 9 9 9 --ts 300
phono3py-load --lbte --tt WTE-GPU --mesh 9 9 9 --ts 300
```

可用 `CUDA_VISIBLE_DEVICES` 选择 GPU。显式调用 `phono3py_gpu.register()` 可以重新注册，并在依赖不匹配时获得直接的异常信息。无需安装 `phono3py-wte`；本插件内保留其公式、属性和输出适配代码及许可证。

`init_phph_interaction()` 默认也会加速：插件注册时，将上游 `Interaction._get_all_shortest` 替换为 NumPy 批量几何计算，保留最短矢量多重性与严格距离容差；接近舍入边界时使用上游的逐项运算。该步骤不启动 CUDA，Python/CLI 均无须增加调用。

初始化早于 `transport_type` 的选择，因此**插件加载后，同一进程里的 CPU 工作流也使用这项初始化优化**。仅修改进程内的辅助函数，不改动安装的 phono3py 文件；每次初始化由当前几何重新计算。对照原始 CPU 初始化时，可在启动程序前设置 `PHONO3PY_GPU_INITIALIZATION=reference`；默认值为 `numpy`。下文旧 CPU 基准的初始化时间来自原实现，不能视为优化后的 CPU 初始化时间。

## 计算范围

| 组件 | 执行位置 |
|---|---|
| FC3 Fourier 变换、r0 平均、声子模式投影、q 置换对称化 | GPU |
| Gaussian 与线性四面体积分权重 | GPU；整数邻居映射由上游生成 |
| 三声子线宽、Normal/Umklapp 分解、碰撞矩阵行 | GPU |
| 同位素散射 | GPU |
| 动力学矩阵及导数、群速度、WTE 有限差分与速度算符 | GPU；WTE 基底约定见下文 |
| 热容、RTA 热导率、WTE P/C 项 | GPU；公开结果在主机保存 |
| LBTE 简并平均、对称化、稠密本征分解/伪逆与矩阵乘法 | GPU |
| 全网格 LBTE 碰撞矩阵的对称性展开 | GPU |
| r0 平均所需的三原子最短距离表 | CPU NumPy 批量计算 |
| 结构与网格拓扑、IFC 拟合、文件读写、少量标量装配 | 上游 CPU |

FC3 和声子数据在一次 calculator 计算中复用；triplet 临时张量按可用显存分批，所有温度共用同一批 interaction。LBTE 稠密矩阵逐个 sigma/温度移入 GPU 求解，仍需要主机存放上游要求的完整碰撞矩阵。显存不足时报错，不会悄悄把重计算切回 CPU。四面体整数邻点在 calculator 内缓存；对于零积分权重的 band 三元组，最后一级模式投影直接跳过，前两级仍使用稠密矩阵收缩。`is_full_pp` 和 q 置换对称化仍按上游要求计算完整相互作用。

**数值约定。** 默认四面体插值未平均简并带权重，换用另一组合法的简并本征矢可能改变结果；恰好落在四面体顶点的频率也对舍入约定敏感。因此 calculator 保留上游的网格声子频率、本征矢，以及 WTE 未扰动 q 点的投影基底。这是显式的兼容性步骤。独立的 `Harmonic.solve` 提供 GPU 本征求解，但不在默认工作流中替换这些基底。WTE 有限差分步长、简并阈值、半宽/全宽换算和 LBTE 伪逆截断均沿用上游。

**边界。** WTE 要求所有声子分支；可约碰撞矩阵的 WTE 相干项尚未在所参考上游实现，本插件保留该限制。`solve_collective_phonon` 同样沿用上游的不支持状态。不同本征求解器的本征矢相位、简并空间表示可以不同，因此验证物理收缩量、谱和张量，不要求本征矢逐元素相同。

所固定上游版本在 `write_LBTE_solution=True` 时会于求解后写全局 `collision-m*.hdf5`，其中的矩阵已被本征矢覆盖；不要把该文件当作原始碰撞矩阵重启。需要重启时使用 `write_collision=True` 写出的逐 q 文件，并避免生成同名的全局文件。逐 q 碰撞矩阵、gamma 和 pp 的读写复用已纳入测试。

## 验证与计时

在 GPU 节点运行：

```bash
python -m pytest tests -q
python benchmarks/transport.py --mesh 5 --conventional
python benchmarks/transport.py --mesh 9 --conventional --lbte
```

计算接口和 benchmark 默认都使用四面体积分，无需设置 σ。要运行 Gaussian 对照，在对应命令末尾加 `--sigma 0.1`。Gaussian 的 σ 是能量守恒 δ 函数的数值展宽参数，与计算得到的物理线宽不同。

测试数据来自上游测试集：Si、NaCl，以及带已有线宽的 60 支声子 NaMgF₃。CPU 基准使用上游计算器与原始 WTE 公式；仅为较旧 WTE 的单 q 接口增加批量调用适配。测试覆盖两种积分、NAC、相互作用、四种热导率、非零相干项、可约 LBTE、伪逆选项和文件契约。

2026-11-17，V100-SXM2 16 GB、CUDA Toolkit 12.4.1、CuPy 13.6 上，**81 项测试通过**。覆盖部分能带/全零权重投影、交换声子分支、分批边界、四种输运方法的多卡对照、顺序回调、worker 异常退出，以及多温度的分布式本征分解和伪逆。新增初始化测试逐元素对照 Si、NaCl、斜晶胞的几何表，并覆盖距离恰好等于容差及相邻浮点数的情况。上一轮另将工作流和文件选项测试强制切为双卡，**31 项均通过**；通过 Ruff 检查与 wheel 打包检查。

下表为优化后的完整 calculator 计时，积分方式均为默认四面体。优化前后使用同一节点 `16v100n10` 的单张 V100；CPU 为 5995WX、上游 C 后端、4 线程。Si 常规胞 24 支声子，T=[100, 300, 600] K；每组 CPU/GPU 各运行三次，表中取第三次：

| 方法与网格 | CPU / s | 优化前 GPU / s | 优化后 GPU / s | 新版相对旧版 GPU | 新版相对 CPU |
|---|---:|---:|---:|---:|---:|
| RTA，5³ | 0.4907 | 0.4351 | 0.1631 | 2.67× | 3.01× |
| LBTE，9³ | 12.6569 | 7.5171 | 2.0095 | 3.74× | 6.30× |

热导率最大绝对差异除以 CPU 最大绝对值分别为 4.17×10⁻¹⁴、1.73×10⁻¹⁴。两组结果保留了全部三次计时和同节点旧版 `e291226` 的原始数据：[`rta-tetra-optimized-v100.json`](benchmarks/rta-tetra-optimized-v100.json)、[`lbte-tetra-optimized-v100.json`](benchmarks/lbte-tetra-optimized-v100.json)。首次调用仍有初始化开销，本次分别为 0.3021 s 和 2.2074 s。

本轮主要优化：

- 缓存四面体整数邻点，复用各温度的 Bose 占据数与碰撞矩阵热因子。
- 用单个 CUDA kernel 完成最后一级模式投影及频率归一化，跳过严格零积分权重项。
- 融合线宽计算；LBTE 对每个 triplet 的两种分支顺序各归约一次，再供对称旋转复用，避免复制大型四维临时张量。
- 合并速度张量的旋转求和，并跳过非简并单支的 1×1 本征求解。

四面体分段公式、阈值、声子基底与 FP64 约定保持不变。早期四面体计时保留在 [`rta-tetra-v100.json`](benchmarks/rta-tetra-v100.json)、[`lbte-tetra-v100.json`](benchmarks/lbte-tetra-v100.json)；早期 Gaussian 计时保留在 [`rta-v100.json`](benchmarks/rta-v100.json)、[`lbte-v100.json`](benchmarks/lbte-v100.json)，它们均是优化前的历史结果。

这些数字只代表所列算例，不是所有材料或网格的性能保证。WTE 的数值对照包含 NaMgF₃ 的非零相干项；上述性能表未对 WTE 宣称加速比。

`benchmarks/transport.py` 在同一节点比较完整 calculator 的墙钟时间，包含设备到 NumPy 的结果复制，不包括 IFC 拟合。首轮和后续轮次都保留，并检查热导率误差。小体系可能因 kernel 启动、传输和 Python 调度而比 CPU 慢；不能从单个 CUDA 内核的速度推断整体加速比。

## 可选的跨 q 批处理

默认仍为逐 q 路径。可以启用实验性的跨 q 批处理，原有 Python/CLI 参数不变：

```bash
PHONO3PY_GPU_Q_BATCH_SIZE=8 \
PHONO3PY_GPU_TRIPLET_BATCH_SIZE=256 \
python your_workflow.py
```

`PHONO3PY_GPU_Q_BATCH_SIZE` 是每组初始 q 点的数量上限，默认 1；`PHONO3PY_GPU_TRIPLET_BATCH_SIZE` 是每次 interaction kernel 的 triplet 数量上限，默认 64。这两个参数独立控制，后者也可用于逐 q 路径；显存预算会进一步限制实际批量。提高上限可能增加显存使用，不保证每个体系都有收益。

实现将不同初始 q 点的 triplets 交错排列，让同一次 FC3、模式投影、积分权重和线宽 kernel 处理多个 q，并按 q 分别归约。LBTE 的碰撞矩阵行也一起组装。整数网格/对称性信息仍在 CPU 逐 q 准备，最后的全局 LBTE 求解保持原流程。RTA、LBTE 及对应两种 WTE 共用这条路径。

以下选项仍自动使用逐 q GPU 路径：RTA 的 `read_gamma`、完整/平均/读取/写出 pp 和 gamma detail；LBTE 的完整/读取 pp 或逐 q 回调（包括写 pp、写碰撞矩阵）；显式指定 NAC q 方向。`read_collision` 直接复用原来的文件求解流程。RTA 的普通 gamma 输出及回调顺序保留。开启日志时会说明选项导致的回退。

2026-11-17，同节点 `16v100n05`，单张 V100-SXM2 16 GB、CPU 4 线程，Si 9³ 网格、T=[100, 300, 600] K、四面体积分。每种设置预热一次，随后三次完整 calculator 计时取中位数；以下比较均为 **GPU 与 GPU**：

| 算例 | 逐 q、64 triplets / s | 8 q、64 triplets / s | 8 q、256 triplets / s | 两项调整合计加速 |
|---|---:|---:|---:|---:|
| Si 原胞，6 支，RTA | 0.4279 | 0.3600 | 0.2477 | 1.73× |
| Si 常规胞，24 支，RTA | 0.8488 | 0.7926 | 0.7748 | 1.10× |
| Si 常规胞，24 支，LBTE | 2.0023 | 1.8306 | 1.7672 | 1.13× |

仅跨 q 合并、保持 64 triplets 上限时，三者分别加速 1.19×、1.07×、1.09×。6 支 RTA 的较大收益还有提高 triplet 批量的贡献；单独将逐 q 路径提高到 256 triplets，其耗时已降至 0.2887 s。LBTE 的散射/矩阵行阶段从 0.9177 s 降至 0.7205 s，其他阶段限制了总加速。

实际记录确认每次 kernel 最多包含 8 个不同初始 q。上述所有批量设置相对逐 q 参考的最大归一化差异：κ < 3.3×10⁻¹⁴、γ < 1.4×10⁻¹⁵。测试还覆盖 Gaussian、简并权重平均、部分能带、交换分支、可约碰撞矩阵、分批边界和回调状态。当前数据支持把跨 q 作为可选优化，尚不足以为所有体系选择统一默认批量。

完整 1/2/4/8 q、64/256 triplets 扫描、阶段计时及显存池记录见 [`q-batch-v100.json`](benchmarks/q-batch-v100.json)。显存字段是计算结束时 CuPy 默认池与 FC3 私有池保留的容量，并非整个设备的瞬时峰值。复现：

```bash
python benchmarks/q_batch.py --mesh 9
python benchmarks/q_batch.py --mesh 9 --conventional
python benchmarks/q_batch.py --mesh 9 --conventional --lbte
# 将上述命令分别追加 --triplet-batch 256 可完成第二组对照。
```

SAI 环境的单 GPU 测试脚本是 `benchmarks/run-tests.sbatch`，使用仓库内 `.venv`；其他环境应使用自己的调度器参数。

## FC3 数据复用与多 GPU

默认 FC3 Fourier 实现已改为 `tiled`：每个 CUDA block 将一块 FC3 读入共享内存，供最多 16 个 triplets 复用，保留原来的 FP64 累加顺序。它在逐 q 和跨 q 路径上都生效，无需修改 Python API。`PHONO3PY_GPU_FOURIER=reference` 可恢复原 kernel；`PHONO3PY_GPU_FOURIER_TILE` 可调整复用宽度，默认 16、最大 32。

另提供实验性的 `PHONO3PY_GPU_FOURIER=gemm`，用 FP64 批量矩阵乘法完成 Fourier 收缩；要求各原子三元组的有效 FC3 配对数相同。NNO 上 tile=32 的 GEMM 比原 kernel 快，但仍慢于共享内存版本；tile=8 甚至变慢，因此不作为默认。四面体权重的 warp 共享实验没有明显收益，未纳入最终实现。Si 的 6/24 分支检查未发现明显性能退化，原始计时见 `benchmarks/nno/si-*.json`。

启用多卡独立 q 调度：

```bash
PHONO3PY_GPU_DEVICES=0,1,2,3 python your_workflow.py
```

编号是 `CUDA_VISIBLE_DEVICES` 内的逻辑编号；不设置时使用当前单卡。每张卡有独立进程和常驻 FC3/声子数据，按 triplet 数优先调度较重的 q 组，空闲卡继续取任务。组内仍可使用 `PHONO3PY_GPU_Q_BATCH_SIZE`，默认 1 即可跨卡并行。CPU 快照暂存在系统临时目录，结束或异常后清理；每个 worker 都需要一份主机 IFC 和显存数据，不是将 FC3 分片到多卡。

RTA、LBTE、WTE-RTA、WTE-LBTE 共用该调度器。整数拓扑在 CPU 准备，速度和热流算符在主 GPU 上计算，结果按原全网格索引归并，不改变不可约权重或网格归一化。RTA 回调在 worker 完成后按原 q 顺序触发；上述需逐 q 文件/内部状态的选项继续在主 GPU 执行，并在日志中说明。`multi_gpu_stats` 保留各卡 q 数、triplet 数及耗时。

LBTE 默认仍在主 GPU 上进行全局稠密求解。也可以将矩阵列分布到多卡，启用可选的 cuSOLVERMg 后端：

```bash
PHONO3PY_GPU_DEVICES=0,1,2,3 \
PHONO3PY_GPU_LBTE_SOLVER=mg python your_lbte_workflow.py
```

该路径使用 CUDA 12 的 [`cusolverMgSyevd`](https://docs.nvidia.com/cuda/archive/12.3.1/cusolver/index.html#cusolvermgsyevd)，需要动态链接器能找到 `libcusolverMg.so.11`。矩阵、本征矢及后续伪逆收缩按列分块；简并平均和对称化也分块，避免先把完整矩阵复制到单卡。公开的主机碰撞矩阵仍保留，**每卡还需要库查询得到的额外 workspace**；显存预算不能只按矩阵大小除以 GPU 数估算。超过可用显存会报出所需 workspace，不自动退回 CPU。NNO 6³、维数 11520 的四卡实测 workspace 为每卡 2.47 GiB；未验证原 12³ LBTE 能在四张 16 GB 卡上完成。

## 真实 NaNbO₃ 算例

使用已有 `../../LD_kappa-strained-size3` 的 Pmc2₁ 结构及 FC2/FC3，原始文件只读。计算原胞 20 个原子、60 支声子，IFC 超胞为 3³、540 个原子；紧凑 FC3 形状为 `(20, 540, 540, 3, 3, 3)`。温度 300 K，四面体积分，无 NAC、同位素或边界散射。每次计算均重新计算三声子散射。

RTA/WTE 保留原 case 的 12³ 网格、343 个不可约 q 点。完整 CPU/GPU LBTE 对照采用 6³、64 个不可约 q 点：原 12³ 的稠密 LBTE 矩阵维数为 61740，单份 FP64 矩阵就需要 28.4 GiB，超过本次 V100 的 16 GB 显存。6³ 用于接口、数值和性能验证，不代表网格收敛。

2026-11-17，单张 V100-SXM2 16 GB、CPU 5995WX 4 线程。LBTE 三组运行在同一节点 `16v100n04`；RTA/WTE 运行在同一节点 `16v100n02`。每组是独立进程的一次完整 calculator 计时，CUDA 上下文预先初始化；包含 FC3 打包、散射、全局求解和 NumPy 结果复制，输入读取及 `init_phph_interaction` 另计。

| 方法与网格 | CPU / s | 逐 q GPU / s | 8 q GPU / s |
|---|---:|---:|---:|
| LBTE，6³ | 833.44 | 33.25 | 32.07 |
| RTA，12³ | 未运行完整 CPU 基准 | 1068.74 | 1067.77 |
| WTE-RTA，12³ | 未运行完整 CPU 基准 | — | 已取消，无完整 GPU 结果 |

LBTE 的 calculator 加速为 25.06×（逐 q）和 25.99×（8 q）。计入输入读取和声子初始化，CPU / GPU 逐 q / GPU 8 q 分别为 872.03 / 71.20 / 68.21 s，对应 12.25× / 12.78×；不包含 Python 导入和 CUDA 上下文建立。原有 CPU 初始化约 31 s，已占 GPU 流程相当一部分。跨 q 与提高 triplet 批量合计仅改善约 3.7%，单次运行不足以区分这样的小收益与运行波动。

12³ RTA 的两种 GPU 设置耗时基本相同，仅差约 0.09%；κ 完全相同，γ 的最大绝对差为 4.44×10⁻¹⁶ THz。这个 60 分支真实 case 不支持跨 q 能带来明显额外加速的判断，保留其可选设置。

LBTE 的 κ、γ、碰撞矩阵本征值及附带 RTA κ 均与同版本 CPU 对照，最大归一化误差低于 2.1×10⁻¹³。12³ 逐 q RTA 得到 κxx / κyy / κzz = 1.495287207 / 1.770294170 / 2.029361786 W/(m·K)，与原 case 历史结果的最大绝对差为 1.8×10⁻⁹ W/(m·K)。历史结果使用较早的 phono3py/phonopy；历史耗时不作为同版本 CPU 基准。

8 q 设置请求最多 256 triplets，仍会受到显存预算限制；本次 LBTE / RTA 实际上限为 132 / 131，实际 kernel 确认覆盖最多 8 个初始 q。完整阶段计时、张量、线宽和本征值保存在 [`benchmarks/nno/`](benchmarks/nno/)。另有两个 12³ 单 q 的当前 C/GPU 线宽对照，归一化误差低于 10⁻¹⁵，见 [`q-probe.json`](benchmarks/nno/q-probe.json)；这些单 q 耗时不能直接外推到完整 CPU 计算。

本轮新增的 FC3 优化在 12³ 网格的两个代表 q 上做同节点交替计时，预热后各三次取中位数：

| 初始 BZ q | triplets | 原单卡完整散射 / s | 新单卡完整散射 / s | 加速 |
|---|---:|---:|---:|---:|
| 1 | 294 | 1.535 | 0.716 | 2.14× |
| 340 | 518 | 2.709 | 1.261 | 2.15× |

这是单 q 的散射阶段，包含四面体权重、FC3、投影和线宽；不包括 IFC 打包、网格初始化和热导率归约。γ 与原 kernel 对照通过。原路径约 88% 时间在 FC3 Fourier，模式投影中的矩阵收缩仅约 1%，因此收益主要来自 FC3 复用。详见 [`profile-tiled-t16.json`](benchmarks/nno/profile-tiled-t16.json)，不能把单 q 加速比直接当作完整 12³ 加速比。

同节点 `16v100n22`，NNO **9³、125 个不可约 q、300 K**，新版 Fourier、逐 q、64 triplets，分别运行一次完整 RTA calculator：

| V100 数量 | calculator / s | 相对新版单卡 | 加读取和初始化 / s |
|---|---:|---:|---:|
| 1 | 89.42 | 1.00× | 127.03 |
| 2 | 49.78 | 1.80× | 85.81 |
| 4 | 29.11 | 3.07× | 64.70 |

计时包含 worker 启动、快照传递和结果归并；三组 κ 完全相同，γ 最大绝对差 2.22×10⁻¹⁶ THz。四卡各处理约 8550–8800 个 triplets，worker 计算时间约 21.7–21.9 s。加读取与初始化后的四卡加速为 1.96×，CPU 初始化限制了整体收益。

同节点 `16v100n81`，NNO **6³ LBTE**，新版单卡 calculator 为 **22.18 s**，四卡散射加 `mg` 全局求解为 **19.27 s**（1.15×）。其中全局处理/求解阶段为 6.76 / 8.48 s；这个尺寸下分布式求解开销超过收益，因此 `mg` 保持可选。四卡结果相对同版本 CPU 的 κ 最大绝对差 3.89×10⁻¹³ W/(m·K)。

同一节点另完成 **6³ WTE-RTA 四卡**，全新散射的 calculator 为 **10.15 s**。复用其 γ，仅用 CPU 重算 WTE 速度与热导率作独立公式对照，κ / κP / κC 的最大绝对差为 1.86×10⁻⁹ / 1.80×10⁻⁹ / 5.87×10⁻¹¹ W/(m·K)。该 CPU 对照不包含重新计算散射，不能用于宣称 CPU/GPU 完整 WTE 加速比。12³ WTE 的已取消任务未恢复。

接入默认 NumPy 初始化后，在同节点 `16v100n27` 再次对照 **9³ RTA、四张 V100、CPU 4 线程**。两次均为独立进程、重新计算散射，其余设置相同：

| 阶段 | 原初始化 / s | NumPy 初始化 / s |
|---|---:|---:|
| 输入读取与结构构建 | 7.24 | 5.16 |
| `init_phph_interaction()` | 29.92 | 1.08 |
| 完整 calculator | 30.10 | 29.55 |
| 上述三项合计 | 67.26 | 35.79 |

初始化实测加速 **27.78×**，减少 28.84 s；合计耗时减少约 46.8%。合计也包含两次输入读取的差异，因此 1.88× 总加速不能全部归于初始化。完整几何表 SHA256 相同，γ 和 κ 逐元素相同。容差边界处的逐项复核使正式实现略慢于最初不带该检查的 0.715 s 几何函数原型。原始记录为 [`initreference`](benchmarks/nno/rta-gpu-m9-q1-t64-g4-tiled16-initreference.json) 和 [`initnumpy`](benchmarks/nno/rta-gpu-m9-q1-t64-g4-tiled16-initnumpy.json)。

新旧阶段计时和检查结果均在 [`benchmarks/nno/`](benchmarks/nno/)，汇总见 [`comparison.json`](benchmarks/nno/comparison.json)。新记录包含源码 SHA256；`plugin_commit` 是运行时父提交，测试时改动尚未提交。以上网格用于实现验证，均不代表 κ 的网格收敛。

复现需要上述外部 case 的原始 IFC。四卡测试的示例：

```bash
sbatch --gpus-per-node=2 benchmarks/run-tests.sbatch tests
PHONO3PY_GPU_FOURIER=tiled PHONO3PY_GPU_FOURIER_TILE=16 \
sbatch --gpus-per-node=4 --qos=rush-gpu benchmarks/run-nno.sbatch scale
PHONO3PY_GPU_FOURIER=tiled PHONO3PY_GPU_FOURIER_TILE=16 \
sbatch --gpus-per-node=4 --qos=rush-gpu benchmarks/run-nno.sbatch solve
sbatch benchmarks/run-nno.sbatch profile
# 同节点比较原初始化与默认 NumPy 初始化，均重新计算完整 9³ RTA：
sbatch --gpus-per-node=4 --qos=rush-gpu benchmarks/run-nno.sbatch init
# 检查仓库内已完成的结果；不要求已取消的 12³ WTE 输出：
.venv/bin/python benchmarks/check_nno.py
```

## 代码与许可

`factory.py` 是依赖上游私有注册表的集中适配层，`initialization.py` 接管初始化的几何辅助函数；`interaction.py`、`scattering.py`、`collision.py` 分别处理相互作用、散射和全局 LBTE；`multigpu.py` 管理独立 q 的多卡调度，`eigensolver.py` 对接分布式稠密求解；`harmonic.py`、`velocity.py` 与 `kappa.py` 处理谐波和热流算符。

`fourier.py` 基于原 kappa 项目的 GPU Fourier 实现扩展；模式投影及后续散射现在在 GPU 上完成。`_wte.py` 与 `tests/reference_velocity.py` 保留 WTE 上游公式/参考代码。四面体 CUDA 分段公式来自 phonopy。许可证与版权声明见 `LICENSE`、`LICENSE.phono3py`、`LICENSE.phonopy`、`LICENSE.wte`。
