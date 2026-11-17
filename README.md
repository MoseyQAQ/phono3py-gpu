# phono3py-gpu

phono3py 的单 GPU、FP64 计算插件，提供普通 BTE-RTA、直接求解 LBTE、WTE-RTA、WTE-LBTE。结构、网格、参数、calculator 生命周期和 HDF5 读写沿用上游，替换重计算组件。公开结果仍是 NumPy 数组。

当前为 `0.1.0`。支持 NVIDIA CUDA 12 + CuPy 13.6；不包含多 GPU、迭代 LBTE 或四声子散射。

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

2026-11-17，V100-SXM2 16 GB、CUDA Toolkit 12.4.1、CuPy 13.6 上，**56 项测试通过**，包括部分能带/全零权重投影、交换声子分支及分批边界的碰撞矩阵逐元素对照。另通过 Ruff 检查与 wheel 打包检查。

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

## 代码与许可

`factory.py` 是依赖上游私有注册表的集中适配层；`interaction.py`、`scattering.py`、`collision.py` 分别处理相互作用、散射和全局 LBTE；`harmonic.py`、`velocity.py` 与 `kappa.py` 处理谐波和热流算符。

`fourier.py` 基于原 kappa 项目的 GPU Fourier 实现扩展；模式投影及后续散射现在在 GPU 上完成。`_wte.py` 与 `tests/reference_velocity.py` 保留 WTE 上游公式/参考代码。四面体 CUDA 分段公式来自 phonopy。许可证与版权声明见 `LICENSE`、`LICENSE.phono3py`、`LICENSE.phonopy`、`LICENSE.wte`。
