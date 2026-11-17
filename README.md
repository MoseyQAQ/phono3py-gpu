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

FC3 和声子数据在一次 calculator 计算中复用；triplet 临时张量按可用显存分批，所有温度共用同一批 interaction。LBTE 稠密矩阵逐个 sigma/温度移入 GPU 求解，仍需要主机存放上游要求的完整碰撞矩阵。显存不足时报错，不会悄悄把重计算切回 CPU。当前仍计算批内的完整 band 三元组，再应用零权重掩码；稀疏投影是后续性能优化项。

**数值约定。** 默认四面体插值未平均简并带权重，换用另一组合法的简并本征矢可能改变结果；恰好落在四面体顶点的频率也对舍入约定敏感。因此 calculator 保留上游的网格声子频率、本征矢，以及 WTE 未扰动 q 点的投影基底。这是显式的兼容性步骤。独立的 `Harmonic.solve` 提供 GPU 本征求解，但不在默认工作流中替换这些基底。WTE 有限差分步长、简并阈值、半宽/全宽换算和 LBTE 伪逆截断均沿用上游。

**边界。** WTE 要求所有声子分支；可约碰撞矩阵的 WTE 相干项尚未在所参考上游实现，本插件保留该限制。`solve_collective_phonon` 同样沿用上游的不支持状态。不同本征求解器的本征矢相位、简并空间表示可以不同，因此验证物理收缩量、谱和张量，不要求本征矢逐元素相同。

所固定上游版本在 `write_LBTE_solution=True` 时会于求解后写全局 `collision-m*.hdf5`，其中的矩阵已被本征矢覆盖；不要把该文件当作原始碰撞矩阵重启。需要重启时使用 `write_collision=True` 写出的逐 q 文件，并避免生成同名的全局文件。逐 q 碰撞矩阵、gamma 和 pp 的读写复用已纳入测试。

## 验证与计时

在 GPU 节点运行：

```bash
python -m pytest tests -q
python benchmarks/transport.py --mesh 5 --conventional
python benchmarks/transport.py --mesh 5 --conventional --lbte
```

测试数据来自上游测试集：Si、NaCl，以及带已有线宽的 60 支声子 NaMgF₃。CPU 基准使用上游计算器与原始 WTE 公式；仅为较旧 WTE 的单 q 接口增加批量调用适配。测试覆盖两种积分、NAC、相互作用、四种热导率、非零相干项、可约 LBTE、伪逆选项和文件契约。

2026-11-17，V100-SXM2 16 GB、CUDA Toolkit 12.4.1、CuPy 13.6 上，**39 项测试通过**。另通过 Ruff 检查与 wheel 打包检查。下表是最终代码在同一节点的完整 calculator 计时，CPU 使用 4 线程，Si 常规胞 24 支声子，Gaussian σ=0.1 THz，T=[100, 300, 600] K；取第二次运行，原始两次数据见相应 JSON：

| 方法与网格 | CPU / s | GPU / s | 加速比 | 热导率最大差异 / CPU 最大绝对值 |
|---|---:|---:|---:|---:|
| RTA，5³ | 1.7898 | 0.4218 | 4.24× | 4.98×10⁻¹⁴ |
| LBTE，9³ | 37.7213 | 7.6317 | 4.94× | 8.28×10⁻¹⁵ |

原始结果：[`rta-v100.json`](benchmarks/rta-v100.json)、[`lbte-v100.json`](benchmarks/lbte-v100.json)。这些数字只代表所列算例，不是所有材料或网格的性能保证。WTE 的数值对照包含 NaMgF₃ 的非零相干项；上述性能表未对 WTE 宣称加速比。

`benchmarks/transport.py` 在同一节点比较完整 calculator 的墙钟时间，包含设备到 NumPy 的结果复制，不包括 IFC 拟合。首轮和后续轮次都保留，并检查热导率误差。小体系可能因 kernel 启动、传输和 Python 调度而比 CPU 慢；不能从单个 CUDA 内核的速度推断整体加速比。

SAI 环境的单 GPU 测试脚本是 `benchmarks/run-tests.sbatch`，使用仓库内 `.venv`；其他环境应使用自己的调度器参数。

## 代码与许可

`factory.py` 是依赖上游私有注册表的集中适配层；`interaction.py`、`scattering.py`、`collision.py` 分别处理相互作用、散射和全局 LBTE；`harmonic.py`、`velocity.py` 与 `kappa.py` 处理谐波和热流算符。

`fourier.py` 基于原 kappa 项目的 GPU Fourier 实现扩展；模式投影及后续散射现在在 GPU 上完成。`_wte.py` 与 `tests/reference_velocity.py` 保留 WTE 上游公式/参考代码。四面体 CUDA 分段公式来自 phonopy。许可证与版权声明见 `LICENSE`、`LICENSE.phono3py`、`LICENSE.phonopy`、`LICENSE.wte`。
