# phono3py-gpu

[English](README.md) | 简体中文

- **phono3py 插件**，可直接沿用现有的 phono3py 工作流。
- 使用 NVIDIA GPU 加速晶格热导率计算，支持**单卡和多卡并行**。

## 安装

环境要求：

- Python 3.11 或更新版本
- NVIDIA GPU 及兼容的驱动
- 与 CuPy 安装包匹配的 CUDA Toolkit
- Git 和 C/C++ 编译器，用于构建依赖

先根据 CUDA 环境选择**一个**安装选项：

| CUDA Toolkit | 安装选项 |
|---|---|
| 11.2–11.8 | `cuda11` |
| 12.x | `cuda12` |
| 13.x | `cuda13` |

以下以 **CUDA 12** 为例，其他版本请替换 `cuda12`：

```bash
git clone https://github.com/MoseyQAQ/phono3py-gpu.git
cd phono3py-gpu
pip3 install '.[cuda12]'
```

安装时从固定的 Git 提交构建 phono3py 和 phonopy。这两个版本已与 PyPI 发布的 phonors 0.8.0 一起验证，具体提交见 [pyproject.toml](pyproject.toml)。

若已有兼容的 CuPy，改用 `pip3 install .` 即可。

## 快速开始

### CLI

```bash
# 在计算目录中运行，需准备好相互匹配的输入文件：
# phono3py_params.yaml：结构和超胞
# fc2.hdf5：二阶力常数
# fc3.hdf5：三阶力常数
# 输入准备方法：https://phonopy.github.io/phono3py/

# 在 300 K 下使用 GPU 运行 BTE-RTA 计算
phono3py --br --tt GPU --mesh 9 9 9 --ts 300
```

结果写入计算目录中的 `kappa-*.hdf5` 文件。

### API

```python
import phono3py

ph3 = phono3py.load(
    "phono3py_params.yaml",
    fc2_filename="fc2.hdf5",
    fc3_filename="fc3.hdf5",
)
ph3.mesh_numbers = [9, 9, 9]
ph3.init_phph_interaction()
ph3.run_thermal_conductivity(
    temperatures=[300],
    transport_type="GPU", # 关键参数
    write_kappa=True,
)
print(ph3.thermal_conductivity.kappa)
```

在已有 Python 计算中，通过 `transport_type="GPU"` 或 `"WTE-GPU"` 选择本插件。

## 支持的输运方法

| 方法 | Python `transport_type` | Python `is_LBTE` | 命令行参数 |
|---|---|---|---|
| BTE-RTA | `"GPU"` | `False`（默认） | `--br --tt GPU` |
| LBTE | `"GPU"` | `True` | `--lbte --tt GPU` |
| WTE-RTA | `"WTE-GPU"` | `False`（默认） | `--br --tt WTE-GPU` |
| WTE-LBTE | `"WTE-GPU"` | `True` | `--lbte --tt WTE-GPU` |

插件已包含 WTE 功能，无需另外安装 phono3py-wte。

如果需要在 GPU LBTE 或 WTE-LBTE 结束后用 Python 读取末 q 点的三声子相互作用强度，请设置 `PHONO3PY_GPU_RETAIN_PP=1`，详见[进阶文档](docs/advanced.md#performance-tuning)。

## 使用多张 GPU

计算使用同一台机器上所有可见的 GPU。如需指定设备，在启动前设置 `CUDA_VISIBLE_DEVICES`；例如使用 GPU 0 和 1：

```bash
CUDA_VISIBLE_DEVICES=0,1 phono3py --br --tt GPU --mesh 9 9 9 --ts 300
```

只用一张卡时设置 `CUDA_VISIBLE_DEVICES=0`。同一设置也适用于 Python 脚本。LBTE 的全局求解默认仍使用一张 GPU，详见[内存要求与求解器选项](docs/advanced.md#out-of-memory)。

## 性能

CPU：AMD Ryzen Threadripper PRO 5995WX（8 线程）；GPU：NVIDIA Tesla V100（16 GB）。耗时单位为秒。

单卡列为预热后三次运行的中位数，使用 300 K、FP64 和四面体积分。
计时沿用原表口径，包含相互作用初始化、FC3 打包、散射及输运求解，
排除输入读取和 CUDA 上下文预热。
CPU 和多卡列保留 `8285225` 的测量结果，本次单卡更新没有重新测量这些列。

| 算例 | 声子分支 | mesh | CPU | 1 GPU | 2 GPU | 4 GPU |
|---|---:|---|---:|---:|---:|---:|
| Si · BTE-RTA | 24 | 19³ | 296.6 | 27.7 | 18.8 | 11.2 |
| AlN · BTE-RTA | 12 | 25×25×17 | 562.3 | 64.7 | 39.5 | 24.8 |
| ABO · LBTE | 60 | 6×6×5 | 441.7 | 12.3 | 11.8 | 10.2 |
| ABO · WTE-RTA | 60 | 6×6×5 | 451.8 | 9.2 | 8.5 | 7.9 |

## 帮助与贡献

- [问题排查与进阶用法](docs/advanced.md)：GPU 识别、多卡、内存和调参。
- [phono3py 文档](https://phonopy.github.io/phono3py/)：输入准备和输运计算参数。
- [GitHub Issues](https://github.com/MoseyQAQ/phono3py-gpu/issues)：提问、报告问题或提出功能建议。

欢迎贡献代码。安装与测试方法见[开发说明](docs/advanced.md#development)。

## 许可与致谢

本项目采用 [BSD 3-Clause 许可证](LICENSE)。

项目基于 [phono3py](https://github.com/phonopy/phono3py)、[phonopy](https://github.com/phonopy/phonopy) 和 [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte)。相应声明保存在 [LICENSE.phono3py](LICENSE.phono3py)、[LICENSE.phonopy](LICENSE.phonopy) 和 [LICENSE.wte](LICENSE.wte) 中。
