# phono3py-gpu

[English](README.md) | 简体中文

## 特点

- **phono3py 插件。** 无缝衔接现有工作流，沿用输入文件、Python 脚本和命令行用法，只需选择 GPU 输运选项。
- **GPU 加速。** 使用一张或多张 NVIDIA GPU 计算晶格热导率。

## 安装

环境要求：

- Python 3.11 或更新版本
- NVIDIA GPU 及兼容的驱动
- 与 CuPy 安装包匹配的 CUDA Toolkit（11.2–11.8、12.x 或 13.x）
- Git 和 C/C++ 编译器，用于构建依赖

建议安装到独立环境：

```bash
git clone https://github.com/MoseyQAQ/phono3py-gpu.git
cd phono3py-gpu
python -m venv .venv
source .venv/bin/activate
python -m pip install '.[cuda12]'
```

根据 CUDA 环境选择**一个**安装选项：

| CUDA Toolkit | 安装命令 |
|---|---|
| 11.2–11.8 | `python -m pip install '.[cuda11]'` |
| 12.x | `python -m pip install '.[cuda12]'` |
| 13.x | `python -m pip install '.[cuda13]'` |

若已有兼容的 CuPy，使用 `python -m pip install .` 即可复用。每个环境只安装一种 CuPy 包。依赖兼容性和已测试配置请参阅[安装说明（英文）](docs/advanced.md#installation-and-compatibility)。

## 快速开始

进入已有的 phono3py 计算目录，准备好相互匹配的输入文件：

```text
phono3py_params.yaml
fc2.hdf5
fc3.hdf5
```

YAML 文件描述结构和超胞，两个 HDF5 文件分别保存对应的二阶、三阶力常数。输入准备方法见 [phono3py 文档](https://phonopy.github.io/phono3py/)。

### 命令行

在 300 K 下运行 BTE-RTA 计算：

```bash
phono3py-load --br --tt GPU --mesh 9 9 9 --ts 300
```

结果写入计算目录中的 `kappa-*.hdf5` 文件。请根据体系选择网格和温度。

### Python

将以下代码保存为计算目录中的 `run_kappa.py`，然后运行 `python run_kappa.py`：

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
    transport_type="GPU",
    write_kappa=True,
)
print(ph3.thermal_conductivity.kappa)
```

热导率的单位为 W/(m·K)。在已有 Python 计算中，通过 `transport_type="GPU"` 或 `"WTE-GPU"` 选择本插件。

## 使用方法

### 选择输运方法

| 方法 | Python `transport_type` | Python `is_LBTE` | 命令行参数 |
|---|---|---|---|
| BTE-RTA | `"GPU"` | `False`（默认） | `--br --tt GPU` |
| LBTE | `"GPU"` | `True` | `--lbte --tt GPU` |
| WTE-RTA | `"WTE-GPU"` | `False`（默认） | `--br --tt WTE-GPU` |
| WTE-LBTE | `"WTE-GPU"` | `True` | `--lbte --tt WTE-GPU` |

例如，运行 WTE-LBTE：

```bash
phono3py-load --lbte --tt WTE-GPU --mesh 9 9 9 --ts 300
```

插件已包含 WTE 功能，无需另外安装 phono3py-wte。积分默认采用四面体方法。

### 使用多张 GPU

默认使用一张 GPU。在同一台机器上使用两张 GPU：

```bash
PHONO3PY_GPU_DEVICES=0,1 python run_kappa.py
```

命令行同样支持此设置：

```bash
PHONO3PY_GPU_DEVICES=0,1 phono3py-load --br --tt GPU --mesh 9 9 9 --ts 300
```

设备编号以 `CUDA_VISIBLE_DEVICES` 中可见的 GPU 为准。显存要求、批处理和 LBTE 求解器选项见[高级配置（英文）](docs/advanced.md)。

## 性能

## 帮助与贡献

- [高级配置（英文）](docs/advanced.md)：GPU 选择、内存、兼容性和问题排查。
- [phono3py 文档](https://phonopy.github.io/phono3py/)：输入准备和输运计算参数。
- [GitHub Issues](https://github.com/MoseyQAQ/phono3py-gpu/issues)：提问、报告问题或提出功能建议。

欢迎贡献代码。安装与测试方法见[开发说明（英文）](docs/advanced.md#development)。

## 许可与致谢

本项目采用 [BSD 3-Clause 许可证](LICENSE)。

项目基于 [phono3py](https://github.com/phonopy/phono3py)、[phonopy](https://github.com/phonopy/phonopy) 和 [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte)。相应声明保存在 [LICENSE.phono3py](LICENSE.phono3py)、[LICENSE.phonopy](LICENSE.phonopy) 和 [LICENSE.wte](LICENSE.wte) 中。
