# phono3py-gpu

English | [简体中文](README_CN.md)

- **A phono3py plugin** that works with your existing phono3py workflows.
- **Single- and multi-GPU acceleration** for lattice thermal conductivity on NVIDIA GPUs.
- **10–50× speedups in selected benchmarks**.

## Installation

Requirements:

- Python >=3.11
- phono3py >=4.11.0
- phonopy >=4.11.0
- phonors >=0.9.0
- CuPy >=13.6 (CUDA 11: <14)
- An NVIDIA GPU with a compatible driver
- CUDA Toolkit matching your CuPy installation

Choose **one** CUDA option for your environment:

| CUDA Toolkit | Installation option |
|---|---|
| 11.2–11.8 | `cuda11` |
| 12.x | `cuda12` |
| 13.x | `cuda13` |

The example below uses **CUDA 12**. Replace `cuda12` if needed:

```bash
git clone https://github.com/MoseyQAQ/phono3py-gpu.git
cd phono3py-gpu
pip3 install '.[cuda12]'
```

If a compatible CuPy is already installed, use `pip3 install .` instead.

## Quick start

### CLI

```bash
# Run in a calculation directory containing matching input files:
# phono3py_params.yaml: structure and supercells
# fc2.hdf5: second-order force constants
# fc3.hdf5: third-order force constants
# Input preparation: https://phonopy.github.io/phono3py/

# BTE-RTA at 300 K using GPU
phono3py --br --tt GPU --mesh 9 9 9 --ts 300
```

Results are written to a `kappa-*.hdf5` file in the calculation directory.

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
    transport_type="GPU", # Important
    write_kappa=True,
)
print(ph3.thermal_conductivity.kappa)
```

In an existing Python calculation, select the plugin with `transport_type="GPU"` or `"WTE-GPU"`.

## Supported transport methods

| Method | Python `transport_type` | Python `is_LBTE` | CLI options |
|---|---|---|---|
| BTE-RTA | `"GPU"` | `False` (default) | `--br --tt GPU` |
| LBTE | `"GPU"` | `True` | `--lbte --tt GPU` |
| WTE-RTA | `"WTE-GPU"` | `False` (default) | `--br --tt WTE-GPU` |
| WTE-LBTE | `"WTE-GPU"` | `True` | `--lbte --tt WTE-GPU` |

WTE is included; no separate phono3py-wte installation is needed.

To read the final q point’s three-phonon interaction strengths in Python after GPU LBTE or WTE-LBTE, set `PHONO3PY_GPU_RETAIN_PP=1`. [Details](docs/advanced.md#performance-tuning)

## Use multiple GPUs

Calculations use all visible GPUs on the same machine. Set `CUDA_VISIBLE_DEVICES` before starting to select GPUs; for example, to use GPUs 0 and 1:

```bash
CUDA_VISIBLE_DEVICES=0,1 phono3py --br --tt GPU --mesh 9 9 9 --ts 300
```

Use `CUDA_VISIBLE_DEVICES=0` for one GPU. The same setting works with Python scripts. For LBTE, the global solve still uses one GPU by default; see [memory requirements and solver options](docs/advanced.md#out-of-memory).

## Performance

CPU: AMD Ryzen Threadripper PRO 5995WX (8 threads). GPU: NVIDIA Tesla V100 (16 GB). Times are in seconds.

| Case | Phonon branches | Mesh | CPU | 1 GPU | 2 GPUs | 4 GPUs |
|---|---:|---|---:|---:|---:|---:|
| Si · BTE-RTA | 24 | 19³ | 296.6 | 27.7 | 18.8 | 11.2 |
| AlN · BTE-RTA | 12 | 25×25×17 | 562.3 | 64.7 | 39.5 | 24.8 |
| ABO · LBTE | 60 | 6×6×5 | 441.7 | 12.3 | 11.8 | 10.2 |
| ABO · WTE-RTA | 60 | 6×6×5 | 451.8 | 9.2 | 8.5 | 7.9 |

## Help and contributing

- [Troubleshooting and advanced usage](docs/advanced.md): GPU detection, multiple GPUs, memory, and tuning.
- [phono3py documentation](https://phonopy.github.io/phono3py/): input preparation and transport settings.
- [GitHub Issues](https://github.com/MoseyQAQ/phono3py-gpu/issues): questions, bug reports, and feature requests.

Contributions are welcome. See [development setup](docs/advanced.md#development) for installation and testing instructions.

## License and acknowledgements

Distributed under the [BSD 3-Clause license](LICENSE).

This project builds on [phono3py](https://github.com/phonopy/phono3py), [phonopy](https://github.com/phonopy/phonopy), and [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte). Their notices are included in [LICENSE.phono3py](LICENSE.phono3py), [LICENSE.phonopy](LICENSE.phonopy), and [LICENSE.wte](LICENSE.wte).
