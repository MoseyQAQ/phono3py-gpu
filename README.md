# phono3py-gpu

English | [简体中文](README_CN.md)

## Features

- **A phono3py plugin.** Keep your existing inputs, Python scripts, and command-line workflow; select the GPU transport option to get started.
- **GPU acceleration.** Run lattice thermal conductivity calculations on one or more NVIDIA GPUs.

## Installation

Requires Python 3.11 or newer, an NVIDIA GPU with a compatible driver, and the CUDA 12 Toolkit. Git and C/C++ compilers are needed to build the dependencies.

Install in a dedicated environment:

```bash
git clone https://github.com/MoseyQAQ/phono3py-gpu.git
cd phono3py-gpu
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

The installer selects compatible phono3py and phonopy revisions. See [installation notes](docs/advanced.md#installation-and-compatibility) for environment troubleshooting.

## Quick start

Start in a phono3py calculation directory containing these matching input files:

```text
phono3py_params.yaml
fc2.hdf5
fc3.hdf5
```

The YAML file describes your structure and supercells; the HDF5 files contain its second- and third-order force constants. For input preparation, see the [phono3py documentation](https://phonopy.github.io/phono3py/).

### Command line

Run a BTE-RTA calculation at 300 K:

```bash
phono3py-load --br --tt GPU --mesh 9 9 9 --ts 300
```

Results are written to a `kappa-*.hdf5` file in the calculation directory. Choose the mesh and temperatures for your system.

### Python

Save the following as `run_kappa.py` in the same calculation directory and run it with `python run_kappa.py`:

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

The returned thermal conductivity is in W/(m·K). In an existing Python calculation, select the plugin with `transport_type="GPU"` or `"WTE-GPU"`.

## Usage

### Choose a transport method

| Method | Python `transport_type` | Python `is_LBTE` | CLI options |
|---|---|---|---|
| BTE-RTA | `"GPU"` | `False` (default) | `--br --tt GPU` |
| LBTE | `"GPU"` | `True` | `--lbte --tt GPU` |
| WTE-RTA | `"WTE-GPU"` | `False` (default) | `--br --tt WTE-GPU` |
| WTE-LBTE | `"WTE-GPU"` | `True` | `--lbte --tt WTE-GPU` |

For example, run WTE-LBTE with:

```bash
phono3py-load --lbte --tt WTE-GPU --mesh 9 9 9 --ts 300
```

WTE is included; no separate phono3py-wte installation is needed. Integration defaults to the tetrahedron method.

### Use multiple GPUs

By default, calculations use one GPU. To use two GPUs on the same machine:

```bash
PHONO3PY_GPU_DEVICES=0,1 python run_kappa.py
```

The same setting works with the CLI:

```bash
PHONO3PY_GPU_DEVICES=0,1 phono3py-load --br --tt GPU --mesh 9 9 9 --ts 300
```

Device numbers refer to GPUs visible through `CUDA_VISIBLE_DEVICES`. See [advanced configuration](docs/advanced.md) for memory requirements, batching, and LBTE solver options.

## Performance

## Help and contributing

- [Advanced configuration](docs/advanced.md): GPU selection, memory, compatibility, and troubleshooting.
- [phono3py documentation](https://phonopy.github.io/phono3py/): input preparation and transport settings.
- [GitHub Issues](https://github.com/MoseyQAQ/phono3py-gpu/issues): questions, bug reports, and feature requests.

Contributions are welcome. See [development setup](docs/advanced.md#development) for installation and testing instructions.

## License and acknowledgements

Distributed under the [BSD 3-Clause license](LICENSE).

This project builds on [phono3py](https://github.com/phonopy/phono3py), [phonopy](https://github.com/phonopy/phonopy), and [phono3py-wte](https://github.com/MSimoncelli/phono3py-wte). Their notices are included in [LICENSE.phono3py](LICENSE.phono3py), [LICENSE.phonopy](LICENSE.phonopy), and [LICENSE.wte](LICENSE.wte).
