# Advanced configuration

[English README](../README.md) | [中文 README](../README_CN.md)

## Installation and compatibility

The dependencies are specified in [pyproject.toml](../pyproject.toml). The plugin currently requires development revisions of phono3py and phonopy. Phono3py 4.10.0 lacks the grid-point selection and bulk velocity interfaces it uses; phonopy 4.10.0 lacks `phonopy.phonon.velocity_matrix`, which the required phono3py revision imports. Installing the plugin selects the compatible revisions automatically. These pins can be replaced with release lower bounds once both projects publish the required interfaces.

Select one of the `cuda11`, `cuda12`, or `cuda13` extras to install the matching CuPy wheel. All require CuPy 13.6 or newer; CUDA 11 uses the CuPy 13 series because CuPy 14 dropped CUDA 11 support. A plain installation (`pip install .`) leaves CuPy selection to you, including when using conda. Do not install multiple CuPy distributions in the same environment. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) and [CuPy 13 guide](https://docs.cupy.dev/en/v13.6.0/install.html).

The GPU test suite has been validated with Python 3.12, CUDA 12.4, and CuPy 13.6 and 14.2. CUDA 11 and 13 installation options have not yet been validated by this project's GPU test suite. CUDA 13 requires Turing or newer GPUs; Volta GPUs such as V100 should use CUDA 12 (see [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)).

CUDA runtime libraries, including NVRTC, must match the selected CuPy package and be available to the Python process. On a cluster, load the appropriate CUDA environment before starting a calculation. Check that CuPy can see your allocated GPUs:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

If `GPU` or `WTE-GPU` is not recognized, confirm that the CLI and Python use the environment where the plugin is installed. Calling the following explicitly reports registration errors:

```python
import phono3py_gpu

phono3py_gpu.register()
```

## Configuration options

Set these environment variables before starting a calculation. The defaults are suitable for getting started.

| Variable | Default | Purpose |
|---|---|---|
| `PHONO3PY_GPU_DEVICES` | Current GPU | Comma-separated logical GPU IDs, such as `0,1`. |
| `PHONO3PY_GPU_Q_BATCH_SIZE` | `1` | Maximum initial q points processed together in each worker. |
| `PHONO3PY_GPU_TRIPLET_BATCH_SIZE` | `64` | Maximum triplets per interaction batch; available memory may reduce it. |
| `PHONO3PY_GPU_LBTE_SOLVER` | `single` | Global LBTE solver: `single` or optional `mg`. |
| `PHONO3PY_GPU_INITIALIZATION` | `numpy` | Geometry initialization: `numpy` or upstream `reference`. |
| `PHONO3PY_GPU_FOURIER` | `tiled` | FC3 Fourier implementation: `tiled`, `reference`, or experimental `gemm`. |
| `PHONO3PY_GPU_FOURIER_TILE` | `16` | Fourier reuse width; `tiled` supports values from 1 to 32. |

Increasing batch sizes raises memory use and may not improve runtime. For example:

```bash
PHONO3PY_GPU_Q_BATCH_SIZE=8 \
PHONO3PY_GPU_TRIPLET_BATCH_SIZE=256 \
python run_kappa.py
```

`gemm` requires the same number of retained FC3 atom pairs in every primitive-atom triplet block. Use `tiled` for the default implementation or `reference` when comparing Fourier implementations.

## GPU selection and memory

GPU IDs are relative to `CUDA_VISIBLE_DEVICES`. For example, to expose physical devices 2 and 3 and use both:

```bash
CUDA_VISIBLE_DEVICES=2,3 PHONO3PY_GPU_DEVICES=0,1 python run_kappa.py
```

Multi-GPU scattering distributes initial q points across workers on one machine. Each worker needs its own host and device copies of the force constants and phonon data. Allow space for a temporary CPU snapshot in the system temporary directory.

File and state-dependent options use the primary GPU for scattering, with a message when logging is enabled:

- RTA: `read_gamma`, full/average/constant-average interaction strengths, reading or writing interaction strengths, and detailed gamma output.
- LBTE: full interaction strengths, reading interaction strengths, and per-q callbacks, including collision or interaction-strength output.
- Both: an explicit NAC q direction.

`read_collision` uses the existing collision-file workflow. RTA callbacks in the multi-GPU path run in the original q order after the workers finish.

### LBTE memory and the optional multi-GPU solver

The default global LBTE solve runs on the primary GPU, even when scattering uses several GPUs. The dense collision matrix must fit in host memory, and the device also needs solver workspace.

To distribute the global solve across the selected GPUs:

```bash
PHONO3PY_GPU_DEVICES=0,1 \
PHONO3PY_GPU_LBTE_SOLVER=mg python run_kappa.py
```

This requires the CUDA 12 `libcusolverMg.so.11` library to be available to the dynamic linker. Matrix storage is distributed, but each GPU also needs the workspace requested by the solver. Available memory therefore cannot be estimated by dividing the matrix size by the GPU count. Allocation failures are reported; the solver does not automatically switch to CPU execution.

The `mg` solver is optional and can be slower for small matrices. Multi-GPU execution currently operates within a single machine.

## Initialization and numerical settings

The plugin speeds up the geometry preparation in `init_phph_interaction()` using NumPy. This applies to both CPU and GPU workflows in the same process once the plugin is registered. To use the original initialization for comparisons:

```bash
PHONO3PY_GPU_INITIALIZATION=reference python run_kappa.py
```

Calculations use double precision. Mesh phonon frequencies and eigenvectors follow the upstream convention, including the treatment of degenerate modes. Default integration uses the tetrahedron method; Gaussian broadening remains available through phono3py's `sigmas` setting or CLI `--sigma` option.

WTE requires all phonon branches. WTE coherence with a reducible collision matrix and `solve_collective_phonon` are not supported by the current integration.

## Saving and restarting calculations

Use `write_kappa=True` to save thermal conductivity through the Python API. The CLI writes the corresponding `kappa-*.hdf5` output as usual. Gamma and interaction-strength files retain the phono3py formats.

For LBTE restarts, use the per-q files produced by `write_collision=True`. With the supported upstream revision, `write_LBTE_solution=True` writes a global collision file after the solve, when its matrix has already been overwritten by eigenvectors. Keep `write_LBTE_solution=False` when preparing collision restart files, and avoid a conflicting global collision file in the restart directory.

## Development

From the repository root, install the editable package and test dependencies:

```bash
python -m pip install -e '.[cuda12,test]'
python -m pytest tests -q
```

Replace `cuda12` with your CUDA option, or use `.[test]` if CuPy is already installed. Run the test suite on a machine with CUDA GPUs. Tests requiring two GPUs are skipped when fewer are visible. Include your Python, CUDA, CuPy, and phono3py versions, command or script, and error output when reporting a problem.
