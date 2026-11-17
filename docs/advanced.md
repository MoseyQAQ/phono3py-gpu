# Troubleshooting and advanced usage

[English README](../README.md) | [中文 README](../README_CN.md)

## GPU not detected

CUDA runtime libraries, including NVRTC, must match the selected CuPy package and be available to the Python process. On a cluster, load the appropriate CUDA environment before starting a calculation. Check that CuPy can see your allocated GPUs:

```bash
python -c "import cupy; print(cupy.cuda.runtime.getDeviceCount())"
```

If `GPU` or `WTE-GPU` is not recognized, confirm that the CLI and Python use the environment where the plugin is installed. Calling the following explicitly reports registration errors:

```python
import phono3py_gpu

phono3py_gpu.register()
```

## Using multiple GPUs

All GPUs visible to CUDA are used. Set `CUDA_VISIBLE_DEVICES` before starting the process to select devices. For example, to use GPUs 2 and 3:

```bash
CUDA_VISIBLE_DEVICES=2,3 python run_kappa.py
```

CUDA renumbers the selected GPUs as logical devices 0 and 1; logs use these logical IDs. Use `CUDA_VISIBLE_DEVICES=2` to select only GPU 2. These examples use `run_kappa.py` from the [README Python example](../README.md#api) and also work with `phono3py-load`. Multi-GPU execution currently operates within a single machine.

### Why is only one GPU doing the work?

The following settings use only the primary GPU for scattering, even when several GPUs are selected. A message is shown when logging is enabled:

- RTA: `read_gamma`, full/average/constant-average interaction strengths, reading or writing interaction strengths, and detailed gamma output.
- LBTE: full interaction strengths, reading interaction strengths, and per-q callbacks, including collision or interaction-strength output.
- Both: an explicit NAC q direction.

`read_collision` uses the existing collision-file workflow; see [saving and restarting](#saving-and-restarting-calculations).

## Out of memory

Each GPU worker needs its own host and device copies of the force constants and phonon data. Adding GPUs does not split those arrays across devices. Allow space for a temporary CPU snapshot in the system temporary directory.

If you increased the [batch sizes](#performance-tuning), reduce them first. This reduces batch memory use, but does not remove the storage requirements for force constants or the LBTE collision matrix.

### LBTE memory and the optional multi-GPU solver

The default global LBTE solve runs on the primary GPU, even when scattering uses several GPUs. The dense collision matrix must fit in host memory, and the device also needs solver workspace.

For an LBTE calculation (set `is_LBTE=True` in the Python call), distribute the global solve across the selected GPUs with:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PHONO3PY_GPU_LBTE_SOLVER=mg python run_kappa.py
```

This requires the CUDA 12 `libcusolverMg.so.11` library to be available to the dynamic linker. Matrix storage is distributed, but each GPU also needs the workspace requested by the solver. Available memory therefore cannot be estimated by dividing the matrix size by the GPU count. Allocation failures are reported; the solver does not automatically switch to CPU execution.

The `mg` solver is optional and can be slower for small matrices. Use `PHONO3PY_GPU_LBTE_SOLVER=single` for the default solver.

## Saving and restarting calculations

Use `write_kappa=True` to save thermal conductivity through the Python API. The CLI writes the corresponding `kappa-*.hdf5` output as usual. Gamma and interaction-strength files retain the phono3py formats.

For LBTE restarts:

1. Save per-q collision files with `write_collision=True` and `write_LBTE_solution=False`.
2. Restart with `read_collision="all"`, keeping any conflicting global collision file out of the restart directory.

**Avoid multi-temperature restarts from a global collision file:** the current upstream reader repeats the first temperature's gamma at every temperature, which can give incorrect results.

## Performance tuning

The defaults are suitable for getting started. Set these environment variables before starting a calculation:

| Variable | Default | Purpose |
|---|---|---|
| `PHONO3PY_GPU_Q_BATCH_SIZE` | `1` | Maximum initial q points processed together in each worker. |
| `PHONO3PY_GPU_TRIPLET_BATCH_SIZE` | `64` | Maximum triplets per interaction batch; available memory may reduce it. |

Increasing batch sizes raises memory use and may not improve runtime. To try larger batches:

```bash
PHONO3PY_GPU_Q_BATCH_SIZE=8 \
PHONO3PY_GPU_TRIPLET_BATCH_SIZE=256 \
python run_kappa.py
```

## Installation and compatibility

The dependencies are specified in [pyproject.toml](../pyproject.toml). A fresh installation resolves the current `main` branches of phono3py and phonopy; existing environments are not updated automatically. Upstream API changes may require a matching plugin update.

Select one of the `cuda11`, `cuda12`, or `cuda13` extras to install the matching CuPy wheel. All require CuPy 13.6 or newer; CUDA 11 uses the CuPy 13 series because CuPy 14 dropped CUDA 11 support. A plain installation (`pip3 install .`) leaves CuPy selection to you, including when using conda. Do not install multiple CuPy distributions in the same environment. See the [CuPy installation guide](https://docs.cupy.dev/en/stable/install.html) and [CuPy 13 guide](https://docs.cupy.dev/en/v13.6.0/install.html).

The GPU test suite has been validated with Python 3.12, CUDA 12.4, and CuPy 13.6 and 14.2. CUDA 11 and 13 installation options have not yet been validated by this project's GPU test suite. CUDA 13 requires Turing or newer GPUs; Volta GPUs such as V100 should use CUDA 12 (see [NVIDIA's release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/index.html)).

## Numerical settings

Calculations use double precision. Default integration uses the tetrahedron method; Gaussian broadening remains available through phono3py's `sigmas` setting or CLI `--sigma` option.

WTE requires all phonon branches. WTE coherence with a reducible collision matrix is not supported by the current integration.

## Development

From the repository root, install the editable package and test dependencies:

```bash
pip3 install -e '.[cuda12,test]'
python -m pytest tests -q
```

Replace `cuda12` with your CUDA option, or use `.[test]` if CuPy is already installed. Run the test suite on a machine with CUDA GPUs. Tests requiring two GPUs are skipped when fewer are visible. Include your Python, CUDA, CuPy, and phono3py versions, command or script, and error output when reporting a problem.

### Implementation controls

These switches are for implementation comparisons and debugging; normal calculations use the defaults.

| Variable | Default | Purpose |
|---|---|---|
| `PHONO3PY_GPU_INITIALIZATION` | `numpy` | Geometry initialization: `numpy` or upstream `reference`. |
| `PHONO3PY_GPU_FOURIER` | `tiled` | FC3 Fourier implementation: `tiled`, `reference`, or experimental `gemm`. |
| `PHONO3PY_GPU_FOURIER_TILE` | `16` | Fourier reuse width; `tiled` supports values from 1 to 32. |

The NumPy geometry initialization in `init_phph_interaction()` applies to both CPU and GPU workflows in the same process once the plugin is registered. Set `PHONO3PY_GPU_INITIALIZATION=reference` to use the original initialization for comparisons.

`gemm` requires the same number of retained FC3 atom pairs in every primitive-atom triplet block. Use `tiled` for the default implementation or `reference` when comparing Fourier implementations.

Mesh phonon frequencies and eigenvectors follow the upstream convention, including the treatment of degenerate modes. Multi-GPU scattering distributes initial q points across workers; RTA callbacks run in the original q order after the workers finish.
