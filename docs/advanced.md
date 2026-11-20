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

## Why is only one GPU doing the work?

The following settings use only the primary GPU for scattering, even when several GPUs are selected. A message is shown when logging is enabled:

- RTA: `read_gamma`, full/average/constant-average interaction strengths, reading or writing interaction strengths, and detailed gamma output.
- LBTE: full interaction strengths, reading interaction strengths, and per-q callbacks, including collision or interaction-strength output.
- Both: an explicit NAC q direction.

## Out of memory

Each GPU worker keeps a full copy of the force constants and phonon data in host memory and on its GPU. Adding GPUs does not reduce this storage requirement. Workers also need temporary disk space.

If you increased the [batch sizes](#performance-tuning), reduce them first. This reduces batch memory use, but does not remove the storage requirements for force constants or the LBTE collision matrix.

### LBTE memory and the optional multi-GPU solver

The default global LBTE solve runs on the primary GPU, even when scattering uses several GPUs. The dense collision matrix must fit in host memory, and the device also needs solver workspace.

To distribute the global LBTE solve across two GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 \
PHONO3PY_GPU_LBTE_SOLVER=mg \
phono3py --lbte --tt GPU --mesh 9 9 9 --ts 300
```

This requires the CUDA 12 `libcusolverMg.so.11` library to be available to the dynamic linker. Matrix storage is distributed, but each GPU also needs the workspace requested by the solver. Available memory therefore cannot be estimated by dividing the matrix size by the GPU count. Allocation failures are reported; the solver does not automatically switch to CPU execution.

The `mg` solver is optional and can be slower for small matrices. Use `PHONO3PY_GPU_LBTE_SOLVER=single` for the default solver.

## Performance tuning

The defaults are suitable for getting started. Set these environment variables before starting a calculation:

| Variable | Default | Purpose |
|---|---|---|
| `PHONO3PY_GPU_Q_BATCH_SIZE` | `1` | Maximum initial q points processed together in each worker. |
| `PHONO3PY_GPU_TRIPLET_BATCH_SIZE` | `64` | Maximum triplets per interaction batch; available memory may reduce it. |
| `PHONO3PY_GPU_RETAIN_PP` | `1` | GPU LBTE and WTE-LBTE copy only the final q point's interaction strengths and zero mask to the host by default (set to `0` to skip this copy). Per-q callbacks such as `--write-pp` still receive the data at every requested q point. |

For conductivity-only runs, disabling PP retention saves host memory and can reduce runtime when the final q point has many triplets or phonon branches. With `PHONO3PY_GPU_RETAIN_PP=0`, the calculated conductivity is the same and the `interaction_strength` and `zero_value_positions` properties of `ph3.phph_interaction` are both `None` after the run.

Increasing batch sizes raises memory use and may not improve runtime. To try larger batches:

```bash
PHONO3PY_GPU_Q_BATCH_SIZE=8 \
PHONO3PY_GPU_TRIPLET_BATCH_SIZE=256 \
phono3py --br --tt GPU --mesh 9 9 9 --ts 300
```

## Installation and compatibility

Package version requirements are listed in [pyproject.toml](../pyproject.toml). For CuPy setup, see the [installation guide](https://docs.cupy.dev/en/stable/install.html).

## Development

From the repository root, install the editable package and test dependencies:

```bash
pip3 install -e '.[cuda12,test]'
python -m pytest tests -q
```