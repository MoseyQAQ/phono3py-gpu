"""Device arrays and bounded workspaces shared by the transport solvers."""

from functools import lru_cache
from pathlib import Path

import numpy as np


@lru_cache(maxsize=1)
def cupy():
    """Return CuPy after checking that an accessible CUDA device exists."""
    import cupy as cp

    if cp.cuda.runtime.getDeviceCount() == 0:
        raise RuntimeError("phono3py-gpu requires a CUDA GPU.")
    return cp


def host(array):
    """Synchronize a device result at the public NumPy API boundary."""
    if isinstance(array, np.ndarray):
        return array
    return cupy().asnumpy(array)


def eigh(matrix):
    """Diagonalize FP64 real-symmetric or complex-Hermitian device matrices."""
    import cupyx

    cp = cupy()
    with cupyx.errstate(linalg="raise"):
        return cp.linalg.eigh(matrix)


@lru_cache(maxsize=None)
def kernel(filename, name):
    """Compile a packaged CUDA kernel, retaining FP64 arithmetic."""
    code = (Path(__file__).parent / "kernels" / filename).read_text()
    return cupy().RawKernel(code, name, options=("--std=c++11", "--fmad=false"))


def batch_size(bytes_per_item, maximum=128):
    """Bound a temporary batch to one quarter of currently free memory."""
    free, _ = cupy().cuda.runtime.memGetInfo()
    if bytes_per_item > free // 2:
        raise MemoryError("A single GPU work item exceeds available device memory.")
    return max(1, min(maximum, free // 4 // max(1, bytes_per_item)))
