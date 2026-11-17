"""FP64 distributed symmetric eigensolve through CUDA's cuSOLVERMg C API.

The host matrix keeps phono3py's public storage contract. Device matrices use
one-dimensional block-cyclic columns; no device stores the full dense matrix.
"""

import atexit
import ctypes as ct
from functools import lru_cache

import numpy as np

from phono3py_gpu.backend import cupy, host


@lru_cache(maxsize=1)
def _library():
    lib = ct.CDLL("libcusolverMg.so.11")
    p, i, integer64 = ct.c_void_p, ct.c_int, ct.c_int64
    signatures = {
        "Create": [ct.POINTER(p)],
        "Destroy": [p],
        "DeviceSelect": [p, i, ct.POINTER(i)],
        "CreateDeviceGrid": [ct.POINTER(p), i, i, ct.POINTER(i), i],
        "DestroyGrid": [p],
        "CreateMatrixDesc": [
            ct.POINTER(p),
            integer64,
            integer64,
            integer64,
            integer64,
            i,
            p,
        ],
        "DestroyMatrixDesc": [p],
    }
    common = [p, i, i, i, ct.POINTER(p), i, i, p, p, i, i]
    signatures["Syevd_bufferSize"] = common + [ct.POINTER(integer64)]
    signatures["Syevd"] = common + [ct.POINTER(p), integer64, ct.POINTER(i)]
    for name, arguments in signatures.items():
        function = getattr(lib, "cusolverMg" + name)
        function.argtypes, function.restype = arguments, i
    return lib


def _call(name, *arguments):
    status = getattr(_library(), "cusolverMg" + name)(*arguments)
    if status:
        raise RuntimeError(f"cusolverMg{name} failed with status {status}.")


def _pointers(arrays):
    return (ct.c_void_p * len(arrays))(*(a.data.ptr for a in arrays))


@lru_cache(maxsize=None)
def _handle(devices):
    # Keep CUDA library handles alive across sigma/temperature steps, as CuPy
    # does for its dense solvers. CUDA 12.4 Mg fails after destroy/recreate.
    handle = ct.c_void_p()
    _call("Create", ct.byref(handle))
    ids = (ct.c_int * len(devices))(*devices)
    _call("DeviceSelect", handle, len(devices), ids)
    atexit.register(_library().cusolverMgDestroy, handle)
    return handle


class DistributedEigh:
    """Own distributed eigenvectors until the pseudo-inverse contractions finish."""

    def __init__(self, matrix, devices, block=256):
        cp = cupy()
        if matrix.dtype != np.float64 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("cuSOLVERMg expects a real FP64 square matrix.")
        self.devices = devices
        self.n = n = len(matrix)
        self.block = block
        self.values = np.empty(n)
        self.shards, self.columns = [], []
        self.pool = cp.cuda.MemoryPool()
        grid, desc = ct.c_void_p(), ct.c_void_p()
        gpu_ids = (ct.c_int * len(devices))(*devices)
        current = cp.cuda.runtime.getDevice()
        try:
            cp.cuda.Device(devices[0]).use()
            handle = _handle(tuple(devices))
            _call("CreateDeviceGrid", ct.byref(grid), 1, len(devices), gpu_ids, 0)
            _call("CreateMatrixDesc", ct.byref(desc), n, n, n, block, 1, grid)
            for rank, device in enumerate(devices):
                starts = range(rank * block, n, len(devices) * block)
                columns = (
                    np.concatenate([np.arange(s, min(s + block, n)) for s in starts])
                    if len(starts)
                    else np.empty(0, dtype=int)
                )
                with cp.cuda.Device(device), cp.cuda.using_allocator(self.pool.malloc):
                    shard = cp.zeros((n, max(1, len(starts)) * block), order="F")
                    for local, start in enumerate(starts):
                        width = min(block, n - start)
                        shard[:, local * block : local * block + width] = cp.asarray(
                            matrix[:, start : start + width]
                        )
                    cp.cuda.get_current_stream().synchronize()
                self.shards.append(shard)
                self.columns.append(columns)
            common = (
                handle,
                1,
                0,
                n,
                _pointers(self.shards),
                1,
                1,
                desc,
                ct.c_void_p(self.values.ctypes.data),
                1,
                1,
            )
            size, info = ct.c_int64(), ct.c_int()
            _call("Syevd_bufferSize", *common, ct.byref(size))
            self.workspace_bytes_per_device = size.value * 8
            workspace = []
            for device in devices:
                with cp.cuda.Device(device), cp.cuda.using_allocator(self.pool.malloc):
                    self.pool.free_all_blocks()
                    free, _ = cp.cuda.runtime.memGetInfo()
                    if size.value * 8 > free:
                        raise MemoryError(
                            f"cuSOLVERMg needs {size.value * 8 / 2**30:.2f} GiB "
                            f"workspace on GPU {device}; "
                            f"{free / 2**30:.2f} GiB free."
                        )
                    workspace.append(cp.empty(size.value, dtype=cp.float64))
            status = _library().cusolverMgSyevd(
                *common, _pointers(workspace), size, ct.byref(info)
            )
            for device in devices:
                cp.cuda.Device(device).synchronize()
            if status or info.value:
                raise np.linalg.LinAlgError(
                    f"cuSOLVERMg syevd status={status}, info={info.value}, "
                    f"dimension={n}, block={block}."
                )
            workspace.clear()
            for device in devices:
                with cp.cuda.Device(device):
                    self.pool.free_all_blocks()
        finally:
            if desc.value:
                _call("DestroyMatrixDesc", desc)
            if grid.value:
                _call("DestroyGrid", grid)
            cp.cuda.Device(current).use()

    def write_vectors(self, output, transpose=False):
        cp = cupy()
        for rank, device in enumerate(self.devices):
            with cp.cuda.Device(device):
                for local, start in enumerate(
                    range(rank * self.block, self.n, len(self.devices) * self.block)
                ):
                    width = min(self.block, self.n - start)
                    tile = host(
                        self.shards[rank][
                            :, local * self.block : local * self.block + width
                        ]
                    )
                    if transpose:
                        output[start : start + width] = tile.T
                    else:
                        output[:, start : start + width] = tile

    def inverse_values(self, cutoff, method=0):
        valid = (np.abs(self.values) if method == 0 else self.values) > cutoff
        inverse = np.zeros_like(self.values)
        inverse[valid] = 1 / self.values[valid]
        return inverse

    def solve(self, x, inverse):
        cp = cupy()
        partials = []
        for device, shard, columns in zip(
            self.devices, self.shards, self.columns, strict=True
        ):
            with cp.cuda.Device(device):
                v = shard[:, : len(columns)]
                scale = cp.asarray(inverse[columns])
                if x.ndim == 2:
                    scale = scale[:, None]
                partials.append(v @ ((v.T @ cp.asarray(x)) * scale))
        result = np.zeros_like(x)
        for device, partial in zip(self.devices, partials, strict=True):
            with cp.cuda.Device(device):
                result += host(partial)
        return result

    def write_inverse(self, output, inverse):
        cp = cupy()
        for start in range(0, self.n, self.block):
            stop = min(start + self.block, self.n)
            partials = []
            for device, shard, columns in zip(
                self.devices, self.shards, self.columns, strict=True
            ):
                with cp.cuda.Device(device):
                    v = shard[:, : len(columns)]
                    partials.append(
                        (v[start:stop] * cp.asarray(inverse[columns])) @ v.T
                    )
            output[start:stop] = 0
            for device, partial in zip(self.devices, partials, strict=True):
                with cp.cuda.Device(device):
                    output[start:stop] += host(partial)

    def close(self):
        cp = cupy()
        for device in self.devices:
            cp.cuda.Device(device).synchronize()
        self.shards.clear()
        for device in self.devices:
            with cp.cuda.Device(device):
                self.pool.free_all_blocks()
