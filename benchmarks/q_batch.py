"""Measure interleaved q batches against the sequential GPU path.

FP64 tetrahedron integration; each setting is warmed once. Full-calculator and
scattering-loop times include public NumPy copies. IFC fitting is excluded.
"""

import argparse
import gc
import json
import os
import socket
import time
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import numpy as np
import phono3py

from phono3py_gpu.backend import cupy
from phono3py_gpu.factory import GPUCalculatorMixin

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=int, default=9)
parser.add_argument("--conventional", action="store_true")
parser.add_argument("--lbte", action="store_true")
parser.add_argument("--repeat", type=int, default=3)
parser.add_argument("--q-batches", type=int, nargs="+", default=[1, 2, 4, 8])
parser.add_argument("--triplet-batch", type=int, default=64)
args = parser.parse_args()
assert 1 in args.q_batches
cp = cupy()
data = Path(__file__).parents[1] / "tests/data"
kwargs = (
    dict(primitive_matrix=np.eye(3).tolist(), is_compact_fc=False)
    if args.conventional
    else {}
)
base = phono3py.load(
    data / "phono3py_si_pbesol.yaml",
    forces_fc3_filename=data / "FORCES_FC3_si_pbesol",
    lang="C",
    **kwargs,
)
base.mesh_numbers = [args.mesh] * 3
base.sigmas = [None]
original_loop = GPUCalculatorMixin._iterate_grid_points
phase = []


def timed_loop(self, *a, **kw):
    cp.cuda.get_current_stream().synchronize()
    start = time.perf_counter()
    original_loop(self, *a, **kw)
    cp.cuda.get_current_stream().synchronize()
    phase.append(time.perf_counter() - start)


result = dict(
    mesh=[args.mesh] * 3,
    bands=len(base.primitive) * 3,
    lbte=args.lbte,
    temperatures=[100.0, 300.0, 600.0],
    integration="tetrahedron",
    host=socket.gethostname(),
    cpu_threads=int(os.environ.get("OMP_NUM_THREADS", "1")),
    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    triplet_batch_limit=args.triplet_batch,
    runs={},
)
references = {}
os.environ["PHONO3PY_GPU_TRIPLET_BATCH_SIZE"] = str(args.triplet_batch)
with patch.object(GPUCalculatorMixin, "_iterate_grid_points", timed_loop):
    for repeat in range(args.repeat + 1):
        sequence = args.q_batches if repeat % 2 == 0 else args.q_batches[::-1]
        for size in sequence:
            gc.collect()
            cp.get_default_memory_pool().free_all_blocks()
            os.environ["PHONO3PY_GPU_Q_BATCH_SIZE"] = str(size)
            ph = deepcopy(base)
            ph.init_phph_interaction()
            cp.cuda.get_current_stream().synchronize()
            start = time.perf_counter()
            ph.run_thermal_conductivity(
                temperatures=result["temperatures"],
                is_LBTE=args.lbte,
                pinv_solver=3,
                transport_type="GPU",
            )
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter() - start
            calc = ph.thermal_conductivity
            solver = calc._collision_solver if args.lbte else calc._scattering_solver
            arrays = dict(gamma=calc.gamma, kappa=calc.kappa)
            if args.lbte:
                arrays["eigenvalues"] = calc.collision_eigenvalues
            if repeat == 0 and size == 1:
                references = {key: value.copy() for key, value in arrays.items()}
            errors = {}
            for key, value in arrays.items():
                np.testing.assert_allclose(value, references[key], rtol=1e-7, atol=1e-9)
                errors[key] = float(
                    np.max(np.abs(value - references[key]))
                    / max(np.max(np.abs(references[key])), 1e-30)
                )
            reserved = (
                cp.get_default_memory_pool().total_bytes()
                + solver.engine.fourier.pool.total_bytes()
            )
            record = result["runs"].setdefault(
                str(size), dict(seconds=[], scattering_seconds=[], reserved_MiB=[])
            )
            if repeat == 0:
                record["warmup_seconds"] = elapsed
            else:
                record["seconds"].append(elapsed)
                record["scattering_seconds"].append(phase[-1])
                record["reserved_MiB"].append(reserved / 2**20)
            record["relative_errors"] = errors
            record["stats"] = dict(calc.q_batch_stats)
            if size > 1:
                assert calc.q_batch_stats["max_q_per_launch"] > 1
            del solver, calc, ph, arrays
for record in result["runs"].values():
    record["median_seconds"] = float(np.median(record["seconds"]))
    record["median_scattering_seconds"] = float(np.median(record["scattering_seconds"]))
    record["speedup"] = (
        float(np.median(result["runs"]["1"]["seconds"])) / record["median_seconds"]
    )
print(json.dumps(result, indent=2))
