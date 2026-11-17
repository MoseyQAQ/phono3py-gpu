"""Compare CPU/GPU wall time and kappa, including public NumPy result copies.

Run inside a one-GPU allocation. IFC fitting is outside the timed region.
"""

import argparse
import json
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import phono3py

from phono3py_gpu.backend import cupy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--mesh", type=int, default=7)
parser.add_argument(
    "--conventional", action="store_true", help="24 branches instead of 6"
)
parser.add_argument("--lbte", action="store_true")
parser.add_argument("--repeat", type=int, default=2)
args = parser.parse_args()
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
base.sigmas = [0.1]
cp = cupy()
# Warm up context and basic CuPy kernels. The first full run is reported too.
cp.linalg.eigh(cp.eye(8))
cp.cuda.get_current_stream().synchronize()
result = dict(
    mesh=[args.mesh] * 3,
    bands=len(base.primitive) * 3,
    lbte=args.lbte,
    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    temperatures=[100.0, 300.0, 600.0],
    sigma=0.1,
    timings={},
)
outputs = {}
for backend in ["cpu", "gpu"]:
    seconds = []
    for repeat in range(args.repeat):
        ph = deepcopy(base)
        ph.init_phph_interaction()
        start = time.perf_counter()
        ph.run_thermal_conductivity(
            temperatures=result["temperatures"],
            is_LBTE=args.lbte,
            pinv_solver=3,
            transport_type="GPU" if backend == "gpu" else None,
        )
        cp.cuda.get_current_stream().synchronize()
        seconds.append(time.perf_counter() - start)
        outputs[backend] = ph.thermal_conductivity.kappa.copy()
    result["timings"][backend] = seconds
result["relative_kappa_error"] = float(
    np.max(np.abs(outputs["gpu"] - outputs["cpu"])) / np.max(np.abs(outputs["cpu"]))
)
result["speedup_last_run"] = result["timings"]["cpu"][-1] / result["timings"]["gpu"][-1]
np.testing.assert_allclose(outputs["gpu"], outputs["cpu"], rtol=1e-6, atol=1e-6)
print(json.dumps(result, indent=2))
