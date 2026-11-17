"""NaNbO3 Pmc21 transport benchmark using the existing, read-only 3x3x3 IFCs.

FP64, tetrahedron, 300 K; no NAC/isotope/boundary scattering. Unless explicitly
reading gamma, each invocation recomputes scattering. Timed conductivity excludes
IFC reading and Interaction initialization, recorded separately. Run from the
repository root.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from contextlib import ExitStack, chdir
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import phono3py
from ase.io import read
from phono3py import Phono3py
from phono3py.conductivity.calculators import ConductivityCalculatorBase
from phono3py.file_IO import read_fc3_from_hdf5, write_kappa_to_hdf5
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.structure.atoms import PhonopyAtoms

from phono3py_gpu.backend import cupy

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--backend", choices=["cpu", "gpu"], required=True)
parser.add_argument("--method", choices=["rta", "lbte", "wte"], required=True)
parser.add_argument("--mesh", type=int, default=12)
parser.add_argument("--q-batch", type=int, default=1)
parser.add_argument("--triplet-batch", type=int, default=64)
parser.add_argument("--read-gamma-from", type=Path)
parser.add_argument("--devices", type=int, nargs="+")
parser.add_argument("--lbte-solver", choices=["single", "mg"], default="single")
args = parser.parse_args()
if args.backend == "cpu" and args.method == "wte":
    assert args.read_gamma_from is not None
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_workflows import register_cpu_wte

    register_cpu_wte()
assert args.read_gamma_from is None or args.method == "wte"
case = Path("../../LD_kappa-strained-size3")
output = Path("benchmarks/nno")
output.mkdir(exist_ok=True)
stem = (
    f"{args.method}-{args.backend}-m{args.mesh}-q{args.q_batch}-t{args.triplet_batch}"
)
if args.read_gamma_from is not None:
    stem += "-readgamma"
if args.devices is not None:
    os.environ["PHONO3PY_GPU_DEVICES"] = ",".join(map(str, args.devices))
    stem += f"-g{len(args.devices)}"
if args.backend == "gpu":
    stem += "-" + os.environ.get("PHONO3PY_GPU_FOURIER", "tiled")
    stem += os.environ.get("PHONO3PY_GPU_FOURIER_TILE", "16")
if args.lbte_solver == "mg":
    stem += "-mg"
os.environ["PHONO3PY_GPU_LBTE_SOLVER"] = args.lbte_solver
source = hashlib.sha256()
for path in sorted(Path("src/phono3py_gpu").rglob("*")):
    if path.suffix in (".py", ".cu"):
        source.update(str(path).encode())
        source.update(path.read_bytes())
os.environ["PHONO3PY_GPU_Q_BATCH_SIZE"] = str(args.q_batch)
os.environ["PHONO3PY_GPU_TRIPLET_BATCH_SIZE"] = str(args.triplet_batch)
cp = cupy()
cp.linalg.eigh(cp.eye(8))
cp.cuda.get_current_stream().synchronize()
start = time.perf_counter()
atoms = read(case / "01-relax/relaxed.vasp")
unitcell = PhonopyAtoms(
    symbols=atoms.get_chemical_symbols(),
    cell=atoms.cell.array,
    scaled_positions=atoms.get_scaled_positions(),
)
ph = Phono3py(
    unitcell,
    supercell_matrix=np.diag([3, 3, 3]),
    primitive_matrix=np.eye(3),
    symprec=1e-5,
    calculator="vasp",
    cutoff_frequency=1e-4,
    lang="C",
    log_level=1,
)
ph.fc2 = read_force_constants_hdf5(
    case / "02-FC2/fc2.hdf5", p2s_map=ph.primitive.p2s_map
)
ph.fc3 = read_fc3_from_hdf5(case / "03-FC3/fc3.hdf5", p2s_map=ph.primitive.p2s_map)
load_seconds = time.perf_counter() - start
ph.mesh_numbers = [args.mesh] * 3
ph.sigmas = [None]
start = time.perf_counter()
ph.init_phph_interaction()
init_seconds = time.perf_counter() - start
stages = {}
original_run = ConductivityCalculatorBase.run


def measured_run(self, *a, **kw):
    for name in ("_compute_all_velocities", "_iterate_grid_points", "_finalize"):
        method = getattr(self, name)

        def measured(*ma, _method=method, _name=name, **mk):
            cp.cuda.get_current_stream().synchronize()
            begin = time.perf_counter()
            value = _method(*ma, **mk)
            cp.cuda.get_current_stream().synchronize()
            stages[_name] = time.perf_counter() - begin
            print("STAGE", _name, stages[_name], flush=True)
            return value

        setattr(self, name, measured)
    return original_run(self, *a, **kw)


transport = (
    None if args.backend == "cpu" else ("WTE-GPU" if args.method == "wte" else "GPU")
)
if args.backend == "cpu" and args.method == "wte":
    transport = "reference-wte"
print("START", stem, "load", load_seconds, "init", init_seconds, flush=True)
cp.cuda.get_current_stream().synchronize()
start = time.perf_counter()
with ExitStack() as stack:
    if args.read_gamma_from is not None:
        supplied = np.load(args.read_gamma_from)
        directory = stack.enter_context(TemporaryDirectory())
        stack.enter_context(chdir(directory))
        write_kappa_to_hdf5(
            temperature=np.array([300.0]),
            mesh=[args.mesh] * 3,
            gamma=supplied["gamma"][0],
            verbose=False,
        )
    stack.enter_context(patch.object(ConductivityCalculatorBase, "run", measured_run))
    ph.run_thermal_conductivity(
        temperatures=[300.0],
        is_LBTE=args.method == "lbte",
        transport_type=transport,
        pinv_solver=3,
        is_isotope=False,
        boundary_mfp=None,
        read_gamma=args.read_gamma_from is not None,
    )
cp.cuda.get_current_stream().synchronize()
seconds = time.perf_counter() - start
calc = ph.thermal_conductivity
solver = calc._collision_solver if args.method == "lbte" else calc._scattering_solver
engine = getattr(solver, "engine", None)
arrays = dict(
    kappa=calc.kappa,
    gamma=calc.gamma,
    grid_points=calc.grid_points,
    frequency=ph.phph_interaction.phonons.frequencies[calc.grid_points],
    group_velocity=calc.group_velocities,
)
for name in ("kappa_P_RTA", "kappa_C", "kappa_RTA", "collision_eigenvalues"):
    value = getattr(calc, name, None)
    if value is not None:
        arrays[name] = value
for name, value in arrays.items():
    assert np.isfinite(value).all(), name
np.savez_compressed(output / f"{stem}.npz", **arrays)
result = dict(
    case=str(case),
    phase="Pmc21",
    backend=args.backend,
    method=args.method,
    mesh=[args.mesh] * 3,
    ir_q=len(calc.grid_points),
    bands=len(ph.primitive) * 3,
    temperature_K=300.0,
    integration="tetrahedron",
    gamma_source=None if args.read_gamma_from is None else str(args.read_gamma_from),
    cutoff_THz=1e-4,
    q_batch=args.q_batch,
    triplet_batch=args.triplet_batch,
    actual_triplet_batch=None if engine is None else engine.batch_size,
    slurm_job_id=os.environ.get("SLURM_JOB_ID"),
    phono3py_version=phono3py.__version__,
    python_version=platform.python_version(),
    plugin_commit=subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
    plugin_source_sha256=source.hexdigest(),
    devices=args.devices,
    multi_gpu_stats=getattr(calc, "multi_gpu_stats", None),
    lbte_solver=args.lbte_solver,
    eigensolver_stats=(
        getattr(calc._kappa_solver._solver, "eigensolver_stats", None)
        if args.method == "lbte"
        else None
    ),
    fourier=os.environ.get("PHONO3PY_GPU_FOURIER", "tiled"),
    fourier_tile=int(os.environ.get("PHONO3PY_GPU_FOURIER_TILE", "16")),
    host=socket.gethostname(),
    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    cpu_threads=int(os.environ["OMP_NUM_THREADS"]),
    cpu_model=next(
        line.split(":", 1)[1].strip()
        for line in Path("/proc/cpuinfo").read_text().splitlines()
        if line.startswith("model name")
    ),
    load_seconds=load_seconds,
    initialization_seconds=init_seconds,
    conductivity_seconds=seconds,
    stage_seconds=stages,
    max_host_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    kappa=calc.kappa.tolist(),
    q_batch_stats=getattr(calc, "q_batch_stats", None),
)
(output / f"{stem}.json").write_text(json.dumps(result, indent=2) + "\n")
print("RESULT", json.dumps(result), flush=True)
