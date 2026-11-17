"""真实 NNO 的少量 q 点分阶段计时；FP64、12³ 四面体、300 K。

读取已有 IFC，不重做拟合。预热后测量两个代表性 q，避免完整网格重算。
阶段墙钟时间在 CUDA 同步边界测量；project 包含 Fourier 和两步收缩。
"""

import json
import os
import socket
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase.io import read
from phono3py import Phono3py
from phono3py.file_IO import read_fc3_from_hdf5
from phonopy.file_IO import read_force_constants_hdf5
from phonopy.structure.atoms import PhonopyAtoms

import phono3py_gpu.scattering as scattering
from phono3py_gpu.backend import cupy
from phono3py_gpu.harmonic import prepare_phonons
from phono3py_gpu.interaction import InteractionKernel

case = Path("../../LD_kappa-strained-size3")
cp = cupy()
atoms = read(case / "01-relax/relaxed.vasp")
ph = Phono3py(
    PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.cell.array,
        scaled_positions=atoms.get_scaled_positions(),
    ),
    supercell_matrix=np.diag([3, 3, 3]),
    primitive_matrix=np.eye(3),
    symprec=1e-5,
    calculator="vasp",
    cutoff_frequency=1e-4,
    lang="C",
)
ph.fc2 = read_force_constants_hdf5(case / "02-FC2/fc2.hdf5")
ph.fc3 = read_fc3_from_hdf5(case / "03-FC3/fc3.hdf5")
ph.mesh_numbers = [12] * 3
ph.init_phph_interaction()
pp = ph.phph_interaction
_, f, e = prepare_phonons(pp)
engine = InteractionKernel(pp, f, e)
solver = scattering.GPUScatteringSolver(
    pp, engine=engine, sigmas=[None], temperatures=np.array([300.0]), lang="C"
)
fourier = engine.fourier
stages = {}


def measured(name, function):
    def run(*args, **kwargs):
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        value = function(*args, **kwargs)
        cp.cuda.get_current_stream().synchronize()
        stages[name] = stages.get(name, 0.0) + time.perf_counter() - start
        return value

    return run


records = []
for gp in (1, 340):
    method = fourier.method
    fourier.method = "reference"
    expected = solver.compute(gp).gamma.copy()
    fourier.method = method
    solver.compute(gp)
    for repeat in range(3):
        stages.clear()
        with (
            patch.object(fourier, "transform", measured("fourier", fourier.transform)),
            patch.object(engine, "project", measured("project", engine.project)),
            patch.object(cp, "einsum", measured("contractions", cp.einsum)),
            patch.object(
                scattering, "weights", measured("weights", scattering.weights)
            ),
            patch.object(
                scattering,
                "linewidth_detail",
                measured("linewidth", scattering.linewidth_detail),
            ),
        ):
            result = measured("total", solver.compute)(gp)
        np.testing.assert_allclose(result.gamma, expected, rtol=1e-9, atol=1e-12)
        record = dict(gp=gp, repeat=repeat, triplets=len(pp.get_triplets_at_q()[0]))
        record.update(stages)
        fourier.method = "reference"
        cp.cuda.get_current_stream().synchronize()
        start = time.perf_counter()
        solver.compute(gp)
        cp.cuda.get_current_stream().synchronize()
        record["reference_seconds"] = time.perf_counter() - start
        fourier.method = method
        records.append(record)
        print(json.dumps(record), flush=True)
output = dict(
    host=socket.gethostname(),
    device=cp.cuda.runtime.getDeviceProperties(0)["name"].decode(),
    job=os.environ.get("SLURM_JOB_ID"),
    tetrahedron="reference",
    fourier=fourier.method,
    fourier_tile=fourier.tile,
    triplet_batch=engine.batch_size,
    records=records,
)
Path(f"benchmarks/nno/profile-{fourier.method}-t{fourier.tile}.json").write_text(
    json.dumps(output, indent=2) + "\n"
)
