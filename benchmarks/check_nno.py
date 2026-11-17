"""核对已经完成的 NNO 张量、线宽、本征谱及多卡结果；不启动计算。"""

import json
from pathlib import Path

import h5py
import numpy as np

root = Path("benchmarks/nno")
case = Path("../../LD_kappa-strained-size3")
comparisons, records = {}, {}


def load(stem):
    records[stem] = json.loads((root / f"{stem}.json").read_text())
    with np.load(root / f"{stem}.npz") as data:
        return dict(data)


def compare(name, actual, expected, rtol=1e-7, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
    error = float(np.max(np.abs(actual - expected)))
    comparisons[name] = dict(
        max_absolute_error=error,
        normalized_max_error=error / max(float(np.max(np.abs(expected))), 1e-30),
    )


def compare_fields(stem, actual, expected, fields):
    np.testing.assert_array_equal(actual["grid_points"], expected["grid_points"])
    for key in fields:
        compare(stem + ":" + key, actual[key], expected[key])


rta = load("rta-gpu-m12-q1-t64")
compare_fields("rta12:q8", load("rta-gpu-m12-q8-t256"), rta, ("gamma", "kappa"))
with h5py.File(case / "04-kappa/kappa-m121212.hdf5", "r") as f:
    np.testing.assert_array_equal(rta["grid_points"], f["grid_point"][:])
    compare("legacy:rta:kappa", rta["kappa"][0], f["kappa"][:], 1e-5, 1e-7)
    compare("legacy:frequencies", rta["frequency"], f["frequency"][:], 1e-5, 1e-6)

lbte = load("lbte-cpu-m6-q1-t64")
for path in sorted(root.glob("lbte-gpu-m6-*.npz")):
    compare_fields(
        path.stem,
        load(path.stem),
        lbte,
        ("gamma", "kappa", "kappa_RTA", "collision_eigenvalues"),
    )

scale = load("rta-gpu-m9-q1-t64-g1-tiled16")
for count in (2, 4):
    stem = f"rta-gpu-m9-q1-t64-g{count}-tiled16"
    compare_fields(stem, load(stem), scale, ("gamma", "kappa"))
    stats = records[stem]["multi_gpu_stats"]
    assert len(stats["devices"]) == count
    assert sum(v["q_points"] for v in stats["per_device"].values()) == len(
        scale["grid_points"]
    )

init_prefix = "rta-gpu-m9-q1-t64-g4-tiled16-init"
if (root / f"{init_prefix}numpy.npz").exists():
    reference = load(init_prefix + "reference")
    optimized = load(init_prefix + "numpy")
    compare_fields("initialization", optimized, reference, ("gamma", "kappa"))
    assert (
        records[init_prefix + "reference"]["geometry_mask_sha256"]
        == records[init_prefix + "numpy"]["geometry_mask_sha256"]
    )

wte6 = "wte-gpu-m6-q1-t64-g4-tiled16"
if (root / f"{wte6}.npz").exists():
    compare_fields(
        wte6,
        load(wte6),
        load("wte-cpu-m6-q1-t64-readgamma"),
        ("gamma", "kappa", "kappa_P_RTA", "kappa_C"),
    )

result = dict(
    timing_scope="Full calculator; input loading and initialization also recorded.",
    wte_reference_scope=(
        "CPU WTE velocity/kappa with saved GPU gamma; not CPU scattering."
    ),
    wte12_status="Cancelled; no completed GPU WTE result or speedup claimed.",
    comparisons=comparisons,
    timings_seconds={s: r["conductivity_seconds"] for s, r in records.items()},
    timings_with_load_and_initialization_seconds={
        s: sum(
            r[k]
            for k in ("load_seconds", "initialization_seconds", "conductivity_seconds")
        )
        for s, r in records.items()
    },
)
(root / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
