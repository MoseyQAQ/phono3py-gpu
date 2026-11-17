"""Multi-device scattering parity, failure propagation and Fourier contractions."""

from copy import deepcopy

import numpy as np
import pytest

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.fourier import GPUFC3Transform
from phono3py_gpu.multigpu import devices


def test_visible_devices():
    cp = cupy()
    expected = list(range(cp.cuda.runtime.getDeviceCount()))
    with cp.cuda.Device(expected[-1]):
        assert devices() == expected


@pytest.mark.parametrize("average", [False, True])
@pytest.mark.parametrize("method", ["gemm", "tiled"])
def test_fourier_reuse(si, monkeypatch, average, method):
    ph = deepcopy(si)
    ph.mesh_numbers = [4] * 3
    ph._make_r0_average = average
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    tp = pp.get_triplets_at_q()[0]
    monkeypatch.setenv("PHONO3PY_GPU_FOURIER", "reference")
    original = GPUFC3Transform.from_interaction(pp)
    reference = host(original.transform(tp))
    monkeypatch.setenv("PHONO3PY_GPU_FOURIER", method)
    monkeypatch.setenv("PHONO3PY_GPU_FOURIER_TILE", "3")
    optimized = GPUFC3Transform.from_interaction(pp)
    result = host(optimized.transform(tp))
    np.testing.assert_allclose(result, reference, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "lbte,wte", [(False, False), (False, True), (True, False), (True, True)]
)
def test_multi_gpu_transport(si, monkeypatch, lbte, wte):
    visible = devices()
    if len(visible) < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", "2")
    values = []
    for ids in ([0], visible):
        monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: ids)
        ph = deepcopy(si)
        ph.mesh_numbers = [4] * 3
        ph.sigmas = [None, 0.1]
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            temperatures=[0.0, 100.0, 300.0],
            transport_type="WTE-GPU" if wte else "GPU",
            is_LBTE=lbte,
            is_N_U=not lbte,
            pinv_solver=3,
        )
        values.append(ph.thermal_conductivity)
    reference, result = values
    assert result.multi_gpu_stats["devices"] == visible
    assert sum(
        x["q_points"] for x in result.multi_gpu_stats["per_device"].values()
    ) == len(result.grid_points)
    np.testing.assert_allclose(result.gamma, reference.gamma, rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(result.kappa, reference.kappa, rtol=1e-7, atol=1e-7)
    assert result._pp.get_triplets_at_q()[0][0, 0] == result.grid_points[-1]
    if lbte:
        np.testing.assert_allclose(
            result.collision_eigenvalues,
            reference.collision_eigenvalues,
            rtol=1e-8,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            result._pp.interaction_strength,
            reference._pp.interaction_strength,
            rtol=1e-9,
            atol=1e-25,
        )
    else:
        for a, b in zip(result.get_gamma_N_U(), reference.get_gamma_N_U(), strict=True):
            np.testing.assert_allclose(a, b, rtol=1e-9, atol=1e-12)
    if wte:
        np.testing.assert_allclose(
            result.kappa_C, reference.kappa_C, rtol=1e-7, atol=1e-9
        )


def test_worker_failure(si, monkeypatch):
    if cupy().cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_FOURIER", "invalid")
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.init_phph_interaction()
    with pytest.raises(RuntimeError, match="worker failed"):
        ph.run_thermal_conductivity(temperatures=[300.0], transport_type="GPU")


@pytest.mark.parametrize("block", [32, 32, 256])
def test_distributed_eigensolver(block):
    from phono3py_gpu.eigensolver import DistributedEigh

    if cupy().cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Two CUDA devices required.")
    rng = np.random.default_rng(820)
    q, _ = np.linalg.qr(rng.normal(size=(131, 131)))
    spectrum = np.linspace(-5, 8, 131)
    spectrum[50] = 0
    matrix = (q * spectrum) @ q.T
    x = rng.normal(size=(131, 3))
    solver = DistributedEigh(matrix, [0, 1], block=block)
    try:
        v = np.empty_like(matrix)
        solver.write_vectors(v)
        np.testing.assert_allclose(solver.values, np.sort(spectrum), atol=1e-11)
        np.testing.assert_allclose(matrix @ v, v * solver.values, atol=1e-11)
        np.testing.assert_allclose(v.T @ v, np.eye(131), atol=1e-11)
        for method in (0, 1):
            inverse = solver.inverse_values(1e-10, method)
            expected = (v * inverse) @ v.T
            np.testing.assert_allclose(
                solver.solve(x, inverse), expected @ x, rtol=1e-10, atol=1e-10
            )
            result = np.empty_like(matrix)
            solver.write_inverse(result, inverse)
            np.testing.assert_allclose(result, expected, rtol=1e-10, atol=1e-10)
    finally:
        solver.close()


@pytest.mark.parametrize(
    "wte,reducible,pinv,method",
    [
        (False, False, 1, 0),
        (False, False, 3, 1),
        (False, False, 6, 0),
        (False, False, 7, 0),
        (True, False, 3, 0),
        (False, True, 3, 0),
    ],
)
def test_distributed_lbte(si, monkeypatch, wte, reducible, pinv, method):
    if cupy().cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Two CUDA devices required.")
    results = []
    for mode in ("single", "mg"):
        monkeypatch.setenv("PHONO3PY_GPU_LBTE_SOLVER", mode)
        ph = deepcopy(si)
        ph.mesh_numbers = [3] * 3
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            temperatures=[0.0, 100.0, 300.0],
            is_LBTE=True,
            transport_type="WTE-GPU" if wte else "GPU",
            is_reducible_collision_matrix=reducible,
            pinv_solver=pinv,
            pinv_method=method,
        )
        results.append(ph.thermal_conductivity)
    expected, result = results
    np.testing.assert_allclose(result.kappa, expected.kappa, rtol=1e-7, atol=1e-7)
    np.testing.assert_allclose(
        result.kappa_RTA, expected.kappa_RTA, rtol=1e-7, atol=1e-7
    )
    np.testing.assert_allclose(
        result.collision_eigenvalues,
        expected.collision_eigenvalues,
        rtol=1e-8,
        atol=1e-10,
    )
