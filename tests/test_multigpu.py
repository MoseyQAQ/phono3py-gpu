"""Multi-device scattering parity, failure propagation and Fourier contractions."""

from copy import deepcopy

import numpy as np
import pytest

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.fourier import GPUFC3Transform
from phono3py_gpu.multigpu import _load_snapshot, _write_snapshot, devices


def test_visible_devices():
    cp = cupy()
    expected = list(range(cp.cuda.runtime.getDeviceCount()))
    with cp.cuda.Device(expected[-1]):
        assert devices() == expected


def test_worker_state_is_private(si, tmp_path):
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    path = tmp_path / "interaction.pickle"
    with path.open("wb") as stream:
        _write_snapshot(pp, stream)
    with path.open("rb") as stream:
        worker = _load_snapshot(stream)
    for name in ("fc3", "fc3_nonzero_indices", "all_shortest"):
        np.testing.assert_array_equal(getattr(worker, name), getattr(pp, name))
    for name in ("frequencies", "eigenvectors", "degenerate_ids"):
        source = getattr(pp.phonons, name)
        copied = getattr(worker.phonons, name)
        np.testing.assert_array_equal(source, copied)
        assert not np.shares_memory(source, copied)
    worker.set_grid_point(2)
    assert pp.get_triplets_at_q()[0][0, 0] == 1


def test_mapped_snapshot_is_copy_on_write(tmp_path):
    path = tmp_path / "arrays.pickle"
    large = np.arange(200_000, dtype=np.float64)
    payload = dict(large=large, alias=large, small=np.arange(3))
    with path.open("wb") as stream:
        _write_snapshot(payload, stream)
    with path.open("rb") as stream:
        first = _load_snapshot(stream)
    with path.open("rb") as stream:
        second = _load_snapshot(stream)
    assert first["large"] is first["alias"]
    assert not first["large"].flags.owndata
    assert first["large"].ctypes.data % 64 == 0
    first["large"][0] = -1
    first["small"][0] = -1
    np.testing.assert_array_equal(second["large"], large)
    np.testing.assert_array_equal(second["small"], payload["small"])
    with path.open("rb") as stream:
        np.testing.assert_array_equal(_load_snapshot(stream)["large"], large)


@pytest.mark.parametrize("q_batch", [1, 8])
def test_single_task_uses_primary_gpu(si, monkeypatch, q_batch):
    if len(devices()) < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", str(q_batch))
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(
        temperatures=[300.0], transport_type="GPU", is_LBTE=True,
        grid_points=[1], pinv_solver=3,
    )
    assert ph.thermal_conductivity.multi_gpu_stats is None


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
@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("material", ["si", "nacl"])
def test_multi_gpu_transport(
    request, monkeypatch, capsys, lbte, wte, streamed, material
):
    if streamed:
        monkeypatch.setattr("phono3py_gpu.interaction.MIN_STREAMED_BANDS", 1)
        monkeypatch.setattr("phono3py_gpu.interaction.initial_band_block", lambda *_: 2)
        # Apply the same small-fixture tile settings in fresh worker interpreters.
        import phono3py_gpu.multigpu as multigpu

        popen = multigpu.subprocess.Popen

        def launch(args, **kwargs):
            bootstrap = (
                "import sys; import phono3py; "
                "import phono3py_gpu.interaction as interaction; "
                "interaction.MIN_STREAMED_BANDS = 1; "
                "interaction.initial_band_block = lambda *_: 2; "
                "from phono3py_gpu.multigpu import _worker; _worker()"
            )
            return popen([args[0], "-c", bootstrap, *args[-2:]], **kwargs)

        monkeypatch.setattr(multigpu.subprocess, "Popen", launch)
    monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", "1")
    visible = devices()
    if len(visible) < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", "2")
    values = []
    for ids in ([0], visible):
        monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: ids)
        ph = deepcopy(request.getfixturevalue(material))
        ph._log_level = 1
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
        calc = ph.thermal_conductivity
        used = (
            ids
            if len(ids) == 1
            else [
                dev
                for dev in ids
                if calc.multi_gpu_stats["per_device"][dev]["q_points"]
            ]
        )
        method = ("WTE" if wte else "BTE") + ("-LBTE" if lbte else "-RTA")
        runtime = cupy().cuda.runtime
        devices_used = ", ".join(
            f"GPU {dev} ({runtime.getDeviceProperties(dev)['name'].decode()})"
            for dev in used
        )
        output = capsys.readouterr().out
        assert output.count(f"phono3py-gpu: {method} scattering: {devices_used}\n") == 1
        stage = (
            "global solve"
            if lbte
            else ("mode contributions" if wte else "conductivity sum")
        )
        assert output.count(f"phono3py-gpu: {method} {stage}: GPU 0 (") == 1
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


def test_primary_gpu_reuses_solver(si, monkeypatch):
    import phono3py_gpu.multigpu as multigpu

    if len(devices()) < 2:
        pytest.skip("Two CUDA devices required.")
    launched = []
    popen = multigpu.subprocess.Popen

    def launch(args, **kwargs):
        launched.append(int(args[-2]))
        return popen(args, **kwargs)

    monkeypatch.setattr(multigpu.subprocess, "Popen", launch)
    ph = deepcopy(si)
    ph.mesh_numbers = [4] * 3
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(
        temperatures=[50.0],
        transport_type="WTE-GPU",
        is_isotope=True,
    )
    calc = ph.thermal_conductivity
    engine = calc._scattering_solver.engine
    assert sorted(launched) == calc.multi_gpu_stats["devices"][1:]
    assert calc._isotope_solver._device_phonons[1] is engine.eigenvectors
    assert calc._scattering_solver.q_batch_stats is calc.q_batch_stats


def test_worker_task_failure(si, monkeypatch):
    import phono3py_gpu.multigpu as multigpu

    if len(devices()) < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", "1")

    write_snapshot = multigpu._write_snapshot

    def invalid_sigma(payload, stream):
        payload["options"]["sigmas"] = ["invalid"]
        write_snapshot(payload, stream)

    monkeypatch.setattr(multigpu, "_write_snapshot", invalid_sigma)
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.init_phph_interaction()
    with pytest.raises(RuntimeError, match="worker failed"):
        ph.run_thermal_conductivity(temperatures=[300.0], transport_type="GPU")


def test_partial_lbte_keeps_last_requested_row(si, monkeypatch):
    from threading import Lock

    import phono3py_gpu.multigpu as multigpu

    if len(devices()) < 2:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", "1")
    monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", "1")
    expected = []
    for ids in ([0], [0, 1]):
        monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: ids)
        # Force result delivery in reverse request order, independent of timing.
        original_queue = multigpu.queue.Queue

        class ReversedResults(original_queue):
            def __init__(self):
                super().__init__()
                self.pending = []
                self.pending_lock = Lock()

            def put(self, item, *args, **kwargs):
                if isinstance(item[0], str) and item[0] == "result":
                    with self.pending_lock:
                        self.pending.append(item)
                        if len(self.pending) == 2:
                            for result in sorted(
                                self.pending, key=lambda r: r[2][0][0][0], reverse=True
                            ):
                                super().put(result, *args, **kwargs)
                else:
                    super().put(item, *args, **kwargs)

        with monkeypatch.context() as patch:
            patch.setattr(multigpu.queue, "Queue", ReversedResults)
            ph = deepcopy(si)
            ph.mesh_numbers = [3] * 3
            ph.init_phph_interaction()
            ph.run_thermal_conductivity(
                temperatures=[300.0], transport_type="GPU", is_LBTE=True,
                grid_points=[1, 4], pinv_solver=3,
            )
        calc = ph.thermal_conductivity
        expected.append((calc.collision_matrix.copy(), calc.gamma.copy(),
                         ph.phph_interaction.interaction_strength.copy()))
    for actual, reference in zip(expected[1], expected[0], strict=True):
        np.testing.assert_allclose(actual, reference, rtol=1e-9, atol=1e-13)


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
def test_distributed_lbte(si, monkeypatch, capsys, wte, reducible, pinv, method):
    if cupy().cuda.runtime.getDeviceCount() < 2:
        pytest.skip("Two CUDA devices required.")
    results = []
    for mode in ("single", "mg"):
        monkeypatch.setenv("PHONO3PY_GPU_LBTE_SOLVER", mode)
        ph = deepcopy(si)
        ph._log_level = 1
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
        lines = [
            line
            for line in capsys.readouterr().out.splitlines()
            if "global solve:" in line
        ]
        assert len(lines) == 1
        assert ("(cuSOLVERMg)" in lines[0]) == (mode == "mg")
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
