"""Cross-q weights, collision rows and partial-band contracts."""

from copy import deepcopy

import numpy as np
import pytest
from phono3py.phonon3.triplets import get_triplets_integration_weights

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.batching import compute_q_batch
from phono3py_gpu.integration import weights
from phono3py_gpu.interaction import InteractionKernel
from phono3py_gpu.scattering import GPUCollisionSolver


@pytest.mark.parametrize("sigma", [None, 0.1])
def test_mixed_frequency_points(si, sigma):
    cp = cupy()
    ph = deepcopy(si)
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction(
        symmetrize_tetrahedra=True, average_degenerate_weights=True
    )
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    frequencies = pp.phonons.frequencies
    triplets, expected, expected_zero = [], [], []
    for gp in [1, 0, 3]:
        pp.set_grid_point(gp)
        g, zero = get_triplets_integration_weights(
            pp,
            frequencies[gp, [5, 1]],
            sigma,
            sigma_cutoff=3.0,
            is_collision_matrix=True,
            average_degenerate_weights=True,
        )
        triplets.append(pp.get_triplets_at_q()[0])
        expected.append(g)
        expected_zero.append(zero)
    triplets = np.concatenate(triplets)
    g, zero = weights(
        pp,
        triplets,
        cp.asarray(frequencies),
        frequencies[triplets[:, 0]][:, [5, 1]],
        sigma,
        3.0,
        collision=True,
    )
    np.testing.assert_allclose(
        host(g), np.concatenate(expected, axis=1), rtol=1e-10, atol=1e-11
    )
    np.testing.assert_array_equal(host(zero), np.concatenate(expected_zero))


@pytest.mark.parametrize("reducible", [False, True])
def test_interleaved_collision_rows(si, reducible):
    from phono3py.phonon3.collision_matrix import CollisionMatrix
    from phonopy.phonon.grid import get_grid_points_by_rotations

    cp = cupy()
    ph = deepcopy(si)
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    targets = (
        None
        if reducible
        else np.array(
            [
                get_grid_points_by_rotations(int(gp), pp.bz_grid, with_surface=False)
                for gp in pp.bz_grid.grg2bzg
            ]
        )
    )
    collision = CollisionMatrix(pp, rot_grid_points=targets, lang="C")
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    engine.batch_size = 5  # Splits inside q lists and leaves a partial final batch.
    solver = GPUCollisionSolver(
        pp,
        collision,
        engine=engine,
        sigmas=[None, 0.1],
        sigma_cutoff=3.0,
        temperatures=np.array([0.0, 100.0, 300.0]),
    )
    gps = [1, 0, 3]
    expected = [solver.compute(gp) for gp in gps]
    strength = pp.interaction_strength.copy()
    last_triplets = pp.get_triplets_at_q()[0].copy()
    solver.q_batch_stats = dict(triplet_batches=0, max_q_per_launch=0)
    pp.delete_interaction_strength()
    compute_q_batch(solver, gps[:1], collision=True, last_batch=False)
    assert pp.interaction_strength is None
    assert pp.zero_value_positions is None
    results = compute_q_batch(solver, gps, collision=True)
    assert solver.q_batch_stats["max_q_per_launch"] == 3
    for actual, reference in zip(results, expected, strict=True):
        np.testing.assert_allclose(actual.gamma, reference.gamma, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(
            actual.collision_row, reference.collision_row, rtol=1e-9, atol=1e-13
        )
    np.testing.assert_allclose(pp.interaction_strength, strength, rtol=1e-9, atol=1e-25)
    np.testing.assert_array_equal(pp.get_triplets_at_q()[0], last_triplets)


@pytest.mark.parametrize(
    "gpu_count,q_batch,transport", [(1, 1, "GPU"), (1, 2, "WTE-GPU"), (2, 2, "GPU")]
)
def test_lbte_without_retained_pp(si, monkeypatch, gpu_count, q_batch, transport):
    if cupy().cuda.runtime.getDeviceCount() < gpu_count:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: list(range(gpu_count)))
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", str(q_batch))
    monkeypatch.delenv("PHONO3PY_GPU_RETAIN_PP", raising=False)
    ph = deepcopy(si)
    ph.mesh_numbers = [3, 3, 3]
    ph.sigmas = [None, 0.1]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    saved_at = []
    set_strength = type(pp).set_interaction_strength

    def record_snapshot(interaction, strength, g_zero=None):
        if strength is not None:
            saved_at.append(interaction.get_triplets_at_q()[0][0, 0])
        set_strength(interaction, strength, g_zero=g_zero)

    monkeypatch.setattr(type(pp), "set_interaction_strength", record_snapshot)
    options = dict(
        temperatures=[100.0, 300.0],
        transport_type=transport,
        is_LBTE=True,
        pinv_solver=3,
    )
    ph.run_thermal_conductivity(**options)
    ref = ph.thermal_conductivity
    assert saved_at == [ref.grid_points[-1]]
    assert pp.interaction_strength is not None
    assert pp.zero_value_positions is not None

    monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", "0")
    saved_at.clear()
    ph.run_thermal_conductivity(**options)
    result = ph.thermal_conductivity
    assert saved_at == []
    assert pp.interaction_strength is None
    assert pp.zero_value_positions is None
    np.testing.assert_allclose(result.kappa, ref.kappa, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(result.gamma, ref.gamma, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(
        result.collision_eigenvalues, ref.collision_eigenvalues, rtol=1e-10, atol=1e-12
    )


@pytest.mark.parametrize("gpu_count", [1, 2])
def test_partial_bands_and_callbacks(si, monkeypatch, gpu_count):
    from phono3py_gpu.factory import GPUCalculatorMixin

    if cupy().cuda.runtime.getDeviceCount() < gpu_count:
        pytest.skip("Two CUDA devices required.")
    monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: list(range(gpu_count)))
    original = GPUCalculatorMixin._iterate_grid_points
    visited = []

    def iterate(self, callback):
        def check(index):
            assert self._pp.get_triplets_at_q()[0][0, 0] == self.grid_points[index]
            assert self.grid_point_count == len(visited) + 1
            visited.append(index)
            if callback is not None:
                callback(index)

        original(self, check)

    monkeypatch.setattr(GPUCalculatorMixin, "_iterate_grid_points", iterate)
    results = []
    for size in (1, 4):
        visited.clear()
        monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", str(size))
        ph = deepcopy(si)
        ph.band_indices = [[4, 0, 2]]
        ph.mesh_numbers = [4, 4, 4]
        ph.sigmas = [None, 0.1]
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            temperatures=[0.0, 100.0, 300.0],
            transport_type="GPU",
            is_N_U=True,
        )
        results.append(ph.thermal_conductivity)
        assert len(visited) == len(ph.thermal_conductivity.grid_points)
    assert results[1].q_batch_stats["max_q_per_launch"] == 4
    np.testing.assert_allclose(
        results[1].gamma, results[0].gamma, rtol=1e-9, atol=1e-13
    )
    for actual, expected in zip(
        results[1].get_gamma_N_U(), results[0].get_gamma_N_U(), strict=True
    ):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("lbte", [False, True])
def test_band_tiles_preserve_pp_files(si, tmp_path, monkeypatch, lbte):
    from phono3py.file_IO import read_pp_from_hdf5

    monkeypatch.chdir(tmp_path)
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.init_phph_interaction()
    options = dict(
        transport_type="GPU", is_LBTE=lbte, temperatures=[300.0], pinv_solver=3
    )
    ph.run_thermal_conductivity(**options, write_pp=True, output_filename="full")
    ref = ph.thermal_conductivity.gamma.copy()
    monkeypatch.setattr("phono3py_gpu.interaction.MIN_STREAMED_BANDS", 1)
    monkeypatch.setattr("phono3py_gpu.interaction.initial_band_block", lambda *_: 2)
    ph.run_thermal_conductivity(**options, write_pp=True, output_filename="tiles")
    for gp in ph.thermal_conductivity.grid_points:
        full, full_zero = read_pp_from_hdf5(ph.mesh_numbers, gp, filename="full")
        tiled, tiled_zero = read_pp_from_hdf5(ph.mesh_numbers, gp, filename="tiles")
        np.testing.assert_allclose(tiled, full, rtol=1e-9, atol=1e-25)
        np.testing.assert_array_equal(tiled_zero, full_zero)
    ph.run_thermal_conductivity(**options, read_pp=True, input_filename="tiles")
    np.testing.assert_allclose(
        ph.thermal_conductivity.gamma, ref, rtol=1e-9, atol=1e-12
    )
