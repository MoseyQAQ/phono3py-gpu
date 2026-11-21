"""多 σ 复用保持各自掩码、部分分支顺序和受显存限制的分组结果。"""

from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.interaction import InteractionKernel
from phono3py_gpu.scattering import interaction_batches, sigma_groups


@pytest.mark.parametrize("streamed", [False, True])
@pytest.mark.parametrize("cutoff", [None, 3.0])
def test_shared_projection_and_sigma_masks(si, streamed, cutoff, monkeypatch):
    cp = cupy()
    ph = deepcopy(si)
    ph.band_indices = [[4, 0, 2]]
    ph.mesh_numbers = [4] * 3
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    engine.streamed, engine.band_block, engine.batch_size = streamed, 2, 3
    sigmas = [None, 0.1, 0.2]
    method = "project_slice" if streamed else "compute"
    original = getattr(engine, method)
    calls = []

    def project(*args, **kwargs):
        calls.append(args[-1] is None)
        return original(*args, **kwargs)

    monkeypatch.setattr(engine, method, project)
    actual = {}
    for j, start, _, block, g, zero, strength in interaction_batches(
        engine, sigmas, cutoff
    ):
        actual[j, start, block.start] = [host(x) for x in (g, zero, strength)]
    shared_calls = len(calls)
    if streamed and cutoff is None:
        assert all(calls)  # An unbounded Gaussian makes the union support dense.
    calls.clear()
    for j, sigma in enumerate(sigmas):
        for _, start, _, block, g, zero, strength in interaction_batches(
            engine, [sigma], cutoff
        ):
            expected = actual[j, start, block.start]
            np.testing.assert_allclose(host(g), expected[0], rtol=1e-10, atol=1e-11)
            np.testing.assert_array_equal(host(zero), expected[1])
            np.testing.assert_allclose(
                host(strength), expected[2], rtol=1e-9, atol=1e-25
            )
    assert len(calls) == len(sigmas) * shared_calls


@pytest.mark.parametrize("q_batch", [1, 4])
@pytest.mark.parametrize("limit", [1, 2])
def test_bounded_sigma_collision_rows(si, q_batch, limit, monkeypatch):
    cp = cupy()
    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", str(q_batch))
    monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", "1")
    groups = []

    def bounded(engine, sigmas, row_bytes=0, reuse=True):
        engine.fourier
        per_sigma = (
            25 * engine.batch_size * len(engine.pp.band_indices) * engine.nb**2
            + row_bytes
        )
        with patch.object(
            cp.cuda.runtime, "memGetInfo", return_value=(4 * limit * per_sigma, 2**34)
        ):
            selected = list(sigma_groups(engine, sigmas, row_bytes, reuse))
        groups.extend(len(group) for group in selected)
        return iter(selected)

    answers = []
    for bounded_memory in (False, True):
        if bounded_memory:
            monkeypatch.setattr("phono3py_gpu.scattering.sigma_groups", bounded)
            monkeypatch.setattr("phono3py_gpu.batching.sigma_groups", bounded)
        ph = deepcopy(si)
        ph.mesh_numbers = [3] * 3
        ph.sigmas = [None, 0.1, 0.2]
        ph.sigma_cutoff = 3.0
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            transport_type="GPU",
            is_LBTE=True,
            pinv_solver=3,
            temperatures=[0.0, 100.0, 300.0],
        )
        calc, pp = ph.thermal_conductivity, ph.phph_interaction
        answers.append(
            [
                np.array(x)
                for x in (
                    calc.gamma,
                    calc.kappa,
                    calc.collision_eigenvalues,
                    pp.interaction_strength,
                    pp.zero_value_positions,
                )
            ]
        )
    assert max(groups) == limit
    for expected, actual in zip(*answers):
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-12)
