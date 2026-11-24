"""Resident LBTE assembly and solver output contracts against upstream assembly."""

from copy import deepcopy

import numpy as np
import pytest
from phono3py.conductivity.collision_matrix_kernel import (
    IrreducibleCollisionMatrixKernel,
    ReducibleCollisionMatrixKernel,
)

from phono3py_gpu import collision
from phono3py_gpu.backend import cupy, host


@pytest.mark.parametrize("reducible", [False, True])
@pytest.mark.parametrize(
    "solver,inplace,star",
    [(i, False, True) for i in range(1, 8)] + [(3, True, True), (3, False, False)],
)
def test_resident_collision(si, monkeypatch, reducible, solver, inplace, star):
    """Check all steps, including zero T, against independent upstream preparation."""
    if inplace:
        monkeypatch.setattr(collision, "INPLACE_EIGH_BYTES", 0)
    expected = []
    visited = []
    prepare = collision.GPUCollisionMixin._prepare_collision_matrix
    device_matrix = collision.GPUCollisionMixin._device_collision_matrix

    def reference_prepare(self):
        reference = deepcopy(self)
        reference.__class__ = (
            ReducibleCollisionMatrixKernel
            if reducible
            else IrreducibleCollisionMatrixKernel
        )
        reference._prepare_collision_matrix()
        expected.append(reference.collision_matrix)
        return prepare(self)

    def check_device_matrix(self, s, t, weights):
        result = device_matrix(self, s, t, weights)
        n = len(result)
        np.testing.assert_allclose(
            host(result), expected[0][s, t].reshape(n, n), rtol=1e-10, atol=1e-13
        )
        assert result.flags.f_contiguous if inplace else result.flags.c_contiguous
        visited.append((s, t))
        return result

    monkeypatch.setattr(
        collision.GPUCollisionMixin, "_prepare_collision_matrix", reference_prepare
    )
    monkeypatch.setattr(
        collision.GPUCollisionMixin, "_device_collision_matrix", check_device_matrix
    )
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    ph.sigmas = [None, 0.2]
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(
        transport_type="GPU",
        is_LBTE=True,
        pinv_solver=solver,
        is_reducible_collision_matrix=reducible,
        is_kappa_star=star,
        temperatures=[0.0, 100.0, 300.0],
        is_isotope=True,
        boundary_mfp=100.0,
    )
    result = ph.thermal_conductivity
    assert visited == list(np.ndindex(2, 3))
    assert np.isfinite(result.kappa).all()
    assert np.isfinite(result.kappa_RTA).all()
    for s, t in visited:
        n = int(np.sqrt(expected[0][s, t].size))
        matrix = expected[0][s, t].reshape(n, n)
        stored = result.collision_matrix[s, t].reshape(n, n)
        if t == 0:
            np.testing.assert_allclose(stored, matrix, rtol=1e-10, atol=1e-13)
        elif solver in (6, 7):
            if solver == 7:
                inverse = np.linalg.pinv(matrix, rcond=1e-15)
            else:
                values, vectors = np.linalg.eigh(matrix)
                inverse_values = np.zeros_like(values)
                valid = np.abs(values) > 1e-8
                inverse_values[valid] = 1 / values[valid]
                inverse = (vectors * inverse_values) @ vectors.T
            np.testing.assert_allclose(stored, inverse, rtol=1e-7, atol=1e-7)
        else:
            vectors = stored.T if solver in (1, 2, 4, 5) else stored
            values = result.collision_eigenvalues[s, t]
            np.testing.assert_allclose(vectors.T @ vectors, np.eye(n), atol=1e-12)
            np.testing.assert_allclose(matrix @ vectors, vectors * values, atol=1e-12)


@pytest.mark.parametrize("order", ["C", "F"])
def test_device_eigh(order):
    cp = cupy()
    a = np.random.default_rng(7).normal(size=(259, 259))
    matrix = a + a.T
    device = cp.array(matrix, order=order)
    values, vectors = collision.eigh_collision(device)
    if order == "F":
        assert vectors.data.ptr == device.data.ptr
    else:
        np.testing.assert_array_equal(host(device), matrix)
    np.testing.assert_allclose(
        matrix @ host(vectors), host(vectors * values), atol=1e-12
    )


@pytest.mark.parametrize("order", ["C", "F"])
def test_collision_symmetrization(order):
    from phono3py_gpu.backend import kernel

    cp = cupy()
    matrix = np.random.default_rng(3).normal(size=(259, 259))
    device = cp.array(matrix, order=order)
    kernel("collision.cu", "symmetrize_collision")(
        ((device.size + 255) // 256,), (256,), (device, np.int32(len(matrix)))
    )
    np.testing.assert_array_equal(host(device), (matrix + matrix.T) / 2)
