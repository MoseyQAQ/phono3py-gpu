"""Numerical and file contracts outside the default conductivity path."""

from copy import deepcopy

import h5py
import numpy as np
import pytest


def run(si, gpu, **kwargs):
    ph = deepcopy(si)
    ph.mesh_numbers = [3, 3, 3]
    ph.sigmas = [None, 0.2]
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(
        temperatures=[100.0, 300.0], transport_type="GPU" if gpu else None, **kwargs
    )
    return ph.thermal_conductivity


@pytest.mark.parametrize(
    "options",
    [
        dict(is_LBTE=True, is_reducible_collision_matrix=True, pinv_solver=3),
        dict(is_LBTE=True, is_kappa_star=False, pinv_solver=4),
        dict(is_LBTE=True, pinv_solver=5, pinv_method=1, pinv_cutoff=1e-5),
        dict(is_LBTE=True, pinv_solver=7),
        dict(is_LBTE=False, is_kappa_star=False),
        dict(is_LBTE=False, is_full_pp=True),
        dict(is_LBTE=False, gv_delta_q=1e-5),
    ],
)
def test_options(si, options):
    cpu, gpu = [run(si, use_gpu, **options) for use_gpu in (False, True)]
    np.testing.assert_allclose(gpu.gamma, cpu.gamma, rtol=1e-8, atol=1e-11)
    np.testing.assert_allclose(gpu.kappa, cpu.kappa, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize(
    "lbte,wte", [(False, False), (True, False), (False, True), (True, True)]
)
def test_hdf5(si, lbte, wte, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ph = deepcopy(si)
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    options = dict(
        temperatures=[100.0, 300.0],
        is_LBTE=lbte,
        transport_type="WTE-GPU" if wte else "GPU",
        pinv_solver=3,
    )
    ph.run_thermal_conductivity(
        **options, write_kappa=True, write_collision=lbte, write_LBTE_solution=False
    )
    kappa = ph.thermal_conductivity.kappa.copy()
    file = next(tmp_path.glob("kappa*.hdf5"))
    with h5py.File(file) as handle:
        assert handle["kappa"].shape == kappa[0].shape
        np.testing.assert_allclose(handle["kappa"][:], kappa[0])
        assert "gamma" in handle
        if wte:
            assert "kappa_C" in handle
    if lbte:
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(**options, read_collision="all")
    else:
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(**options, read_gamma=True)
    np.testing.assert_allclose(
        ph.thermal_conductivity.kappa, kappa, rtol=1e-9, atol=1e-9
    )


def test_pp_roundtrip(si, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ph = deepcopy(si)
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    options = dict(temperatures=[300.0], transport_type="GPU", grid_points=[1])
    ph.run_thermal_conductivity(**options, write_pp=True)
    gamma = ph.thermal_conductivity.gamma.copy()
    ph.init_phph_interaction()
    ph.run_thermal_conductivity(**options, read_pp=True)
    np.testing.assert_allclose(
        ph.thermal_conductivity.gamma, gamma, rtol=1e-12, atol=1e-12
    )


def test_constant_interaction_and_partial_bands(si):
    results = []
    for gpu in (False, True):
        ph = deepcopy(si)
        ph.band_indices = [[0, 1, 4]]
        ph.mesh_numbers = [3, 3, 3]
        ph.sigmas = [None, 0.1]
        ph.init_phph_interaction(constant_averaged_interaction=1e-6)
        ph.run_thermal_conductivity(
            temperatures=[0.0, 300.0],
            transport_type="GPU" if gpu else None,
            grid_points=[1],
        )
        results.append(ph.thermal_conductivity)
    np.testing.assert_allclose(
        results[1].gamma, results[0].gamma, rtol=1e-9, atol=1e-12
    )


def test_gamma_detail(si, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    arrays = []
    for gpu in (False, True):
        run(
            si,
            gpu,
            grid_points=[1],
            write_gamma_detail=True,
            output_filename="gpu" if gpu else "cpu",
        )
        path = next(
            tmp_path.glob("gamma_detail*.gpu.hdf5" if gpu else "gamma_detail*.cpu.hdf5")
        )
        with h5py.File(path) as handle:
            arrays.append(handle["gamma_detail"][:])
    np.testing.assert_allclose(arrays[1], arrays[0], rtol=1e-9, atol=1e-12)
