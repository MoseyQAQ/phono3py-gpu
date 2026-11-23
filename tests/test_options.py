"""Numerical and file contracts outside the default conductivity path."""

from copy import deepcopy
from dataclasses import replace

import h5py
import numpy as np
import pytest
from phono3py.conductivity.build_components import build_rot_grid_points
from phono3py.file_IO import read_pp_from_hdf5

from phono3py_gpu.collision import create_collision_matrix_kernel


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
@pytest.mark.filterwarnings(r"error:.*\.extra is deprecated:DeprecationWarning")
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
    calc = ph.thermal_conductivity
    expected = {"kappa": calc.kappa.copy()}
    if wte:
        for name in ("kappa_P_RTA", "kappa_C"):
            expected[name] = getattr(calc, name).copy()
        if lbte:
            expected["kappa_P_exact"] = calc.kappa_P_exact.copy()
        velocity_operator = calc._velocity_solver.velocity_operator.copy()
    file = next(tmp_path.glob("kappa*.hdf5"))
    with h5py.File(file) as handle:
        assert handle["kappa"].shape == expected["kappa"][0].shape
        for name, values in expected.items():
            np.testing.assert_allclose(handle[name][:], values[0])
        assert "gamma" in handle
        if wte:
            np.testing.assert_array_equal(
                handle["velocity_operator"][:], velocity_operator
            )
    if lbte:
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(**options, read_collision="all")
    else:
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(**options, read_gamma=True)
    for name, values in expected.items():
        np.testing.assert_allclose(
            getattr(ph.thermal_conductivity, name), values, rtol=1e-9, atol=1e-9
        )
    if wte:
        np.testing.assert_allclose(
            ph.thermal_conductivity.velocity_operator,
            velocity_operator,
            rtol=1e-9,
            atol=1e-9,
        )


@pytest.mark.parametrize("include_gamma", [True, False])
def test_mode_kappa_gamma_selection(si, include_gamma):
    """Only the acoustic modes at Gamma are skipped, even after reordering."""
    cpu = run(si, False, is_LBTE=True)
    settings = cpu._kappa_settings
    points = settings.grid_points[::-1].copy()
    assert points[-1] == settings.bz_grid.gp_Gamma
    if not include_gamma:
        points = points[:-1]
    settings = replace(settings, grid_points=points)
    phonons = cpu._pp.phonons
    kernel = create_collision_matrix_kernel(
        kappa_settings=settings,
        frequencies=phonons.frequencies,
        degenerate_ids=phonons.degenerate_ids,
        rot_grid_points=build_rot_grid_points(settings),
    )
    num_gp, num_band = len(points), phonons.frequencies.shape[1]
    x = np.ones((num_gp * num_band, 3))
    mode_kappa = np.zeros((1, 1, num_gp, num_band, 6))
    kernel._set_mode_kappa(mode_kappa, x, x, num_gp, 0, 0)
    diagonal = mode_kappa[0, 0, :, :, :3]
    if include_gamma:
        assert np.all(diagonal[-1, :3] == 0)
        assert np.all(diagonal[-1, 3:] > 0)
        assert np.all(diagonal[:-1] > 0)
    else:
        assert np.all(diagonal > 0)


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


@pytest.mark.parametrize("transport", ["GPU", "WTE-GPU"])
def test_write_lbte_pp_without_retention(si, tmp_path, monkeypatch, capsys, transport):
    monkeypatch.chdir(tmp_path)
    ph = deepcopy(si)
    ph._log_level = 1
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    for retain in ("1", "0", "default"):
        if retain == "default":
            monkeypatch.delenv("PHONO3PY_GPU_RETAIN_PP", raising=False)
        else:
            monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", retain)
        ph.run_thermal_conductivity(
            temperatures=[300.0],
            transport_type=transport,
            is_LBTE=True,
            pinv_solver=3,
            write_pp=True,
            output_filename=retain,
        )
        from phono3py_gpu.multigpu import devices

        output = capsys.readouterr().out
        if len(devices()) > 1:
            assert (
                f"of {len(devices())} visible GPUs (per-q output or callback)" in output
            )
        pp = ph.phph_interaction
        assert (pp.interaction_strength is not None) == (retain == "1")
        assert (pp.zero_value_positions is not None) == (retain == "1")

    for gp in ph.thermal_conductivity.grid_points:
        ref, ref_zero = read_pp_from_hdf5(ph.mesh_numbers, gp, filename="1")
        for mode in ("0", "default"):
            strength, zero = read_pp_from_hdf5(ph.mesh_numbers, gp, filename=mode)
            np.testing.assert_array_equal(strength, ref)
            np.testing.assert_array_equal(zero, ref_zero)

    # The last requested point can differ from the last point of the full grid.
    requested = ph.thermal_conductivity.grid_points[[-1, 1]]
    monkeypatch.setenv("PHONO3PY_GPU_RETAIN_PP", "1")
    ph.run_thermal_conductivity(
        temperatures=[300.0],
        transport_type=transport,
        is_LBTE=True,
        grid_points=requested,
        pinv_solver=3,
    )
    ref, ref_zero = read_pp_from_hdf5(ph.mesh_numbers, requested[-1], filename="1")
    np.testing.assert_allclose(pp.interaction_strength, ref, rtol=1e-9, atol=1e-25)
    np.testing.assert_array_equal(pp.zero_value_positions, ref_zero)


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
