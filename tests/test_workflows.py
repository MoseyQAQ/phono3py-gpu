"""End-to-end calculator parity against upstream CPU solvers."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from phono3py.conductivity.factory import register_variant
from phono3py.conductivity.grid_point_data import VelocityResult
from reference_kappa import WignerLBTEKappaSolver, WignerRTAKappaSolver

from phono3py_gpu.multigpu import devices


def register_cpu_wte():
    # WTE 6463b23 predates the bulk velocity protocol. Adapt only its call shape.
    path = Path(__file__).parent / "reference_velocity.py"
    spec = importlib.util.spec_from_file_location("wte_reference_operator", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from phono3py.conductivity.velocity_solvers import GroupVelocitySolver
    from phonopy.phonon.degeneracy import degenerate_sets
    from phonopy.phonon.grid import get_qpoints_from_bz_grid_points

    class CPUVelocity(GroupVelocitySolver):
        produces_gv_by_gv = False
        produces_vm_by_vm = True

        def compute(self, grid_points):
            obj = module.VelocityOperator(
                self._pp.dynamical_matrix,
                frequency_factor_to_THz=self._pp.frequency_factor_to_THz,
                q_length=self._gv_delta_q,
                symmetry=self._pp.primitive_symmetry,
            )
            obj.run(get_qpoints_from_bz_grid_points(grid_points, self._pp.bz_grid))
            gv = np.einsum("qiij->qij", obj.velocity_operators).real.copy()
            from phono3py.conductivity.velocity_solvers import VelocityMatrixSolver

            products = []
            for i, gp in enumerate(grid_points):
                op = obj.velocity_operators[i]
                for deg in degenerate_sets(self._pp.phonons.frequencies[gp]):
                    if len(deg) > 1:
                        for d in range(3):
                            gv[i, deg, d] = np.linalg.eigvalsh(
                                op[np.ix_(deg, deg, [d])][:, :, 0]
                            )
                products.append(
                    VelocityMatrixSolver._get_vm_by_vm(
                        self, int(gp), op.transpose(2, 0, 1)
                    )
                )
            return VelocityResult(group_velocities=gv, vm_by_vm=np.asarray(products))

    register_variant(
        "reference-wte",
        make_velocity_solver=lambda ctx: CPUVelocity(
            ctx.interaction,
            is_kappa_star=ctx.kappa_settings.is_kappa_star,
            gv_delta_q=ctx.kappa_settings.gv_delta_q,
        ),
        make_rta_kappa_solver=lambda ctx: WignerRTAKappaSolver(
            ctx.kappa_settings,
            ctx.interaction.phonons.frequencies,
            ctx.interaction.primitive.volume,
        ),
        make_lbte_kappa_solver=lambda ctx: WignerLBTEKappaSolver(
            ctx.collision_matrix_kernel,
            ctx.kappa_settings,
            ctx.interaction.phonons.frequencies,
            ctx.interaction.primitive.volume,
        ),
    )


@pytest.mark.parametrize(
    "lbte,wte", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("sigmas", [[None], [0.1], [None, 0.1, 0.2]])
@pytest.mark.parametrize("q_batch,streamed", [(1, False), (4, False), (4, True)])
def test_conductivity(si, lbte, wte, sigmas, q_batch, streamed, monkeypatch):
    if streamed:
        # Exercise multiple initial-band tiles on the small CPU reference.
        monkeypatch.setattr("phono3py_gpu.interaction.MIN_STREAMED_BANDS", 1)
        monkeypatch.setattr("phono3py_gpu.interaction.initial_band_block", lambda *_: 2)
        monkeypatch.setattr("phono3py_gpu.factory.devices", lambda: [0])

    monkeypatch.setenv("PHONO3PY_GPU_Q_BATCH_SIZE", str(q_batch))
    if wte:
        register_cpu_wte()
    results = []
    for gpu in (False, True):
        ph = deepcopy(si)
        ph.mesh_numbers = [3, 3, 3]
        ph.sigmas = sigmas
        ph.sigma_cutoff = 3.0
        ph.init_phph_interaction()
        transport = (
            ("WTE-GPU" if wte else "GPU") if gpu else ("reference-wte" if wte else None)
        )
        ph.run_thermal_conductivity(
            is_LBTE=lbte,
            temperatures=[0.0, 100.0, 300.0],
            transport_type=transport,
            pinv_solver=3,
            is_N_U=not lbte,
            is_isotope=True,
            boundary_mfp=100.0,
        )
        results.append(ph.thermal_conductivity)
    cpu, gpu = results
    visible = devices()
    groups = (len(gpu.grid_points) + q_batch - 1) // q_batch
    active = visible[:groups] if not streamed else visible[:1]
    if len(active) > 1:
        assert gpu.multi_gpu_stats["devices"] == active
    else:
        assert gpu.multi_gpu_stats is None
    if streamed:
        assert gpu.q_batch_stats["max_q_per_launch"] == 1
    elif q_batch > 1:
        assert gpu.q_batch_stats["max_q_per_launch"] > 1
    np.testing.assert_allclose(gpu.gamma, cpu.gamma, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(
        gpu.gamma_isotope, cpu.gamma_isotope, rtol=1e-7, atol=1e-11
    )
    np.testing.assert_allclose(gpu.kappa, cpu.kappa, rtol=1e-6, atol=1e-6)
    if wte:
        for name in ("kappa_P_RTA", "kappa_C"):
            np.testing.assert_allclose(
                getattr(gpu, name), getattr(cpu, name), rtol=1e-6, atol=1e-6
            )
        if lbte:
            np.testing.assert_allclose(
                gpu.kappa_P_exact, cpu.kappa_P_exact, rtol=1e-6, atol=1e-6
            )
    if lbte:
        np.testing.assert_allclose(
            gpu.collision_eigenvalues, cpu.collision_eigenvalues, rtol=1e-7, atol=1e-9
        )
    else:
        for actual, expected in zip(
            gpu.get_gamma_N_U(), cpu.get_gamma_N_U(), strict=True
        ):
            np.testing.assert_allclose(actual, expected, rtol=1e-7, atol=1e-11)


def test_wte_nonzero_coherence(tmp_path, monkeypatch):
    """60-band perovskite with supplied linewidths exercises a finite C term."""
    import shutil

    import phono3py

    data = Path(__file__).parent / "data"
    shutil.copy(data / "kappa-NaMgF3-m979.hdf5", tmp_path / "kappa-m979.hdf5")
    monkeypatch.chdir(tmp_path)
    register_cpu_wte()
    results = []
    for transport in ["reference-wte", "WTE-GPU"]:
        ph = phono3py.load(
            data / "phono3py_params_NaMgF3.yaml.xz",
            fc2_filename=data / "fc2_NaMgF3.hdf5",
            lang="C",
        )
        ph.mesh_numbers = [9, 7, 9]
        ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            temperatures=[300.0], read_gamma=True, transport_type=transport
        )
        results.append(ph.thermal_conductivity)
    cpu, gpu = results
    assert np.max(cpu.kappa_C) > 0.01
    np.testing.assert_allclose(gpu.kappa, cpu.kappa, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(gpu.kappa_C, cpu.kappa_C, rtol=2e-6, atol=1e-6)
    np.testing.assert_allclose(gpu.kappa_P_RTA, cpu.kappa_P_RTA, rtol=2e-6, atol=1e-6)


@pytest.mark.parametrize(
    "structure,mesh",
    [
        ("si", [3, 3, 3]),
        ("si_full", [3, 3, 3]),
        ("nacl", [3, 3, 3]),
        ("aln", [3, 3, 2]),
    ],
)
@pytest.mark.parametrize(
    "lbte,wte", [(False, False), (True, False), (False, True), (True, True)]
)
def test_upstream_materials(request, structure, mesh, lbte, wte, monkeypatch):
    """Default Rust backend, NAC and anisotropic cells against original CPU."""
    from phono3py.phonon3.interaction import Interaction

    from phono3py_gpu.initialization import reference_all_shortest

    register_cpu_wte()
    base = request.getfixturevalue(structure)
    results, frequencies = [], []
    for gpu in (False, True):
        ph = deepcopy(base)
        ph._lang = "Rust"
        ph.mesh_numbers = mesh
        with monkeypatch.context() as context:
            if not gpu:
                context.setattr(
                    Interaction, "_get_all_shortest", reference_all_shortest
                )
            ph.init_phph_interaction()
        ph.run_thermal_conductivity(
            temperatures=[100.0, 300.0],
            is_LBTE=lbte,
            transport_type=("WTE-GPU" if wte else "GPU")
            if gpu
            else ("reference-wte" if wte else None),
            pinv_solver=3,
            is_isotope=True,
        )
        calc = ph.thermal_conductivity
        results.append(calc)
        frequencies.append(
            ph.phph_interaction.phonons.frequencies[calc.grid_points].copy()
        )
    cpu, gpu = results
    np.testing.assert_allclose(frequencies[1], frequencies[0], rtol=1e-10, atol=1e-8)
    np.testing.assert_allclose(gpu.gamma, cpu.gamma, rtol=1e-7, atol=1e-11)
    np.testing.assert_allclose(
        gpu.gamma_isotope, cpu.gamma_isotope, rtol=1e-7, atol=1e-11
    )
    np.testing.assert_allclose(gpu.kappa, cpu.kappa, rtol=1e-6, atol=1e-6)
    if wte:
        for name in ("kappa_P_RTA", "kappa_C"):
            np.testing.assert_allclose(
                getattr(gpu, name), getattr(cpu, name), rtol=1e-6, atol=1e-6
            )
        if lbte:
            np.testing.assert_allclose(
                gpu.kappa_P_exact, cpu.kappa_P_exact, rtol=1e-6, atol=1e-6
            )
    if lbte:
        np.testing.assert_allclose(
            gpu.collision_eigenvalues, cpu.collision_eigenvalues, rtol=1e-7, atol=1e-9
        )
