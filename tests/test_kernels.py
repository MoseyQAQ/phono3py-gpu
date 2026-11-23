"""CPU/GPU comparisons before calculator integration."""

import numpy as np
import pytest
from phono3py.phonon3.triplets import get_triplets_integration_weights
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.harmonic import Harmonic
from phono3py_gpu.integration import weights
from phono3py_gpu.interaction import InteractionKernel
from phono3py_gpu.velocity import GPUVelocitySolver


@pytest.mark.parametrize("zero_acoustic", [False, True])
@pytest.mark.parametrize("unstable", [False, True])
def test_wte_square_root(zero_acoustic, unstable):
    """Check the spectral root, including degeneracy and complex negative roots."""
    cp = cupy()
    rng = np.random.default_rng(42)
    basis = np.linalg.qr(rng.normal(size=(6, 6)) + 1j * rng.normal(size=(6, 6)))[0]
    values = np.array(
        [-1, -0.25, 0.01, 2.25, 4, 4] if unstable else [0.25, 0.25, 1, 2.25, 4, 4]
    )
    dm = (basis * values) @ basis.conj().T
    roots = np.sqrt(values.astype(complex))
    if zero_acoustic:
        roots[:3] = 0
    expected = (basis * roots) @ basis.conj().T
    solver = GPUVelocitySolver.__new__(GPUVelocitySolver)
    actual = host(solver._sqrt(cp.asarray(dm), zero_acoustic))
    np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("fixture", ["si", "nacl"])
@pytest.mark.parametrize("delta_q", [5e-6, 1e-5])
def test_wte_velocity_square_root(fixture, delta_q, request, monkeypatch):
    """Keep finite-difference velocities at Gamma and finite q, including NAC."""
    from copy import deepcopy

    from phono3py_gpu.backend import eigh

    ph = deepcopy(request.getfixturevalue(fixture))
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    solver = GPUVelocitySolver(
        pp, harmonic=Harmonic(pp.dynamical_matrix), wte=True, gv_delta_q=delta_q
    )
    grid_points = [pp.bz_grid.gp_Gamma, 1]
    actual = solver.compute(grid_points)
    actual_operator = solver.velocity_operator.copy()

    def projected_sqrt(self, dm, zero_acoustic):
        cp = cupy()
        _, e = eigh(dm)
        if zero_acoustic:
            e[:, :3] = 0
        return (e * cp.sqrt(cp.diag(e.conj().T @ dm @ e))) @ e.conj().T

    monkeypatch.setattr(GPUVelocitySolver, "_sqrt", projected_sqrt)
    expected = solver.compute(grid_points)
    np.testing.assert_allclose(
        actual_operator, solver.velocity_operator, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(
        actual.group_velocities, expected.group_velocities, atol=1e-6, rtol=1e-6
    )
    np.testing.assert_allclose(actual.vm_by_vm, expected.vm_by_vm, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("fixture", ["si", "nacl"])
def test_harmonic(fixture, request):
    ph = request.getfixturevalue(fixture)
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction()
    dm = ph.phph_interaction.dynamical_matrix
    gpu = Harmonic(dm)
    qs = np.array([[0, 0, 0], [0.12, 0.2, 0.03], [0.5, 0, 0.5]])
    expected = []
    for q in qs:
        dm.run(q)
        expected.append(dm.dynamical_matrix.copy())
    np.testing.assert_allclose(host(gpu.matrices(qs)), expected, atol=1e-10, rtol=1e-10)
    derivative = DerivativeOfDynamicalMatrix(dm)
    expected = []
    for q in qs:
        derivative.run(q)
        expected.append(derivative.d_dynamical_matrix.copy())
    np.testing.assert_allclose(
        host(gpu.matrices(qs, derivative=True)), expected, atol=1e-8, rtol=1e-8
    )


@pytest.mark.parametrize("sigma", [None, 0.1])
def test_interaction(si, sigma):
    cp = cupy()
    si.mesh_numbers = [4, 4, 4]
    si.init_phph_interaction()
    pp = si.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    f, e = pp.phonons.frequencies, pp.phonons.eigenvectors
    triplets = pp.get_triplets_at_q()[0]
    g_ref, z_ref = get_triplets_integration_weights(pp, f[1], sigma, sigma_cutoff=3.0)
    g, z = weights(pp, triplets, cp.asarray(f), f[1], sigma, 3.0)
    np.testing.assert_allclose(host(g), g_ref, atol=1e-11, rtol=1e-10)
    np.testing.assert_array_equal(host(z), z_ref)
    pp.run(g_zero=z_ref)
    reference = pp.interaction_strength.copy()
    engine = InteractionKernel(pp, cp.asarray(f), cp.asarray(e))
    result = engine.compute(triplets, z)
    np.testing.assert_allclose(host(result), reference, atol=1e-25, rtol=1e-9)


@pytest.mark.parametrize("all_zero", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_masked_partial_projection(si, all_zero, streamed):
    """Check skipped amplitudes and non-contiguous band order against upstream."""
    from copy import deepcopy

    cp = cupy()
    ph = deepcopy(si)
    ph.band_indices = [[4, 0, 2]]
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    triplets = pp.get_triplets_at_q()[0]
    nb = pp.phonons.frequencies.shape[1]
    shape = (len(triplets), len(pp.band_indices), nb, nb)
    zero = np.ones(shape, dtype="byte")
    if not all_zero:
        zero = (np.random.default_rng(41).random(shape) < 0.7).astype("byte")
        zero[0] = 1
    pp.run(g_zero=zero)
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    engine.streamed = streamed
    engine.band_block = 2
    result = host(engine.compute(triplets, cp.asarray(zero)))
    np.testing.assert_allclose(result, pp.interaction_strength, rtol=1e-9, atol=1e-25)
    assert np.count_nonzero(result[zero != 0]) == 0


@pytest.mark.parametrize("support", ["empty", "sparse", "dense"])
def test_packed_second_leg(si, support, monkeypatch):
    from copy import deepcopy

    monkeypatch.setattr("phono3py_gpu.interaction.MIN_STREAMED_BANDS", 1)
    ph = deepcopy(si)
    ph.mesh_numbers = [4] * 3
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    triplets = pp.get_triplets_at_q()[0]
    nb = pp.phonons.frequencies.shape[1]
    zero = np.ones((len(triplets), nb, nb, nb), dtype="byte")
    if support == "sparse":
        zero[:, 4, [1, 5], 2] = 0
        zero[:, 2, 3, [0, 4]] = 0
    elif support == "dense":
        zero[:] = 0
    pp.run(g_zero=zero)
    cp = cupy()
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    bands = np.array([4, 0, 2], dtype="int64")
    for index in [0, len(triplets) - 1]:
        tensor = engine.fourier.components(triplets[index])
        actual = (
            engine.project_slice(
                tensor, triplets[index], bands, cp.asarray(zero[index, bands])
            )
            * pp.unit_conversion_factor
        )
        np.testing.assert_allclose(
            host(actual),
            pp.interaction_strength[index, bands],
            rtol=1e-9,
            atol=1e-25,
        )


@pytest.mark.parametrize("fixture", ["si", "si_full"])
def test_sparse_fc3_packing(fixture, request):
    """Compare nonuniform FC3 blocks and an empty anchor with upstream."""
    from copy import deepcopy

    ph = deepcopy(request.getfixturevalue(fixture))
    mask = np.random.default_rng(17).random(ph.fc3.shape[:3]) < 0.3
    mask[0] = False
    ph.fc3_nonzero_indices = mask.astype("byte")
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    pp.run()
    cp = cupy()
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    result = host(engine.compute(pp.get_triplets_at_q()[0]))
    assert np.any(pp.interaction_strength)
    np.testing.assert_allclose(result, pp.interaction_strength, atol=1e-25, rtol=1e-9)


@pytest.mark.parametrize("reducible", [False, True])
@pytest.mark.parametrize("streamed", [False, True])
def test_collision_batches(si, reducible, streamed):
    """Compare unsymmetrized rows, including swapped legs and batch boundaries."""
    from copy import deepcopy

    from phono3py.conductivity.lbte_collision_solver import LBTECollisionSolver
    from phono3py.phonon3.collision_matrix import CollisionMatrix
    from phonopy.phonon.grid import get_grid_points_by_rotations

    from phono3py_gpu.scattering import GPUCollisionSolver

    cp = cupy()
    ph = deepcopy(si)
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction()
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    targets = None
    if not reducible:
        targets = np.array(
            [
                get_grid_points_by_rotations(int(gp), pp.bz_grid, with_surface=False)
                for gp in pp.bz_grid.grg2bzg
            ]
        )
    collision = CollisionMatrix(pp, rot_grid_points=targets, lang="C")
    options = dict(
        sigmas=[None], sigma_cutoff=None, temperatures=np.array([0.0, 100.0, 300.0])
    )
    expected = LBTECollisionSolver(pp, collision, **options).compute(1)
    engine = InteractionKernel(
        pp, cp.asarray(pp.phonons.frequencies), cp.asarray(pp.phonons.eigenvectors)
    )
    engine.streamed = streamed
    engine.band_block = 2
    for size in (1, 3):
        engine.batch_size = size
        result = GPUCollisionSolver(pp, collision, engine=engine, **options).compute(1)
        np.testing.assert_allclose(result.gamma, expected.gamma, rtol=1e-9, atol=1e-13)
        np.testing.assert_allclose(
            result.collision_row, expected.collision_row, rtol=1e-9, atol=1e-13
        )


@pytest.mark.parametrize("sigma", [None, 0.1])
def test_scattering(si, sigma):
    from phono3py.conductivity.scattering_solvers import RTAScatteringSolver

    from phono3py_gpu.scattering import GPUScatteringSolver

    si.mesh_numbers = [4, 4, 4]
    si.init_phph_interaction()
    pp = si.phph_interaction
    pp.run_phonon_solver()
    engine = InteractionKernel(
        pp,
        cupy().asarray(pp.phonons.frequencies),
        cupy().asarray(pp.phonons.eigenvectors),
    )
    kwargs = dict(
        sigmas=[sigma], temperatures=np.array([0.0, 100.0, 300.0]), is_N_U=True
    )
    cpu = RTAScatteringSolver(pp, **kwargs)
    gpu = GPUScatteringSolver(pp, engine=engine, **kwargs)
    a, b = cpu.compute(1), gpu.compute(1)
    np.testing.assert_allclose(b.gamma, a.gamma, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(gpu.gamma_N, cpu.gamma_N, rtol=1e-9, atol=1e-13)
    np.testing.assert_allclose(gpu.gamma_U, cpu.gamma_U, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("sigma", [None, 0.1, 1.0])
@pytest.mark.parametrize("cutoff", [None, 5.0])
@pytest.mark.parametrize("bands", [None, [4, 0, 2], [3, 2]])
@pytest.mark.parametrize("lang", ["C", "Rust"])
def test_isotope(si, sigma, cutoff, bands, lang):
    from phono3py.conductivity.scattering_solvers import IsotopeScatteringSolver
    from phono3py.other.isotope import Isotope

    from phono3py_gpu.isotope import GPUIsotopeSolver

    si.mesh_numbers = [3, 3, 3]
    si.init_phph_interaction()
    pp = si.phph_interaction
    pp.run_phonon_solver()
    cutoff = pp.cutoff_frequency if cutoff is None else cutoff
    iso = Isotope(
        pp.mesh_numbers,
        pp.primitive,
        bz_grid=pp.bz_grid,
        cutoff_frequency=cutoff,
        band_indices=bands,
        lang=lang,
    )
    iso.set_phonons(pp.phonons, dm=pp.dynamical_matrix)
    cpu = IsotopeScatteringSolver(iso, [sigma])
    gpu = GPUIsotopeSolver(iso, [sigma])
    for gp in [1, 3, 9]:
        a, b = cpu.compute(gp), gpu.compute(gp)
        below_cutoff = pp.phonons.frequencies[gp, iso.band_indices] < cutoff
        assert np.all(b[:, below_cutoff] == 0.0)
        assert np.any(b[:, ~below_cutoff] > 0.0)
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize("symmetrize,average", [(False, False), (True, True)])
@pytest.mark.parametrize("shared", [False, True])
def test_isotope_batches(si, symmetrize, average, shared, monkeypatch):
    """跨批次、简并和多 sigma 的同位素结果与上游 CPU 实现一致。"""
    from phono3py.conductivity.scattering_solvers import IsotopeScatteringSolver
    from phono3py.other.isotope import Isotope

    from phono3py_gpu.isotope import GPUIsotopeSolver

    si.mesh_numbers = [3, 3, 3]
    si.init_phph_interaction()
    pp = si.phph_interaction
    pp.run_phonon_solver()
    iso = Isotope(
        pp.mesh_numbers,
        pp.primitive,
        bz_grid=pp.bz_grid,
        mass_variances=[0.0, 0.001],
        band_indices=[4, 0, 2],
        cutoff_frequency=5.0,
        symmetrize_tetrahedra=symmetrize,
        average_degenerate_weights=average,
        lang="Rust" if symmetrize else "C",
    )
    iso.set_phonons(pp.phonons, dm=pp.dynamical_matrix)
    device_phonons = (
        (
            cupy().asarray(pp.phonons.frequencies),
            cupy().asarray(pp.phonons.eigenvectors),
        )
        if shared
        else None
    )
    monkeypatch.setattr("phono3py_gpu.isotope.batch_size", lambda *a, **k: 2)
    gpu = GPUIsotopeSolver(iso, [None, 0.1, 1.0], device_phonons=device_phonons)
    for gp in [0, 1, 9]:
        actual = gpu.compute(gp)
        expected = np.concatenate(
            [
                IsotopeScatteringSolver(iso, [sigma]).compute(gp)
                for sigma in [None, 0.1, 1.0]
            ]
        )
        np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize(
    "symmetrize,average", [(True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("shared", [False, True])
def test_tetrahedron_options(si, symmetrize, average, shared, monkeypatch):
    from copy import deepcopy

    monkeypatch.setattr(
        "phono3py_gpu.integration.MIN_SHARED_FREQUENCIES", 1 if shared else 100
    )
    ph = deepcopy(si)
    ph.mesh_numbers = [4, 4, 4]
    ph.init_phph_interaction(
        symmetrize_tetrahedra=symmetrize, average_degenerate_weights=average
    )
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    f = pp.phonons.frequencies
    expected, z_ref = get_triplets_integration_weights(
        pp, f[1], None, is_collision_matrix=True, average_degenerate_weights=average
    )
    g, z = weights(
        pp, pp.get_triplets_at_q()[0], cupy().asarray(f), f[1], None, collision=True
    )
    np.testing.assert_allclose(host(g), expected, rtol=1e-10, atol=1e-11)
    np.testing.assert_array_equal(host(z), z_ref)


@pytest.mark.parametrize("r0,symmetrize", [(False, False), (True, True)])
def test_fc3_options(si, r0, symmetrize):
    from copy import deepcopy

    ph = deepcopy(si)
    ph.mesh_numbers = [3, 3, 3]
    ph._make_r0_average = r0
    ph.init_phph_interaction(symmetrize_fc3q=symmetrize)
    pp = ph.phph_interaction
    pp.run_phonon_solver()
    pp.set_grid_point(1)
    pp.run()
    engine = InteractionKernel(
        pp,
        cupy().asarray(pp.phonons.frequencies),
        cupy().asarray(pp.phonons.eigenvectors),
    )
    result = engine.compute(pp.get_triplets_at_q()[0])
    np.testing.assert_allclose(
        host(result), pp.interaction_strength, atol=1e-25, rtol=1e-9
    )


@pytest.mark.parametrize("method", ["wang", "gonze-full"])
def test_nac_variants(nacl, method):
    from copy import deepcopy

    from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL

    ph = deepcopy(nacl)
    params = dict(ph.nac_params)
    params["method"] = "wang" if method == "wang" else "gonze"
    ph.nac_params = params
    ph.mesh_numbers = [3, 3, 3]
    ph.init_phph_interaction()
    dm = ph.phph_interaction.dynamical_matrix
    if method == "gonze-full":
        dm = DynamicalMatrixGL(
            ph.supercell,
            ph.primitive,
            ph.fc2,
            nac_params=params,
            with_full_terms=True,
            lang="C",
        )
    gpu = Harmonic(dm)
    q = np.array([[0.0, 0.0, 0.0], [0.1, 0.21, 0.03]])
    expected = []
    for point in q:
        dm.run(point)
        expected.append(dm.dynamical_matrix.copy())
    np.testing.assert_allclose(host(gpu.matrices(q)), expected, rtol=1e-9, atol=1e-9)
    # Full Ewald derivative is independently checked by central differences.
    expected = []
    step = 1e-6
    for point in q[1:]:
        gradient = []
        for direction in np.eye(3):
            shift = ph.primitive.cell @ direction * step
            dm.run(point + shift)
            plus = dm.dynamical_matrix.copy()
            dm.run(point - shift)
            gradient.append((plus - dm.dynamical_matrix) / (2 * step))
        expected.append(gradient)
    np.testing.assert_allclose(
        host(gpu.matrices(q[1:], derivative=True)), expected, rtol=2e-6, atol=1e-7
    )


def test_band_block_for_free_memory():
    from phono3py_gpu.interaction import initial_band_block

    nb = 720
    for width in (1, 2, 4, 8, 16, 32):
        free = 2 * (16 * nb**3 + width * 128 * nb**2)
        assert initial_band_block(nb, free) == width
        assert initial_band_block(nb, free - 2) == max(1, width // 2)


def test_collision_eigh_in_place():
    from phono3py_gpu.collision import eigh_collision

    a = np.random.default_rng(4).normal(size=(259, 259))
    matrix = a + a.T
    w, v = map(host, eigh_collision(matrix))
    np.testing.assert_allclose(w, np.linalg.eigvalsh(matrix), atol=1e-12)
    np.testing.assert_allclose(matrix @ v, v * w, atol=1e-12)
