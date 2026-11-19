"""CPU/GPU comparisons before calculator integration."""

import numpy as np
import pytest
from phono3py.phonon3.triplets import get_triplets_integration_weights
from phonopy.harmonic.derivative_dynmat import DerivativeOfDynamicalMatrix

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.harmonic import Harmonic
from phono3py_gpu.integration import weights
from phono3py_gpu.interaction import InteractionKernel


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
def test_masked_partial_projection(si, all_zero):
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
    result = host(engine.compute(triplets, cp.asarray(zero)))
    np.testing.assert_allclose(result, pp.interaction_strength, rtol=1e-9, atol=1e-25)
    assert np.count_nonzero(result[zero != 0]) == 0


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
def test_collision_batches(si, reducible):
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


@pytest.mark.parametrize("sigma", [None, 0.1])
def test_isotope(si, sigma):
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
        cutoff_frequency=pp.cutoff_frequency,
    )
    iso.set_phonons(pp.phonons, dm=pp.dynamical_matrix)
    cpu = IsotopeScatteringSolver(iso, [sigma])
    gpu = GPUIsotopeSolver(iso, [sigma])
    for gp in [1, 3, 9]:
        a, b = cpu.compute(gp), gpu.compute(gp)
        np.testing.assert_allclose(b, a, rtol=1e-9, atol=1e-13)


@pytest.mark.parametrize(
    "symmetrize,average", [(True, False), (False, True), (True, True)]
)
def test_tetrahedron_options(si, symmetrize, average):
    from copy import deepcopy

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
