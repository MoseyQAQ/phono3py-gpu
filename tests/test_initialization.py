"""Independent geometry masks, including multiplicities and strict cutoffs."""

from copy import deepcopy

import numpy as np
import pytest
from phono3py import Phono3py
from phono3py.phonon3.interaction import Interaction
from phonopy.structure.atoms import PhonopyAtoms

from phono3py_gpu.initialization import reference_all_shortest


@pytest.mark.parametrize("structure", ["si", "nacl", "skew"])
def test_geometry_mask(request, structure):
    if structure == "skew":
        ph = Phono3py(
            PhonopyAtoms(
                symbols=["Si", "Si"],
                cell=[[3.1, 0, 0], [0.8, 3.4, 0], [0.3, 0.6, 4.2]],
                scaled_positions=[[0, 0, 0], [0.23, 0.32, 0.41]],
            ),
            supercell_matrix=[[2, 1, 0], [0, 2, 0], [0, 0, 2]],
        )
    else:
        ph = deepcopy(request.getfixturevalue(structure))
    ph.mesh_numbers = [3] * 3
    pp = Interaction(ph.primitive, ph._bz_grid, ph.primitive_symmetry)
    actual = pp._all_shortest.copy()
    pp._all_shortest.fill(0)
    reference_all_shortest(pp)
    np.testing.assert_array_equal(actual, pp._all_shortest)


@pytest.mark.parametrize("direction", [-1, 0, 1])
def test_strict_cutoff(si, direction):
    ph = deepcopy(si)
    ph.mesh_numbers = [3] * 3
    pp = Interaction(ph.primitive, ph._bz_grid, ph.primitive_symmetry)
    multi, svecs = pp._multi, pp._svecs
    perms, lattice = pp._primitive.atomic_permutations, pp._primitive.cell
    # Find a finite distance mismatch and place the cutoff exactly at it, or
    # one representable float to either side of the upstream scalar value.
    mismatch = None
    for j in np.flatnonzero(multi[:, 0, 0] == 1):
        jp = pp._primitive.p2p_map[pp._s2p[j]]
        row = np.flatnonzero(perms[:, j] == pp._p2s[jp])[0]
        for k in np.flatnonzero(multi[:, 0, 0] == 1):
            mapped = perms[row, k]
            if multi[mapped, jp, 0] != 1:
                continue
            a = svecs[multi[k, 0, 1]] - svecs[multi[j, 0, 1]]
            b = svecs[multi[mapped, jp, 1]]
            difference = abs(np.linalg.norm(a @ lattice) - np.linalg.norm(b @ lattice))
            if difference > 1e-3:
                mismatch = difference
                break
        if mismatch is not None:
            break
    assert mismatch is not None
    pp._symprec = (
        mismatch if direction == 0 else np.nextafter(mismatch, direction * np.inf)
    )
    pp._all_shortest.fill(0)
    reference_all_shortest(pp)
    expected = pp._all_shortest.copy()
    pp._all_shortest.fill(0)
    pp._get_all_shortest()
    np.testing.assert_array_equal(pp._all_shortest, expected)
