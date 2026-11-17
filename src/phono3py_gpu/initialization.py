"""Batch the geometry mask used by FC3 r0 averaging, without starting CUDA."""

import os

import numpy as np
from phono3py.phonon3.interaction import Interaction

# Preserve the pinned upstream implementation for independent comparisons.
reference_all_shortest = Interaction._get_all_shortest


def all_shortest(interaction):
    """Fill (primitive atom, supercell atom, supercell atom) shortest flags."""
    mode = os.environ.get("PHONO3PY_GPU_INITIALIZATION", "numpy")
    if mode == "reference":
        return reference_all_shortest(interaction)
    if mode != "numpy":
        raise ValueError("PHONO3PY_GPU_INITIALIZATION must be numpy or reference.")

    svecs, multi = interaction._svecs, interaction._multi
    primitive = interaction._primitive
    perms = primitive.atomic_permutations
    s2pp = np.array([primitive.p2p_map[i] for i in interaction._s2p])
    lattice = primitive.cell
    ns, na, _ = multi.shape
    eps = np.finfo(np.float64).eps
    for i in range(na):
        for j in range(ns):
            if multi[j, i, 0] > 1:
                continue
            jp = s2pp[j]
            translation = np.flatnonzero(perms[:, j] == interaction._p2s[jp])
            assert len(translation) == 1
            mapped = perms[translation[0]]
            valid = (multi[:, i, 0] <= 1) & (multi[mapped, jp, 0] <= 1)
            k = np.flatnonzero(valid)
            a = svecs[multi[k, i, 1]] - svecs[multi[j, i, 1]]
            b = svecs[multi[mapped[k], jp, 1]]
            da = np.linalg.norm(a @ lattice, axis=1)
            db = np.linalg.norm(b @ lattice, axis=1)
            difference = np.abs(db - da)
            shortest = difference < interaction._symprec
            # Batch BLAS/norm rounding may differ at the strict cutoff. Use the
            # upstream scalar operations for comparisons within roundoff.
            scale = ((np.abs(a) + np.abs(b)) @ np.abs(lattice)).sum(axis=1)
            boundary = np.abs(difference - interaction._symprec) <= (
                32 * eps * np.maximum(scale, 1)
            )
            for index in np.flatnonzero(boundary):
                distance = np.linalg.norm(a[index] @ lattice)
                mapped_distance = np.linalg.norm(b[index] @ lattice)
                shortest[index] = abs(mapped_distance - distance) < interaction._symprec
            interaction._all_shortest[i, j, k[shortest]] = 1


def install():
    """Install once per process; initialization precedes calculator selection."""
    Interaction._get_all_shortest = all_shortest
