"""Experimental interleaved q batches for three-phonon scattering.

Integer symmetry metadata is prepared sequentially on the host. Each CUDA
interaction batch contains triplets from several initial q points.
"""

import numpy as np
from phono3py.conductivity.grid_point_data import ScatteringResult
from phono3py.conductivity.lbte_collision_solver import LBTECollisionResult

from phono3py_gpu.backend import cupy, host, kernel
from phono3py_gpu.integration import weights
from phono3py_gpu.scattering import average_initial, linewidth_detail


def collision_layout(solver):
    """Return target GR points, rotations and one-q collision row shape."""
    collision, pp = solver._collision, solver._pp
    nb0, nb = len(pp.band_indices), solver.engine.nb
    if collision._is_reducible_collision_matrix:
        targets = np.arange(np.prod(pp.mesh_numbers))[:, None]
        return targets, np.eye(3)[None], (nb0, len(targets), nb), 1
    targets = collision._rot_grid_points
    return (
        targets,
        collision._rotations_cartesian,
        (nb0, 3, len(targets), nb, 3),
        3,
    )


def compute_q_batch(solver, grid_points, collision=False):
    """Compute a bounded group of q points without per-q device synchronization.

    Full/read/averaged pp and detailed file output use the original solver path.
    This routine returns the same per-q result objects as the upstream solvers.
    """
    cp = cupy()
    engine, pp = solver.engine, solver._pp
    nq, nb0, nb = len(grid_points), len(pp.band_indices), engine.nb
    ns, nt = len(solver._sigmas), len(solver._temperatures)
    triplets, multiplicities, maps, swaps = [], [], [], []
    offset = 0
    if collision:
        targets, rotations, rowshape, nc = collision_layout(solver)
    for gp in grid_points:
        if collision:
            solver._collision.set_grid_point(int(gp))
            gp2tp, _, swapped = solver._collision._get_gp2tp_map()
            maps.append(gp2tp[targets] + offset)
            swaps.append(swapped[targets])
        else:
            pp.set_grid_point(int(gp))
        tp, w = pp.get_triplets_at_q()[:2]
        triplets.append(tp)
        multiplicities.append(w)
        offset += len(tp)

    counts = [len(tp) for tp in triplets]
    q_triplets, q_weights = triplets, multiplicities
    # Round-robin order ensures large single-q triplet lists also overlap in
    # the same kernel, instead of mixing only the tail of neighbouring lists.
    order = np.argsort(np.concatenate([np.arange(n) for n in counts]), kind="stable")
    owner = np.repeat(np.arange(nq), counts)[order]
    triplets = np.concatenate(triplets)[order]
    multiplicities = cp.asarray(np.concatenate(multiplicities)[order])
    owners = cp.asarray(owner)
    normal = cp.asarray(
        np.all(pp.bz_grid.addresses[triplets].sum(axis=1) == 0, axis=1), dtype=cp.int8
    )
    channels = 3 if not collision and solver._is_N_U else 1
    gamma = cp.zeros((ns, nt, channels, nq, nb0))
    bands = cp.asarray(pp.band_indices)
    if collision:
        inverse = np.argsort(order)
        mapped = cp.asarray(inverse[np.asarray(maps)])
        swapped = cp.asarray(np.asarray(swaps))
        rotations = cp.asarray(rotations)
        rows = np.empty((ns, nt, nq, *rowshape))

    for j, sigma in enumerate(solver._sigmas):
        if collision:
            row = cp.zeros((nt, nq, *rowshape))
            saved, saved_zero = [], []
        for start in range(0, len(triplets), engine.batch_size):
            end = min(start + engine.batch_size, len(triplets))
            selected = triplets[start:end]
            tp = cp.asarray(selected)
            frequencies = engine.frequencies[tp[:, 0]][:, bands]
            g, zero = weights(
                pp,
                selected,
                engine.frequencies,
                frequencies,
                sigma,
                solver._sigma_cutoff,
                collision=collision,
                vertices=engine.tetrahedron.vertices(selected)
                if sigma is None
                else None,
            )
            strength = engine.compute(selected, zero)
            stats = solver.q_batch_stats
            stats["triplet_batches"] += 1
            stats["max_q_per_launch"] = max(
                stats["max_q_per_launch"], len(np.unique(owner[start:end]))
            )
            for k, temperature in enumerate(solver._temperatures):
                values = linewidth_detail(
                    engine, selected, strength, g, temperature, solver._occupations[k]
                ).sum(axis=(2, 3))
                kernel("scattering.cu", "sum_by_q")(
                    ((channels * nq * nb0 + 127) // 128,),
                    (128,),
                    (
                        gamma[j, k],
                        values,
                        multiplicities[start:end],
                        owners[start:end],
                        normal[start:end],
                        np.int32(len(selected)),
                        np.int32(nb0),
                        np.int32(nq),
                        np.int32(channels),
                    ),
                )
                if not collision or temperature <= 0:
                    continue
                reduced = cp.empty((2, len(selected), nb0, nb))
                kernel("scattering.cu", "collision_reduce")(
                    ((reduced.size + 127) // 128,),
                    (128,),
                    (
                        reduced,
                        strength,
                        g[2],
                        solver._inv_sinh[k],
                        tp,
                        np.int32(len(selected)),
                        np.int32(nb0),
                        np.int32(nb),
                        np.float64(solver._collision.unit_conversion_factor),
                    ),
                )
                kernel("scattering.cu", "collision_rows")(
                    ((row[k].size + 127) // 128,),
                    (128,),
                    (
                        row[k],
                        reduced,
                        mapped,
                        swapped,
                        rotations,
                        np.int32(len(selected)),
                        np.int32(nb0),
                        np.int32(nb),
                        np.int32(targets.shape[0]),
                        np.int32(targets.shape[1]),
                        np.int32(start),
                        np.int32(nc),
                        np.int32(nq),
                    ),
                )
            if collision and j == ns - 1:
                # Preserve the upstream final-q pp state without retaining
                # every q's full interaction tensor on the host.
                last = np.flatnonzero(owner[start:end] == nq - 1)
                if len(last):
                    ids = cp.asarray(last)
                    saved.append(host(strength[ids]))
                    saved_zero.append(host(zero[ids]))
        if collision:
            rows[j] = host(row)
    if collision:
        pp.set_interaction_strength(
            np.concatenate(saved), g_zero=np.concatenate(saved_zero)
        )
    for i, gp in enumerate(grid_points):
        average_initial(gamma[:, :, :, i], pp, int(gp))
    gamma = host(gamma)
    if collision:
        return [
            LBTECollisionResult(gamma=gamma[:, :, 0, i], collision_row=rows[:, :, i])
            for i in range(nq)
        ]
    return [
        dict(
            result=ScatteringResult(gamma=gamma[:, :, 0, i]),
            gamma_N=gamma[:, :, 1, i] if channels == 3 else None,
            gamma_U=gamma[:, :, 2, i] if channels == 3 else None,
            triplets=q_triplets[i],
            weights=q_weights[i],
        )
        for i in range(nq)
    ]
