"""GPU Gaussian and linear-tetrahedron three-phonon integration weights."""

import numpy as np
from phonopy.phonon.grid import get_neighboring_grid_points
from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_gr_grid_address

from phono3py_gpu.backend import cupy, kernel


def average_axis(values, ids, axis, minimum=False):
    """Average or AND-mask each complete degenerate block, in place."""
    cp = cupy()
    for identifier in np.unique(ids):
        indices = np.flatnonzero(ids == identifier)
        if len(indices) < 2:
            continue
        key = [slice(None)] * values.ndim
        key[axis] = cp.asarray(indices)
        block = values[tuple(key)]
        reduced = (
            block.min(axis=axis, keepdims=True)
            if minimum
            else block.mean(axis=axis, keepdims=True)
        )
        values[tuple(key)] = reduced


def weights(
    pp,
    triplets,
    frequencies,
    frequency_points,
    sigma,
    sigma_cutoff=None,
    collision=False,
):
    """Return device (g, g_zero), in the upstream channel and band order."""
    cp = cupy()
    nb = frequencies.shape[1]
    nf = len(frequency_points)
    nc = 3 if collision else 2
    shape = (len(triplets), nf, nb, nb)
    f0 = cp.asarray(frequency_points)
    tp = cp.asarray(triplets)
    if sigma is not None:
        f1 = frequencies[tp[:, 1]][:, None, :, None]
        f2 = frequencies[tp[:, 2]][:, None, None, :]
        omega = f0[None, :, None, None]
        args = (omega - f1 - f2, omega + f1 - f2, omega - f1 + f2)
        terms = [
            cp.exp(-x * x / (2 * sigma * sigma)) / (np.sqrt(2 * np.pi) * sigma)
            for x in args
        ]
        zero = cp.zeros(shape, dtype=cp.int8)
        if sigma_cutoff is not None and sigma_cutoff > 0:
            zero = (
                (cp.abs(args[0]) > sigma_cutoff * sigma)
                & (cp.abs(args[1]) > sigma_cutoff * sigma)
                & (cp.abs(args[2]) > sigma_cutoff * sigma)
            ).astype(cp.int8)
        g = cp.stack(
            [terms[0], terms[1] - terms[2]]
            + ([terms[0] + terms[1] + terms[2]] if collision else [])
        )
        g *= (zero == 0)[None]
    else:
        relative = get_tetrahedra_relative_gr_grid_address(
            pp.bz_grid, pp.symmetrize_tetrahedra
        )
        vertices = np.empty((len(triplets), 2, len(relative), 4), dtype="int64")
        # Integer BZ topology remains upstream; all frequency arithmetic is CUDA.
        for t, triplet in enumerate(triplets):
            vertices[t, 0] = get_neighboring_grid_points(
                int(triplet[1]), relative, pp.bz_grid
            )
            vertices[t, 1] = get_neighboring_grid_points(
                int(triplet[2]), -relative, pp.bz_grid
            )
        g = cp.empty((nc, *shape))
        zero = cp.empty(shape, dtype=cp.int8)
        kernel("tetrahedron.cu", "weights")(
            ((zero.size + 127) // 128,),
            (128,),
            (
                g,
                zero,
                frequencies,
                cp.ascontiguousarray(f0),
                cp.asarray(vertices),
                np.int32(len(triplets)),
                np.int32(nf),
                np.int32(nb),
                np.int32(len(relative)),
                np.int32(nc),
            ),
        )
    if pp.average_degenerate_weights and sigma is None:
        ids = pp.phonons.degenerate_ids
        for t, (_, q1, q2) in enumerate(triplets):
            average_axis(g[:, t], ids[q1], 2)
            average_axis(g[:, t], ids[q2], 3)
            average_axis(zero[t], ids[q1], 1, minimum=True)
            average_axis(zero[t], ids[q2], 2, minimum=True)
    return g, zero
