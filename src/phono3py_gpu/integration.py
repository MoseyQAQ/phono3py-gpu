"""GPU Gaussian and linear-tetrahedron three-phonon integration weights."""

import numpy as np
from phonopy.phonon.grid import get_neighboring_grid_points
from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_gr_grid_address

from phono3py_gpu.backend import cupy, kernel

MIN_SHARED_FREQUENCIES = 24


class TetrahedronTopology:
    """Reuse upstream integer BZ neighbours for a fixed interaction mesh."""

    def __init__(self, pp):
        self.grid = pp.bz_grid
        self.relative = get_tetrahedra_relative_gr_grid_address(
            pp.bz_grid, pp.symmetrize_tetrahedra
        )
        self.neighbours = ({}, {})

    def vertices(self, triplets):
        vertices = np.empty((len(triplets), 2, len(self.relative), 4), dtype="int64")
        for leg, sign in enumerate((1, -1)):
            cache = self.neighbours[leg]
            for q in np.unique(triplets[:, leg + 1]):
                if q not in cache:
                    cache[q] = get_neighboring_grid_points(
                        int(q), sign * self.relative, self.grid
                    )
            vertices[:, leg] = [cache[q] for q in triplets[:, leg + 1]]
        return vertices


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


def channel_bounds(frequencies, vertices):
    """Extrema over all tetrahedron vertices, reused by the initial-band tiles."""
    cp = cupy()
    nt, _, nth, _ = vertices.shape
    nb = frequencies.shape[1]
    bounds = cp.empty((4, nt, nb, nb), dtype=cp.float64)
    kernel("tetrahedron.cu", "channel_bounds")(
        ((nt * nb * nb + 127) // 128,),
        (128,),
        (
            bounds,
            frequencies,
            cp.asarray(vertices),
            np.int32(nt),
            np.int32(nb),
            np.int32(nth),
        ),
    )
    return bounds


def weights(
    pp,
    triplets,
    frequencies,
    frequency_points,
    sigma,
    sigma_cutoff=None,
    collision=False,
    vertices=None,
    bounds=None,
):
    """Return device (g, g_zero), in the upstream channel and band order."""
    cp = cupy()
    f0 = cp.asarray(frequency_points)
    nb = frequencies.shape[1]
    nf = f0.shape[-1]
    nc = 3 if collision else 2
    shape = (len(triplets), nf, nb, nb)
    tp = cp.asarray(triplets)
    if sigma is not None:
        f1 = frequencies[tp[:, 1]][:, None, :, None]
        f2 = frequencies[tp[:, 2]][:, None, None, :]
        omega = f0[:, :, None, None] if f0.ndim == 2 else f0[None, :, None, None]
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
        if vertices is None:
            vertices = TetrahedronTopology(pp).vertices(triplets)
        shared = nf >= MIN_SHARED_FREQUENCIES
        if shared and bounds is None:
            bounds = channel_bounds(frequencies, vertices)
        g = cp.empty((nc, *shape))
        zero = cp.empty(shape, dtype=cp.int8)
        work = len(triplets) * nb * nb * 32 if shared else zero.size
        kernel("tetrahedron.cu", "weights_shared" if shared else "weights")(
            ((work + 127) // 128,),
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
                np.int32(vertices.shape[2]),
                np.int32(nc),
                np.int32(nf if f0.ndim == 2 else 0),
                bounds if bounds is not None else np.uint64(0),
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


def sigma_weights(pp, triplets, frequencies, frequency_points, sigmas, **kwargs):
    """Keep each sigma's weights and combine their nonzero supports."""
    integrations = [
        weights(pp, triplets, frequencies, frequency_points, sigma, **kwargs)
        for sigma in sigmas
    ]
    zero = integrations[0][1]
    if len(integrations) > 1:
        zero = zero.copy()
        for _, mask in integrations[1:]:
            zero &= mask
    return integrations, zero
