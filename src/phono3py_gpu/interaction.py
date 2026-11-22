"""Device-resident FC3 Fourier transforms and phonon-mode contractions."""

import os
from itertools import permutations

import numpy as np

from phono3py_gpu.backend import batch_size, cupy, kernel
from phono3py_gpu.fourier import GPUFC3Transform
from phono3py_gpu.integration import TetrahedronTopology

MIN_STREAMED_BANDS = 192


def initial_band_block(nb, free_bytes):
    """Choose the largest projection tile within the existing half-free budget."""
    reciprocal_bytes = 16 * nb**3
    bytes_per_band = 128 * nb**2
    for width in (32, 16, 8, 4, 2):
        if reciprocal_bytes + width * bytes_per_band <= free_bytes // 2:
            return width
    return 1


class InteractionKernel:
    """Share force constants and phonons between RTA, LBTE and WTE."""

    def __init__(self, pp, frequencies, eigenvectors):
        cp = cupy()
        self.pp = pp
        self.frequencies = frequencies
        self.eigenvectors = eigenvectors
        self.nb = frequencies.shape[1]
        self.mass_eigenvectors = cp.ascontiguousarray(
            eigenvectors
            / cp.sqrt(cp.repeat(cp.asarray(pp.primitive.masses), 3))[None, :, None]
        )
        self._fourier = None
        self._tetrahedron = None
        maximum = int(os.environ.get("PHONO3PY_GPU_TRIPLET_BATCH_SIZE", "64"))
        if maximum < 1:
            raise ValueError("PHONO3PY_GPU_TRIPLET_BATCH_SIZE must be positive.")
        self.band_block = 32
        self.streamed = (
            self.nb >= MIN_STREAMED_BANDS
            and not pp.symmetrize_fc3q
            and os.environ.get("PHONO3PY_GPU_FOURIER", "tiled") == "tiled"
        )
        if self.streamed:
            # One complex reciprocal tensor, plus FP64/complex projection and
            # integration tiles. Work on one triplet at a time in this layout.
            self.band_block = initial_band_block(
                self.nb, cp.cuda.runtime.memGetInfo()[0]
            )
            work_bytes = 16 * self.nb**3 + 128 * self.band_block * self.nb**2
            self.batch_size = batch_size(work_bytes, maximum=1)
        else:
            self.batch_size = batch_size(16 * self.nb**3 * 9, maximum=maximum)

    @property
    def tetrahedron(self):
        if self._tetrahedron is None:
            self._tetrahedron = TetrahedronTopology(self.pp)
        return self._tetrahedron

    @property
    def fourier(self):
        """Allocate FC3 only when a scattering calculation actually needs it."""
        if self._fourier is None:
            self._fourier = GPUFC3Transform.from_interaction(self.pp)
        return self._fourier

    def project_slice(self, tensor, triplet, bands, zero):
        """Contract the three legs for a tile of initial bands in complex128."""
        cp = cupy()
        nb = self.nb
        q0, q1, q2 = triplet
        ids = cp.asarray(bands, dtype=cp.int64)
        packed = None
        if zero is not None and nb >= MIN_STREAMED_BANDS:
            active = cp.any(zero == 0, axis=2)
            width = int(active.sum(axis=1).max())
            if width == 0:
                return cp.zeros(zero.shape, dtype=cp.float64)
            if width < nb // 2:
                packed = self._pack_second_leg(q1, active, width)
        e0 = self.mass_eigenvectors[q0][:, ids]
        first = (e0.T @ tensor.reshape(nb, -1)).reshape(len(bands), nb, nb)
        if packed is not None:
            vectors, pair_map = packed
            second = vectors.swapaxes(1, 2) @ first
            result = cp.empty(zero.shape, dtype=cp.float64)
            kernel("interaction.cu", "packed_projection")(
                ((result.size + 127) // 128,),
                (128,),
                (
                    result,
                    second,
                    self.mass_eigenvectors,
                    self.frequencies,
                    cp.asarray(triplet),
                    ids,
                    zero,
                    pair_map,
                    np.int32(len(bands)),
                    np.int32(nb),
                    np.int32(width),
                    np.float64(self.pp.cutoff_frequency),
                ),
            )
            return result
        second = self.mass_eigenvectors[q1].T @ first
        if zero is not None:
            return self._masked_projection(
                second[None], cp.asarray(triplet)[None], bands, zero[None]
            )[0]
        amplitude = second.reshape(-1, nb) @ self.mass_eigenvectors[q2]
        result = cp.empty((len(bands), nb, nb), dtype=cp.float64)
        kernel("interaction.cu", "amplitude_squared")(
            ((result.size + 255) // 256,),
            (256,),
            (
                result,
                cp.ascontiguousarray(amplitude),
                cp.ascontiguousarray(zero) if zero is not None else np.uint64(0),
                self.frequencies[q0],
                self.frequencies[q1],
                self.frequencies[q2],
                ids,
                np.int32(nb),
                np.int64(result.size),
                np.float64(self.pp.cutoff_frequency),
            ),
        )
        return result

    def _pack_second_leg(self, grid_point, active, width):
        """仅投影有非零积分支撑的第二条声子支，保留原始能带索引。"""
        cp = cupy()
        nb0, nb = active.shape
        selected = cp.empty((nb0, nb), dtype=cp.int32)
        pair_map = cp.empty_like(selected)
        kernel("interaction.cu", "pack_active_bands")(
            ((nb0 + 31) // 32,),
            (32,),
            (selected, pair_map, active, np.int32(nb0), np.int32(nb)),
        )
        vectors = cp.empty((nb0, nb, width), dtype=cp.complex128)
        kernel("interaction.cu", "gather_second_leg")(
            ((vectors.size + 127) // 128,),
            (128,),
            (
                vectors,
                self.mass_eigenvectors[grid_point],
                selected,
                np.int32(nb0),
                np.int32(nb),
                np.int32(width),
            ),
        )
        return vectors, pair_map

    def _masked_projection(self, second, triplets, bands, zero):
        """Skip zero-support final-leg dot products before forming amplitudes."""
        cp = cupy()
        result = cp.empty((len(triplets), len(bands), self.nb, self.nb))
        kernel("interaction.cu", "masked_projection")(
            ((result.size + 127) // 128,),
            (128,),
            (
                result,
                second,
                self.mass_eigenvectors,
                self.frequencies,
                triplets,
                cp.asarray(bands, dtype=cp.int64),
                cp.ascontiguousarray(zero, dtype=cp.int8),
                np.int32(len(triplets)),
                np.int32(len(bands)),
                np.int32(self.nb),
                np.float64(self.pp.cutoff_frequency),
                *(np.int64(s // second.itemsize) for s in second.strides),
            ),
        )
        return result

    def project(self, triplets, bands, zero=None):
        """Contract atom-first reciprocal FC3 without conjugating eigenvectors."""
        cp = cupy()
        n = len(triplets)
        reciprocal = self.fourier.transform(triplets)
        # (atom, atom, atom, cart, cart, cart) -> (component, component, component)
        tensor = reciprocal.transpose(0, 1, 4, 2, 5, 3, 6).reshape(
            n, self.nb, self.nb, self.nb
        )
        tp = cp.asarray(triplets)
        e0, e1 = [self.mass_eigenvectors[tp[:, i]] for i in range(2)]
        e0 = e0[:, :, cp.asarray(bands)]
        a = cp.einsum("tijk,tia->tajk", tensor, e0)
        b = cp.einsum("tajk,tjb->tabk", a, e1)
        if zero is not None:
            return self._masked_projection(b, tp, bands, zero)
        e2 = self.mass_eigenvectors[tp[:, 2]]
        amplitude = cp.einsum("tabk,tkc->tabc", b, e2)
        f0 = self.frequencies[tp[:, 0]][:, cp.asarray(bands)]
        f1 = self.frequencies[tp[:, 1]]
        f2 = self.frequencies[tp[:, 2]]
        denom = f0[:, :, None, None] * f1[:, None, :, None] * f2[:, None, None, :]
        cutoff = self.pp.cutoff_frequency
        valid = (
            (f0[:, :, None, None] > cutoff)
            & (f1[:, None, :, None] > cutoff)
            & (f2[:, None, None, :] > cutoff)
        )
        return cp.where(valid, cp.abs(amplitude) ** 2 / cp.where(valid, denom, 1), 0)

    def compute(self, triplets, zero=None):
        """Compute one bounded triplet batch, preserving q symmetrization."""
        cp = cupy()
        pp = self.pp
        bands = pp.band_indices
        if pp.constant_averaged_interaction is not None:
            value = pp.constant_averaged_interaction / np.prod(pp.mesh_numbers)
            return cp.full((len(triplets), len(bands), self.nb, self.nb), value)
        if self.streamed:
            result = cp.empty((len(triplets), len(bands), self.nb, self.nb))
            for t, triplet in enumerate(triplets):
                tensor = self.fourier.components(triplet)
                for start in range(0, len(bands), self.band_block):
                    block = slice(start, start + self.band_block)
                    mask = None if zero is None else zero[t, block]
                    result[t, block] = self.project_slice(
                        tensor, triplet, bands[block], mask
                    )
                del tensor
            result *= pp.unit_conversion_factor
            return result
        if pp.symmetrize_fc3q:
            result = cp.zeros((len(triplets), len(bands), self.nb, self.nb))
            for permutation in permutations(range(3)):
                values = self.project(triplets[:, permutation], np.arange(self.nb))
                inverse = np.argsort(permutation)
                result += values.transpose(0, *(inverse + 1))[:, cp.asarray(bands)] / 6
        else:
            result = self.project(triplets, bands, zero)
        return result * pp.unit_conversion_factor
