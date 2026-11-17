"""Device-resident FC3 Fourier transforms and phonon-mode contractions."""

from itertools import permutations

import numpy as np

from phono3py_gpu.backend import batch_size, cupy
from phono3py_gpu.fourier import GPUFC3Transform


class InteractionKernel:
    """Share force constants and phonons between RTA, LBTE and WTE."""

    def __init__(self, pp, frequencies, eigenvectors):
        cp = cupy()
        self.pp = pp
        self.frequencies = frequencies
        self.eigenvectors = eigenvectors
        self.nb = frequencies.shape[1]
        self.mass_eigenvectors = (
            eigenvectors
            / cp.sqrt(cp.repeat(cp.asarray(pp.primitive.masses), 3))[None, :, None]
        )
        self._fourier = None
        self.batch_size = batch_size(16 * self.nb**3 * 9, maximum=64)

    @property
    def fourier(self):
        """Allocate FC3 only when a scattering calculation actually needs it."""
        if self._fourier is None:
            self._fourier = GPUFC3Transform.from_interaction(self.pp)
        return self._fourier

    def project(self, triplets, bands):
        """Contract atom-first reciprocal FC3 without conjugating eigenvectors."""
        cp = cupy()
        n = len(triplets)
        reciprocal = self.fourier.transform(triplets)
        # (atom, atom, atom, cart, cart, cart) -> (component, component, component)
        tensor = reciprocal.transpose(0, 1, 4, 2, 5, 3, 6).reshape(
            n, self.nb, self.nb, self.nb
        )
        tp = cp.asarray(triplets)
        e0, e1, e2 = [self.mass_eigenvectors[tp[:, i]] for i in range(3)]
        e0 = e0[:, :, cp.asarray(bands)]
        a = cp.einsum("tijk,tia->tajk", tensor, e0)
        b = cp.einsum("tajk,tjb->tabk", a, e1)
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
        if pp.symmetrize_fc3q:
            result = cp.zeros((len(triplets), len(bands), self.nb, self.nb))
            for permutation in permutations(range(3)):
                values = self.project(triplets[:, permutation], np.arange(self.nb))
                inverse = np.argsort(permutation)
                result += values.transpose(0, *(inverse + 1))[:, cp.asarray(bands)] / 6
        else:
            result = self.project(triplets, bands)
            if zero is not None:
                result *= zero == 0
        return result * pp.unit_conversion_factor
