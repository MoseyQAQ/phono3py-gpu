"""Tamura isotope scattering on CUDA with upstream tetrahedron conventions."""

import numpy as np
from phono3py.conductivity.scattering_solvers import IsotopeScatteringSolver
from phonopy.phonon.grid import get_neighboring_grid_points
from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_gr_grid_address

from phono3py_gpu.backend import batch_size, cupy, host, kernel
from phono3py_gpu.integration import average_axis


class GPUIsotopeSolver(IsotopeScatteringSolver):
    def compute(self, grid_point):
        cp = cupy()
        iso = self.isotope
        iso.set_grid_point(grid_point)
        f = cp.asarray(iso.phonons.frequencies)
        e = cp.asarray(iso.phonons.eigenvectors)
        nb = f.shape[1]
        gps = iso.bz_grid.grg2bzg
        step = batch_size(16 * nb**2 * len(iso._primitive) * 3, maximum=128)
        outputs = []
        for sigma in self._sigmas:
            iso.sigma = sigma
            bands = iso._get_calc_band_indices()
            f0 = cp.ascontiguousarray(f[grid_point, cp.asarray(bands)])
            e0 = e[grid_point][:, cp.asarray(bands)].reshape(-1, 3, len(bands)).conj()
            gamma = cp.zeros(len(bands))
            if sigma is None:
                relative = get_tetrahedra_relative_gr_grid_address(
                    iso.bz_grid, iso.symmetrize_tetrahedra
                )
            for start in range(0, len(gps), step):
                selected = gps[start : start + step]
                ids = cp.asarray(selected)
                overlap = cp.einsum(
                    "acb,qacd->qabd", e0, e[ids].reshape(len(ids), -1, 3, nb)
                )
                strength = (
                    cp.abs(overlap) ** 2
                    * cp.asarray(iso.mass_variances)[None, :, None, None]
                ).sum(axis=1)
                if sigma is None:
                    vertices = np.array(
                        [
                            get_neighboring_grid_points(
                                int(gp), relative.reshape(-1, 3), iso.bz_grid
                            )
                            for gp in selected
                        ],
                        dtype="int64",
                    )
                    w = cp.empty((len(ids), len(bands), nb))
                    count = w.size
                    kernel("tetrahedron.cu", "spectral_weights")(
                        ((count + 127) // 128,),
                        (128,),
                        (
                            w,
                            f,
                            f0,
                            cp.asarray(vertices),
                            np.int32(len(ids)),
                            np.int32(len(bands)),
                            np.int32(nb),
                            np.int32(len(relative)),
                        ),
                    )
                    if iso._averaging:
                        if iso._average_over_initial_bands():
                            average_axis(w, iso.phonons.degenerate_ids[grid_point], 1)
                        for i, gp in enumerate(selected):
                            average_axis(w[i], iso.phonons.degenerate_ids[gp], 1)
                else:
                    delta = f0[None, :, None] - f[ids, None, :]
                    w = cp.exp(-(delta**2) / (2 * sigma * sigma)) / (
                        np.sqrt(2 * np.pi) * sigma
                    )
                valid = f[ids] >= iso._cutoff_frequency
                gamma += (strength * w * valid[:, None, :]).sum(axis=(0, 2))
            gamma *= f0**2 * np.pi / (4 * int(np.prod(iso.bz_grid.D_diag)))
            if iso._average_over_initial_bands():
                average_axis(gamma, iso.phonons.degenerate_ids[grid_point], 0)
                gamma = gamma[cp.asarray(iso.band_indices)]
            iso._gamma = host(gamma)
            outputs.append(iso.gamma)
        return np.asarray(outputs)
