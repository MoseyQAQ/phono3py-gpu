"""Tamura isotope scattering on CUDA with upstream tetrahedron conventions."""

import numpy as np
from phono3py.conductivity.scattering_solvers import IsotopeScatteringSolver
from phonopy.phonon.grid import get_neighboring_grid_points
from phonopy.phonon.tetrahedron_method import get_tetrahedra_relative_gr_grid_address

from phono3py_gpu.backend import batch_size, cupy, host, kernel
from phono3py_gpu.integration import average_axis


class GPUIsotopeSolver(IsotopeScatteringSolver):
    def __init__(self, *args, device_phonons=None, **kwargs):
        super().__init__(*args, **kwargs)
        # 复用同一 Interaction 的规范声子基底，避免再上传整个网格。
        self._device_phonons = device_phonons
        self._vertices = None

    def compute(self, grid_point):
        cp = cupy()
        iso = self.isotope
        iso.set_grid_point(grid_point)
        if self._device_phonons is None:
            f = cp.asarray(iso.phonons.frequencies)
            e = cp.asarray(iso.phonons.eigenvectors)
        else:
            f, e = self._device_phonons
        nb = f.shape[1]
        gps = iso.bz_grid.grg2bzg
        step = batch_size(32 * nb**2, maximum=128)
        mass_variances = cp.asarray(iso.mass_variances)
        outputs = []
        for sigma in self._sigmas:
            iso.sigma = sigma
            bands = iso._get_calc_band_indices()
            f0 = cp.ascontiguousarray(f[grid_point, cp.asarray(bands)])
            e0 = cp.ascontiguousarray(e[grid_point][:, cp.asarray(bands)])
            gamma = cp.zeros(len(bands))
            if sigma is None and self._vertices is None:
                relative = get_tetrahedra_relative_gr_grid_address(
                    iso.bz_grid, iso.symmetrize_tetrahedra
                )
                self._vertices = cp.asarray(
                    np.array(
                        [
                            get_neighboring_grid_points(int(gp), relative, iso.bz_grid)
                            for gp in gps
                        ],
                        dtype="int64",
                    )
                )
            for start in range(0, len(gps), step):
                selected = gps[start : start + step]
                ids = cp.asarray(selected)
                if sigma is None:
                    w = cp.empty((len(ids), len(bands), nb))
                    count = w.size
                    kernel("tetrahedron.cu", "spectral_weights")(
                        ((count + 127) // 128,),
                        (128,),
                        (
                            w,
                            f,
                            f0,
                            self._vertices[start : start + step],
                            np.int32(len(ids)),
                            np.int32(len(bands)),
                            np.int32(nb),
                            np.int32(self._vertices.shape[1]),
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
                kernel("isotope.cu", "weighted_overlap")(
                    ((w.size + 127) // 128,),
                    (128,),
                    (
                        w,
                        e0,
                        e,
                        f,
                        ids,
                        mass_variances,
                        np.int32(len(ids)),
                        np.int32(len(bands)),
                        np.int32(nb),
                        np.float64(iso._cutoff_frequency),
                    ),
                )
                gamma += w.sum(axis=(0, 2))
            gamma *= f0**2 * np.pi / (4 * int(np.prod(iso.bz_grid.D_diag)))
            gamma[f0 < iso._cutoff_frequency] = 0.0
            if iso._average_over_initial_bands():
                average_axis(gamma, iso.phonons.degenerate_ids[grid_point], 0)
                gamma = gamma[cp.asarray(iso.band_indices)]
            iso._gamma = host(gamma)
            outputs.append(iso.gamma)
        return np.asarray(outputs)
