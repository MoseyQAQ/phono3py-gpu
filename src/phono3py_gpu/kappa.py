"""Device reductions for standard BTE and effective linewidths."""

import numpy as np
from phono3py.conductivity.kappa_solvers import RTAKappaSolver

from phono3py_gpu.backend import cupy, host


def effective_gamma(aggregates):
    cp = cupy()
    gamma = cp.asarray(aggregates.gamma).copy()
    if aggregates.gamma_isotope is not None:
        gamma += cp.asarray(aggregates.gamma_isotope)[:, None]
    if aggregates.gamma_boundary is not None:
        gamma += cp.asarray(aggregates.gamma_boundary)[None, None]
    if aggregates.gamma_elph is not None:
        gamma += cp.asarray(aggregates.gamma_elph)
    return gamma


class GPURTAKappaSolver(RTAKappaSolver):
    def finalize(self, aggregates):
        cp = cupy()
        settings = self._kappa_settings
        g = effective_gamma(aggregates)
        cv = cp.asarray(aggregates.mode_heat_capacities)
        vv = cp.asarray(aggregates.gv_by_gv)
        mode = cp.nan_to_num(
            vv[None, None]
            * cv[None, ..., None]
            / (2 * g[..., None])
            * settings.conversion_factor
        )
        f = cp.asarray(
            self._frequencies[settings.grid_points][:, settings.band_indices]
        )
        mode *= (f >= settings.cutoff_frequency)[None, None, ..., None]
        self._kappa = host(mode.sum(axis=(2, 3)) / int(np.prod(settings.mesh_numbers)))
        self._mode_kappa = host(mode)
