"""Device reductions for standard BTE and Wigner transport."""

import numpy as np
from phono3py.conductivity.kappa_solvers import RTAKappaSolver
from phonopy.physical_units import get_physical_units

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


def pair_contributions(f, g, cv, vv, cutoff, conversion):
    """WTE mode pairs; g is a half width in THz, as in phono3py.gamma."""
    cp = cupy()
    f, g, cv, vv = (cp.asarray(x) for x in (f, g, cv, vv))
    omega = f * get_physical_units().THzToEv
    width = g * 2 * get_physical_units().THzToEv
    total = width[..., :, None] + width[..., None, :]
    delta = omega[:, None] - omega[None, :]
    denom = delta**2 + total**2 / 4
    c = cv / cp.where(omega > 0, omega, 1)
    pref = (omega[:, None] + omega[None, :]) * (c[..., :, None] + c[..., None, :]) / 4
    valid = (f[:, None] > cutoff) & (f[None, :] > cutoff) & (denom != 0)
    scalar = cp.where(valid, pref * total / 2 / cp.where(denom != 0, denom, 1), 0)
    return scalar[..., None] * vv * conversion


def make_wigner_rta(*args, **kwargs):
    from phono3py_gpu._wte import WignerRTAKappaSolver

    class GPUWignerRTAKappaSolver(WignerRTAKappaSolver):
        def _compute_mode_kappa(self, aggregates):
            cp = cupy()
            gamma = effective_gamma(aggregates)
            for i, gp in enumerate(self._kappa_settings.grid_points):
                f = cp.asarray(self._frequencies[gp])
                pair = pair_contributions(
                    f,
                    gamma[:, :, i],
                    aggregates.mode_heat_capacities[:, i],
                    aggregates.vm_by_vm[i],
                    self._kappa_settings.cutoff_frequency,
                    self._conversion_factor_WTE,
                ).real
                population = cp.abs(f[:, None] - f[None, :]) < 1e-4
                p = pair * population[..., None]
                self._mode_kappa_P[:, :, i] = host(
                    (p.sum(axis=-2) + p.sum(axis=-3)) / 2
                )
                self._mode_kappa_C[:, :, i] = host(pair * (~population)[..., None])

    return GPUWignerRTAKappaSolver(*args, **kwargs)


def make_wigner_lbte(*args, **kwargs):
    from phono3py_gpu._wte import WignerLBTEKappaSolver

    class GPUWignerLBTEKappaSolver(WignerLBTEKappaSolver):
        def _compute_coherence_kappa_at(self, aggregates, i_sigma, i_temp):
            cp = cupy()
            if self._mode_kappa_C is None:
                return
            total = cp.zeros(6)
            for i, gp in enumerate(self._kappa_settings.grid_points):
                f = cp.asarray(self._frequencies[gp])
                g = self._solver.get_main_diagonal(i, i_sigma, i_temp)
                pair = pair_contributions(
                    f,
                    g,
                    aggregates.mode_heat_capacities[i_temp, i],
                    aggregates.vm_by_vm[i],
                    self._kappa_settings.cutoff_frequency,
                    self._conversion_factor_WTE,
                )
                pair *= (cp.abs(f[:, None] - f[None, :]) > 1e-4)[..., None]
                self._mode_kappa_C[i_sigma, i_temp, i] = host(pair)
                total += pair.sum(axis=(0, 1)).real
            self._kappa_C[i_sigma, i_temp] = host(
                total / int(np.prod(self._kappa_settings.mesh_numbers))
            )

    return GPUWignerLBTEKappaSolver(*args, **kwargs)
