# SPDX-License-Identifier: BSD-3-Clause
"""Adapter for phono3py's calculator factory at f0a2fdd.

Assembly follows upstream factory.py; public metadata and I/O are inherited.
"""

from __future__ import annotations

from phono3py.conductivity.build_components import (
    CalculatorConfig,
    build_lbte_kappa_settings,
    build_rot_grid_points,
    build_rta_kappa_settings,
)
from phono3py.conductivity.calculators import LBTECalculator, RTACalculator
from phono3py.conductivity.factory import _get_lbte_grid_point_indices
from phono3py.conductivity.kappa_solvers import LBTEKappaSolver
from phono3py.phonon3.collision_matrix import CollisionMatrix
from phono3py.phonon3.interaction import Interaction

from phono3py_gpu.collision import create_collision_matrix_kernel
from phono3py_gpu.harmonic import prepare_phonons
from phono3py_gpu.interaction import InteractionKernel
from phono3py_gpu.kappa import GPURTAKappaSolver, make_wigner_lbte, make_wigner_rta
from phono3py_gpu.scattering import GPUCollisionSolver, GPUScatteringSolver
from phono3py_gpu.velocity import GPUHeatCapacitySolver, GPUVelocitySolver


class GPUCalculatorMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._isotope_solver is not None:
            from phono3py_gpu.isotope import GPUIsotopeSolver

            self._isotope_solver = GPUIsotopeSolver(
                self._isotope_solver.isotope,
                self._kappa_settings.sigmas,
                log_level=self._log_level,
            )

    def _build_grid_point_aggregates(self):
        aggregates = super()._build_grid_point_aggregates()
        if self._velocity_solver.velocity_operator is not None:
            aggregates.extra["velocity_operator"] = (
                self._velocity_solver.velocity_operator
            )
        return aggregates


class GPURTACalculator(GPUCalculatorMixin, RTACalculator):
    pass


class GPULBTECalculator(GPUCalculatorMixin, LBTECalculator):
    pass


def _validate_wte(interaction, config, wte):
    if wte and len(interaction.band_indices) != 3 * len(interaction.primitive):
        raise ValueError("WTE requires every phonon branch, as in phono3py-wte.")


def make_rta(
    interaction: Interaction,
    config: CalculatorConfig,
    wte: bool = False,
) -> RTACalculator:
    """Build an RTACalculator from variant component factories."""
    harmonic, f, e = prepare_phonons(interaction)
    engine = InteractionKernel(interaction, f, e)
    _validate_wte(interaction, config, wte)
    kappa_settings = build_rta_kappa_settings(interaction, config)
    scattering_solver = GPUScatteringSolver(
        interaction,
        engine=engine,
        sigmas=config.sigmas,
        temperatures=config.temperatures,
        sigma_cutoff=config.sigma_cutoff,
        is_full_pp=config.is_full_pp,
        use_ave_pp=config.use_ave_pp,
        read_pp=config.read_pp,
        store_pp=config.store_pp,
        pp_filename=config.pp_filename,
        is_N_U=config.is_N_U,
        is_gamma_detail=config.is_gamma_detail,
        log_level=config.log_level,
        lang=config.lang,
    )
    velocity_solver = GPUVelocitySolver(
        interaction,
        harmonic=harmonic,
        wte=wte,
        is_kappa_star=kappa_settings.is_kappa_star,
        gv_delta_q=kappa_settings.gv_delta_q,
        log_level=config.log_level,
    )
    cv_solver = GPUHeatCapacitySolver(interaction, kappa_settings.temperatures)
    kappa_solver = (
        make_wigner_rta(
            kappa_settings,
            interaction.phonons.frequencies,
            interaction.primitive.volume,
            log_level=config.log_level,
        )
        if wte
        else GPURTAKappaSolver(
            kappa_settings,
            interaction.phonons.frequencies,
            log_level=config.log_level,
        )
    )

    return GPURTACalculator(
        interaction,
        velocity_solver=velocity_solver,
        cv_solver=cv_solver,
        scattering_solver=scattering_solver,
        kappa_solver=kappa_solver,
        kappa_settings=kappa_settings,
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_N_U=config.is_N_U,
        is_gamma_detail=config.is_gamma_detail,
        sigma_cutoff_width=config.sigma_cutoff,
        log_level=config.log_level,
        lang=config.lang,
        rust_gp_batch_size=config.rust_gp_batch_size,
    )


def make_lbte(
    interaction: Interaction,
    config: CalculatorConfig,
    wte: bool = False,
) -> LBTECalculator:
    """Build an LBTECalculator from variant component factories."""
    harmonic, f, e = prepare_phonons(interaction)
    engine = InteractionKernel(interaction, f, e)
    _validate_wte(interaction, config, wte)
    kappa_settings = build_lbte_kappa_settings(interaction, config)
    grid_point_indices = _get_lbte_grid_point_indices(
        kappa_settings.grid_points, config.grid_points
    )
    assert interaction.phonons is not None
    frequencies = interaction.phonons.frequencies
    rot_grid_points = build_rot_grid_points(kappa_settings)
    collision = CollisionMatrix(
        interaction,
        rot_grid_points=rot_grid_points,
        is_kappa_star=config.is_kappa_star,
        log_level=config.log_level,
        lang=config.lang,
    )
    collision_solver = GPUCollisionSolver(
        interaction,
        collision,
        engine=engine,
        sigmas=config.sigmas,
        sigma_cutoff=config.sigma_cutoff,
        temperatures=config.temperatures,
        is_full_pp=config.is_full_pp,
        read_pp=config.read_pp,
        pp_filename=config.pp_filename,
        log_level=config.log_level,
    )
    colmat_kernel = create_collision_matrix_kernel(
        kappa_settings=kappa_settings,
        frequencies=frequencies,
        degenerate_ids=interaction.phonons.degenerate_ids,
        rot_grid_points=rot_grid_points,
        solve_collective_phonon=config.solve_collective_phonon,
        pinv_cutoff=config.pinv_cutoff,
        pinv_solver=config.pinv_solver,
        pinv_method=config.pinv_method,
        lang=config.lang,
        log_level=config.log_level,
        is_one_gp_colmat=grid_point_indices is not None,
    )
    velocity_solver = GPUVelocitySolver(
        interaction,
        harmonic=harmonic,
        wte=wte,
        is_kappa_star=kappa_settings.is_kappa_star,
        gv_delta_q=kappa_settings.gv_delta_q,
        log_level=config.log_level,
    )
    cv_solver = GPUHeatCapacitySolver(interaction, kappa_settings.temperatures)
    kappa_solver = (
        make_wigner_lbte(
            colmat_kernel,
            kappa_settings,
            frequencies,
            interaction.primitive.volume,
            is_reducible_collision_matrix=kappa_settings.is_reducible_collision_matrix,
            log_level=config.log_level,
        )
        if wte
        else LBTEKappaSolver(colmat_kernel, kappa_settings, config.log_level)
    )

    return GPULBTECalculator(
        interaction,
        velocity_solver=velocity_solver,
        cv_solver=cv_solver,
        collision_solver=collision_solver,
        kappa_solver=kappa_solver,
        kappa_settings=kappa_settings,
        is_isotope=config.is_isotope,
        mass_variances=config.mass_variances,
        is_full_pp=config.is_full_pp,
        sigma_cutoff_width=config.sigma_cutoff,
        log_level=config.log_level,
        lang=config.lang,
        grid_point_indices=grid_point_indices,
    )
