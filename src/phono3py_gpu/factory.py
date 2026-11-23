# SPDX-License-Identifier: BSD-3-Clause
"""Adapter for phono3py's calculator factory.

Assembly follows upstream factory.py; public metadata and I/O are inherited.
"""

from __future__ import annotations

import os

import numpy as np
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

from phono3py_gpu.backend import batch_size, cupy, report_devices
from phono3py_gpu.batching import collision_layout, compute_q_batch
from phono3py_gpu.collision import create_collision_matrix_kernel
from phono3py_gpu.harmonic import prepare_phonons
from phono3py_gpu.interaction import InteractionKernel
from phono3py_gpu.kappa import GPURTAKappaSolver, make_wigner_lbte, make_wigner_rta
from phono3py_gpu.multigpu import devices
from phono3py_gpu.multigpu import iterate as iterate_multi_gpu
from phono3py_gpu.scattering import GPUCollisionSolver, GPUScatteringSolver
from phono3py_gpu.velocity import GPUHeatCapacitySolver, GPUVelocitySolver


class GPUCalculatorMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self._isotope_solver is not None:
            from phono3py_gpu.isotope import GPUIsotopeSolver

            solver = (
                self._collision_solver
                if isinstance(self, LBTECalculator)
                else self._scattering_solver
            )
            self._isotope_solver = GPUIsotopeSolver(
                self._isotope_solver.isotope,
                self._kappa_settings.sigmas,
                log_level=self._log_level,
                device_phonons=(solver.engine.frequencies, solver.engine.eigenvectors),
            )

    def _build_grid_point_aggregates(self):
        if self._velocity_solver.velocity_operator is not None:
            self._kappa_solver.velocity_operator = (
                self._velocity_solver.velocity_operator
            )
        return super()._build_grid_point_aggregates()

    def _iterate_grid_points(self, on_grid_point):
        """Opt-in q batching; stateful file paths retain upstream iteration."""
        requested = int(os.environ.get("PHONO3PY_GPU_Q_BATCH_SIZE", "1"))
        collision = isinstance(self, LBTECalculator)
        solver = self._collision_solver if collision else self._scattering_solver
        self.q_batch_stats = dict(q_batches=0, triplet_batches=0, max_q_per_launch=0)
        solver.q_batch_stats = self.q_batch_stats
        self.multi_gpu_stats = None
        reason = ""
        if self._pp.nac_q_direction is not None:
            reason = "directional NAC"
        elif collision and on_grid_point is not None:
            reason = "per-q output or callback"
        elif solver._read_pp:
            reason = "reading PP"
        elif solver._is_full_pp:
            reason = "full PP requested"
        elif not collision and self._read_gamma:
            reason = "reading linewidths"
        elif not collision and solver._requires_full_gamma_path():
            reason = "per-q scattering options"
        method = "WTE" if self._velocity_solver.wte else "BTE"
        stage = f"{method}-{'LBTE' if collision else 'RTA'} scattering"
        ids = devices()
        indices = list(self._get_grid_point_indices())
        if collision and indices:
            self._pp.delete_interaction_strength()
        if not reason and indices:
            if collision and (len(ids) > 1 or requested > 1):
                _, _, shape, _ = collision_layout(solver)
                bytes_per_q = int(np.prod(shape)) * len(solver._temperatures) * 8
                requested = batch_size(bytes_per_q, maximum=max(1, requested))
            groups = (len(indices) + max(1, requested) - 1) // max(1, requested)
            ids = ids[:groups]
        if len(ids) > 1 and not reason and indices:
            iterate_multi_gpu(
                self, solver, ids, indices, max(1, requested), collision, on_grid_point
            )
            if self._log_level:
                per_device = self.multi_gpu_stats["per_device"]
                used = [dev for dev in ids if per_device[dev]["q_points"]]
                report_devices(stage, used)
            return
        if self._log_level and indices:
            if collision or not self._read_gamma or solver.use_ave_pp:
                note = f" of {len(ids)} visible GPUs ({reason})" if len(ids) > 1 else ""
                report_devices(stage, ids[:1], note)
        if requested <= 1 or reason:
            if requested > 1 and self._log_level:
                print(
                    "phono3py-gpu: using the per-q path for the requested options."
                )
            if collision:
                return self._iterate_lbte_points(indices, on_grid_point)
            return super()._iterate_grid_points(on_grid_point)
        for start in range(0, len(indices), requested):
            selected = indices[start : start + requested]
            gps = self._kappa_settings.grid_points[selected]
            if self._log_level:
                print("phono3py-gpu: GPU batch: %d q points." % len(gps), flush=True)
            results = compute_q_batch(
                solver,
                gps,
                collision=collision,
                last_batch=start + len(selected) == len(indices),
            )
            self.q_batch_stats["q_batches"] += 1
            for offset, (index, result) in enumerate(
                zip(selected, results, strict=True)
            ):
                index = int(index)
                if collision:
                    self._gamma[:, :, index] = result.gamma
                    self._kappa_solver.store(index, result)
                else:
                    self._store_scattering_result(
                        index, result["result"], result["gamma_N"], result["gamma_U"]
                    )
                    self._gamma_detail_at_q = None
                    solver._gamma_N, solver._gamma_U = (
                        result["gamma_N"],
                        result["gamma_U"],
                    )
                    self._pp._triplets_at_q = result["triplets"]
                    self._pp._weights_at_q = result["weights"]
                self._grid_point_count = start + offset + 1
                if self._log_level:
                    if collision:
                        self._show_log(index, self._gv[index])
                    else:
                        self._show_log(index, self._gv[index], None)
                if on_grid_point is not None:
                    on_grid_point(index)


class GPURTACalculator(GPUCalculatorMixin, RTACalculator):
    def _finalize(self):
        if self._log_level:
            stage = (
                "WTE-RTA mode contributions"
                if self._velocity_solver.wte
                else "BTE-RTA conductivity sum"
            )
            report_devices(stage, [cupy().cuda.runtime.getDevice()])
        super()._finalize()


class GPULBTECalculator(GPUCalculatorMixin, LBTECalculator):
    def _iterate_lbte_points(self, indices, on_grid_point):
        """Expose each callback's PP, retaining only the final requested snapshot."""
        solver = self._collision_solver
        retain_final = solver.retain_pp
        try:
            for count, index in enumerate(indices, 1):
                keep = retain_final and count == len(indices)
                solver.retain_pp = keep or on_grid_point is not None
                index = int(index)
                self._compute_at_grid_point(index)
                self._grid_point_count = count
                if on_grid_point is not None:
                    on_grid_point(index)
                    if not keep:
                        self._pp.delete_interaction_strength()
        finally:
            solver.retain_pp = retain_final
            if not retain_final:
                self._pp.delete_interaction_strength()


def _validate_wte(interaction, config, wte):
    if wte and len(interaction.band_indices) != 3 * len(interaction.primitive):
        raise ValueError("WTE requires every phonon branch, as in phono3py-wte.")


def make_rta(
    interaction: Interaction,
    config: CalculatorConfig,
    wte: bool = False,
) -> RTACalculator:
    """Build an RTACalculator from variant component factories."""
    cupy().cuda.Device(devices()[0]).use()
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
    cupy().cuda.Device(devices()[0]).use()
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
        retain_pp=os.environ.get("PHONO3PY_GPU_RETAIN_PP", "0") != "0",
        sigmas=config.sigmas,
        sigma_cutoff=config.sigma_cutoff,
        temperatures=config.temperatures,
        is_full_pp=config.is_full_pp,
        read_pp=config.read_pp,
        pp_filename=config.pp_filename,
        log_level=config.log_level,
    )
    colmat_kernel = create_collision_matrix_kernel(
        wte=wte,
        kappa_settings=kappa_settings,
        frequencies=frequencies,
        degenerate_ids=interaction.phonons.degenerate_ids,
        rot_grid_points=rot_grid_points,
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
