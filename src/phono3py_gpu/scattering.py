"""GPU three-phonon linewidths and collision rows with bounded batches."""

import numpy as np
from phono3py.conductivity.grid_point_data import ScatteringResult
from phono3py.conductivity.lbte_collision_solver import (
    LBTECollisionResult,
    LBTECollisionSolver,
)
from phono3py.conductivity.scattering_solvers import RTAScatteringSolver
from phono3py.file_IO import read_pp_from_hdf5
from phono3py.phonon3.imag_self_energy import ImagSelfEnergy
from phonopy.physical_units import get_physical_units

from phono3py_gpu.backend import cupy, host, kernel
from phono3py_gpu.integration import channel_bounds, sigma_weights


def average_initial(values, pp, gp):
    """Average fully selected degenerate subspaces in upstream band order."""
    cp = cupy()
    ids = pp.phonons.degenerate_ids[gp]
    for identifier in np.unique(ids):
        full = np.flatnonzero(ids == identifier)
        if len(full) > 1 and np.isin(full, pp.band_indices).all():
            selected = cp.asarray(np.flatnonzero(np.isin(pp.band_indices, full)))
            values[..., selected] = values[..., selected].mean(axis=-1, keepdims=True)
    return values


def occupations(frequency, temperature, cutoff):
    cp = cupy()
    if temperature <= 0:
        return cp.zeros_like(frequency)
    u = get_physical_units()
    x = cp.where(frequency > cutoff, frequency, 1.0) * u.THzToEv / (u.KB * temperature)
    return 1 / cp.expm1(x)


def linewidth_detail(engine, triplets, strength, g, temperature, occupation=None):
    """Unweighted per-triplet gamma in THz, with one fused CUDA kernel."""
    cp = cupy()
    if occupation is None:
        occupation = occupations(
            engine.frequencies, temperature, engine.pp.cutoff_frequency
        )
    result = cp.empty(strength.shape)
    kernel("scattering.cu", "linewidth")(
        ((result.size + 127) // 128,),
        (128,),
        (
            result,
            cp.ascontiguousarray(strength),
            g,
            engine.frequencies,
            occupation,
            cp.asarray(triplets),
            np.int32(len(triplets)),
            np.int32(strength.shape[1]),
            np.int32(engine.nb),
            np.float64(engine.pp.cutoff_frequency),
            np.float64(ImagSelfEnergy(engine.pp).unit_conversion_factor),
        ),
    )
    return result


def collision_layout(solver):
    """Return target GR points, rotations and one-q collision row shape."""
    collision, pp = solver._collision, solver._pp
    nb0, nb = len(pp.band_indices), solver.engine.nb
    if collision._is_reducible_collision_matrix:
        targets = np.arange(np.prod(pp.mesh_numbers))[:, None]
        return targets, np.eye(3)[None], (nb0, len(targets), nb), 1
    targets = collision._rot_grid_points
    return (
        targets,
        collision._rotations_cartesian,
        (nb0, 3, len(targets), nb, 3),
        3,
    )


def reduce_collision(strength, g, inv_sinh, triplets, unit):
    """归约两个散射方向，返回 (2, triplet, band0, band) 张量。"""
    cp = cupy()
    nt, nb0, nb = strength.shape[:3]
    reduced = cp.empty((2, nt, nb0, nb))
    kernel("scattering.cu", "collision_reduce")(
        ((reduced.size + 127) // 128,),
        (128,),
        (
            reduced,
            cp.ascontiguousarray(strength),
            g[2],
            inv_sinh,
            cp.asarray(triplets),
            np.int32(nt),
            np.int32(nb0),
            np.int32(nb),
            np.float64(unit),
        ),
    )
    return reduced


def add_collision_rows(
    row, reduced, triplet_map, swapped, rotations, *, start, num_cartesian, num_q
):
    """按 triplet 映射和笛卡尔旋转，将归约结果累加到 collision 行。"""
    nt, nb0, nb = reduced.shape[1:]
    ni, nr = triplet_map.shape[-2:]
    kernel("scattering.cu", "collision_rows")(
        ((row.size + 127) // 128,),
        (128,),
        (
            row,
            reduced,
            triplet_map,
            swapped,
            rotations,
            np.int32(nt),
            np.int32(nb0),
            np.int32(nb),
            np.int32(ni),
            np.int32(nr),
            np.int32(start),
            np.int32(num_cartesian),
            np.int32(num_q),
        ),
    )


def sigma_groups(engine, sigmas, row_bytes=0, reuse=True):
    """Bound extra sigma weights and collision rows to a quarter of free memory."""
    width = 1
    if reuse and len(sigmas) > 1 and engine.pp.constant_averaged_interaction is None:
        # Account for resident FC3 before choosing how many sigmas to retain.
        engine.fourier
        bands = engine.band_block if engine.streamed else len(engine.pp.band_indices)
        elements = engine.batch_size * bands * engine.nb**2
        per_sigma = 25 * elements + row_bytes  # Three FP64 channels and a byte mask.
        width = max(
            1, min(len(sigmas), cupy().cuda.runtime.memGetInfo()[0] // 4 // per_sigma)
        )
    for first in range(0, len(sigmas), width):
        yield range(first, min(first + width, len(sigmas)))


def interaction_batches(
    engine,
    sigmas,
    sigma_cutoff,
    collision=False,
    full=False,
    read_pp=False,
    filename=None,
    averaged=None,
):
    """Yield device weights and interactions without a full-triplet tensor."""
    cp = cupy()
    pp = engine.pp
    triplets = pp.get_triplets_at_q()[0]
    gp = triplets[0, 0]
    loaded = loaded_zero = None
    if read_pp:
        assert len(sigmas) == 1
        loaded, loaded_zero = read_pp_from_hdf5(
            pp.mesh_numbers,
            grid_point=int(gp),
            sigma=sigmas[0],
            sigma_cutoff=sigma_cutoff,
            filename=filename,
        )
    bands = pp.band_indices
    # An unbounded Gaussian makes the shared support dense; keep the GEMM path.
    dense = full or (
        (sigma_cutoff is None or sigma_cutoff <= 0)
        and any(sigma is not None for sigma in sigmas)
    )
    # A streamed batch shares one reciprocal FC3 across all its initial-band tiles.
    triplet_step = 1 if engine.streamed else engine.batch_size
    band_step = engine.band_block if engine.streamed else len(bands)
    for start in range(0, len(triplets), triplet_step):
        batch = slice(start, start + triplet_step)
        selected = triplets[batch]
        vertices = engine.tetrahedron.vertices(selected) if None in sigmas else None
        bounds = None
        tensor = None
        if engine.streamed:
            if vertices is not None:
                bounds = channel_bounds(engine.frequencies, vertices)
            if (
                loaded is None
                and averaged is None
                and pp.constant_averaged_interaction is None
            ):
                tensor = engine.fourier.components(selected[0])
        for first in range(0, len(bands), band_step):
            block = slice(first, first + band_step)
            ids = bands[block]
            integrations, zero = sigma_weights(
                pp,
                selected,
                engine.frequencies,
                engine.frequencies[gp, cp.asarray(ids)],
                sigmas,
                sigma_cutoff=sigma_cutoff,
                collision=collision,
                vertices=vertices,
                bounds=bounds,
            )
            if loaded is not None:
                if loaded_zero is not None and not collision:
                    if not cp.array_equal(zero, cp.asarray(loaded_zero[batch, block])):
                        raise ValueError("Inconsistency found in g_zero.")
                strength = cp.asarray(loaded[batch, block])
            elif averaged is not None:
                value = cp.asarray(averaged[block])[:, None, None]
                strength = cp.broadcast_to(value / np.prod(pp.mesh_numbers), zero.shape)
            elif pp.constant_averaged_interaction is not None:
                strength = cp.full(
                    zero.shape,
                    pp.constant_averaged_interaction / np.prod(pp.mesh_numbers),
                )
            elif engine.streamed:
                strength = engine.project_slice(
                    tensor, selected[0], ids, None if dense else zero[0]
                )[None]
                strength *= pp.unit_conversion_factor
            else:
                strength = engine.compute(selected, None if full else zero)
            shared = strength
            masked = (
                not full
                and loaded is None
                and averaged is None
                and pp.constant_averaged_interaction is None
                and not pp.symmetrize_fc3q
            )
            for j, (g, zero) in enumerate(integrations):
                strength = (
                    cp.where(zero, 0, shared) if masked and len(sigmas) > 1 else shared
                )
                yield j, start, selected, block, g, zero, strength
            del integrations, shared, strength
        del tensor


class GPUScatteringSolver(RTAScatteringSolver):
    """Upstream RTA solver contract with a shared CUDA interaction engine."""

    def __init__(self, *args, engine, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self._occupations = [
            occupations(engine.frequencies, t, engine.pp.cutoff_frequency)
            for t in self._temperatures
        ]

    @property
    def supports_rust_batching(self):
        return False

    def compute(self, grid_point, averaged_pp_interaction=None):
        cp = cupy()
        pp = self._pp
        pp.set_grid_point(grid_point)
        triplets, multiplicities = pp.get_triplets_at_q()[:2]
        ns, nt, nb0 = len(self._sigmas), len(self._temperatures), len(pp.band_indices)
        gamma = cp.zeros((ns, nt, nb0))
        if self._is_N_U:
            normal, umklapp = cp.zeros_like(gamma), cp.zeros_like(gamma)
        detail = None
        if self._is_gamma_detail:
            detail = np.zeros((nt, len(triplets), nb0, self.engine.nb, self.engine.nb))
        average = cp.zeros(nb0) if self._is_full_pp or self._use_const_ave_pp else None
        if self._use_ave_pp and averaged_pp_interaction is None:
            raise ValueError("use_ave_pp requires averaged_pp_interaction.")
        if self._store_pp:
            shape = (len(triplets), nb0, self.engine.nb, self.engine.nb)
            saved = np.empty(shape, dtype=np.float64)
            saved_zero = np.empty(shape, dtype=np.int8)
        for group in sigma_groups(
            self.engine, self._sigmas, reuse=not self._read_pp and not self._use_ave_pp
        ):
            for j, start, selected, block, g, zero, strength in interaction_batches(
                self.engine,
                [self._sigmas[i] for i in group],
                self._sigma_cutoff,
                full=self._is_full_pp,
                read_pp=self._read_pp,
                filename=self._pp_filename,
                averaged=averaged_pp_interaction if self._use_ave_pp else None,
            ):
                j += group.start
                end = start + len(selected)
                w = cp.asarray(multiplicities[start:end])
                if self._is_N_U:
                    is_normal = cp.asarray(
                        np.all(pp.bz_grid.addresses[selected].sum(axis=1) == 0, axis=1)
                    )
                if average is not None and j == 0:
                    average[block] += (strength.sum(axis=(2, 3)) * w[:, None]).sum(
                        axis=0
                    ) / self.engine.nb**2
                for k, temperature in enumerate(self._temperatures):
                    values = linewidth_detail(
                        self.engine,
                        selected,
                        strength,
                        g,
                        temperature,
                        self._occupations[k],
                    )
                    weighted = values.sum(axis=(2, 3)) * w[:, None]
                    gamma[j, k, block] += weighted.sum(axis=0)
                    if self._is_N_U:
                        normal[j, k, block] += (weighted * is_normal[:, None]).sum(
                            axis=0
                        )
                        umklapp[j, k, block] += (weighted * (~is_normal)[:, None]).sum(
                            axis=0
                        )
                    if detail is not None:
                        detail[k, start:end, block] = host(values)
                if self._store_pp and j == ns - 1:
                    saved[start:end, block] = host(strength)
                    saved_zero[start:end, block] = host(zero)
        if self._store_pp:
            pp.set_interaction_strength(saved, g_zero=saved_zero)
        average_initial(gamma, pp, grid_point)
        if self._is_N_U:
            average_initial(normal, pp, grid_point)
            average_initial(umklapp, pp, grid_point)
        self._gamma_N = host(normal) if self._is_N_U else None
        self._gamma_U = host(umklapp) if self._is_N_U else None
        self._gamma_detail_at_q = detail
        result = host(average) if average is not None else averaged_pp_interaction
        return ScatteringResult(gamma=host(gamma), averaged_pp_interaction=result)


class GPUCollisionSolver(LBTECollisionSolver):
    """GPU collision rows using upstream integer symmetry maps."""

    def __init__(self, *args, engine, retain_pp=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine
        self.retain_pp = retain_pp
        if not retain_pp:
            self._pp.delete_interaction_strength()
        self._occupations = [
            occupations(engine.frequencies, t, engine.pp.cutoff_frequency)
            for t in self._temperatures
        ]

        cp = cupy()
        u = get_physical_units()
        valid = engine.frequencies > engine.pp.cutoff_frequency
        f = cp.where(valid, engine.frequencies, 1)
        self._inv_sinh = [
            cp.where(valid, 1 / cp.sinh(f * u.THzToEv / (2 * u.KB * t)), 0)
            if t > 0
            else cp.zeros_like(f)
            for t in self._temperatures
        ]

    def compute(self, grid_point):
        cp = cupy()
        pp, collision = self._pp, self._collision
        collision.set_grid_point(grid_point)
        triplets, multiplicities = pp.get_triplets_at_q()[:2]
        gp2tp, _, swapped = collision._get_gp2tp_map()
        nb0, nb = len(pp.band_indices), self.engine.nb
        ns, nt = len(self._sigmas), len(self._temperatures)
        targets, rotations, rowshape, nc = collision_layout(self)
        gamma = cp.zeros((ns, nt, nb0))
        rows = np.zeros((ns, nt, *rowshape))
        average = cp.zeros(nb0) if self._is_full_pp else None
        unit = collision.unit_conversion_factor
        target_triplets = cp.asarray(gp2tp[targets])
        target_swapped = cp.asarray(swapped[targets])
        device_rotations = cp.asarray(rotations)
        if self.retain_pp:
            shape = (len(triplets), nb0, nb, nb)
            saved = np.empty(shape, dtype=np.float64)
            saved_zero = np.empty(shape, dtype=np.int8)
        for group in sigma_groups(
            self.engine,
            self._sigmas,
            row_bytes=nt * int(np.prod(rowshape)) * 8,
            reuse=not self._read_pp,
        ):
            row = cp.zeros((len(group), nt, *rowshape))
            for j, start, selected, block, g, zero, strength in interaction_batches(
                self.engine,
                [self._sigmas[i] for i in group],
                self._sigma_cutoff,
                collision=True,
                full=self._is_full_pp,
                read_pp=self._read_pp,
                filename=self._pp_filename,
            ):
                j += group.start
                end = start + len(selected)
                w = cp.asarray(multiplicities[start:end])
                if average is not None and j == 0:
                    average[block] += (strength.sum(axis=(2, 3)) * w[:, None]).sum(
                        axis=0
                    ) / nb**2
                for k, temperature in enumerate(self._temperatures):
                    values = linewidth_detail(
                        self.engine,
                        selected,
                        strength,
                        g,
                        temperature,
                        self._occupations[k],
                    )
                    gamma[j, k, block] += (values.sum(axis=(2, 3)) * w[:, None]).sum(
                        axis=0
                    )
                    if temperature <= 0:
                        continue
                    reduced = reduce_collision(
                        strength, g, self._inv_sinh[k], selected, unit
                    )
                    add_collision_rows(
                        row[j - group.start, k, block],
                        reduced,
                        target_triplets,
                        target_swapped,
                        device_rotations,
                        start=start,
                        num_cartesian=nc,
                        num_q=1,
                    )
                if self.retain_pp and j == ns - 1:
                    saved[start:end, block] = host(strength)
                    saved_zero[start:end, block] = host(zero)
            rows[group.start : group.stop] = host(row)
            del row
        if self.retain_pp:
            pp.set_interaction_strength(saved, g_zero=saved_zero)
        average_initial(gamma, pp, grid_point)
        return LBTECollisionResult(
            gamma=host(gamma),
            collision_row=rows,
            averaged_pp=None if average is None else host(average),
        )
