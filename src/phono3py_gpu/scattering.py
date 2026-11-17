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

from phono3py_gpu.backend import cupy, host
from phono3py_gpu.integration import weights


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


def linewidth_detail(engine, triplets, strength, g, temperature):
    """Unweighted per-triplet gamma in THz."""
    cp = cupy()
    tp = cp.asarray(triplets)
    f1 = engine.frequencies[tp[:, 1]]
    f2 = engine.frequencies[tp[:, 2]]
    n1 = occupations(f1, temperature, engine.pp.cutoff_frequency)
    n2 = occupations(f2, temperature, engine.pp.cutoff_frequency)
    factor = (n1[:, None, :, None] + n2[:, None, None, :] + 1) * g[0] + (
        n1[:, None, :, None] - n2[:, None, None, :]
    ) * g[1]
    valid = (f1[:, None, :, None] > engine.pp.cutoff_frequency) & (
        f2[:, None, None, :] > engine.pp.cutoff_frequency
    )
    return strength * factor * valid * ImagSelfEnergy(engine.pp).unit_conversion_factor


def interaction_batches(
    engine,
    sigma,
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
        loaded, loaded_zero = read_pp_from_hdf5(
            pp.mesh_numbers,
            grid_point=int(gp),
            sigma=sigma,
            sigma_cutoff=sigma_cutoff,
            filename=filename,
        )
    for start in range(0, len(triplets), engine.batch_size):
        selected = triplets[start : start + engine.batch_size]
        g, zero = weights(
            pp,
            selected,
            engine.frequencies,
            engine.frequencies[gp, cp.asarray(pp.band_indices)],
            sigma,
            sigma_cutoff,
            collision,
        )
        if loaded_zero is not None and not collision:
            expected_zero = loaded_zero[start : start + len(selected)]
            if not cp.array_equal(zero, cp.asarray(expected_zero)):
                raise ValueError("Inconsistency found in g_zero.")
        if loaded is not None:
            strength = cp.asarray(loaded[start : start + len(selected)])
        elif averaged is not None:
            shape = (len(selected), len(pp.band_indices), engine.nb, engine.nb)
            strength = cp.broadcast_to(
                cp.asarray(averaged)[None, :, None, None] / np.prod(pp.mesh_numbers),
                shape,
            )
        else:
            strength = engine.compute(selected, None if full else zero)
        yield start, selected, g, zero, strength


class GPUScatteringSolver(RTAScatteringSolver):
    """Upstream RTA solver contract with a shared CUDA interaction engine."""

    def __init__(self, *args, engine, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine

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
        normal, umklapp = cp.zeros_like(gamma), cp.zeros_like(gamma)
        detail = None
        if self._is_gamma_detail:
            detail = np.zeros((nt, len(triplets), nb0, self.engine.nb, self.engine.nb))
        average = cp.zeros(nb0) if self._is_full_pp or self._use_const_ave_pp else None
        if self._use_ave_pp and averaged_pp_interaction is None:
            raise ValueError("use_ave_pp requires averaged_pp_interaction.")
        for j, sigma in enumerate(self._sigmas):
            saved, saved_zero = [], []
            for start, selected, g, zero, strength in interaction_batches(
                self.engine,
                sigma,
                self._sigma_cutoff,
                full=self._is_full_pp,
                read_pp=self._read_pp,
                filename=self._pp_filename,
                averaged=averaged_pp_interaction if self._use_ave_pp else None,
            ):
                end = start + len(selected)
                w = cp.asarray(multiplicities[start:end])
                is_normal = cp.asarray(
                    np.all(pp.bz_grid.addresses[selected].sum(axis=1) == 0, axis=1)
                )
                if average is not None and j == 0:
                    average += (strength.sum(axis=(2, 3)) * w[:, None]).sum(
                        axis=0
                    ) / self.engine.nb**2
                for k, temperature in enumerate(self._temperatures):
                    values = linewidth_detail(
                        self.engine, selected, strength, g, temperature
                    )
                    weighted = values.sum(axis=(2, 3)) * w[:, None]
                    gamma[j, k] += weighted.sum(axis=0)
                    if self._is_N_U:
                        normal[j, k] += (weighted * is_normal[:, None]).sum(axis=0)
                        umklapp[j, k] += (weighted * (~is_normal)[:, None]).sum(axis=0)
                    if detail is not None:
                        detail[k, start:end] = host(values)
                if self._store_pp:
                    saved.append(host(strength))
                    saved_zero.append(host(zero))
            if self._store_pp:
                pp.set_interaction_strength(
                    np.concatenate(saved), g_zero=np.concatenate(saved_zero)
                )
        for array in (gamma, normal, umklapp):
            average_initial(array, pp, grid_point)
        self._gamma_N = host(normal) if self._is_N_U else None
        self._gamma_U = host(umklapp) if self._is_N_U else None
        self._gamma_detail_at_q = detail
        result = host(average) if average is not None else averaged_pp_interaction
        return ScatteringResult(gamma=host(gamma), averaged_pp_interaction=result)


class GPUCollisionSolver(LBTECollisionSolver):
    """GPU collision rows using upstream integer symmetry maps."""

    def __init__(self, *args, engine, **kwargs):
        super().__init__(*args, **kwargs)
        self.engine = engine

    def compute(self, grid_point):
        cp = cupy()
        pp, collision = self._pp, self._collision
        collision.set_grid_point(grid_point)
        triplets, multiplicities = pp.get_triplets_at_q()[:2]
        gp2tp, tp2s, swapped = collision._get_gp2tp_map()
        reducible = collision._is_reducible_collision_matrix
        nb0, nb = len(pp.band_indices), self.engine.nb
        ns, nt = len(self._sigmas), len(self._temperatures)
        if reducible:
            targets = np.arange(len(gp2tp))[:, None]
            rotations = np.eye(3)[None]
            rowshape = (nb0, len(targets), nb)
        else:
            targets = collision._rot_grid_points
            rotations = collision._rotations_cartesian
            rowshape = (nb0, 3, len(targets), nb, 3)
        gamma = cp.zeros((ns, nt, nb0))
        rows = np.zeros((ns, nt, *rowshape))
        average = cp.zeros(nb0) if self._is_full_pp else None
        unit = collision.unit_conversion_factor
        u = get_physical_units()
        for j, sigma in enumerate(self._sigmas):
            row = cp.zeros((nt, *rowshape))
            saved, saved_zero = [], []
            for start, selected, g, zero, strength in interaction_batches(
                self.engine,
                sigma,
                self._sigma_cutoff,
                collision=True,
                full=self._is_full_pp,
                read_pp=self._read_pp,
                filename=self._pp_filename,
            ):
                end = start + len(selected)
                w = cp.asarray(multiplicities[start:end])
                if average is not None and j == 0:
                    average += (strength.sum(axis=(2, 3)) * w[:, None]).sum(
                        axis=0
                    ) / nb**2
                out_i, out_r = np.nonzero(
                    (gp2tp[targets] >= start) & (gp2tp[targets] < end)
                )
                target = targets[out_i, out_r]
                ti = cp.asarray(gp2tp[target] - start)
                partner = cp.asarray(tp2s[target])
                sw = cp.asarray(swapped[target]).astype(bool)
                for k, temperature in enumerate(self._temperatures):
                    values = linewidth_detail(
                        self.engine, selected, strength, g, temperature
                    )
                    gamma[j, k] += (values.sum(axis=(2, 3)) * w[:, None]).sum(axis=0)
                    if temperature <= 0 or not len(target):
                        continue
                    f = self.engine.frequencies[partner]
                    valid = f > pp.cutoff_frequency
                    inv = cp.where(
                        valid,
                        1
                        / cp.sinh(
                            cp.where(valid, f, 1) * u.THzToEv / (2 * u.KB * temperature)
                        ),
                        0,
                    )
                    product = strength[ti] * g[2, ti]
                    product = cp.where(
                        sw[:, None, None, None], product.swapaxes(2, 3), product
                    )
                    reduced = (product * inv[:, None, None, :]).sum(axis=3) * unit
                    if reducible:
                        row[k, :, cp.asarray(out_i), :] = reduced
                    else:
                        for r in np.unique(out_r):
                            take = np.flatnonzero(out_r == r)
                            ids = cp.asarray(out_i[take])
                            block = (
                                reduced[cp.asarray(take), :, None, :, None]
                                * cp.asarray(rotations[r])[None, None, :, None, :]
                            )
                            row[k][:, :, ids, :, :] += block.transpose(1, 2, 0, 3, 4)
                saved.append(host(strength))
                saved_zero.append(host(zero))
            rows[j] = host(row)
            pp.set_interaction_strength(
                np.concatenate(saved), g_zero=np.concatenate(saved_zero)
            )
        average_initial(gamma, pp, grid_point)
        return LBTECollisionResult(
            gamma=host(gamma),
            collision_row=rows,
            averaged_pp=None if average is None else host(average),
        )
