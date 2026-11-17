"""CUDA group velocities and the finite-difference Wigner velocity operator."""

import numpy as np
from phono3py.conductivity.grid_point_data import HeatCapacityResult, VelocityResult
from phono3py.conductivity.heat_capacity_solvers import ModeHeatCapacitySolver
from phono3py.conductivity.utils import VOIGT_INDEX_PAIRS
from phono3py.conductivity.velocity_solvers import GroupVelocitySolver
from phonopy.phonon.degeneracy import degenerate_sets
from phonopy.phonon.grid import (
    get_grid_points_by_rotations,
    get_qpoints_from_bz_grid_points,
)
from phonopy.physical_units import get_physical_units

from phono3py_gpu.backend import cupy, eigh, host


class GPUVelocitySolver(GroupVelocitySolver):
    """Bulk velocity protocol; finite differences and eigensolutions stay on CUDA."""

    def __init__(self, *args, harmonic, wte=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.harmonic = harmonic
        self.wte = wte
        self.produces_vm_by_vm = wte
        self.produces_gv_by_gv = not wte
        self.velocity_operator = None

    def _derivatives(self, q, length, square_root=False):
        cp = cupy()
        results = []
        for direction in np.eye(3):
            dq = self._pp.primitive.cell @ direction * length
            zero_acoustic = np.linalg.norm(q) < np.linalg.norm(dq)
            if square_root and zero_acoustic and self.harmonic.is_gl:
                # WTE specifies the NAC approach direction even when |q| is
                # below the dynamical matrix's Gamma tolerance.
                plus, minus = [
                    self.harmonic.matrices(
                        [point], q_direction=point / np.linalg.norm(point)
                    )[0]
                    for point in (q + dq, q - dq)
                ]
            else:
                plus, minus = self.harmonic.matrices([q + dq, q - dq])
            if square_root:
                plus = self._sqrt(plus, zero_acoustic)
                minus = self._sqrt(minus, zero_acoustic)
            results.append((plus - minus) / (2 * length))
        return cp.stack(results)

    def _sqrt(self, dm, zero_acoustic):
        cp = cupy()
        _, e = eigh(dm)
        if zero_acoustic:
            e[:, :3] = 0
        # Match WTE's complex square root, including unstable modes.
        roots = cp.sqrt(cp.diag(e.conj().T @ dm @ e))
        return (e * roots) @ e.conj().T

    def compute(self, grid_points):
        cp = cupy()
        pp = self._pp
        qs = get_qpoints_from_bz_grid_points(np.asarray(grid_points), pp.bz_grid)
        bands = cp.asarray(pp.band_indices)
        rotations = cp.asarray(self._rotations_cartesian)
        first, second = np.asarray(VOIGT_INDEX_PAIRS).T
        first, second = cp.asarray(first), cp.asarray(second)
        velocities, products, operators = [], [], []
        for gp, q in zip(grid_points, qs, strict=True):
            if self.wte:
                # Supplied per-band linewidths require WTE's canonical basis,
                # including the arbitrary rotation within degenerate spaces.
                self._pp.dynamical_matrix.run(q)
                val_host, e_host = np.linalg.eigh(
                    self._pp.dynamical_matrix.dynamical_matrix
                )
                val, e = cp.asarray(val_host), cp.asarray(e_host)
            else:
                val, e = eigh(self.harmonic.matrices([q])[0])
            freq = cp.sign(val) * cp.sqrt(cp.abs(val)) * pp.frequency_factor_to_THz
            if self.wte:
                derivative = self._derivatives(q, self._gv_delta_q or 5e-6, True)
                op = (
                    e.conj().T[None] @ derivative @ e[None]
                ) * pp.frequency_factor_to_THz
                op = (op + op.swapaxes(-1, -2).conj()) / 2
                gv = cp.diagonal(op, axis1=1, axis2=2).real.T.copy()
                for deg in degenerate_sets(host(freq)):
                    if len(deg) > 1:
                        ids = cp.asarray(deg)
                        gv[ids] = eigh(op[:, ids[:, None], ids[None, :]])[0].T
                vector = op.transpose(1, 2, 0)[bands]
                operators.append(host(vector))
            else:
                derivative = (
                    self.harmonic.matrices([q], derivative=True)[0]
                    if self._gv_delta_q is None
                    else self._derivatives(q, self._gv_delta_q)
                )
                direction = cp.asarray([1.0, 2.0, 3.0]) / np.sqrt(14)
                lifting = cp.einsum("d,dij->ij", direction, derivative)
                if self._gv_delta_q is not None:
                    dq = pp.primitive.cell @ host(direction) * self._gv_delta_q
                    mats = self.harmonic.matrices([q + dq, q - dq])
                    lifting = (mats[0] - mats[1]) / (2 * self._gv_delta_q)
                rotated = e.copy()
                for deg in degenerate_sets(host(freq)):
                    if len(deg) == 1:
                        continue
                    ids = cp.asarray(deg)
                    block = e[:, ids]
                    _, rotation = eigh(block.conj().T @ lifting @ block)
                    rotated[:, ids] = block @ rotation
                gv = cp.einsum("ij,dij->jd", rotated.conj(), derivative @ rotated).real
                gv *= cp.where(
                    freq > 1e-4,
                    pp.frequency_factor_to_THz**2
                    / (2 * cp.where(freq > 1e-4, freq, 1)),
                    0,
                )[:, None]
                rmean = self._velocity_obj._get_r_cart_mean(q - np.rint(q))
                gv = gv @ cp.asarray(rmean).T
                vector = gv[bands]
            velocities.append(host(gv[bands]))
            if self._is_kappa_star:
                mapped = get_grid_points_by_rotations(
                    int(gp), pp.bz_grid, with_surface=False
                )
                multi = len(mapped) // len(np.unique(mapped))
            else:
                multi = 1
            v = cp.einsum("...a,rba->r...b", vector, rotations)
            outer = (v[..., first] * v[..., second].conj()).sum(axis=0) / multi
            products.append(host(outer))
        gv = np.asarray(velocities).reshape(len(grid_points), len(bands), 3)
        if self.wte:
            self.velocity_operator = np.asarray(operators)
            return VelocityResult(
                group_velocities=gv,
                vm_by_vm=np.asarray(products),
                extra={"velocity_operator": self.velocity_operator},
            )
        return VelocityResult(group_velocities=gv, gv_by_gv=np.asarray(products))


class GPUHeatCapacitySolver(ModeHeatCapacitySolver):
    """Einstein heat capacities, keeping upstream cutoffs and zero-T convention."""

    def compute(self, grid_points):
        cp = cupy()
        u = get_physical_units()
        f = cp.asarray(
            self._pp.phonons.frequencies[grid_points][:, self._pp.band_indices]
        )
        t = cp.asarray(self._temperatures)[:, None, None]
        x = f[None] * u.THzToEv / (u.KB * cp.where(t > 0, t, 1))
        # Stable form of x**2 * exp(x) / expm1(x)**2.
        z = cp.exp(-x)
        cv = u.KB * x**2 * z / cp.expm1(-x) ** 2
        cv = cp.where(
            (t > 0) & (f[None] > self._pp.cutoff_frequency) & cp.isfinite(cv), cv, 0
        )
        return HeatCapacityResult(heat_capacities=host(cv))
