"""GPU dense LBTE solve with upstream assembly and result containers."""

import numpy as np
from phono3py.conductivity.collision_matrix_kernel import (
    IrreducibleCollisionMatrixKernel,
    ReducibleCollisionMatrixKernel,
)
from phonopy.physical_units import get_physical_units

from phono3py_gpu.backend import cupy, eigh, host
from phono3py_gpu.integration import average_axis


class GPUCollisionMixin:
    """Device algebra shared by the irreducible and full-mesh kernels."""

    def _average_collision_matrix_by_degeneracy(self):
        cp = cupy()
        ir = isinstance(self, IrreducibleCollisionMatrixKernel)
        groups = list(
            self._iter_ir_degenerate_ids() if ir else self._iter_mesh_degenerate_ids()
        )
        for s, t in self._iter_sigma_temp():
            matrix = cp.asarray(self._collision_matrix[s, t])
            for i, ids in groups:
                average_axis(matrix[i], ids, 0)
            for i, ids in groups:
                if ir:
                    average_axis(matrix[:, :, :, i], ids, 3)
                else:
                    average_axis(matrix[:, :, i], ids, 2)
            self._collision_matrix[s, t] = host(matrix)

    def _symmetrize_collision_matrix(self):
        cp = cupy()
        n = self._get_symmetrization_size()
        for s, t in self._iter_sigma_temp():
            matrix = cp.asarray(self._collision_matrix[s, t].reshape(n, n))
            self._collision_matrix[s, t] = host((matrix + matrix.T) / 2).reshape(
                self._collision_matrix.shape[2:]
            )

    def _iter_kappa_at_sigmas(self, weights):
        cp = cupy()
        n = self._get_symmetrization_size()
        solver = self._get_colmat_solver()
        for s, t in self._iter_sigma_temp():
            if self._kappa_settings.temperatures[t] <= 0:
                continue
            self._set_kappa_RTA_at_step(s, t, weights)
            matrix = cp.asarray(self._collision_matrix[s, t].reshape(n, n))
            if solver == 7:
                self._gpu_inverse = cp.linalg.pinv(matrix, rcond=1e-15)
                stored = self._gpu_inverse
            else:
                self._gpu_eigenvalues, self._gpu_eigenvectors = eigh(matrix)
                self._collision_eigenvalues[s, t] = host(self._gpu_eigenvalues)
                stored = (
                    self._gpu_eigenvectors.T
                    if solver in (1, 2, 4, 5)
                    else self._gpu_eigenvectors
                )
            self._collision_matrix[s, t] = host(stored).reshape(
                self._collision_matrix.shape[2:]
            )
            self._set_kappa_at_step(s, t, weights)
            num_mesh = int(np.prod(self._kappa_settings.mesh_numbers))
            self._kappa[s, t] /= num_mesh
            self._kappa_RTA[s, t] /= num_mesh
            yield s, t
        self._gpu_eigenvectors = self._gpu_eigenvalues = self._gpu_inverse = None

    def _solve_Y(self, solver, v, X, i_sigma, i_temp):
        cp = cupy()
        ir = isinstance(self, IrreducibleCollisionMatrixKernel)
        x = cp.asarray(X.ravel() if ir else X)
        if solver == 7:
            y = self._gpu_inverse @ x
        else:
            w, v = self._gpu_eigenvalues, self._gpu_eigenvectors
            valid = (cp.abs(w) if self._pinv_method == 0 else w) > self._pinv_cutoff
            inverse = cp.where(valid, 1 / cp.where(valid, w, 1), 0)
            projection = v.T @ x
            y = v @ (projection * (inverse if ir else inverse[:, None]))
            if solver == 6:
                pinv = (v * inverse) @ v.T
                self._collision_matrix[i_sigma, i_temp] = host(pinv).reshape(
                    self._collision_matrix.shape[2:]
                )
        return host(y.reshape(-1, 3))

    def _get_X(self, i_temp, weights):
        cp = cupy()
        t = self._kappa_settings.temperatures[i_temp]
        f = cp.asarray(self._get_X_frequencies())
        u = get_physical_units()
        if t <= 0:
            return np.zeros_like(self._gv.reshape(-1, 3))
        valid = f > self._kappa_settings.cutoff_frequency
        x = cp.where(valid, f, 1) * u.THzToEv / (2 * u.KB * t)
        scale = cp.where(valid, f * u.THzToEv / (cp.sinh(x) * 4 * u.KB * t * t), 0)
        return host(
            (
                cp.asarray(self._gv)
                * scale[..., None]
                * cp.asarray(weights)[:, None, None]
            ).reshape(-1, 3)
        )

    def _set_mode_kappa(self, mode_kappa, X, Y, num_grid_points, i_sigma, i_temp):
        cp = cupy()
        nb = self._frequencies.shape[1]
        x, y = (cp.asarray(a).reshape(num_grid_points, nb, 3) for a in (X, Y))
        tensor = cp.zeros((num_grid_points, nb, 3, 3))
        for rotation in self._rotations_cartesian:
            r = cp.asarray(rotation)
            vx, vy = x @ r.T, y @ r.T
            tensor += vx[..., :, None] * vy[..., None, :]
        tensor += tensor.swapaxes(-1, -2).copy()
        pairs = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))
        result = cp.stack([tensor[..., a, b] for a, b in pairs], axis=-1)
        # Preserve the upstream Gamma selection and output ordering.
        gamma = np.all(
            self._kappa_settings.bz_grid.addresses[:num_grid_points] == 0, axis=1
        )
        result[cp.asarray(gamma), :3] = 0
        t = self._kappa_settings.temperatures[i_temp]
        result *= (
            self._kappa_settings.conversion_factor * get_physical_units().KB * t * t
        )
        mode_kappa[i_sigma, i_temp] = host(result)


class GPUIrreducibleCollisionKernel(
    GPUCollisionMixin, IrreducibleCollisionMatrixKernel
):
    def _get_collision_weights(self):
        cp = cupy()
        weights = self._get_ir_weights()
        w = cp.asarray(weights)
        for s, t in self._iter_sigma_temp():
            matrix = cp.asarray(self._collision_matrix[s, t])
            matrix *= (
                w[:, None, None, None, None, None] * w[None, None, None, :, None, None]
            )
            self._collision_matrix[s, t] = host(matrix)
        return weights


class GPUReducibleCollisionKernel(GPUCollisionMixin, ReducibleCollisionMatrixKernel):
    def _expand_collisions(self, ir_gr_grid_points, rot_grid_points):
        """Expand collision rows on device using upstream integer rotation maps."""
        cp = cupy()
        rotations = cp.asarray(rot_grid_points)
        for s, t in self._iter_sigma_temp():
            matrix = cp.asarray(self._collision_matrix[s, t])
            for gp in ir_gr_grid_points:
                multiplicity = np.count_nonzero(rot_grid_points[:, gp] == gp)
                source = matrix[gp].copy() / multiplicity
                matrix[gp] = 0
                for r, mapping in enumerate(rot_grid_points):
                    # Each mapping is a permutation, so this scatter has no races.
                    matrix[mapping[gp]][:, rotations[r], :] += source
            self._collision_matrix[s, t] = host(matrix)


def create_collision_matrix_kernel(*, rot_grid_points=None, **kwargs):
    if rot_grid_points is None:
        return GPUReducibleCollisionKernel(**kwargs)
    return GPUIrreducibleCollisionKernel(rot_grid_points=rot_grid_points, **kwargs)
