"""FP64 dynamical matrices and analytical Cartesian derivatives on CUDA."""

import numpy as np
from phonopy.harmonic.dynamical_matrix import DynamicalMatrixGL, DynamicalMatrixWang

from phono3py_gpu.backend import batch_size, cupy, eigh


class Harmonic:
    """Cache a snapshot of FC2, shortest vectors and NAC data on one device."""

    def __init__(self, dm):
        cp = cupy()
        self.dm = dm
        self.primitive = dm.primitive
        self.na = len(dm.primitive)
        self.nb = 3 * self.na
        self.cell = cp.asarray(dm.primitive.cell)
        self.recip = cp.linalg.inv(self.cell)
        self.mass = cp.asarray(dm.normalization_masses)
        self.p2s = dm.primitive.p2s_map
        self.s2p = dm.primitive.s2p_map
        self.groups = [np.flatnonzero(self.s2p == p) for p in self.p2s]
        self.svecs, self.multi = dm.primitive.get_smallest_vectors()
        self.sv = cp.asarray(self.svecs)
        self.count = cp.asarray(self.multi[:, :, 0])
        self.start = cp.asarray(self.multi[:, :, 1])
        self.is_gl = isinstance(dm, DynamicalMatrixGL)
        self.is_wang = isinstance(dm, DynamicalMatrixWang)
        if self.is_gl and dm.short_range_force_constants is None:
            dm.make_Gonze_nac_dataset()
        fc = dm.short_range_force_constants if self.is_gl else dm.force_constants
        self.fc = cp.asarray(fc if fc.shape[0] != fc.shape[1] else fc[self.p2s])
        self.group_indices = cp.asarray(np.asarray(self.groups))
        self.fc_blocks = self.fc[:, self.group_indices]
        if self.is_gl or self.is_wang:
            self.born = cp.asarray(dm.born)
            self.eps = cp.asarray(dm.dielectric_constant)
        if self.is_gl:
            self.gs = cp.asarray(dm._G_list)
            phase = cp.exp(
                2j * cp.pi * (self.gs @ cp.asarray(dm.primitive.positions).T)
            )
            self.gphase = phase[:, :, None] * phase[:, None, :].conj()

    def _phase(self, q, derivative=False):
        cp = cupy()
        values = cp.exp(2j * cp.pi * (q @ self.sv.T))
        if derivative:
            values = values[:, None, :] * (2j * cp.pi * (self.sv @ self.cell).T)[None]
        # Prefix sums permit arbitrary packed vector order and multiplicities.
        prefix = cp.concatenate(
            (
                cp.zeros((*values.shape[:-1], 1), values.dtype),
                cp.cumsum(values, axis=-1),
            ),
            axis=-1,
        )
        return (
            prefix[..., self.start + self.count] - prefix[..., self.start]
        ) / self.count

    def matrices(self, qpoints, derivative=False, q_direction=None):
        """Return D(q), or dD/dq_cart, with q as the first axis."""
        cp = cupy()
        q = cp.asarray(qpoints, dtype=cp.float64).reshape(-1, 3)
        n = len(q)
        shape = (n, 3, self.nb, self.nb) if derivative else (n, self.nb, self.nb)
        phases = self._phase(q, derivative)
        base_phases = self._phase(q) if derivative and self.is_wang else phases
        nac, dnac = self._wang(q, q_direction) if self.is_wang else (None, None)
        mass = cp.sqrt(self.mass[:, None] * self.mass[None, :])
        if derivative:
            grouped = phases[:, :, self.group_indices, :].transpose(0, 1, 4, 2, 3)
            block = cp.einsum("qdijs,ijsab->qdiajb", grouped, self.fc_blocks)
            if nac is not None:
                extra = grouped.sum(axis=-1)[..., None, None] * nac[:, None]
                plain = base_phases[:, self.group_indices, :].transpose(0, 3, 1, 2)
                extra += plain.sum(axis=-1)[:, None, ..., None, None] * dnac
                block += extra.transpose(0, 1, 2, 4, 3, 5)
            block /= mass[None, None, :, None, :, None]
        else:
            grouped = phases[:, self.group_indices, :].transpose(0, 3, 1, 2)
            block = cp.einsum("qijs,ijsab->qiajb", grouped, self.fc_blocks)
            if nac is not None:
                extra = grouped.sum(axis=-1)[..., None, None] * nac
                block += extra.transpose(0, 1, 3, 2, 4)
            block /= mass[None, :, None, :, None]
        result = block.reshape(shape)
        if self.is_gl:
            result += self._gonze(q, derivative, q_direction)
        return (result + result.swapaxes(-1, -2).conj()) * 0.5

    def _wang(self, q, q_direction):
        cp = cupy()
        qc = q @ self.recip.T
        if q_direction is not None:
            qc = cp.broadcast_to(cp.asarray(q_direction) @ self.recip.T, qc.shape)
        valid = cp.linalg.norm(qc, axis=1) > self.dm.Q_DIRECTION_TOLERANCE
        b = cp.einsum("qi,ij,qj->q", qc, self.eps, qc, optimize=True)
        b = cp.where(valid, b, 1)
        a = cp.einsum("qb,iba->qia", qc, self.born)
        outer = cp.einsum("qia,qjb->qijab", a, a)
        scale = self.dm.nac_factor / (len(self.s2p) // self.na)
        nac = outer / b[:, None, None, None, None]
        dnac = cp.empty((len(q), 3, self.na, self.na, 3, 3))
        for d in range(3):
            top = cp.einsum("ia,qjb->qijab", self.born[:, d], a) + cp.einsum(
                "qia,jb->qijab", a, self.born[:, d]
            )
            db = 2 * (qc @ self.eps)[:, d]
            dnac[:, d] = (
                top / b[:, None, None, None, None]
                - outer * (db / b**2)[:, None, None, None, None]
            )
        return nac * (valid * scale)[:, None, None, None, None], dnac * (valid * scale)[
            :, None, None, None, None, None
        ]

    def _gonze(self, q, derivative, q_direction):
        cp = cupy()
        dm = self.dm
        # G summation is batched independently of q and atom pairs.
        Q = self.gs[None] + (q @ self.recip.T)[:, None]
        valid = cp.linalg.norm(Q, axis=-1) >= dm.Q_DIRECTION_TOLERANCE
        zero = ~valid
        if q_direction is not None:
            direction = cp.asarray(q_direction) @ self.recip.T
            if float(cp.linalg.norm(direction)) > 0:
                Q = cp.where(zero[..., None], direction, Q)
                valid = cp.ones_like(valid)
        v = Q @ self.eps
        b = cp.sum(Q * v, axis=-1)
        b = cp.where(valid, b, 1)
        c = cp.exp(-b / (4 * dm._Lambda**2))
        if q_direction is not None and not derivative:
            c = cp.where(zero, 1.0, c)
        c *= valid
        if derivative:
            eye = cp.eye(3)
            t = (
                eye[None, None, :, :, None] * Q[:, :, None, None, :]
                + Q[:, :, None, :, None] * eye[None, None, :, None, :]
            ) / b[..., None, None, None]
            k = v * (2 / b**2 + 1 / (2 * dm._Lambda**2 * b))[..., None]
            t -= Q[:, :, None, :, None] * Q[:, :, None, None, :] * k[..., :, None, None]
            t *= c[..., None, None, None]
            out = cp.einsum(
                "qgdbc,gij,iba,jce->qdiaje",
                t,
                self.gphase,
                self.born,
                self.born,
                optimize=True,
            )
            out = out.reshape(len(q), 3, self.na, 3, self.na, 3)
        else:
            t = Q[..., :, None] * Q[..., None, :] * (c / b)[..., None, None]
            out = cp.einsum(
                "qgbc,gij,iba,jce->qiaje",
                t,
                self.gphase,
                self.born,
                self.born,
                optimize=True,
            )
            if not dm._with_full_terms:
                for i in range(self.na):
                    out[:, i, :, i, :] -= cp.asarray(dm._dd_q0[i])
        out *= dm.nac_factor
        if dm._with_full_terms:
            # Full Ewald real-space terms use the same stored H tensor.
            real = self._gonze_real(q, derivative)
            out += real
            if not derivative:
                drift = cp.asarray(
                    dm._dd_q0 + dm._dd_limiting * self.na + dm._dd_real_q0
                )
                for i in range(self.na):
                    out[:, i, :, i, :] += cp.asarray(dm._dd_limiting) - drift[i]
        mass = 1 / cp.sqrt(self.mass[:, None] * self.mass[None, :])
        out *= (
            mass[None, :, None, :, None]
            if not derivative
            else mass[None, None, :, None, :, None]
        )
        return out.reshape(
            (len(q), 3, self.nb, self.nb) if derivative else (len(q), self.nb, self.nb)
        )

    def _gonze_real(self, q, derivative):
        cp = cupy()
        dm = self.dm
        shape = (
            (len(q), 3, self.na, 3, self.na, 3)
            if derivative
            else (len(q), self.na, 3, self.na, 3)
        )
        result = cp.zeros(shape, dtype=cp.complex128)
        phase = cp.exp(2j * cp.pi * (q @ self.sv.T))
        h = -(dm._Lambda**3) * cp.asarray(dm._H) * cp.linalg.det(self.eps) ** (-0.5)
        for s in range(len(self.s2p)):
            i = int(dm._s2pp_map[s])
            for j in range(self.na):
                count, start = self.multi[s, j]
                val = phase[:, start, None, None] * h[:, :, start]
                val = (val + val.swapaxes(-1, -2).conj()) / (2 * count)
                if derivative:
                    # Differentiate before imposing the Hermitian block convention.
                    raw = phase[:, start, None, None] * h[:, :, start]
                    dval = (
                        raw[:, None]
                        * (2j * cp.pi * (self.sv[start] @ self.cell))[
                            None, :, None, None
                        ]
                    )
                    result[:, :, i, :, j, :] += (
                        dval + dval.swapaxes(-1, -2).conj()
                    ) / (2 * count)
                else:
                    result[:, i, :, j, :] += val
        return result

    def solve(self, qpoints, factor):
        """Solve a bounded batch and return frequencies and column eigenvectors."""
        cp = cupy()
        qpoints = np.asarray(qpoints).reshape(-1, 3)
        n = len(qpoints)
        freq = cp.empty((n, self.nb))
        vectors = cp.empty((n, self.nb, self.nb), dtype=cp.complex128)
        step = batch_size(16 * self.nb**2 * 6 + 16 * len(self.svecs), maximum=256)
        for start in range(0, n, step):
            vals, vecs = eigh(self.matrices(qpoints[start : start + step]))
            freq[start : start + step] = cp.sign(vals) * cp.sqrt(cp.abs(vals)) * factor
            vectors[start : start + step] = vecs
        return freq, vectors


def prepare_phonons(pp):
    """Upload canonical upstream phonons without changing their band convention.

    Tetrahedron interpolation with unaveraged degenerate bands is basis
    dependent. Frequencies exactly on tetrahedron vertices also require the
    same rounding convention. The inexpensive mesh eigensolutions therefore
    remain upstream; FC3 contraction, scattering, velocities, and LBTE solves
    use CUDA. The standalone Harmonic.solve method supplies GPU eigensolutions
    for callers that do not require this canonical convention.
    """
    cp = cupy()
    pp.nac_q_direction = None
    pp.run_phonon_solver_at_gamma()
    if not pp.phonon_all_done:
        pp.run_phonon_solver()
    return (
        Harmonic(pp.dynamical_matrix),
        cp.asarray(pp.phonons.frequencies),
        cp.asarray(pp.phonons.eigenvectors),
    )
