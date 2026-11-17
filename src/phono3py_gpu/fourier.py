# SPDX-License-Identifier: BSD-3-Clause
"""FP64 CUDA FC3 Fourier transform for the shared GPU interaction engine.

The phase, multiplicity, pre-phase and r0-average conventions match phonors
0.6.0. Each warp sums one atom triplet's 27 Cartesian components in the
original supercell-pair order. No fast-math or frequency approximation is used.
"""

from __future__ import annotations

import time

import numpy as np

_KERNELS = r"""
extern "C" __global__ void phases(
    const double* q, const double* sv, const long long* multi,
    double2* out, int na, int ns, int nq) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= nq * na * ns) return;
    int s = index % ns, a = (index / ns) % na, g = index / (ns * na);
    long long count = multi[(s * na + a) * 2];
    long long start = multi[(s * na + a) * 2 + 1];
    double re = 0., im = 0.;
    for (long long v = start; v < start + count; ++v) {
        double angle = 6.2831853071795864769 *
            (q[g*3]*sv[v*3] + q[g*3+1]*sv[v*3+1] + q[g*3+2]*sv[v*3+2]);
        double si, co;
        sincos(angle, &si, &co);
        re += co; im += si;
    }
    out[index] = make_double2(re / count, im / count);
}
extern "C" __global__ void transform(
    const double* fc, const long long* offsets, const int* pairs,
    const signed char* shortest, const double2* phase,
    const long long* triplets, double2* out,
    int na, int ns, int batch, int average) {
    int cart = threadIdx.x % 32;
    int atom = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    int t = blockIdx.y, leg = blockIdx.z, blocks = na*na*na;
    if (cart >= 27 || atom >= blocks) return;
    int anchor = atom / (na*na);
    int qa = (leg == 1) ? 0 : 1;
    int qb = (leg == 2) ? 0 : 2;
    long long p1 = (triplets[t*3+qa]*na + anchor)*ns;
    long long p2 = (triplets[t*3+qb]*na + anchor)*ns;
    double re = 0., im = 0.;
    for (long long p = offsets[atom]; p < offsets[atom+1]; ++p) {
        if (average && leg > 0 && shortest[p]) continue;
        double2 x = phase[p1 + pairs[p*2]], y = phase[p2 + pairs[p*2+1]];
        double weight = (average && leg == 0 && shortest[p]) ? 3. : 1.;
        double f = fc[p*32+cart];
        re += ((x.x*y.x - x.y*y.y)*weight)*f;
        im += ((x.x*y.y + x.y*y.x)*weight)*f;
    }
    out[((long long)leg*batch+t)*blocks*27+atom*27+cart] = make_double2(re, im);
}
__device__ double2 premultiply(double2 x, const double* q,
    const double* origin, const long long* triplet, int a) {
    double angle = 0.;
    for (int d = 0; d < 3; ++d)
        angle += origin[a*3+d] *
            (q[triplet[0]*3+d]+q[triplet[1]*3+d]+q[triplet[2]*3+d]);
    double si, co;
    sincos(angle * 6.2831853071795864769, &si, &co);
    return make_double2(x.x*co-x.y*si, x.x*si+x.y*co);
}
extern "C" __global__ void assemble(
    const double2* legs, const double* q, const double* origin,
    const long long* triplets, double2* out, int na, int batch, int average) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    int size = na*na*na*27;
    if (index >= (long long)batch*size) return;
    int t = index / size, elem = index % size, atom = elem / 27;
    int i = atom/(na*na), j = (atom/na)%na, k = atom%na;
    int cart = elem%27, l = cart/9, m = (cart/3)%3, n = cart%3;
    double2 v = premultiply(legs[index], q, origin, triplets+t*3, i);
    if (average) {
        long long stride = (long long)batch*size;
        double2 w = premultiply(legs[stride+t*size+((j*na+i)*na+k)*27+m*9+l*3+n],
            q, origin, triplets+t*3, j);
        double2 z = premultiply(legs[2*stride+t*size+((k*na+j)*na+i)*27+n*9+m*3+l],
            q, origin, triplets+t*3, k);
        v = make_double2(((v.x+w.x)+z.x)/3., ((v.y+w.y)+z.y)/3.);
    }
    out[index] = v;
}
"""


class GPUFC3Transform:
    """Cache a snapshot of FC3 and BZ phases on the current CuPy device.

    Input FC3/mappings must remain unchanged for this object's lifetime.
    ``transform`` returns atom-first complex128 tensors on the device. Calls are
    synchronous and bounded to ``batch_size`` triplets in the interaction path.
    """

    def __init__(
        self,
        fc3,
        nonzero,
        svecs,
        multi,
        p2s,
        s2p,
        all_shortest,
        qpoints,
        make_r0_average=True,
        batch_size=128,
    ):
        import cupy as cp

        if fc3.dtype != np.float64:
            raise TypeError("GPU FC3 transform requires float64 force constants.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        self.batch_size = batch_size
        self.na, self.ns = len(p2s), len(s2p)
        self.average = bool(make_r0_average)
        self.pool = cp.cuda.MemoryPool()
        self.device = cp.cuda.Device().id
        self.timings = dict(transform=0.0, projection=0.0, calls=0)
        start = time.perf_counter()
        compact = fc3.shape[0] != fc3.shape[1]
        groups = [np.flatnonzero(s2p == p) for p in p2s]
        offsets = [0]
        values, pairs, flags = [], [], []
        # Keep native j-then-k summation order; omit only explicit nonzero masks.
        for a in range(self.na):
            row = a if compact else p2s[a]
            for js in groups:
                for ks in groups:
                    jj, kk = np.meshgrid(js, ks, indexing="ij")
                    jj, kk = jj.ravel(), kk.ravel()
                    keep = nonzero[row, jj, kk] != 0
                    jj, kk = jj[keep], kk[keep]
                    values.append(fc3[row, jj, kk].reshape(-1, 27))
                    pairs.append(np.column_stack((jj, kk)))
                    flags.append(all_shortest[a, jj, kk])
                    offsets.append(offsets[-1] + len(jj))
        packed = np.zeros((offsets[-1], 32), dtype="double")
        packed[:, :27] = np.concatenate(values)
        with cp.cuda.using_allocator(self.pool.malloc):
            self.fc = cp.asarray(packed)
            self.offsets = cp.asarray(offsets, dtype=cp.int64)
            self.pairs = cp.asarray(np.concatenate(pairs), dtype=cp.int32)
            self.shortest = cp.asarray(np.concatenate(flags), dtype=cp.int8)
            self.qpoints = cp.asarray(qpoints, dtype=cp.float64, order="C")
            self.origins = cp.asarray(svecs[multi[p2s, 0, 1]], order="C")
            self.phase = cp.empty((len(qpoints), self.na, self.ns), cp.complex128)
            # FP64 FMA is disabled to retain the native summation convention.
            module = cp.RawModule(code=_KERNELS, options=("--fmad=false",))
            phase_kernel = module.get_function("phases")
            self.kernel = module.get_function("transform")
            self.assemble = module.get_function("assemble")
            sv = cp.asarray(svecs, dtype=cp.float64, order="C")
            multiplicity = cp.asarray(multi, dtype=cp.int64, order="C")
            phase_kernel(
                ((self.phase.size + 255) // 256,),
                (256,),
                (
                    self.qpoints,
                    sv,
                    multiplicity,
                    self.phase,
                    np.int32(self.na),
                    np.int32(self.ns),
                    np.int32(len(qpoints)),
                ),
            )
            cp.cuda.get_current_stream().synchronize()
        self.setup_seconds = time.perf_counter() - start

    def transform(self, triplets):
        """Transform a batch, returning a device tensor."""
        import cupy as cp

        if cp.cuda.Device().id != self.device:
            raise RuntimeError("GPU FC3 cache belongs to a different CUDA device.")
        batch = len(triplets)
        shape = (batch, self.na, self.na, self.na, 3, 3, 3)
        if not batch:
            return cp.empty(shape, dtype="complex128")
        with cp.cuda.using_allocator(self.pool.malloc):
            ids = cp.asarray(triplets, dtype=cp.int64, order="C")
            legs = cp.empty((3 if self.average else 1, *shape), cp.complex128)
            result = cp.empty(shape, cp.complex128)
            self.kernel(
                ((self.na**3 + 3) // 4, batch, len(legs)),
                (128,),
                (
                    self.fc,
                    self.offsets,
                    self.pairs,
                    self.shortest,
                    self.phase,
                    ids,
                    legs,
                    np.int32(self.na),
                    np.int32(self.ns),
                    np.int32(batch),
                    np.int32(self.average),
                ),
            )
            self.assemble(
                ((result.size + 255) // 256,),
                (256,),
                (
                    legs,
                    self.qpoints,
                    self.origins,
                    ids,
                    result,
                    np.int32(self.na),
                    np.int32(batch),
                    np.int32(self.average),
                ),
            )
            return result

    @classmethod
    def from_interaction(cls, interaction, **kwargs):
        """Create a cache from a phono3py Interaction object."""
        pp = interaction
        return cls(
            pp.fc3,
            pp._fc3_nonzero_indices,
            pp._svecs,
            pp._multi,
            pp._p2s,
            pp._s2p,
            pp._all_shortest,
            pp.bz_grid.addresses @ pp.bz_grid.QDinv.T,
            make_r0_average=pp._make_r0_average,
            **kwargs,
        )
