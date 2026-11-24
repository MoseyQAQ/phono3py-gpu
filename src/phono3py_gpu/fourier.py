# SPDX-License-Identifier: BSD-3-Clause
"""FP64 CUDA FC3 Fourier transform for the shared GPU interaction engine.

The phase, multiplicity, pre-phase and r0-average conventions match phonors
0.6.0. Each warp sums one atom triplet's 27 Cartesian components in the
original supercell-pair order. No fast-math or frequency approximation is used.
"""

from __future__ import annotations

import os
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
// Each block reuses a coalesced FC3 tile across several triplets. The inner
// supercell-pair accumulation order remains identical to transform().
extern "C" __global__ void transform_tiled(
    const double* fc, const long long* offsets, const int* pairs,
    const signed char* shortest, const double2* phase,
    const long long* triplets, double2* out,
    int na, int ns, int batch, int average) {
    __shared__ double f[32*32];
    __shared__ int js[32], ks[32];
    __shared__ signed char flags[32];
    int cart = threadIdx.x, lane = threadIdx.y*32+cart;
    int atom = blockIdx.x, t = blockIdx.y*blockDim.y+threadIdx.y, leg = blockIdx.z;
    int anchor = atom/(na*na), qa = leg == 1 ? 0 : 1, qb = leg == 2 ? 0 : 2;
    bool valid = t < batch && cart < 27;
    long long p1 = 0, p2 = 0;
    if (valid) {
        p1 = (triplets[t*3+qa]*na+anchor)*ns;
        p2 = (triplets[t*3+qb]*na+anchor)*ns;
    }
    double re = 0., im = 0.;
    long long begin = offsets[atom], end = offsets[atom+1];
    for (long long base = begin; base < end; base += 32) {
        int count = (end-base < 32) ? int(end-base) : 32;
        for (int i=lane; i<count*32; i+=blockDim.y*32)
            f[i] = fc[base*32+i];
        if (lane < count) {
            js[lane] = pairs[(base+lane)*2];
            ks[lane] = pairs[(base+lane)*2+1];
            flags[lane] = shortest[base+lane];
        }
        __syncthreads();
        if (valid) {
            for (int p=0; p<count; ++p) {
                if (average && leg > 0 && flags[p]) continue;
                double2 x = phase[p1+js[p]], y = phase[p2+ks[p]];
                double w = average && leg == 0 && flags[p] ? 3. : 1.;
                double value = f[p*32+cart];
                re += ((x.x*y.x-x.y*y.y)*w)*value;
                im += ((x.x*y.y+x.y*y.x)*w)*value;
            }
        }
        __syncthreads();
    }
    if (valid)
        out[((long long)leg*batch+t)*na*na*na*27+atom*27+cart] = make_double2(re, im);
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
// Evaluate each requested r0 permutation directly in the final component layout.
__device__ double2 component_leg(
    const double* fc, const long long* offsets, const int* pairs,
    const signed char* shortest, const double2* phase, const double2 pre,
    int na, int ns, int i, int j, int k, int cart, int qj, int qk,
    int leg, int average) {
    long long atom = ((long long)i*na+j)*na+k;
    const double2* pj = phase + ((long long)qj*na+i)*ns;
    const double2* pk = phase + ((long long)qk*na+i)*ns;
    double re = 0., im = 0.;
    for (long long p=offsets[atom]; p<offsets[atom+1]; ++p) {
        if (average && leg && shortest[p]) continue;
        double2 x = pj[pairs[2*p]], y = pk[pairs[2*p+1]];
        double w = (average && !leg && shortest[p]) ? 3. : 1.;
        double v = fc[p*32+cart];
        re += ((x.x*y.x-x.y*y.y)*w)*v;
        im += ((x.x*y.y+x.y*y.x)*w)*v;
    }
    return make_double2(re*pre.x-im*pre.y, re*pre.y+im*pre.x);
}
extern "C" __global__ void transform_components(
    const double* fc, const long long* offsets, const int* pairs,
    const signed char* shortest, const double2* phase, const long long* tp,
    const double2* pre, double2* out, int na, int ns, int average) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    int nb=3*na;
    if (index >= (long long)nb*nb*nb) return;
    int x=index/(nb*nb), y=(index/nb)%nb, z=index%nb;
    int i=x/3, j=y/3, k=z/3, l=x%3, m=y%3, n=z%3;
    double2 v=component_leg(fc,offsets,pairs,shortest,phase,pre[i],na,ns,
                            i,j,k,9*l+3*m+n,tp[1],tp[2],0,average);
    if (average) {
        double2 w=component_leg(fc,offsets,pairs,shortest,phase,pre[j],na,ns,
                                j,i,k,9*m+3*l+n,tp[0],tp[2],1,1);
        double2 u=component_leg(fc,offsets,pairs,shortest,phase,pre[k],na,ns,
                                k,j,i,9*n+3*m+l,tp[1],tp[0],2,1);
        v=make_double2(((v.x+w.x)+u.x)/3., ((v.y+w.y)+u.y)/3.);
    }
    out[index]=v;
}
// Reuse each FC3 block across triplets; phase products are the right GEMM operand.
extern "C" __global__ void phase_products(
    const int* pairs, const signed char* shortest, const double2* phase,
    const long long* triplets, double* real, double* imag,
    int na, int ns, int npair, int tile, int start, int count,
    int leg, int average) {
    long long index = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    long long size = (long long)na*na*na*npair*tile;
    if (index >= size) return;
    int t = index % tile;
    long long p = index / tile;
    int atom = p / npair, anchor = atom / (na*na);
    real[index] = imag[index] = 0.;
    if (t >= count || (average && leg > 0 && shortest[p])) return;
    int qa = leg == 1 ? 0 : 1, qb = leg == 2 ? 0 : 2;
    long long p1 = (triplets[(start+t)*3+qa]*na+anchor)*ns;
    long long p2 = (triplets[(start+t)*3+qb]*na+anchor)*ns;
    double2 x = phase[p1+pairs[p*2]], y = phase[p2+pairs[p*2+1]];
    double w = average && leg == 0 && shortest[p] ? 3. : 1.;
    real[index] = (x.x*y.x-x.y*y.y)*w;
    imag[index] = (x.x*y.y+x.y*y.x)*w;
}
extern "C" __global__ void store_gemm(
    const double* real, const double* imag, double2* legs,
    int atoms, int batch, int tile, int start, int count, int leg) {
    long long index = (long long)blockIdx.x*blockDim.x + threadIdx.x;
    if (index >= (long long)atoms*27*count) return;
    int t = index % count, cart = (index/count)%27, atom = index/(count*27);
    long long src = ((long long)atom*32+cart)*tile+t;
    long long dst = (((long long)leg*batch+start+t)*atoms+atom)*27+cart;
    legs[dst] = make_double2(real[src], imag[src]);
}
"""


class GPUFC3Transform:
    """Cache a snapshot of FC3 and BZ phases on the current CuPy device.

    Input FC3/mappings must remain unchanged for this object's lifetime.
    ``transform`` returns atom-first complex128 tensors on the device.
    Triplet batches are bounded by the caller.
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
    ):
        import cupy as cp

        if fc3.dtype != np.float64:
            raise TypeError("GPU FC3 transform requires float64 force constants.")
        self.na, self.ns = len(p2s), len(s2p)
        self.average = bool(make_r0_average)
        self.pool = cp.cuda.MemoryPool()
        self.device = cp.cuda.Device().id
        start = time.perf_counter()
        compact = fc3.shape[0] != fc3.shape[1]
        groups = [np.flatnonzero(s2p == p) for p in p2s]
        # Pure translations give every primitive atom the same number of images.
        # Flatten one anchor in (j atom, k atom, j image, k image) order.
        images = np.asarray(groups, dtype=np.int32)
        js, ks = np.broadcast_arrays(images[:, None, :, None], images[None, :, None, :])
        js, ks = js.ravel(), ks.ravel()
        rows = np.arange(self.na) if compact else p2s
        counts = np.empty((self.na, self.na**2), dtype=np.int64)
        for a, row in enumerate(rows):
            counts[a] = np.count_nonzero(
                nonzero[row, js, ks].reshape(self.na**2, -1), axis=1
            )
        offsets = np.empty(counts.size + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(counts.ravel(), out=offsets[1:])
        self.uniform_pairs = (
            int(counts.flat[0]) if np.all(counts == counts.flat[0]) else 0
        )
        self.method = os.environ.get("PHONO3PY_GPU_FOURIER", "tiled")
        if self.method not in ("reference", "gemm", "tiled"):
            raise ValueError("PHONO3PY_GPU_FOURIER must be reference, tiled or gemm.")
        if self.method == "gemm" and not self.uniform_pairs:
            raise ValueError("GEMM Fourier requires uniform FC3 pair blocks.")
        self.tile = int(os.environ.get("PHONO3PY_GPU_FOURIER_TILE", "16"))
        if self.tile < 1:
            raise ValueError("PHONO3PY_GPU_FOURIER_TILE must be positive.")
        if self.method == "tiled" and self.tile > 32:
            raise ValueError("Tiled Fourier supports at most 32 triplets per block.")
        with cp.cuda.using_allocator(self.pool.malloc):
            n_pairs = int(offsets[-1])
            self.fc = cp.empty((n_pairs, 32), dtype=cp.float64)
            self.pairs = cp.empty((n_pairs, 2), dtype=cp.int32)
            self.shortest = cp.empty(n_pairs, dtype=cp.int8)
            for a, row in enumerate(rows):
                block = slice(offsets[a * self.na**2], offsets[(a + 1) * self.na**2])
                keep = nonzero[row, js, ks] != 0
                j, k = js[keep], ks[keep]
                packed = np.zeros((len(j), 32), dtype=np.float64)
                packed[:, :27] = fc3[row, j, k].reshape(-1, 27)
                self.fc[block].set(packed)
                self.pairs[block].set(np.column_stack((j, k)))
                self.shortest[block].set(
                    np.asarray(all_shortest[a, j, k], dtype=np.int8)
                )
            self.offsets = cp.asarray(offsets, dtype=cp.int64)
            self.qpoints = cp.asarray(qpoints, dtype=cp.float64, order="C")
            self.origins = cp.asarray(svecs[multi[p2s, 0, 1]], order="C")
            self.phase = cp.empty((len(qpoints), self.na, self.ns), cp.complex128)
            # FP64 FMA is disabled to retain the native summation convention.
            module = cp.RawModule(code=_KERNELS, options=("--fmad=false",))
            phase_kernel = module.get_function("phases")
            self.components_kernel = module.get_function("transform_components")
            self.kernel = module.get_function("transform")
            self.tiled_kernel = module.get_function("transform_tiled")
            self.assemble = module.get_function("assemble")
            self.phase_products = module.get_function("phase_products")
            self.store_gemm = module.get_function("store_gemm")
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

    def components(self, triplet):
        """One reciprocal FC3 in component order, including all requested origins."""
        import cupy as cp

        nb = 3 * self.na
        tensor = cp.empty((nb, nb, nb), dtype=cp.complex128)
        tp = cp.asarray(triplet, dtype=cp.int64)
        prephase = cp.exp((2j * np.pi) * (self.origins @ self.qpoints[tp].sum(axis=0)))
        self.components_kernel(
            ((tensor.size + 255) // 256,),
            (256,),
            (
                self.fc,
                self.offsets,
                self.pairs,
                self.shortest,
                self.phase,
                tp,
                prephase,
                tensor,
                np.int32(self.na),
                np.int32(self.ns),
                np.int32(self.average),
            ),
        )
        return tensor

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
            if self.method == "gemm":
                self._gemm(ids, legs, batch)
            elif self.method == "tiled":
                self.tiled_kernel(
                    (self.na**3, (batch + self.tile - 1) // self.tile, len(legs)),
                    (32, self.tile),
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
            else:
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

    def _gemm(self, ids, legs, batch):
        """FP64 batched GEMM；固定大小相位块避免显存随尾批形状累积。"""
        import cupy as cp

        atoms, tile, npair = self.na**3, self.tile, self.uniform_pairs
        fc = self.fc.reshape(atoms, npair, 32).swapaxes(1, 2)
        real = cp.empty((atoms, npair, tile), dtype=cp.float64)
        imag = cp.empty_like(real)
        for start in range(0, batch, tile):
            count = min(tile, batch - start)
            for leg in range(len(legs)):
                self.phase_products(
                    ((real.size + 255) // 256,),
                    (256,),
                    (
                        self.pairs,
                        self.shortest,
                        self.phase,
                        ids,
                        real,
                        imag,
                        np.int32(self.na),
                        np.int32(self.ns),
                        np.int32(npair),
                        np.int32(tile),
                        np.int32(start),
                        np.int32(count),
                        np.int32(leg),
                        np.int32(self.average),
                    ),
                )
                r, i = cp.matmul(fc, real), cp.matmul(fc, imag)
                self.store_gemm(
                    ((atoms * 27 * count + 255) // 256,),
                    (256,),
                    (
                        r,
                        i,
                        legs,
                        np.int32(atoms),
                        np.int32(batch),
                        np.int32(tile),
                        np.int32(start),
                        np.int32(count),
                        np.int32(leg),
                    ),
                )

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
