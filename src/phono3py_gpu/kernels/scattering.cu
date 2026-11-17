// FP64 three-phonon reductions in the upstream band and channel order.
extern "C" __global__ void linewidth(
    double* out, const double* strength, const double* g,
    const double* freq, const double* occupation, const long long* triplets,
    int nt, int nb0, int nb, double cutoff, double unit) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)nt * nb0 * nb * nb;
    if (index >= size) return;
    int k = index % nb, j = (index / nb) % nb;
    int t = index / (nb0 * nb * nb);
    long long q1 = triplets[3*t+1]*nb+j, q2 = triplets[3*t+2]*nb+k;
    double value = 0;
    if (freq[q1] > cutoff && freq[q2] > cutoff) {
        double n1 = occupation[q1], n2 = occupation[q2];
        value = strength[index] * ((n1+n2+1)*g[index] +
                                  (n1-n2)*g[size+index]) * unit;
    }
    out[index] = value;
}

// Reduce both q1/q2 orientations once per irreducible triplet. The symmetry
// rotations reuse these values instead of replicating a four-index tensor.
extern "C" __global__ void collision_reduce(
    double* out, const double* strength, const double* g,
    const double* inv_sinh, const long long* triplets,
    int nt, int nb0, int nb, double unit) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)nt * nb0 * nb;
    if (index >= 2*size) return;
    int j = index % nb, b = (index / nb) % nb0;
    int t = (index / (nb*nb0)) % nt, swapped = index / size;
    long long partner = triplets[3*t+(swapped ? 1 : 2)];
    long long base = ((long long)t*nb0+b)*nb*nb;
    double value = 0;
    for (int k=0; k<nb; ++k) {
        long long p = base + (swapped ? k*nb+j : j*nb+k);
        value += strength[p] * g[p] * inv_sinh[partner*nb+k];
    }
    out[index] = value * unit;
}

// One thread owns each output element: no atomics or duplicated rotations.
extern "C" __global__ void collision_rows(
    double* row, const double* reduced, const long long* triplet_index,
    const signed char* swapped, const double* rotations,
    int nt, int nb0, int nb, int ni, int nr, int start, int nc, int nq) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)nq*nb0*nc*ni*nb*nc;
    if (index >= size) return;
    long long rest = index;
    int c = rest % nc; rest /= nc;
    int j = rest % nb; rest /= nb;
    int i = rest % ni; rest /= ni;
    int a = rest % nc, b = (rest / nc) % nb0;
    int q = rest / (nc*nb0);
    double value = 0;
    for (int r=0; r<nr; ++r) {
        long long target = ((long long)q*ni+i)*nr+r;
        long long t = triplet_index[target]-start;
        if (t < 0 || t >= nt) continue;
        long long p = (((long long)swapped[target]*nt+t)*nb0+b)*nb+j;
        value += reduced[p] * (nc == 1 ? 1.0 : rotations[r*9+a*3+c]);
    }
    row[index] += value;
}

// Each q and initial branch owns its sum; interleaved triplets retain their
// original order within q. No floating-point atomics are needed.
extern "C" __global__ void sum_by_q(
    double* out, const double* values, const long long* multiplicity,
    const long long* owner, const signed char* normal,
    int nt, int nb0, int nq, int channels) {
    int index = blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= channels*nq*nb0) return;
    int b = index % nb0, q = (index/nb0) % nq;
    int channel = index/(nb0*nq);
    double value = 0;
    for (int t=0; t<nt; ++t) {
        if (owner[t] != q) continue;
        if (channel == 1 && !normal[t]) continue;
        if (channel == 2 && normal[t]) continue;
        value += values[(long long)t*nb0+b]*multiplicity[t];
    }
    out[index] += value;
}
