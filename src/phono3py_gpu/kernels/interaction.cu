// Last mode contraction: exact integration zeros never need an amplitude.
extern "C" __global__ void masked_projection(
    double* out, const double2* partial, const double2* eigenvectors,
    const double* freq, const long long* triplets, const long long* bands,
    const signed char* zero, int nt, int nb0, int nb, double cutoff,
    long long s0, long long s1, long long s2, long long s3) {
    long long index = (long long)blockIdx.x*blockDim.x+threadIdx.x;
    if (index >= (long long)nt*nb0*nb*nb) return;
    int c = index % nb, b = (index/nb) % nb;
    int a = (index/(nb*nb)) % nb0, t = index/(nb0*nb*nb);
    out[index] = 0;
    if (zero[index]) return;
    long long q0 = triplets[t*3], q1 = triplets[t*3+1], q2 = triplets[t*3+2];
    double f0 = freq[q0*nb+bands[a]], f1 = freq[q1*nb+b], f2 = freq[q2*nb+c];
    if (f0 <= cutoff || f1 <= cutoff || f2 <= cutoff) return;
    long long base = t*s0+a*s1+b*s2;
    double re = 0, im = 0;
    for (int k=0; k<nb; ++k) {
        double2 x = partial[base+k*s3], y = eigenvectors[(q2*nb+k)*nb+c];
        re += x.x*y.x-x.y*y.y;
        im += x.x*y.y+x.y*y.x;
    }
    out[index] = (re*re+im*im)/(f0*f1*f2);
}

extern "C" __global__ void amplitude_squared(
    double* out, const double2* amplitude, const signed char* zero,
    const double* f0, const double* f1, const double* f2, const long long* bands,
    int nb, long long size, double cutoff) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;
    double x = f0[bands[i/(nb*nb)]], y = f1[(i/nb)%nb], z = f2[i%nb];
    if ((zero && zero[i]) || x <= cutoff || y <= cutoff || z <= cutoff) {
        out[i] = 0.;
    } else {
        double2 v = amplitude[i];
        out[i] = (v.x*v.x + v.y*v.y)/(x*y*z);
    }
}
