// Accumulate Tamura overlaps only at nonzero integration weights.
extern "C" __global__ void weighted_overlap(
    double* weights, const double2* e0, const double2* eigenvectors,
    const double* frequencies, const long long* grid_points,
    const double* mass_variances, int nq, int nb0, int nb, double cutoff) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)nq * nb0 * nb) return;
    int b = index % nb, a = (index / nb) % nb0;
    long long q = grid_points[index / (nb0 * nb)];
    double weight = weights[index];
    weights[index] = 0.;
    if (weight == 0. || frequencies[q * nb + b] < cutoff) return;
    double strength = 0.;
    for (int atom = 0; atom < nb / 3; ++atom) {
        double variance = mass_variances[atom];
        if (variance == 0.) continue;
        double re = 0., im = 0.;
        for (int cart = 0; cart < 3; ++cart) {
            int row = atom * 3 + cart;
            double2 x = e0[row * nb0 + a];
            double2 y = eigenvectors[(q * nb + row) * nb + b];
            re += x.x * y.x + x.y * y.y;
            im += x.x * y.y - x.y * y.x;
        }
        strength += variance * (re * re + im * im);
    }
    weights[index] = weight * strength;
}
