// Each thread owns one symmetric pair in either row- or column-major storage.
extern "C" __global__ void symmetrize_collision(double* matrix, int n) {
    long long index = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= (long long)n * n) return;
    int row = index / n, column = index % n;
    if (row >= column) return;
    long long transposed = (long long)column * n + row;
    double mean = (matrix[index] + matrix[transposed]) / 2;
    matrix[index] = mean;
    matrix[transposed] = mean;
}
