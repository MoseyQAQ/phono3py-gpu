/* SPDX-License-Identifier: BSD-3-Clause
 * Phonopy tetrahedron formulas, ported to CUDA; see LICENSE.phonopy. */
typedef long long int64_t;
#define THM_EPSILON 1e-10
#define warning_print(...)
static __device__ int64_t sort_omegas(double v[4]);
static __device__ double norm_squared_d3(const double a[3]);
static __device__ void multiply_matrix_vector_dl3(double v[3], const double a[3][3],
                                       const int64_t b[3]);
static __device__ double _f(const int64_t n, const int64_t m, const double omega,
                 const double vertices_omegas[4]);
static __device__ double _J(const int64_t i, const int64_t ci, const double omega,
                 const double vertices_omegas[4]);
static __device__ double _I(const int64_t i, const int64_t ci, const double omega,
                 const double vertices_omegas[4]);
static __device__ double _n(const int64_t i, const double omega,
                 const double vertices_omegas[4]);
static __device__ double _g(const int64_t i, const double omega,
                 const double vertices_omegas[4]);
static __device__ double _n_0(void);
static __device__ double _n_1(const double omega, const double vertices_omegas[4]);
static __device__ double _n_2(const double omega, const double vertices_omegas[4]);
static __device__ double _n_3(const double omega, const double vertices_omegas[4]);
static __device__ double _n_4(void);
static __device__ double _g_0(void);
static __device__ double _g_1(const double omega, const double vertices_omegas[4]);
static __device__ double _g_2(const double omega, const double vertices_omegas[4]);
static __device__ double _g_3(const double omega, const double vertices_omegas[4]);
static __device__ double _g_4(void);
static __device__ double _J_0(void);
static __device__ double _J_10(const double omega, const double vertices_omegas[4]);
static __device__ double _J_11(const double omega, const double vertices_omegas[4]);
static __device__ double _J_12(const double omega, const double vertices_omegas[4]);
static __device__ double _J_13(const double omega, const double vertices_omegas[4]);
static __device__ double _J_20(const double omega, const double vertices_omegas[4]);
static __device__ double _J_21(const double omega, const double vertices_omegas[4]);
static __device__ double _J_22(const double omega, const double vertices_omegas[4]);
static __device__ double _J_23(const double omega, const double vertices_omegas[4]);
static __device__ double _J_30(const double omega, const double vertices_omegas[4]);
static __device__ double _J_31(const double omega, const double vertices_omegas[4]);
static __device__ double _J_32(const double omega, const double vertices_omegas[4]);
static __device__ double _J_33(const double omega, const double vertices_omegas[4]);
static __device__ double _J_4(void);
static __device__ double _I_0(void);
static __device__ double _I_10(const double omega, const double vertices_omegas[4]);
static __device__ double _I_11(const double omega, const double vertices_omegas[4]);
static __device__ double _I_12(const double omega, const double vertices_omegas[4]);
static __device__ double _I_13(const double omega, const double vertices_omegas[4]);
static __device__ double _I_20(const double omega, const double vertices_omegas[4]);
static __device__ double _I_21(const double omega, const double vertices_omegas[4]);
static __device__ double _I_22(const double omega, const double vertices_omegas[4]);
static __device__ double _I_23(const double omega, const double vertices_omegas[4]);
static __device__ double _I_30(const double omega, const double vertices_omegas[4]);
static __device__ double _I_31(const double omega, const double vertices_omegas[4]);
static __device__ double _I_32(const double omega, const double vertices_omegas[4]);
static __device__ double _I_33(const double omega, const double vertices_omegas[4]);
static __device__ double _I_4(void);

static __device__ int64_t sort_omegas(double v[4]) {
    int64_t i;
    double w[4];

    i = 0;

    if (v[0] > v[1]) {
        w[0] = v[1];
        w[1] = v[0];
        i = 1;
    } else {
        w[0] = v[0];
        w[1] = v[1];
    }

    if (v[2] > v[3]) {
        w[2] = v[3];
        w[3] = v[2];
    } else {
        w[2] = v[2];
        w[3] = v[3];
    }

    if (w[0] > w[2]) {
        v[0] = w[2];
        v[1] = w[0];
        if (i == 0) {
            i = 4;
        }
    } else {
        v[0] = w[0];
        v[1] = w[2];
    }

    if (w[1] > w[3]) {
        v[3] = w[1];
        v[2] = w[3];
        if (i == 1) {
            i = 3;
        }
    } else {
        v[3] = w[3];
        v[2] = w[1];
        if (i == 1) {
            i = 5;
        }
    }

    if (v[1] > v[2]) {
        w[1] = v[1];
        v[1] = v[2];
        v[2] = w[1];
        if (i == 4) {
            i = 2;
        }
        if (i == 5) {
            i = 1;
        }
    } else {
        if (i == 4) {
            i = 1;
        }
        if (i == 5) {
            i = 2;
        }
    }
    return i;
}

static __device__ double _f(const int64_t n, const int64_t m, const double omega,
                 const double vertices_omegas[4]) {
    double delta;
    delta = vertices_omegas[n] - vertices_omegas[m];

#ifdef THM_EPSILON
    if (fabs(delta) < THM_EPSILON) {
        return 0;
    }
#endif

    return ((omega - vertices_omegas[m]) / delta);
}

static __device__ double _J(const int64_t i, const int64_t ci, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _J_0();
        case 1:
            switch (ci) {
                case 0:
                    return _J_10(omega, vertices_omegas);
                case 1:
                    return _J_11(omega, vertices_omegas);
                case 2:
                    return _J_12(omega, vertices_omegas);
                case 3:
                    return _J_13(omega, vertices_omegas);
            }
        case 2:
            switch (ci) {
                case 0:
                    return _J_20(omega, vertices_omegas);
                case 1:
                    return _J_21(omega, vertices_omegas);
                case 2:
                    return _J_22(omega, vertices_omegas);
                case 3:
                    return _J_23(omega, vertices_omegas);
            }
        case 3:
            switch (ci) {
                case 0:
                    return _J_30(omega, vertices_omegas);
                case 1:
                    return _J_31(omega, vertices_omegas);
                case 2:
                    return _J_32(omega, vertices_omegas);
                case 3:
                    return _J_33(omega, vertices_omegas);
            }
        case 4:
            return _J_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" J is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static __device__ double _I(const int64_t i, const int64_t ci, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _I_0();
        case 1:
            switch (ci) {
                case 0:
                    return _I_10(omega, vertices_omegas);
                case 1:
                    return _I_11(omega, vertices_omegas);
                case 2:
                    return _I_12(omega, vertices_omegas);
                case 3:
                    return _I_13(omega, vertices_omegas);
            }
        case 2:
            switch (ci) {
                case 0:
                    return _I_20(omega, vertices_omegas);
                case 1:
                    return _I_21(omega, vertices_omegas);
                case 2:
                    return _I_22(omega, vertices_omegas);
                case 3:
                    return _I_23(omega, vertices_omegas);
            }
        case 3:
            switch (ci) {
                case 0:
                    return _I_30(omega, vertices_omegas);
                case 1:
                    return _I_31(omega, vertices_omegas);
                case 2:
                    return _I_32(omega, vertices_omegas);
                case 3:
                    return _I_33(omega, vertices_omegas);
            }
        case 4:
            return _I_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" I is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static __device__ double _n(const int64_t i, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _n_0();
        case 1:
            return _n_1(omega, vertices_omegas);
        case 2:
            return _n_2(omega, vertices_omegas);
        case 3:
            return _n_3(omega, vertices_omegas);
        case 4:
            return _n_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" n is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

static __device__ double _g(const int64_t i, const double omega,
                 const double vertices_omegas[4]) {
    switch (i) {
        case 0:
            return _g_0();
        case 1:
            return _g_1(omega, vertices_omegas);
        case 2:
            return _g_2(omega, vertices_omegas);
        case 3:
            return _g_3(omega, vertices_omegas);
        case 4:
            return _g_4();
    }

    warning_print("******* Warning *******\n");
    warning_print(" g is something wrong. \n");
    warning_print("******* Warning *******\n");
    warning_print("(line %d, %s).\n", __LINE__, __FILE__);

    return 0;
}

/* omega < omega1 */
static __device__ double _n_0(void) { return 0.0; }

/* omega1 < omega < omega2 */
static __device__ double _n_1(const double omega, const double vertices_omegas[4]) {
    return (_f(1, 0, omega, vertices_omegas) *
            _f(2, 0, omega, vertices_omegas) *
            _f(3, 0, omega, vertices_omegas));
}

/* omega2 < omega < omega3 */
static __device__ double _n_2(const double omega, const double vertices_omegas[4]) {
    return (
        _f(3, 1, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas) +
        _f(3, 0, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas) *
            _f(2, 1, omega, vertices_omegas) +
        _f(3, 0, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) *
            _f(1, 2, omega, vertices_omegas));
}

/* omega2 < omega < omega3 */
static __device__ double _n_3(const double omega, const double vertices_omegas[4]) {
    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas));
}

/* omega4 < omega */
static __device__ double _n_4(void) { return 1.0; }

/* omega < omega1 */
static __device__ double _g_0(void) { return 0.0; }

/* omega1 < omega < omega2 */
static __device__ double _g_1(const double omega, const double vertices_omegas[4]) {
    return (3 * _f(1, 0, omega, vertices_omegas) *
            _f(2, 0, omega, vertices_omegas) /
            (vertices_omegas[3] - vertices_omegas[0]));
}

/* omega2 < omega < omega3 */
static __device__ double _g_2(const double omega, const double vertices_omegas[4]) {
    return (
        3 / (vertices_omegas[3] - vertices_omegas[0]) *
        (_f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
         _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas)));
}

/* omega3 < omega < omega4 */
static __device__ double _g_3(const double omega, const double vertices_omegas[4]) {
    return (3 * _f(1, 3, omega, vertices_omegas) *
            _f(2, 3, omega, vertices_omegas) /
            (vertices_omegas[3] - vertices_omegas[0]));
}

/* omega4 < omega */
static __device__ double _g_4(void) { return 0.0; }

static __device__ double _J_0(void) { return 0.0; }

static __device__ double _J_10(const double omega, const double vertices_omegas[4]) {
    return (1.0 + _f(0, 1, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) +
            _f(0, 3, omega, vertices_omegas)) /
           4;
}

static __device__ double _J_11(const double omega, const double vertices_omegas[4]) {
    return _f(1, 0, omega, vertices_omegas) / 4;
}

static __device__ double _J_12(const double omega, const double vertices_omegas[4]) {
    return _f(2, 0, omega, vertices_omegas) / 4;
}

static __device__ double _J_13(const double omega, const double vertices_omegas[4]) {
    return _f(3, 0, omega, vertices_omegas) / 4;
}

static __device__ double _J_20(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (1.0 + _f(0, 3, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                (1.0 + _f(0, 3, omega, vertices_omegas) +
                 _f(0, 2, omega, vertices_omegas))) /
           4 / n;
}

static __device__ double _J_21(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (1.0 + _f(1, 3, omega, vertices_omegas) +
                 _f(1, 2, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (_f(1, 3, omega, vertices_omegas) +
                 _f(1, 2, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas)) /
           4 / n;
}

static __device__ double _J_22(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                (_f(2, 1, omega, vertices_omegas) +
                 _f(2, 0, omega, vertices_omegas))) /
           4 / n;
}

static __device__ double _J_23(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_2(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 1, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                _f(3, 1, omega, vertices_omegas) +
            _f(3, 0, omega, vertices_omegas) *
                _f(1, 3, omega, vertices_omegas) *
                _f(2, 1, omega, vertices_omegas) *
                (_f(3, 1, omega, vertices_omegas) +
                 _f(3, 0, omega, vertices_omegas)) +
            _f(3, 0, omega, vertices_omegas) *
                _f(2, 0, omega, vertices_omegas) *
                _f(1, 2, omega, vertices_omegas) *
                _f(3, 0, omega, vertices_omegas)) /
           4 / n;
}

static __device__ double _J_30(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static __device__ double _J_31(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static __device__ double _J_32(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas)) /
           4 / n;
}

static __device__ double _J_33(const double omega, const double vertices_omegas[4]) {
    double n;
    n = _n_3(omega, vertices_omegas);

#ifdef THM_EPSILON
    if (n < THM_EPSILON) {
        return 0;
    }
#endif

    return (1.0 - _f(0, 3, omega, vertices_omegas) *
                      _f(1, 3, omega, vertices_omegas) *
                      _f(2, 3, omega, vertices_omegas) *
                      (1.0 + _f(3, 0, omega, vertices_omegas) +
                       _f(3, 1, omega, vertices_omegas) +
                       _f(3, 2, omega, vertices_omegas))) /
           4 / n;
}

static __device__ double _J_4(void) { return 0.25; }

static __device__ double _I_0(void) { return 0.0; }

static __device__ double _I_10(const double omega, const double vertices_omegas[4]) {
    return (_f(0, 1, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) +
            _f(0, 3, omega, vertices_omegas)) /
           3;
}

static __device__ double _I_11(const double omega, const double vertices_omegas[4]) {
    return _f(1, 0, omega, vertices_omegas) / 3;
}

static __device__ double _I_12(const double omega, const double vertices_omegas[4]) {
    return _f(2, 0, omega, vertices_omegas) / 3;
}

static __device__ double _I_13(const double omega, const double vertices_omegas[4]) {
    return _f(3, 0, omega, vertices_omegas) / 3;
}

static __device__ double _I_20(const double omega, const double vertices_omegas[4]) {
    double f12_20, g;
    f12_20 =
        _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas);
    g = f12_20 +
        _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas);

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(0, 3, omega, vertices_omegas) +
            _f(0, 2, omega, vertices_omegas) * f12_20 / g) /
           3;
}

static __device__ double _I_21(const double omega, const double vertices_omegas[4]) {
    double f13_21, g;
    f13_21 =
        _f(1, 3, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas);
    g = _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
        f13_21;

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(1, 2, omega, vertices_omegas) +
            _f(1, 3, omega, vertices_omegas) * f13_21 / g) /
           3;
}

static __device__ double _I_22(const double omega, const double vertices_omegas[4]) {
    double f12_20, g;
    f12_20 =
        _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas);
    g = f12_20 +
        _f(2, 1, omega, vertices_omegas) * _f(1, 3, omega, vertices_omegas);

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(2, 1, omega, vertices_omegas) +
            _f(2, 0, omega, vertices_omegas) * f12_20 / g) /
           3;
}

static __device__ double _I_23(const double omega, const double vertices_omegas[4]) {
    double f13_21, g;
    f13_21 =
        _f(1, 3, omega, vertices_omegas) * _f(2, 1, omega, vertices_omegas);
    g = _f(1, 2, omega, vertices_omegas) * _f(2, 0, omega, vertices_omegas) +
        f13_21;

#ifdef THM_EPSILON
    if (g < THM_EPSILON) {
        return 0;
    }
#endif

    return (_f(3, 0, omega, vertices_omegas) +
            _f(3, 1, omega, vertices_omegas) * f13_21 / g) /
           3;
}

static __device__ double _I_30(const double omega, const double vertices_omegas[4]) {
    return _f(0, 3, omega, vertices_omegas) / 3;
}

static __device__ double _I_31(const double omega, const double vertices_omegas[4]) {
    return _f(1, 3, omega, vertices_omegas) / 3;
}

static __device__ double _I_32(const double omega, const double vertices_omegas[4]) {
    return _f(2, 3, omega, vertices_omegas) / 3;
}

static __device__ double _I_33(const double omega, const double vertices_omegas[4]) {
    return (_f(3, 0, omega, vertices_omegas) +
            _f(3, 1, omega, vertices_omegas) +
            _f(3, 2, omega, vertices_omegas)) /
           3;
}

static __device__ double _I_4(void) { return 0.0; }

// vertices: (triplet, 2, tetrahedron, 4); each work item is one band triple.
extern "C" __global__ void weights(
    double* out, signed char* zero, const double* freq, const double* f0,
    const long long* vertices, int nt, int nb0, int nb, int nth, int nc,
    int frequency_stride) {
    long long idx = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    long long size = (long long)nt*nb0*nb*nb;
    if (idx >= size) return;
    int k = idx % nb, j = (idx/nb)%nb, b = (idx/(nb*nb))%nb0;
    int t = idx/(nb0*nb*nb);
    double total[3] = {0., 0., 0.};
    bool active = false;
    for (int ch=0; ch<3; ++ch) {
        for (int h=0; h<nth; ++h) {
            double v[4];
            for (int l=0; l<4; ++l) {
                long long a = vertices[((t*2)*nth+h)*4+l];
                long long c = vertices[((t*2+1)*nth+h)*4+l];
                double x = fmax(freq[a*nb+j], 0.);
                double y = fmax(freq[c*nb+k], 0.);
                v[l] = ch==0 ? x+y : (ch==1 ? -x+y : x-y);
            }
            int center = sort_omegas(v);
            double w = f0[t*frequency_stride+b];
            if (w>=v[0] && w<=v[3]) active=true;
            int region = w<v[0] ? 0 : w<v[1] ? 1 : w<v[2] ? 2 : w<v[3] ? 3 : 4;
            total[ch] += _I(region, center, w, v)*_g(region, w, v);
        }
        total[ch] *= 4.0/nth;
    }
    out[idx] = total[0];
    out[size+idx] = total[1]-total[2];
    if (nc==3) out[2*size+idx] = total[0]+total[1]+total[2];
    zero[idx] = !active;
}

extern "C" __global__ void spectral_weights(
    double *out, const double *freq, const double *f0,
    const long long *vertices, int nq, int nb0, int nb, int nth) {
    long long index = blockIdx.x*(long long)blockDim.x+threadIdx.x;
    if (index >= (long long)nq*nb0*nb) return;
    int b = index%nb;
    int a = (index/nb)%nb0;
    int q = index/(nb*nb0);
    double sum = 0;
    for (int t=0;t<nth;t++) {
        double v[4];
        for (int j=0;j<4;j++) v[j]=freq[vertices[(q*nth+t)*4+j]*nb+b];
        int center = sort_omegas(v);
        double w=f0[a];
        int region = w<v[0]?0:w<v[1]?1:w<v[2]?2:w<v[3]?3:4;
        sum += _I(region,center,w,v)*_g(region,w,v);
    }
    out[index]=sum*4.0/nth;
}
