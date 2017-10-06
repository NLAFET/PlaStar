/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> c d s
 *
 **/

#include "core_blas.h"
// #include "plasma_internal.h"
#include "plasma_types.h"
#include "core_lapack.h"

#include "starpu.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
__attribute__((weak))
void core_zgeswp(plasma_enum_t colrow,
                 plasma_desc_t A, int k1, int k2, const int *iPiv, int incx)
{
    //================
    // PlasmaRowwise
    //================
    if (colrow == PlasmaRowwise) {
        if (incx > 0) {
            for (int m = k1-1; m <= k2-1; m += incx) {
                if (iPiv[m]-1 != m) {
                    int m1 = m;
                    int m2 = iPiv[m]-1;

                    int lda1 = plasma_tile_mmain(A, m1/A.mb);
                    int lda2 = plasma_tile_mmain(A, m2/A.mb);

                    cblas_zswap(A.n,
                                A(m1/A.mb, 0) + m1%A.mb, lda1,
                                A(m2/A.mb, 0) + m2%A.mb, lda2);
                }
            }
        }
        else {
            for (int m = k2-1; m >= k1-1; m += incx) {
                if (iPiv[m]-1 != m) {
                    int m1 = m;
                    int m2 = iPiv[m]-1;

                    int lda1 = plasma_tile_mmain(A, m1/A.mb);
                    int lda2 = plasma_tile_mmain(A, m2/A.mb);

                    cblas_zswap(A.n,
                                A(m1/A.mb, 0) + m1%A.mb, lda1,
                                A(m2/A.mb, 0) + m2%A.mb, lda2);
                }
            }
        }
    }
    //===================
    // PlasmaColumnwise
    //===================
    else {
        if (incx > 0) {
            for (int n = k1-1; n <= k2-1; n += incx) {
                if (iPiv[n]-1 != n) {
                    int n1 = n;
                    int n2 = iPiv[n]-1;

                    int lda0 = plasma_tile_mmain(A, 0);

                    cblas_zswap(A.m,
                                A(0, n1/A.nb) + (n1%A.nb)*lda0, 1,
                                A(0, n2/A.nb) + (n2%A.nb)*lda0, 1);
                }
            }
        }
        else {
            for (int n = k2-1; n >= k1-1; n += incx) {
                if (iPiv[n]-1 != n) {
                    int n1 = n;
                    int n2 = iPiv[n]-1;

                    int lda0 = plasma_tile_mmain(A, 0);

                    cblas_zswap(A.m,
                                A(0, n1/A.nb) + (n1%A.nb)*lda0, 1,
                                A(0, n2/A.nb) + (n2%A.nb)*lda0, 1);
                }
            }
        }
    }
}

/******************************************************************************/
// StarPU GESWP CPU kernel
static void core_starpu_cpu_zgeswp(void *descr[], void *cl_arg)
{
    plasma_enum_t colrow;
    plasma_desc_t A;
    //plasma_complex64_t *mtrxA;
    int k1, k2, incx;
    int *iPiv;

    // Unpack tile data
    //mtrxA = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    iPiv  = (int *)                STARPU_VECTOR_GET_PTR(descr[1]);

    // Unpack scalar parameters
    starpu_codelet_unpack_args(cl_arg, &A, &colrow, &k1, &k2, &incx);

    // Call kernel
    core_zgeswp(colrow, A, k1, k2, iPiv, incx);
}

/******************************************************************************/
// StarPU codelet
struct starpu_codelet core_starpu_codelet_zgeswp = {
    .cpu_func = core_starpu_cpu_zgeswp,
    .nbuffers = 2,
    .name     = "zgeswp"
};

/******************************************************************************/
// StarPU task inserter
void core_starpu_zgeswp(plasma_enum_t colrow, plasma_desc_t A, starpu_data_handle_t *pnlA,
                        int k1, int k2, starpu_data_handle_t *pnlPiv, int incx,
                        plasma_sequence_t *sequence, plasma_request_t *request)
{
    starpu_insert_task(
        &core_starpu_codelet_zgeswp,
        STARPU_VALUE,    &colrow,    sizeof(plasma_enum_t),
        STARPU_VALUE,    &A,         sizeof(plasma_desc_t),
        STARPU_RW,       pnlA,
        STARPU_VALUE,    &k1,        sizeof(int),
        STARPU_VALUE,    &k2,        sizeof(int),
        STARPU_R,        pnlPiv,
        STARPU_VALUE,    &incx,      sizeof(int),
        STARPU_NAME,     "zgeswp",
        0);
}
