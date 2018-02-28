/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/
#include "test.h"
#include "flops.h"
#include "core_blas.h"
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <mpi.h>

#define COMPLEX

#define A(i_, j_) A[(i_) + (size_t)lda*(j_)]

/***************************************************************************//**
 *
 * @brief Tests ZPOTRF.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zpotrf(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_UPLO   ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_ZEROCOL].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int n = param[PARAM_DIM].dim.n;

    int lda = imax(1, n + param[PARAM_PADA].i);

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);

    //================================================================
    // Allocate and initialize arrays.
    //================================================================
    plasma_complex64_t *A =
        (plasma_complex64_t*)malloc((size_t)lda*n*sizeof(plasma_complex64_t));
    assert(A != NULL);

    int seed[] = {0, 0, 0, 1};
    lapack_int retval;
    retval = LAPACKE_zlarnv(1, seed, (size_t)lda*n, A);
    assert(retval == 0);

    //================================================================
    // Make the A matrix symmetric/Hermitian positive definite.
    // It increases diagonal by n, and makes it real.
    // It sets Aji = conj( Aij ) for j < i, that is, copy lower
    // triangle to upper triangle.
    //================================================================
    for (int i = 0; i < n; i++) {
        A(i, i) = creal(A(i, i)) + n;
        for (int j = 0; j < i; j++) {
            A(j, i) = conj(A(i, j));
        }
    }

    int zerocol = param[PARAM_ZEROCOL].i;
    if (zerocol >= 0 && zerocol < n)
        memset(&A[zerocol*lda], 0, n*sizeof(plasma_complex64_t));

    plasma_complex64_t *Aref = NULL;
    if (test) {
        Aref = (plasma_complex64_t*)malloc(
            (size_t)lda*n*sizeof(plasma_complex64_t));
        assert(Aref != NULL);

        memcpy(Aref, A, (size_t)lda*n*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    plasma_context_t *plasma = plasma_context_self();
    MPI_Barrier(plasma->comm);
    plasma_time_t start = MPI_Wtime();
    int plainfo = plasma_zpotrf(uplo, n, A, lda);
    MPI_Barrier(plasma->comm);
    plasma_time_t stop = MPI_Wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = plasma->time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / plasma->time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        double error;
        int success;
        int my_rank;
        MPI_Comm_rank(plasma->comm, &my_rank);
        if (my_rank == 0) {
            int lapinfo = LAPACKE_zpotrf(LAPACK_COL_MAJOR,
                                         lapack_const(uplo), n,
                                         Aref, lda);
            if (lapinfo == 0) {
                plasma_complex64_t zmone = -1.0;
                cblas_zaxpy((size_t)lda*n, CBLAS_SADDR(zmone), Aref, 1, A, 1);

                double work[1];
                double Anorm = LAPACKE_zlanhe_work(
                    LAPACK_COL_MAJOR, 'F', lapack_const(uplo), n, Aref, lda, work);
                error = LAPACKE_zlange_work(
                    LAPACK_COL_MAJOR, 'F', n, n, A, lda, work);
                if (Anorm != 0)
                    error /= Anorm;

                success = error < tol;
            }
            else {
                if (plainfo == lapinfo) {
                    error = 0.0;
                    success = 1;
                }
                else {
                    error = INFINITY;
                    success = 0;
                }
            }
        }
        MPI_Bcast(&error,   1, MPI_DOUBLE, 0, plasma->comm);
        MPI_Bcast(&success, 1, MPI_INT,    0, plasma->comm);

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = success;
    }

    //================================================================
    // Free arrays.
    //================================================================
    free(A);
    if (test)
        free(Aref);
}
