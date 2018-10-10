/**
 * @file
 *
 *  PLASTAR = StarPU + PLASMA.
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 * @precisions normal z -> s d c
 *
 **/
#include "test.h"
#include "flops.h"
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

#include <mkl_scalapack.h>
#include <mkl_blacs.h>

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
    param[PARAM_IB     ].used = true; // used as the subtiling by PLASMA
    param[PARAM_ZEROCOL].used = true;
    param[PARAM_P      ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t uplo = plasma_uplo_const(param[PARAM_UPLO].c);

    int n = param[PARAM_DIM].dim.n;

    int nb = param[PARAM_NB].i;

    int test = param[PARAM_TEST].c == 'y';
    double tol = param[PARAM_TOL].d * LAPACKE_dlamch('E');

    int p = param[PARAM_P].i;

    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, nb);
    plasma_set(PlasmaIb, param[PARAM_IB].i);
    plasma_set(PlasmaProcessRows, p);

    //================================================================
    // Prepare ScaLAPACK environment.
    //================================================================
    plasma_context_t *plasma = plasma_context_self();

    // Initialize BLACS.
    int bl_rank, bl_nproc;
    blacs_pinfo_(&bl_rank, &bl_nproc);

    // Index of the BLACS context.
    int icontxt = -1;
    int what    =  0;
    int ictxt;
    blacs_get_(&icontxt, &what, &ictxt);

    // Initialize BLACS grid.
    int ip, jp;
    blacs_gridinit_(&ictxt, "Column-major", &(plasma->p), &(plasma->q));
    blacs_gridinfo_(&ictxt, &(plasma->p), &(plasma->q), &ip, &jp);

    // Initialize ScaLAPACK descriptors.
    int scal_descA[9];

    //================================================================
    // Allocate and initialize arrays.
    //================================================================

    int irsrc = 0;
    int icsrc = 0;
    int info;

    // A
    int m_loc = numroc_(&n, &nb, &ip, &irsrc, &(plasma->p));
    int llda = imax(1, m_loc);
    descinit_(scal_descA, &n, &n, &nb, &nb, &irsrc, &icsrc, &ictxt,
              &llda, &info);
    int n_loc = numroc_(&n, &nb, &jp, &icsrc, &(plasma->q));
    plasma_complex64_t *A_loc =
        (plasma_complex64_t*)malloc((size_t)llda*n_loc*sizeof(plasma_complex64_t));
    assert(A_loc != NULL);

    // Seed for ScaLAPACK matrix generation.
    int iaseed = 100;
    // Row and column offsets for local matrix generation.
    int iroff = 0;
    int icoff = 0;

    pzmatgen_(&ictxt, "Hermitian", "Diagonal", &n, &n, &nb, &nb,
              A_loc, &llda, &irsrc, &icsrc, &iaseed,
              &iroff, &m_loc, &icoff, &n_loc,
              &ip, &jp, &(plasma->p), &(plasma->q));

    // Copy C for testing purposes.
    plasma_complex64_t *Aref_loc = NULL;
    if (test) {
        Aref_loc = (plasma_complex64_t*)
            malloc((size_t)llda*n_loc*sizeof(plasma_complex64_t));
        assert(Aref_loc != NULL);

        memcpy(Aref_loc, A_loc, (size_t)llda*n_loc*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    MPI_Barrier(plasma->comm);
    plasma_time_t start = MPI_Wtime();

    int plainfo = plasma_zpotrf(uplo, n, A_loc, llda);

    MPI_Barrier(plasma->comm);
    
    plasma_time_t stop = MPI_Wtime();
    plasma_time_t time = stop-start;

    // set time to computation only time
    time = plasma->time;
    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zpotrf(n) / time / 1e9;

    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        double error;
        int success;

        int ia = 1; 
        int ja = 1;
        double work[1];

        char *uploc = (uplo == PlasmaLower) ? "L" : "U";

        int lapinfo;
        pzpotrf_(uploc, &n,
                Aref_loc, &ia, &ja, scal_descA, 
                &lapinfo); 

        if (lapinfo == 0) {
            plasma_complex64_t zmone = -1.0;
            cblas_zaxpy((size_t)llda*n_loc, CBLAS_SADDR(zmone), Aref_loc, 1, A_loc, 1);

            double Anorm = pzlanhe_("F", uploc,
                                    &n, Aref_loc, &ia, &ja, scal_descA,
                                    work);
            MPI_Bcast(&Anorm, 1, MPI_DOUBLE, 0, plasma->comm);

            error = pzlange_("F", &n, &n, A_loc, &ia, &ja,
                             scal_descA, work);
            MPI_Bcast(&error, 1, MPI_DOUBLE, 0, plasma->comm);


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

        param[PARAM_ERROR].d   = error;
        param[PARAM_SUCCESS].i = success;
    }

    // Destroy the process grid.
    blacs_gridexit_(&ictxt);

    //================================================================
    // Free arrays.
    //================================================================
    free(A_loc);
    if (test)
        free(Aref_loc);
}
