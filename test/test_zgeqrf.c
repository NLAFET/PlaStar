/**
 *
 * @file
 *
 *  PLASTAR = STARPU + PLASMA.
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
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <omp.h>
#include <mpi.h>

#include <mkl_scalapack.h>
#include <mkl_pblas.h>
#include <mkl_blacs.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGEQRF.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets flags in param indicating which parameters are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgeqrf(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N;
    param[PARAM_PADA   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_IB     ].used = true;
    param[PARAM_HMODE  ].used = true;
    param[PARAM_P      ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    int m = param[PARAM_DIM].dim.m;
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
    //plasma_set(PlasmaIb, param[PARAM_IB].i); // Set IB = NB/4
    plasma_set(PlasmaIb, param[PARAM_NB].i / 4); // Set IB = NB/4
    if (param[PARAM_HMODE].c == 't')
        plasma_set(PlasmaHouseholderMode, PlasmaTreeHouseholder);
    else
        plasma_set(PlasmaHouseholderMode, PlasmaFlatHouseholder);

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
    int m_loc = numroc_(&m, &nb, &ip, &irsrc, &(plasma->p));
    int llda = imax(1, m_loc);
    descinit_(scal_descA, &m, &n, &nb, &nb, &irsrc, &icsrc, &ictxt,
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

    pzmatgen_(&ictxt, "General", "No diagonal preference", &m, &n, &nb, &nb,
              A_loc, &llda, &irsrc, &icsrc, &iaseed,
              &iroff, &m_loc, &icoff, &n_loc,
              &ip, &jp, &(plasma->p), &(plasma->q));

    // Copy C for testing purposes.
    plasma_complex64_t *Aref_loc = NULL;
    if (test) {
        Aref_loc = (plasma_complex64_t*)
            malloc((size_t)llda*n_loc*sizeof(plasma_complex64_t));
        assert(Aref_loc != NULL);

        memcpy(Aref_loc,A_loc, (size_t)llda*n_loc*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Prepare the descriptor for matrix T.
    //================================================================
    plasma_desc_t T;

    //================================================================
    // Run and time PLASMA.
    //================================================================
    MPI_Barrier(plasma->comm);
    plasma_time_t start = MPI_Wtime();

    plasma_zgeqrf(m, n, A_loc, llda, &T);

    MPI_Barrier(plasma->comm);
    plasma_time_t stop = MPI_Wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgeqrf(m, n) / time / 1e9;

    //=================================================================
    // Test results by checking orthogonality of Q and precision of Q*R
    //=================================================================
    if (test) {

        double error;
        double ortho;
        int success;
        int my_rank;
        MPI_Comm_rank(plasma->comm, &my_rank);

        // Check the orthogonality of Q
        int minmn = imin(m, n);

        // Q
        int iq = 1; 
        int jq = 1;
        int scal_descQ[9];
        int Qm_loc = numroc_(&m, &nb, &ip, &irsrc, &(plasma->p));
        int lldq = imax(1, Qm_loc);
        descinit_(scal_descQ, &m, &minmn, &nb, &nb, &irsrc, &icsrc, &ictxt,
                  &lldq, &info);
        int Qn_loc = numroc_(&minmn, &nb, &jp, &icsrc, &(plasma->q));
        plasma_complex64_t *Q_loc =
            (plasma_complex64_t*)malloc((size_t)lldq*Qn_loc*sizeof(plasma_complex64_t));
        assert(Q_loc != NULL);

        // Build Q.
        plasma_zungqr(m, minmn, minmn, A_loc, llda, T, Q_loc, lldq);

        // Identity
        int scal_descI[9];
        int Im_loc = numroc_(&minmn, &nb, &ip, &irsrc, &(plasma->p));
        int lldi = imax(1, Im_loc);
        descinit_(scal_descI, &minmn, &minmn, &nb, &nb, &irsrc, &icsrc, &ictxt,
                  &lldi, &info);
        int In_loc = numroc_(&minmn, &nb, &jp, &icsrc, &(plasma->q));
        plasma_complex64_t *I_loc =
            (plasma_complex64_t*)malloc((size_t)lldi*In_loc*sizeof(plasma_complex64_t));
        assert(I_loc != NULL);

        // Build the identity matrix
        int ii = 1; 
        int ji = 1;
        plasma_complex64_t zone  = 1.0;
        plasma_complex64_t zzero = 0.0;
        pzlaset_("G", &minmn, &minmn, &zzero, &zone, I_loc, &ii, &ji, scal_descI);

        // Perform Id - Q^H * Q
        double one  = 1.0;
        double mone = -1.0;
        pzherk_("U", "C", &minmn, &m, &mone, Q_loc, &iq, &jq, scal_descQ, 
                                      &one,  I_loc, &ii, &ji, scal_descI);

        size_t wsize = Im_loc + 2 * In_loc + lldi + lldi*nb;
        double *work = (double *) malloc((size_t)wsize*sizeof(double));

        // |Id - Q^H * Q|_oo
        ortho = pzlanhe_("I", "U", &minmn, I_loc, &ii, &ji, scal_descI, work);
        MPI_Bcast(&ortho, 1, MPI_DOUBLE, 0, plasma->comm);

        // normalize the result
        // |Id - Q^H * Q|_oo / n
        ortho /= minmn;

        free(I_loc);
        free(Q_loc);

        // Check the accuracy of A - Q * R

        // Extract the R.
        int scal_descR[9];
        int Rm_loc = numroc_(&m, &nb, &ip, &irsrc, &(plasma->p));
        int lldr = imax(1, Rm_loc);
        descinit_(scal_descR, &m, &n, &nb, &nb, &irsrc, &icsrc, &ictxt,
                  &lldr, &info);
        int Rn_loc = numroc_(&n, &nb, &jp, &icsrc, &(plasma->q));
        plasma_complex64_t *R_loc =
            (plasma_complex64_t*)malloc((size_t)lldr*Rn_loc*sizeof(plasma_complex64_t));
        assert(R_loc != NULL);

        int ir = 1; 
        int jr = 1;
        pzlaset_("L", &m, &n, &zzero, &zzero, R_loc, &ir, &jr, scal_descR);

        int ia = 1; 
        int ja = 1;
        pzlacpy_("U", &m, &n, A_loc, &ia, &ja, scal_descA,
                              R_loc, &ir, &jr, scal_descR);

        // Compute Q * R.
        plasma_zunmqr(PlasmaLeft, PlasmaNoTrans, m, n, minmn,
                     A_loc, llda, T, R_loc, lldr);

        // Compute the difference.
        // R = A - Q*R
        plasma_complex64_t zmone = -1.0;
        pzgeadd_( "N", &m, &n, &zone,  Aref_loc, &ia, &ja, scal_descA,
                               &zmone, R_loc,    &ir, &jr, scal_descR);

        // |A|_oo
        double normA = pzlange_("I", &m, &n, Aref_loc, &ia, &ja, scal_descA,
                                work);
        MPI_Bcast(&normA,   1, MPI_DOUBLE, 0, plasma->comm);


        // |A - Q*R|_oo
        error = pzlange_("I", &m, &n, R_loc, &ir, &jr, scal_descR, work);
        MPI_Bcast(&error,   1, MPI_DOUBLE, 0, plasma->comm);

        // normalize the result
        // |A-QR|_oo / (|A|_oo * n)
        error /= (normA * n);

        success = (error < tol && ortho < tol);

        //printf("ortho = %e, error = %e \n", ortho, error);

        free(work);
        free(R_loc);

        param[PARAM_ERROR].d   = error;
        param[PARAM_ORTHO].d   = ortho;
        param[PARAM_SUCCESS].i = success;
    }

    // Destroy the process grid.
    blacs_gridexit_(&ictxt);

    //================================================================
    // Free arrays.
    //================================================================
    plasma_desc_destroy(&T);
    free(A_loc);
    if (test)
        free(Aref_loc);
}
