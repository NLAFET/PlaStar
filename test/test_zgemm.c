/**
1;4205;0c *
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
#include "core_lapack.h"
#include "plasma.h"

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include <omp.h>
#include <mpi.h>

#include <mkl_scalapack.h>
#include <mkl_blacs.h>
#include <mkl_pblas.h>

#define COMPLEX

/***************************************************************************//**
 *
 * @brief Tests ZGEMM.
 *
 * @param[in,out] param - array of parameters
 * @param[in]     run - whether to run test
 *
 * Sets used flags in param indicating parameters that are used.
 * If run is true, also runs test and stores output parameters.
 ******************************************************************************/
void test_zgemm(param_value_t param[], bool run)
{
    //================================================================
    // Mark which parameters are used.
    //================================================================
    param[PARAM_TRANSA ].used = true;
    param[PARAM_TRANSB ].used = true;
    param[PARAM_DIM    ].used = PARAM_USE_M | PARAM_USE_N | PARAM_USE_K;
    param[PARAM_ALPHA  ].used = true;
    param[PARAM_BETA   ].used = true;
    param[PARAM_PADA   ].used = true;
    param[PARAM_PADB   ].used = true;
    param[PARAM_PADC   ].used = true;
    param[PARAM_NB     ].used = true;
    param[PARAM_P      ].used = true;
    if (! run)
        return;

    //================================================================
    // Set parameters.
    //================================================================
    plasma_enum_t transa = plasma_trans_const(param[PARAM_TRANSA].c);
    plasma_enum_t transb = plasma_trans_const(param[PARAM_TRANSB].c);

    int m = param[PARAM_DIM].dim.m;
    int n = param[PARAM_DIM].dim.n;
    int k = param[PARAM_DIM].dim.k;

    int Am, An;
    int Bm, Bn;
    int Cm, Cn;

    if (transa == PlasmaNoTrans) {
        Am = m;
        An = k;
    }
    else {
        Am = k;
        An = m;
    }
    if (transb == PlasmaNoTrans) {
        Bm = k;
        Bn = n;
    }
    else {
        Bm = n;
        Bn = k;
    }
    Cm = m;
    Cn = n;

    int test = param[PARAM_TEST].c == 'y';
    double eps = LAPACKE_dlamch('E');

#ifdef COMPLEX
    plasma_complex64_t alpha = param[PARAM_ALPHA].z;
    plasma_complex64_t beta  = param[PARAM_BETA].z;
#else
    double alpha = creal(param[PARAM_ALPHA].z);
    double beta  = creal(param[PARAM_BETA].z);
#endif

    int nb = param[PARAM_NB].i;
    
    int p = param[PARAM_P].i;
    
    //================================================================
    // Set tuning parameters.
    //================================================================
    plasma_set(PlasmaTuning, PlasmaDisabled);
    plasma_set(PlasmaNb, param[PARAM_NB].i);
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
    int scal_descA[9], scal_descB[9], scal_descC[9];

    //================================================================
    // Allocate and initialize arrays.
    //================================================================

    int irsrc = 0;
    int icsrc = 0;
    int info;

    // A
    int Am_loc = numroc_(&Am, &nb, &ip, &irsrc, &(plasma->p));
    int llda = imax(1, Am_loc);
    descinit_(scal_descA, &Am, &An, &nb, &nb, &irsrc, &icsrc, &ictxt,
              &llda, &info);
    int An_loc = numroc_(&An, &nb, &jp, &icsrc, &(plasma->q));
    plasma_complex64_t *A_loc =
        (plasma_complex64_t*)malloc((size_t)llda*An_loc*sizeof(plasma_complex64_t));
    assert(A_loc != NULL);

    // B
    int Bm_loc = numroc_(&Bm, &nb, &ip, &irsrc, &(plasma->p));
    int lldb = imax(1, Bm_loc);
    descinit_(scal_descB, &Bm, &Bn, &nb, &nb, &irsrc, &icsrc, &ictxt,
              &lldb, &info);
    int Bn_loc = numroc_(&Bn, &nb, &jp, &icsrc, &(plasma->q));
    plasma_complex64_t *B_loc =
        (plasma_complex64_t*)malloc((size_t)lldb*Bn_loc*sizeof(plasma_complex64_t));
    assert(B_loc != NULL);

    // C
    int Cm_loc = numroc_(&Cm, &nb, &ip, &irsrc, &(plasma->p));
    int lldc = imax(1, Cm_loc);
    descinit_(scal_descC, &Cm, &Cn, &nb, &nb, &irsrc, &icsrc, &ictxt,
              &lldc, &info);
    int Cn_loc = numroc_(&Cn, &nb, &jp, &icsrc, &(plasma->q));
    plasma_complex64_t *C_loc =
        (plasma_complex64_t*)malloc((size_t)lldc*Cn_loc*sizeof(plasma_complex64_t));
    assert(C_loc != NULL);


    // Seed for ScaLAPACK matrix generation.
    int iaseed = 100;
    // Row and column offsets for local matrix generation.
    int iroff = 0;
    int icoff = 0;

    pzmatgen_(&ictxt, "General", "No diagonal preference", &Am, &An, &nb, &nb,
              A_loc, &llda, &irsrc, &icsrc, &iaseed,
              &iroff, &Am_loc, &icoff, &An_loc,
              &ip, &jp, &(plasma->p), &(plasma->q));

    pzmatgen_(&ictxt, "General", "No diagonal preference", &Bm, &Bn, &nb, &nb,
              B_loc, &lldb, &irsrc, &icsrc, &iaseed,
              &iroff, &Bm_loc, &icoff, &Bn_loc,
              &ip, &jp, &(plasma->p), &(plasma->q));

    pzmatgen_(&ictxt, "General", "No diagonal preference", &Cm, &Cn, &nb, &nb,
              C_loc, &lldc, &irsrc, &icsrc, &iaseed,
              &iroff, &Cm_loc, &icoff, &Cn_loc,
              &ip, &jp, &(plasma->p), &(plasma->q));


    // Copy C for testing purposes.
    plasma_complex64_t *Cref_loc = NULL;
    if (test) {
        Cref_loc = (plasma_complex64_t*)
            malloc((size_t)lldc*Cn_loc*sizeof(plasma_complex64_t));
        assert(Cref_loc != NULL);
        
        memcpy(Cref_loc, C_loc, (size_t)lldc*Cn_loc*sizeof(plasma_complex64_t));
    }

    //================================================================
    // Run and time PLASMA.
    //================================================================
    MPI_Barrier(plasma->comm);
    plasma_time_t start = MPI_Wtime();
    
    plasma_zgemm(
        transa, transb,
        m, n, k,
        alpha, A_loc, llda,
               B_loc, lldb,
         beta, C_loc, lldc);

    MPI_Barrier(plasma->comm);
    plasma_time_t stop = MPI_Wtime();
    plasma_time_t time = stop-start;

    param[PARAM_TIME].d = time;
    param[PARAM_GFLOPS].d = flops_zgemm(m, n, k) / time / 1e9;

    
    //================================================================
    // Test results by comparing to a reference implementation.
    //================================================================
    if (test) {
        // |R - R_ref|_p < gamma_{k+2} * |alpha| * |A|_p * |B|_p +
        //                 gamma_2 * |beta| * |C|_p
        // holds component-wise or with |.|_p as 1, inf, or Frobenius norm.
        // gamma_k = k*eps / (1 - k*eps), but we use
        // gamma_k = sqrt(k)*eps as a statistical average case.
        // Using 3*eps covers complex arithmetic.
        // See Higham, Accuracy and Stability of Numerical Algorithms, ch 2-3.
        double error;
        double work[1];
        int ia = 1; 
        int ja = 1;
        
        double Anorm = pzlange_("F", &Am, &An,
                                A_loc, &ia, &ja, scal_descA,
                                work);
        MPI_Bcast(&Anorm, 1, MPI_DOUBLE, 0, plasma->comm);

        int ib = 1; 
        int jb = 1;
        double Bnorm = pzlange_("F", &Bm, &Bn,
                                B_loc, &ib, &jb, scal_descB,
                                work);
        MPI_Bcast(&Bnorm, 1, MPI_DOUBLE, 0, plasma->comm);

        int ic = 1; 
        int jc = 1;
        double Cnorm = pzlange_("F", &Cm, &Cn,
                                Cref_loc, &ic, &jc, scal_descC,
                                work);
        MPI_Bcast(&Cnorm, 1, MPI_DOUBLE, 0, plasma->comm);
        
        pzgemm_("N", "N", &m, &n, &k,
                &alpha,
                A_loc, &ia, &ja, scal_descA,
                B_loc, &ib, &jb, scal_descB,
                &beta,
                Cref_loc, &ic, &jc, scal_descC);

        plasma_complex64_t zmone = -1.0;
        cblas_zaxpy((size_t)lldc*Cn_loc, CBLAS_SADDR(zmone), Cref_loc, 1, C_loc, 1);
        
        error = pzlange_("F", &Cm, &Cn, C_loc, &ic, &jc,
                         scal_descC, work);
        MPI_Bcast(&error, 1, MPI_DOUBLE, 0, plasma->comm);

                double normalize = sqrt((double)k+2) * cabs(alpha) * Anorm * Bnorm
                         + 2 * cabs(beta) * Cnorm;
        if (normalize != 0)
            error /= normalize;


        param[PARAM_ERROR].d = error;
        param[PARAM_SUCCESS].i = error < 3*eps;
    }
    
    // Destroy the process grid.
    blacs_gridexit_(&ictxt);
    
    //================================================================
    // Free arrays.
    //================================================================
    free(A_loc);
    free(B_loc);
    free(C_loc);
    if (test)
        free(Cref_loc);
}
