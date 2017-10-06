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

#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"
#include "plasma_workspace.h"
#include "core_blas.h"

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
void plasma_pzgetrf(plasma_desc_t A, starpu_data_handle_t *pnlA,
                                     starpu_data_handle_t *pnlPiv,
                    plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    // Read parameters from the context.
    plasma_context_t *plasma = plasma_context_self();

    // Set tiling parameters.
    int ib = plasma->ib;

    int minmtnt = imin(A.mt, A.nt);

    for (int k = 0; k < minmtnt; k++) {

        // plasma_complex64_t *a00, *a20;
        plasma_complex64_t *a00;
        a00 = A(k, k);
        // a20 = A(A.mt-1, k);

        //// Create fake dependencies of the whole panel on its individual tiles.
        //// These tasks are inserted to generate a correct DAG rather than
        //// doing any useful work.
        //for (int m = k+1; m < A.mt-1; m++) {
        //    plasma_complex64_t *amk = A(m, k);
        //    #pragma omp task depend (in:amk[0]) \
        //                     depend (inout:a00[0]) \
        //                     priority(1)
        //    {
        //        // Do some funny work here. It appears so that the compiler
        //        // might not insert the task if it is completely empty.
        //        int l = 1;
        //        l++;
        //    }
        //}

        // int ma00k = (A.mt-k-1)*A.mb;
        // int na00k = plasma_tile_nmain(A, k);
        // int lda20 = plasma_tile_mmain(A, A.mt-1);

        int nvak = plasma_tile_nview(A, k);
        // int mvak = plasma_tile_mview(A, k);
        // int ldak = plasma_tile_mmain(A, k);

        int num_panel_threads = imin(plasma->max_panel_threads,
                                     minmtnt-k);
        // panel
        core_starpu_zgetrf_pnl(A, pnlA[k], pnlPiv[k], k, nvak, ib, num_panel_threads,
                               sequence, request);

        // update
        // core_starpu_zgetrf_upd();

        /*
        for (int n = k+1; n < A.nt; n++) {
            plasma_complex64_t *a01, *a11, *a21;
            a01 = A(k, n);
            a11 = A(k+1, n);
            a21 = A(A.mt-1, n);

            int ma11k = (A.mt-k-2)*A.mb;
            int na11n = plasma_tile_nmain(A, n);
            int lda21 = plasma_tile_mmain(A, A.mt-1);

            int nvan = plasma_tile_nview(A, n);

            #pragma omp task depend(in:a00[0:ma00k*na00k]) \
                             depend(in:a20[0:lda20*nvak]) \
                             depend(in:iPiv[k*A.mb:mvak]) \
                             depend(inout:a01[0:ldak*nvan]) \
                             depend(inout:a11[0:ma11k*na11n]) \
                             depend(inout:a21[0:lda21*nvan]) \
                             priority(n == k+1)
            {
                if (sequence->status == PlasmaSuccess) {
                    // geswp
                    int k1 = k*A.mb+1;
                    int k2 = imin(k*A.mb+A.mb, A.m);
                    plasma_desc_t view =
                        plasma_desc_view(A, 0, n*A.nb, A.m, nvan);
                    core_zgeswp(PlasmaRowwise, view, k1, k2, iPiv, 1);

                    // trsm
                    core_ztrsm(PlasmaLeft, PlasmaLower,
                               PlasmaNoTrans, PlasmaUnit,
                               mvak, nvan,
                               1.0, A(k, k), ldak,
                                    A(k, n), ldak);
                    // gemm
                    for (int m = k+1; m < A.mt; m++) {
                        int mvam = plasma_tile_mview(A, m);
                        int ldam = plasma_tile_mmain(A, m);

                        #pragma omp task priority(n == k+1)
                        {
                            core_zgemm(
                                PlasmaNoTrans, PlasmaNoTrans,
                                mvam, nvan, A.nb,
                                -1.0, A(m, k), ldam,
                                      A(k, n), ldak,
                                1.0,  A(m, n), ldam);
                        }
                    }
                }
                #pragma omp taskwait
            }
        }
        */
    }

    /*
    // Multidependency of the whole iPiv on the individual chunks
    // corresponding to tiles. 
    for (int m = 0; m < minmtnt; m++) {
        // insert dummy task
        #pragma omp task depend (in:iPiv[m*A.mb]) \
                         depend (inout:iPiv[0])
        {
            int l = 1;
            l++;
        }
    }
    */

    // pivoting to the left
    // core_starpu_zgetrf_piv();

    /*
    for (int k = 0; k < minmtnt-1; k++) {
        plasma_complex64_t *a10, *a20;
        a10 = A(k+1, k);
        a20 = A(A.mt-1, k);

        int ma10k = (A.mt-k-2)*A.mb;
        int na00k = plasma_tile_nmain(A, k);
        int lda20 = plasma_tile_mmain(A, A.mt-1);

        int nvak = plasma_tile_nview(A, k);

        #pragma omp task depend(in:iPiv[0:imin(A.m,A.n)]) \
                         depend(inout:a10[0:ma10k*na00k]) \
                         depend(inout:a20[0:lda20*nvak])
        {
            if (sequence->status == PlasmaSuccess) {
                plasma_desc_t view =
                    plasma_desc_view(A, 0, k*A.nb, A.m, A.nb);
                int k1 = (k+1)*A.mb+1;
                int k2 = imin(A.m, A.n);
                core_zgeswp(PlasmaRowwise, view, k1, k2, iPiv, 1);
            }
        }

        // Multidependency of individual tiles on the whole panel.
        for (int m = k+2; m < A.mt-1; m++) {
            plasma_complex64_t *amk = A(m, k);
            #pragma omp task depend (in:a10[0]) \
                             depend (inout:amk[0])
            {
                // Do some funny work here. It appears so that the compiler
                // might not insert the task if it is completely empty.
                int l = 1;
                l++;
            }
        }
    }
    */
}
