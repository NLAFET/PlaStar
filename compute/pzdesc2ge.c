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

//#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)
#define A(m, n) A,m,n
/******************************************************************************/
void plasma_pzdesc2ge(plasma_desc_t A,
                      plasma_complex64_t *pA, int lda,
                      plasma_sequence_t *sequence,
                      plasma_request_t *request)
{
    // Return if failed sequence.
    if (sequence->status != PlasmaSuccess)
        return;

    plasma_complex64_t *f77;

    int x1, y1;
    int x2, y2;
    int n, m, ldt;

    for (m = 0; m < A.mt; m++) {
        ldt = plasma_tile_mmain(A, m);
        for (n = 0; n < A.nt; n++) {
            x1 = n == 0 ? A.j%A.nb : 0;
            y1 = m == 0 ? A.i%A.mb : 0;
            x2 = n == A.nt-1 ? (A.j+A.n-1)%A.nb+1 : A.nb;
            y2 = m == A.mt-1 ? (A.i+A.m-1)%A.mb+1 : A.mb;

            // find local indices
            int m_loc, n_loc;
            plasma_tile_global2local(m, n, A.p, A.q, &m_loc, &n_loc);
            
            f77 = &pA[(size_t)A.nb*lda*n_loc + (size_t)A.mb*m_loc];

            core_starpu_zlacpy(PlasmaGeneral, PlasmaNoTrans, PlasmaBackward,
                               x1, x2, y1, y2,
                               f77, lda,
                               A(m, n), ldt,
                               sequence, request);
        }
    }
}
