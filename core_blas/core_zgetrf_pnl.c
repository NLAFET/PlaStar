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
#include "core_lapack.h"
#include "plasma_barrier.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_types.h"

#include <omp.h>
#include <assert.h>
#include <math.h>

#define A(m, n) (plasma_complex64_t*)plasma_tile_addr(A, m, n)

/******************************************************************************/
__attribute__((weak))
void core_zgetrf_pnl(plasma_desc_t A, plasma_complex64_t *mtrxA, int *iPiv,
                     int k, int nvak, int ib, int num_panel_threads,
                     plasma_sequence_t *sequence, plasma_request_t *request)
{
    /*
    #pragma omp task depend(inout:a00[0:ma00k*na00k]) \
                     depend(inout:a20[0:lda20*nvak]) \
                     depend(out:iPiv[k*A.mb:mvak]) \
                     priority(1)
    */
    volatile int *max_idx = (int*)malloc(num_panel_threads*sizeof(int));
    if (max_idx == NULL)
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);

    volatile plasma_complex64_t *max_val =
        (plasma_complex64_t*)malloc(num_panel_threads*sizeof(
                                    plasma_complex64_t));
    if (max_val == NULL)
        plasma_request_fail(sequence, request, PlasmaErrorOutOfMemory);

    volatile int info = 0;

    plasma_barrier_t barrier;
    plasma_barrier_init(&barrier);

    if (sequence->status == PlasmaSuccess) {
        //#pragma omp taskloop untied shared(barrier) \
        //                     num_tasks(num_panel_threads) \
        //                     priority(2)
        //for (int rank = 0; rank < num_panel_threads; rank++) {
        int rank = 0; 
            {
                plasma_desc_t view =
                    plasma_desc_view(A,
                                     k*A.mb, k*A.nb,
                                     A.m-k*A.mb, nvak);

                core_zgetrf(view, iPiv, ib,
                            rank, num_panel_threads,
                            max_idx, max_val, &info,
                            &barrier);

                if (info != 0)
                    plasma_request_fail(sequence, request, k*A.mb+info);
            }
        //}
    }
    // Wait for nested tasks created inside this one.
    //#pragma omp taskwait

    free((void*)max_idx);
    free((void*)max_val);

    for (int i = k*A.mb+1; i <= imin(A.m, k*A.mb+nvak); i++)
        iPiv[i-1] += k*A.mb;
}

/******************************************************************************/
// StarPU GETRF Panel CPU kernel.
static void core_starpu_cpu_zgetrf_pnl(void *descr[], void *cl_arg)
{
    plasma_desc_t A;
    plasma_complex64_t *mtrxA;
    int *iPiv, k, nvak, ib, num_panel_threads;
    plasma_sequence_t sequence;
    plasma_request_t  request;

    // Unpack data of tiles.
    mtrxA = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    iPiv  = (int *)                STARPU_VECTOR_GET_PTR(descr[1]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &A, &k, &nvak, &ib, &num_panel_threads,
                               &sequence, &request);

    // Call the kernel.
    core_zgetrf_pnl(A, mtrxA, iPiv, k, nvak, ib, num_panel_threads,
                    &sequence, &request);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_zgetrf_pnl = {
    .cpu_func  = core_starpu_cpu_zgetrf_pnl,
    .nbuffers  = 2,
    .name      = "zgetrf_pnl"
};

/******************************************************************************/
// The function for task insertion.
void core_starpu_zgetrf_pnl(plasma_desc_t A, starpu_data_handle_t pnlA,
                                             starpu_data_handle_t pnlPiv,
                            int k, int nvak, int ib, int num_panel_threads,
                            plasma_sequence_t *sequence, plasma_request_t *request)
{
    starpu_insert_task(
        &core_starpu_codelet_zgetrf_pnl,
        STARPU_VALUE,   &A,                    sizeof(plasma_desc_t),
        STARPU_RW,      pnlA,
        STARPU_RW,      pnlPiv,
        STARPU_VALUE,   &k,                    sizeof(int),
        STARPU_VALUE,   &nvak,                 sizeof(int),
        STARPU_VALUE,   &ib,                   sizeof(int),
        STARPU_VALUE,   &num_panel_threads,    sizeof(int),
        STARPU_VALUE,   &sequence,             sizeof(plasma_sequence_t),
        STARPU_VALUE,   &request,              sizeof(plasma_request_t),
        STARPU_NAME,   "zgetrf_pnl",
        0);
}
