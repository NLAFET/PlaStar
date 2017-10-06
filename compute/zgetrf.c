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

#include "plasma.h"
#include "plasma_async.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_tuning.h"
#include "plasma_types.h"
#include "plasma_workspace.h"

/***************************************************************************//**
 *
 ******************************************************************************/
int plasma_zgetrf(int m, int n,
                  plasma_complex64_t *pA, int lda, int *iPiv)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    if (m < 0) {
        plasma_error("illegal value of m");
        return -1;
    }
    if (n < 0) {
        plasma_error("illegal value of n");
        return -2;
    }
    if (lda < imax(1, m)) {
        plasma_error("illegal value of lda");
        return -4;
    }

    // quick return
    if (imin(m, n) == 0)
        return PlasmaSuccess;

    // Tune parameters.
    if (plasma->tuning)
        plasma_tune_getrf(plasma, PlasmaComplexDouble, m, n);

    // Set tiling parameters.
    int nb = plasma->nb;

    // Initialize barrier.
    plasma_barrier_init(&plasma->barrier);

    // Create tile matrix.
    plasma_desc_t A;
    int retval;
    retval = plasma_desc_general_create(PlasmaComplexDouble, nb, nb,
                                        m, n, 0, 0, m, n, &A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_create() failed");
        return retval;
    }

    // Initialize sequence.
    plasma_sequence_t sequence;
    retval = plasma_sequence_init(&sequence);

    // Initialize request.
    plasma_request_t request;
    retval = plasma_request_init(&request);

    // Translate to tile layout.
    plasma_starpu_zge2desc(pA, lda, A, &sequence, &request);

    // Remove StarPU handles for individual tiles.
    plasma_desc_handles_destroy(&A);

    // Create StarPU matrix and vector handles for A and iPiv panels
    int lt = imin(A.mt, A.nt);

    starpu_data_handle_t *pnlA   = malloc(lt * sizeof(starpu_data_handle_t));
    starpu_data_handle_t *pnlPiv = malloc(lt * sizeof(starpu_data_handle_t));

    if (pnlA == NULL) {
        plasma_error("Creation of matrix handles for A panels 'pnlA' failed!");
        return(PlasmaErrorOutOfMemory);
    }

    if (pnlPiv == NULL) {
        plasma_error("Creation of vector handles for iPiv panels 'pnlPiv' failed!");
        return(PlasmaErrorOutOfMemory);
    }

    size_t elemSize = plasma_element_size(A.precision);

    // for each panel
    for (int k = 0; k < lt; k++) {

        int nvak = plasma_tile_nview(A, k);

        starpu_matrix_data_register(&(pnlA[k]), STARPU_MAIN_RAM, (uintptr_t) plasma_tile_addr(A, k, k),
                                    A.m-k*A.mb, A.m-k*A.mb, nvak, elemSize);

        starpu_vector_data_register(&(pnlPiv[k]), STARPU_MAIN_RAM, (uintptr_t) &(iPiv[k*A.mb]),
                                    A.m-k*A.mb, sizeof(iPiv[0]));
    }

    // Call the tile async function.
    plasma_starpu_zgetrf(A, pnlA, pnlPiv, &sequence, &request);

    // Destroy StarPU matrix and vector handles for A and iPiv panels
    for (int k = 0; k < lt; k++) {

        starpu_data_unregister(pnlA[k]);
        starpu_data_unregister(pnlPiv[k]);
    }

    // Create the StarPU handles for individual tiles.
    plasma_desc_handles_create(&A);

    // Revert to LAPACK layout.
    plasma_starpu_zdesc2ge(A, pA, lda, &sequence, &request);


    if (pnlA == NULL) {
        plasma_error("Destruction of matrix handles for A panels 'pnlA' failed!");
        return(PlasmaErrorIllegalValue);
    }

    if (pnlPiv == NULL) {
        plasma_error("Destruction of vector handles for iPiv panels 'pnlPiv' failed!");
        return(PlasmaErrorIllegalValue);
    }

    free(pnlA);  free(pnlPiv);

    // StarPU block: end
    // Synchronize.
    starpu_task_wait_for_all();

    // Free matrix A in tile layout.
    plasma_desc_destroy(&A);

    // Return status.
    int status = sequence.status;
    return status;
}

/***************************************************************************//**
 *
 ******************************************************************************/
void plasma_starpu_zgetrf(plasma_desc_t A, starpu_data_handle_t *pnlA,
                                           starpu_data_handle_t *pnlPiv,
                          plasma_sequence_t *sequence, plasma_request_t *request)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_fatal_error("PLASMA not initialized");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // Check input arguments.
    if (plasma_desc_check(A) != PlasmaSuccess) {
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        plasma_error("invalid A");
        return;
    }
    if (sequence == NULL) {
        plasma_fatal_error("NULL sequence");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }
    if (request == NULL) {
        plasma_fatal_error("NULL request");
        plasma_request_fail(sequence, request, PlasmaErrorIllegalValue);
        return;
    }

    // quick return
    if (A.m == 0 || A.n == 0)
        return;

    // Call the parallel function.
    plasma_pzgetrf(A, pnlA, pnlPiv, sequence, request);
}
