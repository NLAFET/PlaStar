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
#include "plasma_types.h"
#include "plasma_internal.h"
#include "core_lapack.h"

#include "starpu.h"
#include "starpu_mpi.h"

// for memset function
#include <string.h>

/***************************************************************************//**
 *
 * @ingroup core_laset
 *
 *  Sets the elements of the matrix A on the diagonal
 *  to beta and on the off-diagonals to alpha
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          Specifies which elements of the matrix are to be set
 *          - PlasmaUpper: Upper part of A is set;
 *          - PlasmaLower: Lower part of A is set;
 *          - PlasmaUpperLower: ALL elements of A are set.
 *
 * @param[in] m
 *          The number of rows of the matrix A.  m >= 0.
 *
 * @param[in] n
 *         The number of columns of the matrix A.  n >= 0.
 *
 * @param[in] alpha
 *         The constant to which the off-diagonal elements are to be set.
 *
 * @param[in] beta
 *         The constant to which the diagonal elements are to be set.
 *
 * @param[in,out] A
 *         On entry, the m-by-n tile A.
 *         On exit, A has been set accordingly.
 *
 * @param[in] lda
 *         The leading dimension of the array A.  lda >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void core_zlaset(plasma_enum_t uplo, int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t beta,
                 plasma_complex64_t *A, int lda)
{
    if (alpha == 0.0 && beta == 0.0 && uplo == PlasmaGeneral && m == lda) {
        // Use memset to zero continuous memory.
        memset((void*)A, 0, (size_t)m*n*sizeof(plasma_complex64_t));
    }
    else {
        // Use LAPACKE_zlaset_work to initialize the matrix.
        LAPACKE_zlaset_work(LAPACK_COL_MAJOR, lapack_const(uplo),
                            m, n, alpha, beta, A, lda);
    }
}

/******************************************************************************/
// StarPU LASET CPU kernel.
static void core_starpu_cpu_zlaset(void *descr[], void *cl_arg)
{
    plasma_enum_t uplo;
    int mb, nb;
    int i, j;
    int m, n;
    plasma_complex64_t alpha, beta;
    plasma_complex64_t *A;

    // Unpack data of tiles.
    A = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &uplo,
                               &mb, &nb, &i, &j, &m, &n,
                               &alpha, &beta);

    // Call the kernel.
    core_zlaset(uplo, m, n,
                alpha, beta,
                A+i+j*mb, mb);
}

/******************************************************************************/
// StarPU codelets.
struct starpu_codelet core_starpu_codelet_zlaset = {
    .cpu_func  = core_starpu_cpu_zlaset,
    .nbuffers  = 1,
    .name      = "zlaset"
};       

/******************************************************************************/
// The function for task insertion.
#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)
void core_starpu_zlaset(
    plasma_enum_t uplo,
    int mb, int nb,
    int i, int j,
    int m, int n,
    plasma_complex64_t alpha, plasma_complex64_t beta,
    plasma_desc_t A, int Am, int An)
{
    //int owner_A = starpu_mpi_data_get_rank(A);
    int owner_A = (A.tile_owner)(A.p, A.q, Am, An);

    int execution_rank = owner_A;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner_A == my_rank ||
        execution_rank == my_rank) {

        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_zlaset, 
            STARPU_VALUE,    &uplo,              sizeof(plasma_enum_t),
            STARPU_VALUE,    &mb,                sizeof(int),
            STARPU_VALUE,    &nb,                sizeof(int),
            STARPU_VALUE,    &i,                 sizeof(int),
            STARPU_VALUE,    &j,                 sizeof(int),
            STARPU_VALUE,    &m,                 sizeof(int),
            STARPU_VALUE,    &n,                 sizeof(int),
            STARPU_VALUE,    &alpha,             sizeof(plasma_complex64_t),
            STARPU_VALUE,    &beta,              sizeof(plasma_complex64_t),
            STARPU_W,        A(Am, An),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_NAME, "zlaset",
            0);
    }
}
