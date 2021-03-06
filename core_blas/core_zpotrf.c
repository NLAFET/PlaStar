/**
1;4205;0c *
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
#include "core_lapack.h"

#include "starpu.h"
#include "starpu_mpi.h"


/***************************************************************************//**
 *
 * @ingroup core_potrf
 *
 *  Performs the Cholesky factorization of a Hermitian positive definite
 *  matrix A. The factorization has the form
 *
 *    \f[ A = L \times L^H, \f]
 *    or
 *    \f[ A = U^H \times U, \f]
 *
 *  where U is an upper triangular matrix and L is a lower triangular matrix.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of A is stored;
 *          - PlasmaLower: Lower triangle of A is stored.
 *
 * @param[in] n
 *          The order of the matrix A. n >= 0.
 *
 * @param[in,out] A
 *          On entry, the Hermitian positive definite matrix A.
 *          If uplo = PlasmaUpper, the leading N-by-N upper triangular part of A
 *          contains the upper triangular part of the matrix A, and the strictly
 *          lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading N-by-N lower triangular part of A
 *          contains the lower triangular part of the matrix A, and the strictly
 *          upper triangular part of A is not referenced.
 *          On exit, if return value = 0, the factor U or L from the Cholesky
 *          factorization A = U^H*U or A = L*L^H.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,n).
 *
 ******************************************************************************/
__attribute__((weak))
int core_zpotrf(plasma_enum_t uplo,
                 int n,
                 plasma_complex64_t *A, int lda)
{
    return LAPACKE_zpotrf_work(LAPACK_COL_MAJOR,
                               lapack_const(uplo),
                               n,
                               A, lda);
}

/******************************************************************************/
// StarPU POTRF CPU kernel.
static void core_starpu_cpu_zpotrf(void *descr[], void *cl_arg)
{
    plasma_enum_t uplo;
    int n;
    plasma_complex64_t *A;
    int lda;

    // Unpack data of tiles.
    A = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
   
    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &uplo, &n, &lda);

    // Call the kernel.
    core_zpotrf(uplo, n, A, lda);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_zpotrf = {
    .cpu_func  = core_starpu_cpu_zpotrf,
    .nbuffers  = 1,
    .name      = "zpotrf"
};       

/******************************************************************************/
// The function for task insertion.

#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)

void core_starpu_zpotrf(plasma_enum_t uplo,
                        int n,
                        plasma_desc_t A, int Am, int An, int lda,
                        plasma_sequence_t *sequence, plasma_request_t *request)
{
    //int owner = starpu_mpi_data_get_rank(A);
    int owner = (A.tile_owner)(A.p, A.q, Am, An);

    int execution_rank = owner;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner == my_rank) {
        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_zpotrf,
            STARPU_VALUE,    &uplo,              sizeof(plasma_enum_t),
            STARPU_VALUE,    &n,                 sizeof(int),
            STARPU_RW,       A(Am, An),
            STARPU_VALUE,    &lda,               sizeof(int),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_PRIORITY, STARPU_MAX_PRIO,
            STARPU_NAME, "zpotrf",
            0);
    }
}
