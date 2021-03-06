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
#include "core_lapack.h"

#include "starpu.h"
#include "starpu_mpi.h"

/***************************************************************************//**
 *
 * @ingroup core_syrk
 *
 *  Performs one of the symmetric rank k operations
 *
 *    \f[ C = \alpha A \times A^T + \beta C, \f]
 *    or
 *    \f[ C = \alpha A^T \times A + \beta C, \f]
 *
 *  where alpha and beta are scalars, C is an n-by-n symmetric
 *  matrix, and A is an n-by-k matrix in the first case and a k-by-n
 *  matrix in the second case.
 *
 *******************************************************************************
 *
 * @param[in] uplo
 *          - PlasmaUpper: Upper triangle of C is stored;
 *          - PlasmaLower: Lower triangle of C is stored.
 *
 * @param[in] trans
 *          - PlasmaNoTrans: \f[ C = \alpha A \times A^T + \beta C; \f]
 *          - PlasmaTrans:   \f[ C = \alpha A^T \times A + \beta C. \f]
 *
 * @param[in] n
 *          The order of the matrix C. n >= 0.
 *
 * @param[in] k
 *          If trans = PlasmaNoTrans, number of columns of the A matrix;
 *          if trans = PlasmaTrans, number of rows of the A matrix.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          A is an lda-by-ka matrix.
 *          If trans = PlasmaNoTrans, ka = k;
 *          if trans = PlasmaTrans,   ka = n.
 *
 * @param[in] lda
 *          The leading dimension of the array A.
 *          If trans = PlasmaNoTrans, lda >= max(1, n);
 *          if trans = PlasmaTrans,   lda >= max(1, k).
 *
 * @param[in] beta
 *          The scalar beta.
 *
 * @param[in,out] C
 *          C is an ldc-by-n matrix.
 *          On exit, the uplo part of the matrix is overwritten
 *          by the uplo part of the updated matrix.
 *
 * @param[in] ldc
 *          The leading dimension of the array C. ldc >= max(1, n).
 *
 ******************************************************************************/
__attribute__((weak))
void core_zsyrk(plasma_enum_t uplo, plasma_enum_t trans,
                int n, int k,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                plasma_complex64_t beta,        plasma_complex64_t *C, int ldc)
{
    cblas_zsyrk(CblasColMajor,
                (CBLAS_UPLO)uplo, (CBLAS_TRANSPOSE)trans,
                n, k,
                CBLAS_SADDR(alpha), A, lda,
                CBLAS_SADDR(beta),  C, ldc);
}

/******************************************************************************/
// StarPU SYRK CPU kernel.
static void core_starpu_cpu_zsyrk(void *descr[], void * cl_arg)
{
    plasma_enum_t uplo,  trans;
    int n,  k;
    plasma_complex64_t alpha, beta;
    plasma_complex64_t *A, *C;
    int lda, ldc;

    // Unpack data of tiles.
    A = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    C = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[1]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &uplo, &trans, &n, &k, &alpha, &lda,
                               &beta, &ldc);

    // Call the SYRK kernel.
    core_zsyrk(uplo, trans,
               n, k,
               alpha, A, lda,
               beta,  C, ldc);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_zsyrk = {
    .cpu_func = core_starpu_cpu_zsyrk,
    .nbuffers = 2,
    .name     = "zsyrk"
};
    
/******************************************************************************/
// The function for task insertion.
#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)
#define C(m, n) (starpu_data_handle_t) plasma_desc_handle(C, m, n)

void core_starpu_zsyrk(
    plasma_enum_t uplo, plasma_enum_t trans,
    int n, int k,
    plasma_complex64_t alpha, plasma_desc_t A, int Am, int An, int lda,
    plasma_complex64_t beta,  plasma_desc_t C, int Cm, int Cn, int ldc,
    plasma_sequence_t *sequence, plasma_request_t *request)
{
    //int owner_A = starpu_mpi_data_get_rank(A);
    //int owner_C = starpu_mpi_data_get_rank(C);
    int owner_A = (A.tile_owner)(A.p, A.q, Am, An);
    int owner_C = (C.tile_owner)(C.p, C.q, Cm, Cn);

    int execution_rank = owner_C;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner_A == my_rank ||
        owner_C == my_rank ||
        execution_rank == my_rank) {
        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_zsyrk,
            STARPU_VALUE,    &uplo,        sizeof(plasma_enum_t),
            STARPU_VALUE,    &trans,       sizeof(plasma_enum_t),
            STARPU_VALUE,    &n,           sizeof(int),
            STARPU_VALUE,    &k,           sizeof(int),
            STARPU_VALUE,    &alpha,       sizeof(plasma_complex64_t),
            STARPU_R,        A(Am, An),
            STARPU_VALUE,    &lda,         sizeof(int),
            STARPU_VALUE,    &beta,        sizeof(plasma_complex64_t),
            STARPU_RW,       C(Cm, Cn),
            STARPU_VALUE,    &ldc,         sizeof(int),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_NAME,     "zsyrk",
            0);
    }
}
