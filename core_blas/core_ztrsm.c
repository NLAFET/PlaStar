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
 * @ingroup core_trsm
 *
 *  Solves one of the matrix equations
 *
 *    \f[ op( A )\times X  = \alpha B, \f] or
 *    \f[ X \times op( A ) = \alpha B, \f]
 *
 *  where op( A ) is one of:
 *    \f[ op( A ) = A,   \f]
 *    \f[ op( A ) = A^T, \f]
 *    \f[ op( A ) = A^H, \f]
 *
 *  alpha is a scalar, X and B are m-by-n matrices, and
 *  A is a unit or non-unit, upper or lower triangular matrix.
 *  The matrix X overwrites B.
 *
 *******************************************************************************
 *
 * @param[in] side
 *          - PlasmaLeft:  op(A)*X = B,
 *          - PlasmaRight: X*op(A) = B.
 *
 * @param[in] uplo
 *          - PlasmaUpper: A is upper triangular,
 *          - PlasmaLower: A is lower triangular.
 *
 * @param[in] transa
 *          - PlasmaNoTrans:   A is not transposed,
 *          - PlasmaTrans:     A is transposed,
 *          - PlasmaConjTrans: A is conjugate transposed.
 *
 * @param[in] diag
 *          - PlasmaNonUnit: A has non-unit diagonal,
 *          - PlasmaUnit:    A has unit diagonal.
 *
 * @param[in] m
 *          The number of rows of the matrix B. m >= 0.
 *
 * @param[in] n
 *          The number of columns of the matrix B. n >= 0.
 *
 * @param[in] alpha
 *          The scalar alpha.
 *
 * @param[in] A
 *          The lda-by-ka triangular matrix,
 *          where ka = m if side = PlasmaLeft,
 *            and ka = n if side = PlasmaRight.
 *          If uplo = PlasmaUpper, the leading k-by-k upper triangular part
 *          of the array A contains the upper triangular matrix, and the
 *          strictly lower triangular part of A is not referenced.
 *          If uplo = PlasmaLower, the leading k-by-k lower triangular part
 *          of the array A contains the lower triangular matrix, and the
 *          strictly upper triangular part of A is not referenced.
 *          If diag = PlasmaUnit, the diagonal elements of A are also not
 *          referenced and are assumed to be 1.
 *
 * @param[in] lda
 *          The leading dimension of the array A. lda >= max(1,k).
 *
 * @param[in,out] B
 *          On entry, the ldb-by-n right hand side matrix B.
 *          On exit, if return value = 0, the ldb-by-n solution matrix X.
 *
 * @param[in] ldb
 *          The leading dimension of the array B. ldb >= max(1,m).
 *
 ******************************************************************************/
__attribute__((weak))
void core_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                plasma_enum_t transa, plasma_enum_t diag,
                int m, int n,
                plasma_complex64_t alpha, const plasma_complex64_t *A, int lda,
                                                plasma_complex64_t *B, int ldb)
{
    cblas_ztrsm(CblasColMajor,
                (CBLAS_SIDE)side, (CBLAS_UPLO)uplo,
                (CBLAS_TRANSPOSE)transa, (CBLAS_DIAG)diag,
                m, n,
                CBLAS_SADDR(alpha), A, lda,
                                    B, ldb);
}

/******************************************************************************/
// StarPU TRSM CPU kernel.
static void core_starpu_cpu_ztrsm(void *descr[], void *cl_arg)
{
    plasma_enum_t side, uplo;
    plasma_enum_t transa, diag;
    int m,  n;
    plasma_complex64_t alpha;
    plasma_complex64_t *A, *B;
    int lda, ldb;

    // Unpack data of tiles.
    A = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    B = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[1]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &side, &uplo, &transa, &diag, &m, &n,
                               &alpha, &lda, &ldb);

    // Call the kernel.
    core_ztrsm(side, uplo,
               transa, diag,
               m, n,
               alpha, A, lda,
               B, ldb);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_ztrsm = {
    .cpu_func = core_starpu_cpu_ztrsm,
    .nbuffers = 2,
    .name     = "ztrsm"
};

/******************************************************************************/

#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)
#define B(m, n) (starpu_data_handle_t) plasma_desc_handle(B, m, n)

// The function for task insertion.
void core_starpu_ztrsm(
    plasma_enum_t side, plasma_enum_t uplo,
    plasma_enum_t transa, plasma_enum_t diag,
    int m, int n,
    plasma_complex64_t alpha, plasma_desc_t A, int Am, int An, int lda,
                              plasma_desc_t B, int Bm, int Bn, int ldb,
    plasma_sequence_t *sequence, plasma_request_t *request)

{
    //int owner_A = starpu_mpi_data_get_rank(A);
    //int owner_B = starpu_mpi_data_get_rank(B);
    int owner_A = (A.tile_owner)(A.p, A.q, Am, An);
    int owner_B = (B.tile_owner)(B.p, B.q, Bm, Bn);

    int execution_rank = owner_B;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner_A == my_rank ||
        owner_B == my_rank ||
        execution_rank == my_rank) {
        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_ztrsm,
            STARPU_VALUE,    &side,        sizeof(plasma_enum_t),
            STARPU_VALUE,    &uplo,        sizeof(plasma_enum_t),
            STARPU_VALUE,    &transa,      sizeof(plasma_enum_t),
            STARPU_VALUE,    &diag,        sizeof(plasma_enum_t),
            STARPU_VALUE,    &m,           sizeof(int),
            STARPU_VALUE,    &n,           sizeof(int),
            STARPU_VALUE,    &alpha,       sizeof(plasma_complex64_t),
            STARPU_R,        A(Am, An),
            STARPU_VALUE,    &lda,         sizeof(int),
            STARPU_RW,       B(Bm, Bn),
            STARPU_VALUE,    &ldb,         sizeof(int),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_NAME,     "ztrsm",
            0);
    }
}
