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

/***************************************************************************//**
 *
 * @ingroup core_unmqr
 *
 *  Overwrites the general m-by-n tile C with
 *
 *                                    side = PlasmaLeft      side = PlasmaRight
 *    trans = PlasmaNoTrans              Q * C                  C * Q
 *    trans = Plasma_ConjTrans         Q^H * C                  C * Q^H
 *
 *  where Q is a unitary matrix defined as the product of k
 *  elementary reflectors
 *    \f[
 *        Q = H(1) H(2) ... H(k)
 *    \f]
 *  as returned by core_zgeqrt. Q is of order m if side = PlasmaLeft
 *  and of order n if side = PlasmaRight.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight : apply Q or Q^H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans    :  No transpose, apply Q;
 *         - Plasma_ConjTrans :  Transpose, apply Q^H.
 *
 * @param[in] m
 *         The number of rows of the tile C.  m >= 0.
 *
 * @param[in] n
 *         The number of columns of the tile C.  n >= 0.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *         If side = PlasmaLeft,  m >= k >= 0;
 *         if side = PlasmaRight, n >= k >= 0.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in] A
 *         Dimension:  (lda,k)
 *         The i-th column must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k,
 *         as returned by core_zgeqrt in the first k columns of its
 *         array argument A.
 *
 * @param[in] lda
 *         The leading dimension of the array A.
 *         If side = PlasmaLeft,  lda >= max(1,m);
 *         if side = PlasmaRight, lda >= max(1,n).
 *
 * @param[in] T
 *         The ib-by-k triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param[in,out] C
 *         On entry, the m-by-n tile C.
 *         On exit, C is overwritten by Q*C or Q^T*C or C*Q^T or C*Q.
 *
 * @param[in] ldc
 *         The leading dimension of the array C. ldc >= max(1,m).
 *
 * @param work
 *         Auxiliary workspace array of length
 *         ldwork-by-n  if side == PlasmaLeft
 *         ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,m)  if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int core_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                int m, int n, int k, int ib,
                const plasma_complex64_t *A,    int lda,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *C,    int ldc,
                      plasma_complex64_t *work, int ldwork)
{
    // Check input arguments.
    if (side != PlasmaLeft && side != PlasmaRight) {
        coreblas_error("illegal value of side");
        return -1;
    }

    int nq; // order of Q
    int nw; // dimension of work

    if (side == PlasmaLeft) {
        nq = m;
        nw = n;
    }
    else {
        nq = n;
        nw = m;
    }

    if (trans != PlasmaNoTrans && trans != Plasma_ConjTrans) {
        coreblas_error("illegal value of trans");
        return -2;
    }
    if (m < 0) {
        coreblas_error("illegal value of m");
        return -3;
    }
    if (n < 0) {
        coreblas_error("illegal value of n");
        return -4;
    }
    if (k < 0 || k > nq) {
        coreblas_error("illegal value of k");
        return -5;
    }
    if (ib < 0) {
        coreblas_error("illegal value of ib");
        return -6;
    }
    if (A == NULL) {
        coreblas_error("NULL A");
        return -7;
    }
    if (lda < imax(1, nq) && nq > 0) {
        coreblas_error("illegal value of lda");
        return -8;
    }
    if (T == NULL) {
        coreblas_error("NULL T");
        return -9;
    }
    if (ldt < imax(1, ib)) {
        coreblas_error("illegal value of ldt");
        return -10;
    }
    if (C == NULL) {
        coreblas_error("NULL C");
        return -11;
    }
    if (ldc < imax(1, m) && m > 0) {
        coreblas_error("illegal value of ldc");
        return -12;
    }
    if (work == NULL) {
        coreblas_error("NULL work");
        return -13;
    }
    if (ldwork < imax(1, nw) && nw > 0) {
        coreblas_error("illegal value of ldwork");
        return -14;
    }

    // quick return
    if (m == 0 || n == 0 || k == 0)
        return PlasmaSuccess;

    int i1, i3;

    if ((side == PlasmaLeft  && trans != PlasmaNoTrans) ||
        (side == PlasmaRight && trans == PlasmaNoTrans)) {
        i1 = 0;
        i3 = ib;
    }
    else {
        i1 = ((k-1)/ib)*ib;
        i3 = -ib;
    }

    for (int i = i1; i > -1 && i < k; i += i3) {
        int kb = imin(ib, k-i);
        int ic = 0;
        int jc = 0;
        int ni = n;
        int mi = m;

        if (side == PlasmaLeft) {
            // H or H^H is applied to C(i:m,1:n).
            mi = m - i;
            ic = i;
        }
        else {
            // H or H^H is applied to C(1:m,i:n).
            ni = n - i;
            jc = i;
        }

        // Apply H or H^H.
        LAPACKE_zlarfb_work(LAPACK_COL_MAJOR,
                            lapack_const(side),
                            lapack_const(trans),
                            lapack_const(PlasmaForward),
                            lapack_const(PlasmaColumnwise),
                            mi, ni, kb,
                            &A[lda*i+i], lda,
                            &T[ldt*i], ldt,
                            &C[ldc*jc+ic], ldc,
                            work, ldwork);
    }

    return PlasmaSuccess;
}

/******************************************************************************/
// The function to be run as a task.
static void core_starpu_cpu_zunmqr(void *descr[], void *cl_arg)
{
    plasma_enum_t side, trans;
    int m, n, k, ib;
    plasma_complex64_t *A, *T, *C;
    int lda, ldt, ldc;
    plasma_workspace_t work;

    // Unpack data of tiles.
    A = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    T = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[1]);
    C = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[2]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &side, &trans, &m, &n, &k, &ib,
                               &lda, &ldt, &ldc, &work);

    // Prepare workspaces.
    int id = starpu_worker_get_id();
    plasma_complex64_t *W = (plasma_complex64_t*) work.spaces[id];

    int ldwork = side == PlasmaLeft ? n : m;

    // Call the kernel.
    core_zunmqr(side, trans,
                m, n, k, ib,
                A, lda,
                T, ldt,
                C, ldc,
                W, ldwork);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_zunmqr = {
    .cpu_func  = core_starpu_cpu_zunmqr,
    .nbuffers  = 3,
    .name      = "zunmqr"
};

/******************************************************************************/
// The function for inserting a task.
#define A(m, n) (starpu_data_handle_t) plasma_desc_handle(A, m, n)
#define T(m, n) (starpu_data_handle_t) plasma_desc_handle(T, m, n)
#define C(m, n) (starpu_data_handle_t) plasma_desc_handle(C, m, n)

void core_starpu_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                        int m, int n, int k, int ib,
                        plasma_desc_t A, int Am, int An, int lda,
                        plasma_desc_t T, int Tm, int Tn, int ldt,
                        plasma_desc_t C, int Cm, int Cn, int ldc,
                        plasma_workspace_t work,
                        plasma_sequence_t *sequence, plasma_request_t *request)
{
    //int owner_A = starpu_mpi_data_get_rank(A(m,n));
    //int owner_T = starpu_mpi_data_get_rank(T(m,n));
    //int owner_C = starpu_mpi_data_get_rank(C(m,n));
    int owner_A = (A.tile_owner)(A.p, A.q, Am, An);
    int owner_C = (C.tile_owner)(C.p, C.q, Cm, Cn);
    int owner_T = (T.tile_owner)(T.p, T.q, Tm, Tn);

    assert(owner_T == owner_A);

    int execution_rank = owner_C;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner_A == my_rank ||
        owner_T == my_rank ||
        owner_C == my_rank ||
        execution_rank == my_rank) {

        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_zunmqr,
            STARPU_VALUE,    &side,              sizeof(plasma_enum_t),
            STARPU_VALUE,    &trans,             sizeof(plasma_enum_t),
            STARPU_VALUE,    &m,                 sizeof(int),
            STARPU_VALUE,    &n,                 sizeof(int),
            STARPU_VALUE,    &k,                 sizeof(int),
            STARPU_VALUE,    &ib,                sizeof(int),
            STARPU_R,        A(Am, An),
            STARPU_VALUE,    &lda,               sizeof(int),
            STARPU_R,        T(Tm, Tn),
            STARPU_VALUE,    &ldt,               sizeof(int),
            STARPU_RW,       C(Cm, Cn),
            STARPU_VALUE,    &ldc,               sizeof(int),
            STARPU_VALUE,    &work,              sizeof(plasma_workspace_t),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_NAME, "zunmqr",
            0);
    }
}
