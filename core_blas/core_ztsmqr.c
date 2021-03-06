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
 * @ingroup core_tsmqr
 *
 *  Overwrites the general m1-by-n1 tile A1 and
 *  m2-by-n2 tile A2 with
 *
 *                                side = PlasmaLeft        side = PlasmaRight
 *    trans = PlasmaNoTrans            Q * | A1 |           | A1 A2 | * Q
 *                                         | A2 |
 *
 *    trans = Plasma_ConjTrans       Q^H * | A1 |           | A1 A2 | * Q^H
 *                                         | A2 |
 *
 *  where Q is a complex unitary matrix defined as the product of k
 *  elementary reflectors
 *
 *    Q = H(1) H(2) . . . H(k)
 *
 *  as returned by core_ztsqrt.
 *
 *******************************************************************************
 *
 * @param[in] side
 *         - PlasmaLeft  : apply Q or Q^H from the Left;
 *         - PlasmaRight :  apply Q or Q^H from the Right.
 *
 * @param[in] trans
 *         - PlasmaNoTrans    : Apply Q;
 *         - Plasma_ConjTrans : Apply Q^H.
 *
 * @param[in] m1
 *         The number of rows of the tile A1. m1 >= 0.
 *
 * @param[in] n1
 *         The number of columns of the tile A1. n1 >= 0.
 *
 * @param[in] m2
 *         The number of rows of the tile A2. m2 >= 0.
 *         m2 = m1 if side == PlasmaRight.
 *
 * @param[in] n2
 *         The number of columns of the tile A2. n2 >= 0.
 *         n2 = n1 if side == PlasmaLeft.
 *
 * @param[in] k
 *         The number of elementary reflectors whose product defines
 *         the matrix Q.
 *
 * @param[in] ib
 *         The inner-blocking size.  ib >= 0.
 *
 * @param[in,out] A1
 *         On entry, the m1-by-n1 tile A1.
 *         On exit, A1 is overwritten by the application of Q.
 *
 * @param[in] lda1
 *         The leading dimension of the array A1. lda1 >= max(1,m1).
 *
 * @param[in,out] A2
 *         On entry, the m2-by-n2 tile A2.
 *         On exit, A2 is overwritten by the application of Q.
 *
 * @param[in] lda2
 *         The leading dimension of the tile A2. lda2 >= max(1,m2).
 *
 * @param[in] V
 *         The i-th row must contain the vector which defines the
 *         elementary reflector H(i), for i = 1,2,...,k, as returned by
 *         core_ZTSQRT in the first k columns of its array argument V.
 *
 * @param[in] ldv
 *         The leading dimension of the array V. ldv >= max(1,k).
 *
 * @param[in] T
 *         The ib-by-k triangular factor T of the block reflector.
 *         T is upper triangular by block (economic storage);
 *         The rest of the array is not referenced.
 *
 * @param[in] ldt
 *         The leading dimension of the array T. ldt >= ib.
 *
 * @param work
 *         Auxiliary workspace array of length
 *         ldwork-by-n1 if side == PlasmaLeft
 *         ldwork-by-ib if side == PlasmaRight
 *
 * @param[in] ldwork
 *         The leading dimension of the array work.
 *             ldwork >= max(1,ib) if side == PlasmaLeft
 *             ldwork >= max(1,m1) if side == PlasmaRight
 *
 *******************************************************************************
 *
 * @retval PlasmaSuccess successful exit
 * @retval < 0 if -i, the i-th argument had an illegal value
 *
 ******************************************************************************/
__attribute__((weak))
int core_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                int m1, int n1, int m2, int n2, int k, int ib,
                      plasma_complex64_t *A1,   int lda1,
                      plasma_complex64_t *A2,   int lda2,
                const plasma_complex64_t *V,    int ldv,
                const plasma_complex64_t *T,    int ldt,
                      plasma_complex64_t *work, int ldwork)
{
    // Check input arguments.
    if (side != PlasmaLeft && side != PlasmaRight) {
        coreblas_error("illegal value of side");
        return -1;
    }
    if (trans != PlasmaNoTrans && trans != Plasma_ConjTrans) {
        coreblas_error("illegal value of trans");
        return -2;
    }
    if (m1 < 0) {
        coreblas_error("illegal value of m1");
        return -3;
    }
    if (n1 < 0) {
        coreblas_error("illegal value of n1");
        return -4;
    }
    if (m2 < 0 || (m2 != m1 && side == PlasmaRight)) {
        coreblas_error("illegal value of m2");
        return -5;
    }
    if (n2 < 0 || (n2 != n1 && side == PlasmaLeft)) {
        coreblas_error("illegal value of n2");
        return -6;
    }
    if (k < 0 ||
        (side == PlasmaLeft  && k > m1) ||
        (side == PlasmaRight && k > n1)) {
        coreblas_error("illegal value of k");
        return -7;
    }
    if (ib < 0) {
        coreblas_error("illegal value of ib");
        return -8;
    }
    if (A1 == NULL) {
        coreblas_error("NULL A1");
        return -9;
    }
    if (lda1 < imax(1, m1)) {
        coreblas_error("illegal value of lda1");
        return -10;
    }
    if (A2 == NULL) {
        coreblas_error("NULL A2");
        return -11;
    }
    if (lda2 < imax(1, m2)) {
        coreblas_error("illegal value of lda2");
        return -12;
    }
    if (V == NULL) {
        coreblas_error("NULL V");
        return -13;
    }
    if (ldv < imax(1, side == PlasmaLeft ? m2 : n2)) {
        coreblas_error("illegal value of ldv");
        return -14;
    }
    if (T == NULL) {
        coreblas_error("NULL T");
        return -15;
    }
    if (ldt < imax(1, ib)) {
        coreblas_error("illegal value of ldt");
        return -16;
    }
    if (work == NULL) {
        coreblas_error("NULL work");
        return -17;
    }
    if (ldwork < imax(1, side == PlasmaLeft ? ib : m1)) {
        coreblas_error("illegal value of ldwork");
        return -18;
    }

    // quick return
    if (m1 == 0 || n1 == 0 || m2 == 0 || n2 == 0 || k == 0 || ib == 0)
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
        int mi = m1;
        int ni = n1;

        if (side == PlasmaLeft) {
            // H or H^H is applied to C(i:m,1:n).
            mi = m1 - i;
            ic = i;
        }
        else {
            // H or H^H is applied to C(1:m,i:n).
            ni = n1 - i;
            jc = i;
        }

        // Apply H or H^H (NOTE: core_zparfb used to be core_ztsrfb).
        core_zparfb(side, trans, PlasmaForward, PlasmaColumnwise,
                    mi, ni, m2, n2, kb, 0,
                    &A1[lda1*jc+ic], lda1,
                    A2, lda2,
                    &V[ldv*i], ldv,
                    &T[ldt*i], ldt,
                    work, ldwork);
    }

    return PlasmaSuccess;
}

/******************************************************************************/
// The function to be run as a task.
static void core_starpu_cpu_ztsmqr(void *descr[], void *cl_arg)
{
    plasma_enum_t side, trans;
    int m1, n1, m2, n2, k, ib;
    plasma_complex64_t *A1, *A2, *V, *T;
    int lda1, lda2, ldv, ldt;
    plasma_workspace_t work;

    // Unpack data of tiles.
    A1 = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[0]);
    A2 = (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[1]);
    V =  (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[2]);
    T =  (plasma_complex64_t *) STARPU_MATRIX_GET_PTR(descr[3]);

    // Unpack scalar parameters.
    starpu_codelet_unpack_args(cl_arg, &side, &trans, &m1, &n1, &m2, &n2,
                               &k, &ib, &lda1, &lda2, &ldv, &ldt, &work);

    // Prepare workspaces.
    int id = starpu_worker_get_id();
    plasma_complex64_t *W = (plasma_complex64_t*) work.spaces[id];

    int ldwork = side == PlasmaLeft ? ib : m1; // TODO: double check

    // Call the kernel.
    core_ztsmqr(side, trans,
                m1, n1, m2, n2, k, ib,
                A1, lda1,
                A2, lda2,
                V,  ldv,
                T,  ldt,
                W,  ldwork);
}

/******************************************************************************/
// StarPU codelet.
struct starpu_codelet core_starpu_codelet_ztsmqr = {
    .cpu_func  = core_starpu_cpu_ztsmqr,
    .nbuffers  = 4,
    .name      = "ztsmqr"
};

/******************************************************************************/
// The function for inserting a task.

#define A1(m, n) (starpu_data_handle_t) plasma_desc_handle(A1, m, n)
#define A2(m, n) (starpu_data_handle_t) plasma_desc_handle(A2, m, n)
#define V(m, n)  (starpu_data_handle_t) plasma_desc_handle(V, m, n)
#define T(m, n)  (starpu_data_handle_t) plasma_desc_handle(T, m, n)

void core_starpu_ztsmqr(plasma_enum_t side, plasma_enum_t trans,
                        int m1, int n1, int m2, int n2, int k, int ib,
                        plasma_desc_t A1, int A1m, int A1n, int lda1,
                        plasma_desc_t A2, int A2m, int A2n, int lda2,
                        plasma_desc_t V,  int Vm,  int Vn,  int ldv,
                        plasma_desc_t T,  int Tm,  int Tn,  int ldt,
                        plasma_workspace_t work,
                        plasma_sequence_t *sequence, plasma_request_t *request)
{
    //int owner_A1 = starpu_mpi_data_get_rank(A1);
    //int owner_A2 = starpu_mpi_data_get_rank(A2);
    //int owner_V  = starpu_mpi_data_get_rank(V);
    //int owner_T  = starpu_mpi_data_get_rank(T);
    int owner_A1 = (A1.tile_owner)(A1.p, A1.q, A1m, A1n);
    int owner_A2 = (A2.tile_owner)(A2.p, A2.q, A2m, A2n);
    int owner_V = (V.tile_owner)(V.p, V.q, Vm, Vn);
    int owner_T = (T.tile_owner)(T.p, T.q, Tm, Tn);

    assert(owner_T == owner_V);

    int execution_rank = owner_A2;

    int my_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);

    if (owner_A1 == my_rank ||
        owner_A2 == my_rank ||
        owner_V  == my_rank ||
        owner_T  == my_rank ||
        execution_rank == my_rank) {

        starpu_mpi_insert_task(
            MPI_COMM_WORLD,
            &core_starpu_codelet_ztsmqr,
            STARPU_VALUE,    &side,              sizeof(plasma_enum_t),
            STARPU_VALUE,    &trans,             sizeof(plasma_enum_t),
            STARPU_VALUE,    &m1,                sizeof(int),
            STARPU_VALUE,    &n1,                sizeof(int),
            STARPU_VALUE,    &m2,                sizeof(int),
            STARPU_VALUE,    &n2,                sizeof(int),
            STARPU_VALUE,    &k,                 sizeof(int),
            STARPU_VALUE,    &ib,                sizeof(int),
            STARPU_RW,       A1(A1m, A1n),
            STARPU_VALUE,    &lda1,              sizeof(int),
            STARPU_RW,       A2(A2m, A2n),
            STARPU_VALUE,    &lda2,              sizeof(int),
            STARPU_R,        V(Vm, Vn),
            STARPU_VALUE,    &ldv,               sizeof(int),
            STARPU_R,        T(Tm, Tn),
            STARPU_VALUE,    &ldt,               sizeof(int),
            STARPU_VALUE,    &work,              sizeof(plasma_workspace_t),
            STARPU_EXECUTE_ON_NODE, execution_rank,
            STARPU_NAME, "ztsmqr",
            0);
    }
}
