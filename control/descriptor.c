/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#include "plasma_types.h"
#include "plasma_context.h"
#include "plasma_descriptor.h"
#include "plasma_internal.h"
#include "plasma_distributed.h"
#include "core_blas.h"

#include <starpu.h>
#include <starpu_mpi.h>

static int32_t descriptor_id = 0;

/******************************************************************************/
int plasma_desc_general_create(plasma_enum_t precision, int mb, int nb,
                               int lm, int ln, int i, int j, int m, int n,
                               plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // pad the trailing tiles
    int mt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    int nt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);
    int mp = mt*mb;
    int np = nt*nb;
    
    // Initialize the descriptor.
    int retval = plasma_desc_general_init(precision, NULL, mb, nb,
                                          mp, np, i, j, m, n, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_init() failed");
        return retval;
    }
    // Check the descriptor.
    retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }

    // Add MPI properties to descriptors.
    retval = plasma_desc_set_dist(plasma->comm, plasma->p, plasma->q,
                                  plasma_owner_2D_block_cyclic, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_set_dist() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    size_t size = (size_t)A->gml*A->gnl*
                  plasma_element_size(A->precision);
    A->matrix = malloc(size);
    if (A->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Create the StarPU handles for individual tiles.
    plasma_desc_handles_create(A);

    return PlasmaSuccess;
}

/******************************************************************************/
/* int plasma_desc_general_band_create(plasma_enum_t precision, plasma_enum_t uplo, */
/*                                     int mb, int nb, int lm, int ln, */
/*                                     int i, int j, int m, int n, int kl, int ku, */
/*                                     plasma_desc_t *A) */
/* { */
/*     plasma_context_t *plasma = plasma_context_self(); */
/*     if (plasma == NULL) { */
/*         plasma_error("PLASMA not initialized"); */
/*         return PlasmaErrorNotInitialized; */
/*     } */
/*     // Initialize the descriptor. */
/*     int retval = plasma_desc_general_band_init(precision, uplo, NULL, mb, nb, */
/*                                                lm, ln, i, j, m, n, kl, ku, A); */
/*     if (retval != PlasmaSuccess) { */
/*         plasma_error("plasma_desc_general_band_init() failed"); */
/*         return retval; */
/*     } */
/*     // Check the descriptor. */
/*     retval = plasma_desc_check(*A); */
/*     if (retval != PlasmaSuccess) { */
/*         plasma_error("plasma_desc_check() failed"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     // Allocate the matrix. */
/*     size_t size = (size_t)A->gm*A->gn* */
/*                   plasma_element_size(A->precision); */
/*     A->matrix = malloc(size); */
/*     if (A->matrix == NULL) { */
/*         plasma_error("malloc() failed"); */
/*         return PlasmaErrorOutOfMemory; */
/*     } */
/*     return PlasmaSuccess; */
/* } */

/******************************************************************************/
int plasma_desc_triangular_create(plasma_enum_t precision, plasma_enum_t uplo, int mb, int nb,
                                  int lm, int ln, int i, int j, int m, int n,
                                  plasma_desc_t *A)
{
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

     // pad the trailing tiles
    int mt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    int nt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);
    int mp = mt*mb;
    int np = nt*nb;
    
    // Initialize the descriptor.
    int retval = plasma_desc_triangular_init(precision, uplo, NULL, mb, nb,
                                             mp, np, i, j, m, n, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_general_init() failed");
        return retval;
    }
    // Check the descriptor.
    retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }

    // Add MPI properties to descriptors.
    retval = plasma_desc_set_dist(plasma->comm, plasma->p, plasma->q,
                                 plasma_owner_2D_block_cyclic, A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_set_dist() failed");
        return PlasmaErrorIllegalValue;
    }
    // Allocate the matrix.
    /* int lm1 = lm/mb; */
    /* int ln1 = ln/nb; */
    /* int mnt = (ln1*(1+lm1))/2; */
    /* size_t size = (size_t)(mnt*mb*nb + (lm * (ln%nb)))* */
    /*               plasma_element_size(A->precision); */

    int mnt = A->ntl*A->mtl;
    size_t size = (size_t)(mnt*mb*nb) *
        plasma_element_size(A->precision);
    A->matrix = malloc(size);
    if (A->matrix == NULL) {
        plasma_error("malloc() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Create the StarPU handles for individual tiles.
    plasma_desc_handles_create(A);

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_destroy(plasma_desc_t *A)
{
    // Remove StarPU handles for individual tiles.
    plasma_desc_handles_destroy(A);

    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }
    free(A->matrix);
    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_general_init(plasma_enum_t precision, void *matrix,
                             int mb, int nb, int lm, int ln, int i, int j,
                             int m, int n, plasma_desc_t *A)
{
    // type and precision
    A->type = PlasmaGeneral;
    A->precision = precision;

    // pointer and offsets
    A->matrix = matrix;
    A->A21 = (size_t)(lm - lm%mb) * (ln - ln%nb);
    A->A12 = (size_t)(     lm%mb) * (ln - ln%nb) + A->A21;
    A->A22 = (size_t)(lm - lm%mb) * (     ln%nb) + A->A12;

    // tile parameters
    A->mb = mb;
    A->nb = nb;

    // main matrix parameters
    A->gm = lm;
    A->gn = ln;

    A->gmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    A->gnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    A->i = i;
    A->j = j;
    A->m = m;
    A->n = n;

    A->mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    A->nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    // band parameters
    A->kl = m-1;
    A->ku = n-1;
    A->klt = A->mt;
    A->kut = A->nt;

    return PlasmaSuccess;
}

/******************************************************************************/
/* int plasma_desc_general_band_init(plasma_enum_t precision, plasma_enum_t uplo, */
/*                                   void *matrix, int mb, int nb, int lm, int ln, */
/*                                   int i, int j, int m, int n, int kl, int ku, */
/*                                   plasma_desc_t *A) */
/* { */
/*     // Init parameters for a general matrix. */
/*     int retval = plasma_desc_general_init(precision, matrix, mb, nb, */
/*                                           lm, ln, i, j, m, n, A); */
/*     if (retval != PlasmaSuccess) { */
/*         plasma_error("plasma_desc_general_init() failed"); */
/*         return retval; */
/*     } */
/*     // Change matrix type to band. */
/*     A->type = PlasmaGeneralBand; */
/*     A->uplo = uplo; */

/*     // Initialize band matrix parameters. */
/*     // bandwidth */
/*     A->kl = kl; */
/*     A->ku = ku; */

/*     // number of tiles within band, 1+ for diagonal */
/*     if (uplo == PlasmaGeneral) { */
/*         A->klt = 1+(i+kl + mb-1)/mb - i/mb; */
/*         A->kut = 1+(i+ku+kl + nb-1)/nb - i/nb; */
/*     } */
/*     else if (uplo == PlasmaUpper) { */
/*         A->klt = 1; */
/*         A->kut = 1+(i+ku + nb-1)/nb - i/nb; */
/*     } */
/*     else { */
/*         A->klt = 1+(i+kl + mb-1)/mb - i/mb; */
/*         A->kut = 1; */
/*     } */
/*     return PlasmaSuccess; */
/* } */

/******************************************************************************/
int plasma_desc_triangular_init(plasma_enum_t precision, plasma_enum_t uplo, void *matrix,
                                int mb, int nb, int lm, int ln, int i, int j,
                                int m, int n, plasma_desc_t *A)
{
    // only for square matrix..
    if (lm != ln) {
        plasma_error("invalid lm or ln");
    }
    // type and precision
    A->type = uplo;
    A->precision = precision;

    // pointer and offsets
    int lm1 = lm/mb;
    int ln1 = ln/nb;
    //int mnt = (ln1*(1+lm1))/2;
    int mnt = ln1*lm1;
    A->matrix = matrix;
    A->A21 = (size_t)(mb * nb) * mnt; // only for PlasmaLower
    A->A12 = (size_t)(mb * nb) * mnt; // only for PlasmaUpper
    A->A22 = (size_t)(lm - lm%mb) * (ln%nb) + A->A12;

    // tile parameters
    A->mb = mb;
    A->nb = nb;

    // main matrix parameters
    A->gm = lm;
    A->gn = ln;

    A->gmt = (lm%mb == 0) ? (lm/mb) : (lm/mb+1);
    A->gnt = (ln%nb == 0) ? (ln/nb) : (ln/nb+1);

    // submatrix parameters
    A->i = i;
    A->j = j;
    A->m = m;
    A->n = n;

    A->mt = (m == 0) ? 0 : (i+m-1)/mb - i/mb + 1;
    A->nt = (n == 0) ? 0 : (j+n-1)/nb - j/nb + 1;

    // band parameters
    A->kl = m-1;
    A->ku = n-1;
    A->klt = A->mt;
    A->kut = A->nt;

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_check(plasma_desc_t A)
{
    if (A.type == PlasmaGeneral || 
        A.type == PlasmaUpper || 
        A.type == PlasmaLower) {
        return plasma_desc_general_check(A);
    }
    /* else if (A.type == PlasmaGeneralBand) { */
    /*     return plasma_desc_general_band_check(A); */
    /* } */
    else {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
}

/******************************************************************************/
int plasma_desc_general_check(plasma_desc_t A)
{
    if (A.precision != PlasmaRealFloat &&
        A.precision != PlasmaRealDouble &&
        A.precision != PlasmaComplexFloat &&
        A.precision != PlasmaComplexDouble  ) {
        plasma_error("invalid matrix type");
        return PlasmaErrorIllegalValue;
    }
    if (A.mb <= 0 || A.nb <= 0) {
        plasma_error("negative tile dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((A.m < 0) || (A.n < 0)) {
        plasma_error("negative matrix dimension");
        return PlasmaErrorIllegalValue;
    }
    if ((A.gm < A.m) || (A.gn < A.n)) {
        plasma_error("invalid leading dimensions");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i > 0 && A.i >= A.gm) ||
        (A.j > 0 && A.j >= A.gn)) {
        plasma_error("beginning of the matrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if (A.i+A.m > A.gm || A.j+A.n > A.gn) {
        plasma_error("submatrix out of bounds");
        return PlasmaErrorIllegalValue;
    }
    if ((A.i % A.mb != 0) || (A.j % A.nb != 0)) {
        plasma_error("submatrix not aligned to a tile");
        return PlasmaErrorIllegalValue;
    }
    return PlasmaSuccess;
}

/******************************************************************************/
/* int plasma_desc_general_band_check(plasma_desc_t A) */
/* { */
/*     if (A.precision != PlasmaRealFloat && */
/*         A.precision != PlasmaRealDouble && */
/*         A.precision != PlasmaComplexFloat && */
/*         A.precision != PlasmaComplexDouble  ) { */
/*         plasma_error("invalid matrix type"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if (A.mb <= 0 || A.nb <= 0) { */
/*         plasma_error("negative tile dimension"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if ((A.m < 0) || (A.n < 0)) { */
/*         plasma_error("negative matrix dimension"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if (A.gn < A.n) { */
/*         plasma_error("invalid leading column dimensions"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if ((A.uplo == PlasmaGeneral && */
/*          A.gm < A.mb*((2*A.kl+A.ku+A.mb)/A.mb)) || */
/*         (A.uplo == PlasmaUpper && */
/*          A.gm < A.mb*((A.ku + A.mb)/A.mb)) || */
/*         (A.uplo == PlasmaUpper && */
/*          A.gm < A.mb*((A.kl + A.mb)/A.mb))) { */
/*         plasma_error("invalid leading row dimensions"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if ((A.i > 0 && A.i >= A.gm) || */
/*         (A.j > 0 && A.j >= A.gn)) { */
/*         plasma_error("beginning of the matrix out of bounds"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if (A.j+A.n > A.gn) { */
/*         plasma_error("submatrix out of bounds"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     if ((A.i % A.mb != 0) || (A.j % A.nb != 0)) { */
/*         plasma_error("submatrix not aligned to a tile"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */

/*     if (A.kl+1 > A.m || A.ku+1 > A.n) { */
/*         plasma_error("band width larger than matrix dimension"); */
/*         return PlasmaErrorIllegalValue; */
/*     } */
/*     return PlasmaSuccess; */
/* } */

/******************************************************************************/
plasma_desc_t plasma_desc_view(plasma_desc_t A, int i, int j, int m, int n)
{
    if ((A.i+i+m) > A.gm)
        plasma_fatal_error("rows out of bound");

    if ((A.j+j+n) > A.gn)
        plasma_fatal_error("columns out of bound");

    plasma_desc_t B = A;
    int mb = A.mb;
    int nb = A.nb;

    // submatrix parameters
    B.i = A.i + i;
    B.j = A.j + j;
    B.m = m;
    B.n = n;

    // submatrix derived parameters
    B.mt = (m == 0) ? 0 : (B.i+m-1)/mb - B.i/mb + 1;
    B.nt = (n == 0) ? 0 : (B.j+n-1)/nb - B.j/nb + 1;

    return B;
}

/******************************************************************************/
int plasma_descT_create(plasma_desc_t A, int ib, plasma_enum_t householder_mode,
                        plasma_desc_t *T)
{
    // T uses tiles ib x nb, typically, ib < nb, and these tiles are
    // rectangular. This dimension is the same for QR and LQ factorizations.
    int mb = ib;
    int nb = A.nb;

    // Number of tile rows and columns in T is the same as for T.
    int mt = A.mt;
    int nt = A.nt;
    // nt is doubled for tree-reduction QR and LQ
    if (householder_mode == PlasmaTreeHouseholder) {
        nt = 2*nt;
    }

    // Dimension of the matrix as whole multiples of the tiles.
    int m = mt*mb;
    int n = nt*nb;

    // Create the descriptor using the standard function.
    int retval = plasma_desc_general_create(A.precision, mb, nb, m, n,
                                            0, 0, m, n, T);
    return retval;
}

/******************************************************************************/
int plasma_desc_handles_create(plasma_desc_t *A)
{
    // Get PLASMA context.
    plasma_context_t *plasma = plasma_context_self();
    if (plasma == NULL) {
        plasma_error("PLASMA not initialized");
        return PlasmaErrorNotInitialized;
    }

    // Check the descriptor.
    int retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }

    // Allocate the handles.
    size_t size = (size_t) A->gmt * A->gnt * sizeof(starpu_data_handle_t *);
    A->tile_handles = (starpu_data_handle_t **) malloc(size);
    if (A->tile_handles == NULL) {
        plasma_error("plasma_desc_handles_create() failed");
        return PlasmaErrorOutOfMemory;
    }

    // Initialize the handles.
    for (int j = A->j; j < A->nt; j++) {
        for (int i = A->i; i < A->mt; i++) {
            int index = j*A->gmt + i;
            (A->tile_handles)[index] = NULL;
        }
    }

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_handles_destroy(plasma_desc_t *A)
{
    // Check the descriptor.
    int retval = plasma_desc_check(*A);
    if (retval != PlasmaSuccess) {
        plasma_error("plasma_desc_check() failed");
        return PlasmaErrorIllegalValue;
    }

    int rank;
    MPI_Comm_rank(A->comm, &rank);

    // Unregister the handles to tiles.
    for (int j = A->j; j < A->nt; j++) {
        for (int i = A->i; i < A->mt; i++) {

            int index = j*A->gmt + i;

            starpu_data_handle_t *handle = (A->tile_handles)[index];

            if (handle != NULL) {
                starpu_data_unregister(*handle);
                free(handle);
            }
        }
    }

    // Destroy the handles.
    if (A->tile_handles == NULL) {
        plasma_error("plasma_desc_handles_destroy() failed");
        return PlasmaErrorIllegalValue;
    }

    free(A->tile_handles);

    return PlasmaSuccess;
}

/******************************************************************************/
int plasma_desc_set_dist(MPI_Comm comm, int p, int q,
                         int (*tile_owner)(int p, int q, int i, int j),
                         plasma_desc_t *A)
{
    int rank, nproc;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &nproc);
    
    
    // Find process coordinates based on the rank.
    int ip, jp;
    plasma_rank2coords(rank, p, &ip, &jp);

    // Number of local rows and columns
    int lm = plasma_numroc(A->m, A->mb, ip, 0, p);
    int ln = plasma_numroc(A->n, A->nb, jp, 0, q);

    // Add MPI properties to descriptors.
    A->comm = comm;
    A->p = p;
    A->q = q;
    A->tile_owner = tile_owner; 

    // Number of local rows and columns.
    A->ml = lm;
    A->nl = ln;

    // Pad the sizes
    A->mtl = (lm%A->mb == 0) ? (lm/A->mb) : (lm/A->mb+1); // number of local tiles
    A->ntl = (ln%A->nb == 0) ? (ln/A->nb) : (ln/A->nb+1); // number of local tiles
    A->gml = A->mtl*A->mb; // padded to whole tiles
    A->gnl = A->ntl*A->nb; // padded to whole tiles

    // Increase the counter and set descriptor ID.
    A->id = descriptor_id++;
    
    return PlasmaSuccess;
}

/******************************************************************************/
starpu_data_handle_t plasma_desc_handle(plasma_desc_t A, int m, int n)
{

    int32_t index = n*A.gmt + m;
    int32_t id    = A.id;

    // Concatenate the id and the index into one long integer.
    // Make sure the sizes fit into the joint integer.
    assert(index < 8388607); // maximum number of tiles per descriptor
    assert(   id < 127);     // maximal supported number of descriptors in use
    int64_t plasma_data_tag = id << 24 | index;

    size_t eltsize = plasma_element_size(A.precision);

    int rank;
    MPI_Comm_rank(A.comm, &rank);

    // Initialize the handle on demand.
    if ((A.tile_handles)[index] == NULL) {
        (A.tile_handles)[index] = malloc(sizeof(starpu_data_handle_t));

        starpu_data_handle_t *handle = (A.tile_handles)[index];
        int remote_memory = -1;
        plasma_complex64_t *data_pointer = NULL;
        int owner = (A.tile_owner)(A.p, A.q, m, n);
        if (rank == owner) {
            remote_memory = STARPU_MAIN_RAM;
            data_pointer = plasma_tile_addr(A, m, n);
        }

        int ldai  = plasma_tile_mmain(A, m);
        int nmain = plasma_tile_nmain(A, n);

        starpu_matrix_data_register(handle,
                                    remote_memory,
                                    (uintptr_t) data_pointer,
                                    ldai, ldai, nmain, eltsize);

        starpu_mpi_data_register(*handle, plasma_data_tag, owner);
    }

    return *((A.tile_handles)[index]);
}

/******************************************************************************/
int plasma_desc_populate_nonlocal_tiles(plasma_desc_t *A)
{
    int rank;
    MPI_Comm_rank(A->comm, &rank);
    
    for (int i = 0; i < A->mt; i++) {
        for (int j = 0; j < A->nt; j++) {
            int owner = (A->tile_owner)(A->p, A->q,  i, j);
            plasma_complex64_t *pointer = plasma_tile_addr(*A, i, j);

            int naj = plasma_tile_nview(*A, j);
            int lda = plasma_tile_mmain(*A, i);

            MPI_Datatype mpi_type;
            if      (A->precision == PlasmaComplexDouble) {
                mpi_type = MPI_DOUBLE_COMPLEX;
            }
            else if (A->precision == PlasmaComplexFloat) {
                mpi_type = MPI_COMPLEX;
            }
            else if (A->precision == PlasmaRealDouble) {
                mpi_type = MPI_DOUBLE;
            }
            else if (A->precision == PlasmaRealFloat) {
                mpi_type = MPI_FLOAT;
            }
            else {
                return PlasmaErrorIllegalValue;
            }

            MPI_Bcast(pointer, lda*naj, mpi_type,
                      owner, MPI_COMM_WORLD);
        }
    }

    return PlasmaSuccess;
}


int plasma_starpu_data_acquire(plasma_desc_t *A)
{
    int rank;
    MPI_Comm_rank(A->comm, &rank);

    for (int i = 0; i < A->mt; i++) {
        for (int j = 0; j < A->nt; j++) {
            int index = j*A->gmt + i;
            starpu_data_handle_t *handle = (A->tile_handles)[index];
            int owner = (A->tile_owner)(A->p, A->q, i, j);

            if ((owner != rank) || (*handle == NULL)) {
                continue;
            }
            starpu_data_acquire(*handle, STARPU_R);
            starpu_data_release(*handle);
        }
    }
    return PlasmaSuccess;
}

