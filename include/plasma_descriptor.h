/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_DESCRIPTOR_H
#define ICL_PLASMA_DESCRIPTOR_H

#include "plasma_types.h"
#include "plasma_error.h"
#include "plasma_distributed.h"

#include <stdlib.h>
#include <assert.h>

#include <starpu.h>
#include <starpu_mpi.h>
#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 * @ingroup plasma_descriptor
 *
 * Tile matrix descriptor.
 *
 *              n1      n2
 *         +----------+---+
 *         |          |   |    m1 = lm - (lm%mb)
 *         |          |   |    m2 = lm%mb
 *     m1  |    A11   |A12|    n1 = ln - (ln%nb)
 *         |          |   |    n2 = ln%nb
 *         |          |   |
 *         +----------+---+
 *     m2  |    A21   |A22|
 *         +----------+---+
 *
 **/
typedef struct {
    // matrix properties
    plasma_enum_t type;      ///< general, general band, etc.
    plasma_enum_t uplo;      ///< upper, lower, etc.
    plasma_enum_t precision; ///< precision of the matrix

    // pointer and offsets
    void *matrix; ///< pointer to the beginning of the matrix
    size_t A21;   ///< pointer to the beginning of A21
    size_t A12;   ///< pointer to the beginning of A12
    size_t A22;   ///< pointer to the beginning of A22

    // tile parameters
    int mb; ///< number of rows in a tile
    int nb; ///< number of columns in a tile

    // main matrix parameters
    int gm;  ///< number of rows of the entire matrix
    int gn;  ///< number of columns of the entire matrix
    int gmt; ///< number of tile rows of the entire matrix
    int gnt; ///< number of tile columns of the entire matrix

    // submatrix parameters
    int i;  ///< row index to the beginning of the submatrix
    int j;  ///< column index to the beginning of the submatrix
    int m;  ///< number of rows of the submatrix
    int n;  ///< number of columns of the submatrix
    int mt; ///< number of tile rows of the submatrix
    int nt; ///< number of tile columns of the submatrix

    // submatrix parameters for a band matrix
    int kl;  ///< number of rows below the diagonal
    int ku;  ///< number of rows above the diagonal
    int klt; ///< number of tile rows below the diagonal tile
    int kut; ///< number of tile rows above the diagonal tile
             ///  includes the space for potential fills, i.e., kl+ku

    // array of StarPU handles to tiles
    starpu_data_handle_t **tile_handles;

    // related to distribution
    MPI_Comm comm; ///< MPI_Communicator
    int p;   ///< number of rows in a 2D processor grid
    int q;   ///< number of columns in a 2D processor grid
    // function describing ownership of a tile
    int (*tile_owner) (int p, int q, int i, int j);
    int id;  ///< id for distributed tag asignment

    // Properties for distributed matrix.
    int mtl; ///< number of local tile rows
    int ntl; ///< number of local tile columns

    int ml; ///< number of local rows
    int nl; ///< number of local columns

    int gml; ///< local number of rows padded to whole tiles
    int gnl; ///< local number of columns padded to whole tiles
    
} plasma_desc_t;

/******************************************************************************/
static inline size_t plasma_element_size(int type)
{
    switch (type) {
    case PlasmaByte:          return          1;
    case PlasmaInteger:       return   sizeof(int);
    case PlasmaRealFloat:     return   sizeof(float);
    case PlasmaRealDouble:    return   sizeof(double);
    case PlasmaComplexFloat:  return 2*sizeof(float);
    case PlasmaComplexDouble: return 2*sizeof(double);
    default: assert(0);
    }
}

/******************************************************************************/
static inline void *plasma_tile_addr_general(plasma_desc_t A, int m, int n) 
{
    // find local indices
    int m_loc, n_loc;
    plasma_tile_global2local(m, n, A.p, A.q, &m_loc, &n_loc);

    size_t eltsize = plasma_element_size(A.precision);
    size_t offset = 0;

    offset = (m_loc + A.mtl * n_loc)*A.mb*A.nb;
    //printf("m, n, offset = %d, %d, %d \n",m, n, offset);

    return (void*)((char*)A.matrix + (offset*eltsize));
}
    
/******************************************************************************/
/* static inline void *plasma_tile_addr_triangle(plasma_desc_t A, int m, int n) */
//static inline void *plasma_tile_addr_triangle(plasma_desc_t A, int m, int n)
//{
//    // find local indices
//    int ml = m/A.p;
//    int nl = n/A.q;
//
//    size_t eltsize = plasma_element_size(A.precision);
//    size_t offset = 0;
//
//    if (A.type == PlasmaUpper) {
//        offset = A.mb*A.nb*(ml + (nl * (nl + 1))/2);
//    } 
//    else {
//        offset = A.mb*A.nb*((ml - nl) + (nl * (2*A.mtl - (nl-1)))/2);
//    }
//
//    return (void*)((char*)A.matrix + (offset*eltsize));
//}

/******************************************************************************/
static inline void *plasma_tile_addr(plasma_desc_t A, int m, int n)
{
    return plasma_tile_addr_general(A, m, n);
}
    
/***************************************************************************//**
 *
 *  Returns the height of the tile with vertical position k.
 *
 */
static inline int plasma_tile_mmain(plasma_desc_t A, int k)
{
    (void) k;
    return A.mb;
}    
    
/***************************************************************************//**
 *
 *  Returns the width of the tile with horizontal position k.
 *
 */
static inline int plasma_tile_nmain(plasma_desc_t A, int k)
{
    (void) k;
    return A.nb;
}

    
/***************************************************************************//**
 *
 *  Returns the height of the portion of the submatrix occupying the tile
 *  at vertical position k.
 *
 */
static inline int plasma_tile_mview(plasma_desc_t A, int k)
{
    if (k < A.mt-1)
        return A.mb;
    else
        if ((A.i+A.m)%A.mb == 0)
            return A.mb;
        else
            return (A.i+A.m)%A.mb;
}

/***************************************************************************//**
 *
 *  Returns the width of the portion of the submatrix occupying the tile
 *  at horizontal position k.
 *
 */
static inline int plasma_tile_nview(plasma_desc_t A, int k)
{
    if (k < A.nt-1)
        return A.nb;
    else
        if ((A.j+A.n)%A.nb == 0)
            return A.nb;
        else
            return (A.j+A.n)%A.nb;
}

/******************************************************************************/
int plasma_desc_general_create(plasma_enum_t dtyp, int mb, int nb,
                               int lm, int ln, int i, int j, int m, int n,
                               plasma_desc_t *A);

int plasma_desc_general_band_create(plasma_enum_t dtyp, plasma_enum_t uplo,
                                    int mb, int nb, int lm, int ln,
                                    int i, int j, int m, int n, int kl, int ku,
                                    plasma_desc_t *A);

int plasma_desc_triangular_create(plasma_enum_t dtyp, plasma_enum_t uplo, int mb, int nb,
                                  int lm, int ln, int i, int j, int m, int n,
                                  plasma_desc_t *A);

int plasma_desc_destroy(plasma_desc_t *A);

int plasma_desc_general_init(plasma_enum_t precision, void *matrix,
                             int mb, int nb, int lm, int ln, int i, int j,
                             int m, int n, plasma_desc_t *A);

int plasma_desc_general_band_init(plasma_enum_t precision, plasma_enum_t uplo,
                                  void *matrix, int mb, int nb, int lm, int ln,
                                  int i, int j, int m, int n, int kl, int ku,
                                  plasma_desc_t *A);

int plasma_desc_triangular_init(plasma_enum_t precision, plasma_enum_t uplo, void *matrix,
                                int mb, int nb, int lm, int ln, int i, int j,
                                int m, int n, plasma_desc_t *A);

int plasma_desc_check(plasma_desc_t A);
int plasma_desc_general_check(plasma_desc_t A);
int plasma_desc_general_band_check(plasma_desc_t A);

plasma_desc_t plasma_desc_view(plasma_desc_t A, int i, int j, int m, int n);

int plasma_descT_create(plasma_desc_t A, int ib, plasma_enum_t householder_mode,
                        plasma_desc_t *T);

// StarPU related stuff
int plasma_desc_handles_create(plasma_desc_t *A);
int plasma_desc_handles_destroy(plasma_desc_t *A);

int plasma_desc_set_dist(MPI_Comm comm, int p, int q,
                         int (*tile_owner)(int p, int q,  int i, int j),
                         plasma_desc_t *A);

//static inline void *plasma_desc_handle_addr(plasma_desc_t A, int m, int n) {
//    return (A.tile_handles)[n*A.gmt + m];
//}

starpu_data_handle_t plasma_desc_handle(plasma_desc_t A, int m, int n);

int plasma_desc_populate_nonlocal_tiles(plasma_desc_t *A);
int plasma_starpu_data_acquire(plasma_desc_t *A);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DESCRIPTOR_H
