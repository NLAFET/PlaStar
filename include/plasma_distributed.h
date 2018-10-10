/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 *
 **/
#ifndef ICL_PLASMA_DISTRIBUTED_H
#define ICL_PLASMA_DISTRIBUTED_H

#include <stdlib.h>
#include <assert.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef STARPU_USE_MPI
#define STARPU_USE_MPI 1
#endif

// Given number of processes, find a nice p and q for the cartesian grid.
static inline void plasma_get_process_grid_dimension(int nproc, int *p, int *q)
{
    int div = 0;
    for (int i = 1; i <= (int) sqrt(nproc); i++) {
        // Find the largest divisor.
        if (nproc%i == 0) {
            div = i;
        }
    }
    assert(div != 0);

    // Set the grid dimensions.
    *p = div; 
    *q = nproc/div; 

    assert(*p * *q == nproc);
}

int plasma_mpi_init();

int plasma_mpi_finalize();

// *******************
// Data distributions.
// *******************
// 2D block cyclic - get rank from process coordinates
static inline int plasma_coords2rank(int ip, int jp, int p)
{
    // with column major numbering of processors into 2D grid:
    return ip + p*jp;
}

// 2D block cyclic - get process coordinates in process grid
static inline void plasma_rank2coords(int rank, int p, int *ip, int *jp)
{
    // with column major numbering of processors into 2D grid:
    *ip = rank % p;
    *jp = rank / p;
}

// 2D block cyclic - ownership of a tile
static inline int plasma_owner_2D_block_cyclic(int p, int q, int i, int j)
{
    // process coordinates for the tile
    int ip = (i % p);
    int jp = (j % q);

    int owner = plasma_coords2rank(ip, jp, p);
    return owner;
}

// 2D block cyclic - global tile indices to local tile indices
static inline void plasma_tile_global2local(int i, int j, int p, int q,
                                            int *i_loc, int *j_loc)
{
    // local index of the tile in the local matrix
    *i_loc = i / p;
    *j_loc = j / q;
}

static inline int plasma_numroc(int n, int nb, int iproc, int isrcproc, int nprocs)
{
//  Purpose
//  =======
//
//  NUMROC computes the NUMber of Rows Or Columns of a distributed
//  matrix owned by the process indicated by IPROC.
//  Converted from the ScaLAPACK function.
//
//  Arguments
//  =========
//
//  N         (global input) INTEGER
//            The number of rows/columns in distributed matrix.
//
//  NB        (global input) INTEGER
//            Block size, size of the blocks the distributed matrix is
//            split into.
//
//  IPROC     (local input) INTEGER
//            The coordinate of the process whose local array row or
//            column is to be determined.
//
//  ISRCPROC  (global input) INTEGER
//            The coordinate of the process that possesses the first
//            row or column of the distributed matrix.
//
//  NPROCS    (global input) INTEGER
//            The total number processes over which the matrix is
//            distributed.

     // Figure PROC's distance from source process
     int mydist = (nprocs+iproc-isrcproc) % nprocs;

     // Figure the total number of whole NB blocks N is split up into
     int nblocks = n / nb;

     // Figure the minimum number of rows/cols a process can have
     int numroc = (nblocks/nprocs) * nb;

     // See if there are any extra blocks
     int extrablks = nblocks % nprocs;

     // If I have an extra block
     if      (mydist < extrablks) {
         numroc = numroc + nb;
     }
     else if (mydist == extrablks) {
         // If I have last block, it may be a partial block
         numroc = numroc + n%nb;
     }

     return numroc;
}    

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DISTRIBUTED_H
