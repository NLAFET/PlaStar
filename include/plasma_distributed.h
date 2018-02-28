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
// root owns everything
static inline int plasma_owner_root(int p, int q, int m, int n, int i, int j)
{
    int owner = 0;
    return owner;
}

// 1D chunks
static inline int plasma_owner_1D_chunks(int p, int q, int m, int n, int i, int j)
{
    int nproc = p*q;
    int chunk = (n + nproc - 1) / nproc;
    assert(chunk > 0);

    int owner = j / chunk;
    return owner;
}

// 2D chunks
static inline int plasma_owner_2D_chunks(int p, int q, int m, int n, int i, int j)
{
    int chunk_m = (m + p - 1) / p;
    int chunk_n = (n + q - 1) / q;
    assert(chunk_m > 0 && chunk_n > 0);

    int owner = (j / chunk_n)*q + (i / chunk_m);
    return owner;
}

// 1D block cyclic
static inline int plasma_owner_1D_block_cyclic(int p, int q, int m, int n, int i, int j)
{
    int nproc = p*q;
    int owner = j % nproc;
    return owner;
}

// 2D block cyclic
static inline int plasma_owner_2D_block_cyclic(int p, int q, int m, int n, int i, int j)
{
    int owner = (i % p) * q + (j % q);
    return owner;
}

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_DISTRIBUTED_H
