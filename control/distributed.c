/**
 *
 * @file
 *
 *  PLASMA is a software package provided by:
 *  University of Tennessee, US,
 *  University of Manchester, UK.
 **/

#include "plasma_context.h"
#include "plasma_distributed.h"

#include <math.h>
#include <assert.h>
#include <stdio.h>

#include <mpi.h>

int plasma_mpi_init()
{
    plasma_context_t *plasma;
    plasma = plasma_context_self();

    int flag = 0; 
    int provided = 0;

    MPI_Initialized(&flag);
    if (!flag) {
        MPI_Init_thread(NULL, NULL, MPI_THREAD_MULTIPLE, &provided);
    }

    plasma->mpi_outer_init = flag;
    plasma->comm           = MPI_COMM_WORLD;

    // Set default process grid P and Q
    int nproc;
    MPI_Comm_size(plasma->comm, &nproc);
    plasma_get_process_grid_dimension(nproc, &(plasma->p), &(plasma->q));
    
    return PlasmaSuccess;
}

int plasma_mpi_finalize()
{
    plasma_context_t *plasma;
    plasma = plasma_context_self();

    if (!plasma->mpi_outer_init) {
        MPI_Finalize();
    }

    return PlasmaSuccess;
}
