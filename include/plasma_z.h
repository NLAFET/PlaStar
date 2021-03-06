/**
 *
 * @file
 *
 *  PLASMA header.
 *  PLASMA is a software package provided by Univ. of Tennessee,
 *  Univ. of Manchester, Univ. of California Berkeley and
 *  Univ. of Colorado Denver.
 *
 * @precisions normal z -> s d c
 *
 **/
#ifndef ICL_PLASMA_Z_H
#define ICL_PLASMA_Z_H

#include "plasma_async.h"
#include "plasma_barrier.h"
#include "plasma_descriptor.h"
#include "plasma_workspace.h"

#ifdef __cplusplus
extern "C" {
#endif

/***************************************************************************//**
 *  Standard interface.
 **/
int plasma_zgemm(plasma_enum_t transa, plasma_enum_t transb,
                 int m, int n, int k,
                 plasma_complex64_t alpha, plasma_complex64_t *pA, int lda,
                                           plasma_complex64_t *pB, int ldb,
                 plasma_complex64_t beta,  plasma_complex64_t *pC, int ldc);

int plasma_zgeqrf(int m, int n,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t *T);

int plasma_zgetrf(int m, int n,
                  plasma_complex64_t *pA, int lda, int *iPiv);

int plasma_zlacpy(plasma_enum_t uplo, plasma_enum_t transa,
                  int m, int n,
                  plasma_complex64_t *pA, int lda,
                  plasma_complex64_t *pB, int ldb);

double plasma_zlange(plasma_enum_t norm,
                     int m, int n,
                     plasma_complex64_t *pA, int lda);

int plasma_zgeswp(plasma_enum_t colrow,
                  int m, int n,
                  plasma_complex64_t *pA, int lda,
                  int *ipiv, int incx);

int plasma_zpotrf(plasma_enum_t uplo,
                  int n,
                  plasma_complex64_t *pA, int lda);

int plasma_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                 plasma_enum_t transa, plasma_enum_t diag,
                 int m, int n,
                 plasma_complex64_t alpha, plasma_complex64_t *pA, int lda,
                                           plasma_complex64_t *pB, int ldb);

int plasma_zungqr(int m, int n, int k,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t T,
                  plasma_complex64_t *pQ, int ldq);

int plasma_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                  int m, int n, int k,
                  plasma_complex64_t *pA, int lda,
                  plasma_desc_t T,
                  plasma_complex64_t *pC, int ldc);

/***************************************************************************//**
 *  Tile asynchronous interface.
 **/
void plasma_starpu_zdesc2ge(plasma_desc_t A,
                            plasma_complex64_t *pA, int lda,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request);

void plasma_starpu_zdesc2tr(plasma_desc_t A,
                            plasma_complex64_t *pA, int lda,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request);

void plasma_starpu_zge2desc(plasma_complex64_t *pA, int lda,
                            plasma_desc_t A,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request);

void plasma_starpu_zgemm(plasma_enum_t transa, plasma_enum_t transb,
                         plasma_complex64_t alpha, plasma_desc_t A,
                                                   plasma_desc_t B,
                         plasma_complex64_t beta,  plasma_desc_t C,
                         plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_zgeqrf(plasma_desc_t A, plasma_desc_t T,
                          plasma_workspace_t work,
                          plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_zgetrf(plasma_desc_t A, starpu_data_handle_t *pnlA,
                                           starpu_data_handle_t *pnlPiv,
                          plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_zpotrf(plasma_enum_t uplo, plasma_desc_t A,
                          plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_ztr2desc(plasma_complex64_t *pA, int lda,
                            plasma_desc_t A,
                            plasma_sequence_t *sequence,
                            plasma_request_t *request);

void plasma_starpu_ztrsm(plasma_enum_t side, plasma_enum_t uplo,
                        plasma_enum_t transa, plasma_enum_t diag,
                        plasma_complex64_t alpha, plasma_desc_t A,
                                                plasma_desc_t B,
                        plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_zungqr(plasma_desc_t A, plasma_desc_t T,
                          plasma_desc_t Q, plasma_workspace_t work,
                          plasma_sequence_t *sequence, plasma_request_t *request);

void plasma_starpu_zunmqr(plasma_enum_t side, plasma_enum_t trans,
                          plasma_desc_t A, plasma_desc_t T,
                          plasma_desc_t C, plasma_workspace_t work,
                          plasma_sequence_t *sequence, plasma_request_t *request);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif // ICL_PLASMA_Z_H
