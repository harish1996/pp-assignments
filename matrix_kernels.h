#ifndef __MATRIX_KERNELS_H
#define __MATRIX_KERNELS_H

#include <stdio.h>
#include <stdlib.h>

#define MAX_BLOCK_SIZE (1<<4)

struct matrices{
	float *A,*B;
};

__global__ void pblockmultiply(float *A, float *B, float *C, int side );

__global__ void pmatrixadd( float *A, float *B, float *C, int side );

#endif
