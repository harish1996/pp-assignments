#include "matrix_kernels.h"

/**
 * Parallel matrix multiplication routines
 */

#define MAX_BLOCK_SIZE (1<<4)

__device__ void copyblock( struct matrices *g, struct matrices *shared, int K, int side )
{
	int I = blockIdx.x;
	int J = blockIdx.y;
	int Ioffset = (I)*MAX_BLOCK_SIZE;
	int Joffset = (J)*MAX_BLOCK_SIZE;
	int Koffset = (K)*MAX_BLOCK_SIZE;

	int i = threadIdx.x;
	int j = threadIdx.y;
		
	shared->A[ i*MAX_BLOCK_SIZE + j ] = 0;
	shared->B[ i*MAX_BLOCK_SIZE + j ] = 0;
	
	if( Ioffset+i < side && Koffset+j < side )
		shared->A[ i*MAX_BLOCK_SIZE + j ] = g->A[ (Ioffset+i)*side + (Koffset+j) ];
	if( Koffset+i < side && Joffset+j < side )
		shared->B[ i*MAX_BLOCK_SIZE + j ] = g->B[ (Koffset+i)*side + (Joffset+j) ];
}

__device__ void updateOriginal( float *shared, float *global, int side )
{
	int I = blockIdx.x;
	int J = blockIdx.y;
	int Ioffset = (I)*MAX_BLOCK_SIZE;
	int Joffset = (J)*MAX_BLOCK_SIZE;

	int i = threadIdx.x;
	int j = threadIdx.y;
	
	if( Ioffset+i < side && Joffset+j < side )
		global[ (Ioffset+i)*side + (Joffset+j) ] = shared[i*MAX_BLOCK_SIZE+j];
}

__device__ void mulblock( float *A, float *B, float *C, int K )
{
	int i = threadIdx.x;
	int j = threadIdx.y;

	int element = i*MAX_BLOCK_SIZE + j;
	int a_in = i*MAX_BLOCK_SIZE;
	int b_in = j;

	for( int k=0; k<MAX_BLOCK_SIZE; k++ ){
		C[ element ] += A[ a_in ] * B[ b_in ];
		a_in += 1;
		b_in += MAX_BLOCK_SIZE;
	}
}

__global__ void pblockmultiply(float *A, float *B, float *C, int side )
{
	__shared__ float blockA[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	__shared__ float blockB[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	__shared__ float blockC[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	
	int total_k = gridDim.x;
	
	int i = threadIdx.x;
	int j = threadIdx.y;
	
	struct matrices input = { A,B }, block={ (float *)&blockA,(float *)&blockB};

	blockC[ i*MAX_BLOCK_SIZE + j ] = 0;
	for( int k=0; k< total_k; k++ ){
		copyblock( &input, &block, k, side );
		mulblock( blockA, blockB, blockC, k );
	}
	updateOriginal( blockC, C, side );
}


__global__ void pmatrixadd( float *A, float *B, float *C, int side )
{
	int i = ( blockIdx.x * blockDim.x + threadIdx.x );
	int j = ( blockIdx.y * blockDim.y + threadIdx.y );
	int el = i*side +j;

	C[ el ] = A[ el ] + B[ el ];
}


