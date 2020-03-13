/************************************************
 * MATRIX MULTIPLICATION and ADDITION.
 * 	Checking which is faster (A+B)^2 or ( A^2 + AB + BA + B^2 )
 *
 * Usage:
 * 	Compile using nvcc -lcudart random_gens.c matrix_kernels.cu matrixmul.cu -o matrixmul
 *	Run using ./matrixmul <size of the matrix>
 *
 * Notes:
 * 	Uncomment line number 157, if you try to run it in CSSC's computation
 *	server
 *
 * Example:
 *	./mat 153
 *
 ************************************************/ 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include "random_gens.h"
#include "matrix_kernels.h"

/**
 * Safety macro
 * 	Convenience macro which checks the output of all CUDA calls, and prints the verbose error incase of any
 */
#ifndef __CUDA_SAFE_CALL
cudaError_t __cuda_error;
#define __CUDA_SAFE_CALL(call) do { __cuda_error = call; if(__cuda_error != cudaSuccess) { fprintf(stderr,"CUDA Error: %s,%s, line %d\n",cudaGetErrorString(__cuda_error), __FILE__, __LINE__ ); exit(-1);} } while(0)
#endif

/**
 * LHS = ( A + B )*( A + B )
 */
void lhs( float *A, float *B, float *C, int side )
{
	float *ga, *gb, *gc;
	int matrix_size = side * side * sizeof(float);
	int dim_thread;
	int num_blocks;
	dim3 block;
	dim3 grid;

	__CUDA_SAFE_CALL( cudaMalloc( &ga, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gc, matrix_size ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, matrix_size, cudaMemcpyHostToDevice ) );	
	__CUDA_SAFE_CALL( cudaMemcpy( gb, B, matrix_size, cudaMemcpyHostToDevice ) );

	/* A + B */
	dim_thread = 32;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;

	pmatrixadd<<<grid,block>>>( ga, gb, gc, side );

	/* ( A + B ) ( A + B ) */
	dim_thread = MAX_BLOCK_SIZE;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pblockmultiply<<<grid,block>>> (gc, gc, ga, side);
	
	__CUDA_SAFE_CALL( cudaMemcpy( C, ga, matrix_size, cudaMemcpyDeviceToHost ) );

	cudaFree( ga );
	cudaFree( gb );
	cudaFree( gc );
}

/**
 * RHS = ( A*A + A*B + B*A + B*B )
 */
void rhs( float *A, float *B, float *C, int side )
{
	float *ga, *gb, *gc1, *gc2;
	int matrix_size = side * side * sizeof(float);
	int dim_thread;
	int num_blocks;
	dim3 block;
	dim3 grid;

	__CUDA_SAFE_CALL( cudaMalloc( &ga, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gc1, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gc2, matrix_size ) );
	
	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, matrix_size, cudaMemcpyHostToDevice ) );	
	__CUDA_SAFE_CALL( cudaMemcpy( gb, B, matrix_size, cudaMemcpyHostToDevice ) );

	/* AA */
	dim_thread = MAX_BLOCK_SIZE;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;

	pblockmultiply<<<grid,block>>>( ga, ga, gc1, side );

	/* AB */
	dim_thread = MAX_BLOCK_SIZE;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pblockmultiply<<<grid,block>>> (ga, gb, gc2, side);
		
	/* AA + AB */
	dim_thread = 32;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pmatrixadd<<<grid,block>>> (gc1, gc2, gc1, side);
	
	/* BA */
	dim_thread = MAX_BLOCK_SIZE;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pblockmultiply<<<grid,block>>> (gb, ga, gc2, side);

	/* AA + AB + BA */
	dim_thread = 32;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pmatrixadd<<<grid,block>>> (gc1, gc2, gc1, side);
	
	/* BB */
	dim_thread = MAX_BLOCK_SIZE;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pblockmultiply<<<grid,block>>> (gb, gb, gc2, side);

	/* AA + AB + BA + BB */
	dim_thread = 32;
	num_blocks = ( side - 1 )/dim_thread + 1;
	grid.x = num_blocks;
	grid.y = num_blocks;
	block.x = dim_thread;
	block.x = dim_thread;
	
	pmatrixadd<<<grid,block>>> (gc1, gc2, gc1, side);
	
	__CUDA_SAFE_CALL( cudaMemcpy( C, ga, matrix_size, cudaMemcpyDeviceToHost ) );

	cudaFree( ga );
	cudaFree( gb );
	cudaFree( gc1 );
	cudaFree( gc2 );
}

void print_matrix( float *A, int side )
{
	//printf("A=\n");
	for( int i=0; i<side; i++ ){
		for( int j=0; j<side; j++ ){
			printf("%5.2f ",A[i*side + j]);
		}
		printf("\n");
	}
}

int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	float *A,*B,*C,*C2;

	int size;		/* Size of the matrix */
	
	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	
	float pms = 0,pms2=0;	/* Parallel and sequential times */

	if( argc != 2 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	
	//if( size <= 12 ) do_print= true;

	A = (float *) malloc( sizeof(float)* size * size );
	B = (float *) malloc( sizeof(float)* size * size );
       	C = (float *) malloc( sizeof(float)* size * size );
	C2 = (float *) malloc( sizeof(float)* size * size );
	
	generate_notso_random_matrix( A, size );
	generate_notso_random_matrix( B, size );

	if( do_print ){
		printf("A=\n");
		print_matrix( A, size );
		printf("B=\n");
		print_matrix( B, size );
	}


	/* Uncomment the below line to run this code in CSSC Computation server.
	   CSSC's 0th device is always occupied and fails to allocate any size of
	   memory consistently.
	*/
	__CUDA_SAFE_CALL( cudaSetDevice(2) );

	/* Timers to time the parallel process */ 
	__CUDA_SAFE_CALL( cudaEventCreate(&start) );
	__CUDA_SAFE_CALL( cudaEventCreate(&stop) );

	/*********************
	 * Start of RHS GPU run
	 *******************/
	__CUDA_SAFE_CALL( cudaEventRecord(start) );
	
	rhs( A, B, C, size );

	/*****************
	 * End of RHS run
	 ****************/
	
	__CUDA_SAFE_CALL( cudaEventRecord(stop) );
	__CUDA_SAFE_CALL( cudaEventSynchronize(stop) );

	__CUDA_SAFE_CALL( cudaEventElapsedTime( &pms, start, stop ) );
	
	/* Timers to time the parallel process */ 
	__CUDA_SAFE_CALL( cudaEventCreate(&start) );
	__CUDA_SAFE_CALL( cudaEventCreate(&stop) );

	/*********************
	 * Start of LHS GPU run
	 *******************/
	__CUDA_SAFE_CALL( cudaEventRecord(start) );

	lhs( A, B, C, size );

	__CUDA_SAFE_CALL( cudaEventRecord(stop) );
	__CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	/*****************
	 * End of GPU code
	 ****************/
	
	__CUDA_SAFE_CALL( cudaEventElapsedTime( &pms2, start, stop ) );

	if( do_print ){
		printf("C=\n");
		print_matrix( C, size );
	}

	if( do_print ){
		printf("C2=\n");
		print_matrix( C2, size );
	}

	printf("%12d % 12f % 12f % 12f\n",size,pms,pms2,pms2/pms);

	free(A);
	free(B);
	free(C);
	free(C2);
}





