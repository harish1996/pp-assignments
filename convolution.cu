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

/**
 * Safety macro
 * 	Convenience macro which checks the output of all CUDA calls, and prints the verbose error incase of any
 */
#ifndef __CUDA_SAFE_CALL
cudaError_t __cuda_error;
#define __CUDA_SAFE_CALL(call) do { __cuda_error = call; if(__cuda_error != cudaSuccess) { fprintf(stderr,"CUDA Error: %s,%s, line %d\n",cudaGetErrorString(__cuda_error), __FILE__, __LINE__ ); exit(-1);} } while(0)
#endif

#define KERNEL_BLOCK_SIZE 64

struct tuple{
	int first;
	int second;
};

__device__ void copyflippedwindow( float* s_img, float* img, struct tuple img_size, struct tuple filter_size, int k )
{
	int i = blockIdx.x;
	int j = blockIdx.y;
		
	int n = k*KERNEL_BLOCK_SIZE + threadIdx.x;

	int I = n / filter_size.second;
	int J = n % filter_size.second;
	I = i - ( I - filter_size.first / 2 );
	J = j - ( J - filter_size.second / 2 );	
	

	if( I < img_size.first && J < img_size.second && I >= 0 && J >= 0 )
		s_img[ threadIdx.x ] = img[ I*img_size.second + J ];
	else
		s_img[ threadIdx.x ] = 0;

	__syncthreads();
}

__device__ void copyfilter( float *s_filter, float *filter, struct tuple img_size, struct tuple filter_size, int k )
{	
	int n = k*KERNEL_BLOCK_SIZE + threadIdx.x;

	int I = n / filter_size.second;
	int J = n % filter_size.second;

	//I = ( I - ( filter_size.first / 2 ) );
	//J = ( J - ( filter_size.second / 2 ) );

	if( I < filter_size.first && J < filter_size.second )
		s_filter[ threadIdx.x ] = filter[ n ];
	else
		s_filter[ threadIdx.x ] = 0;

	__syncthreads();
}

__device__ void outerproduct( float *img, float *flt, float *out )
{
	int i = threadIdx.x;
	out[ i ] = img[ i ]*flt[i];

	__syncthreads();
}

__device__ void reducesum( float *in )
{
	int alive= KERNEL_BLOCK_SIZE >> 1;
	
	while( alive >= 1 ){
		if( threadIdx.x < alive ){
			in[ threadIdx.x ] += in[ threadIdx.x + alive ];
		}
		else
			break;

		alive = alive >> 1;
	}
	__syncthreads();

}	
	//J = threadIdx.y;

__global__ void convolve2d( float *image, float *filter, float *output, struct tuple img_size, struct tuple filter_size )
{
	int i,j;
	__shared__ float s_filter[KERNEL_BLOCK_SIZE];
	__shared__ float s_image[KERNEL_BLOCK_SIZE];
	__shared__ float temp[KERNEL_BLOCK_SIZE];

	i = blockIdx.x;
	j = blockIdx.y;

	//struct tuple out_size;
	
	//int isize = img_size.first * img_size.second;
	int fsize = filter_size.first * filter_size.second;
	int blocks = ( fsize - 1 )/KERNEL_BLOCK_SIZE + 1;
	float accum = 0;

	for( int k=0; k<blocks; k++ ){
		
		copyflippedwindow( (float *)&s_image, image, img_size, filter_size, k );
		copyfilter( (float *)&s_filter, filter, img_size, filter_size, k );
		outerproduct( (float *)&s_image, (float *)&s_filter, (float *)&temp );
		reducesum( (float *)&temp );
		//if( threadIdx.x == 0 )
		accum += temp[0];
		
		__syncthreads();
	}
	
	if( threadIdx.x == 0 && i< img_size.first && j < img_size.second )
		output[ i*img_size.second + j ] = accum;
	
}
	
/**
 * LHS = ( A + B )*( A + B )
 */
void pconvolve2d( float *in, float *filter, float *__restrict__ out, int in_side, int filter_side )
{
	float *ga, *gf, *gb;

	int matrix_size = in_side * in_side * sizeof(float);
	int filter_size = filter_side * filter_side * sizeof(float);

	int dim_thread = KERNEL_BLOCK_SIZE;
	int num_blocks = in_side;
	
	dim3 block(dim_thread);
	dim3 grid(num_blocks,num_blocks);

	struct tuple fsize = {filter_side, filter_side }, isize= {in_side, in_side};

	__CUDA_SAFE_CALL( cudaMalloc( &ga, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gf, filter_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, matrix_size ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, in, matrix_size, cudaMemcpyHostToDevice ) );	
	__CUDA_SAFE_CALL( cudaMemcpy( gf, filter, filter_size, cudaMemcpyHostToDevice ) );

	convolve2d<<<grid,block>>>( ga, gf, gb, isize, fsize );

	__CUDA_SAFE_CALL( cudaMemcpy( out, gb, matrix_size, cudaMemcpyDeviceToHost ) );

	cudaFree( ga );
	cudaFree( gf );
	cudaFree( gb );
}

void print_matrix( float *A, int side )
{
	//printf("A=\n");
	for( int i=0; i<side; i++ ){
		for( int j=0; j<side; j++ ){
			printf("% 5.2f ",A[i*side + j]);
		}
		printf("\n");
	}
}

int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	float *A,*out;
	float filter[]={
		2,0,0,
		0,0,0,
		0,0,1
	};

	int size;		/* Size of the matrix */
	int filter_size;

	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	
	float pms = 0,pms2=0, sms = 0;	/* Parallel and sequential times */

	if( argc != 3 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix> <filter size>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	filter_size = atoi( argv[2] );
	
	if( size <= 12 ) do_print= true;

	A = (float *) malloc( sizeof(float)* size * size );
	//filter = (float *) malloc( sizeof(float) * filter_size * filter_size );
	out = (float *) malloc( sizeof(float)* size * size );
	
	generate_notso_random_matrix( A, size );
	//generate_notso_random_matrix( B, filter_size );

	if( do_print ){
		printf("A=\n");
		print_matrix( A, size );
		printf("filter=\n");
		print_matrix( filter, filter_size );
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
	
	pconvolve2d( A, filter, out, size, filter_size );

	/*****************
	 * End of RHS run
	 ****************/
	
	__CUDA_SAFE_CALL( cudaEventRecord(stop) );
	__CUDA_SAFE_CALL( cudaEventSynchronize(stop) );

	__CUDA_SAFE_CALL( cudaEventElapsedTime( &pms, start, stop ) );
	
	if( do_print ){
		printf("Out=\n");
		print_matrix( out, size );
	}

	//printf("%12d % 12f % 12f % 12f\n",size,pms,pms2,pms2/pms);

	free(A);
	//free(filter);
	free(out);
}





