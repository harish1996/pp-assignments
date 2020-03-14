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

/**
 * Tuple for maintaining coordinates
 */
struct tuple{
	int first;
	int second;
};

/**
 * Copies a section of the window of image corresponding to the filter values which were copied.
 * 
 *	The filter is flipped for convolution. Instead of that, the image window is flipped, and the
 *	filter is kept constant.
 *
 *	To maintain a constant shared memory, blocks of KERNEL_BLOCK_SIZE are copied at once. As blocks
 *	of 64 filter elements are copied at one go, the corresponding image elements are also copied
 *	through this function.
 *
 * Parameters:
 *	s_img 	    - The shared memory space allocated for the image inside the block
 *	img	    - The global memory copy of the image
 *	img_size    - Image dimensions
 *	filter_size - Filter dimensions
 *	k 	    - Block number to be copied.
 */
__device__ void copyflippedwindow( float* s_img, float* img, struct tuple img_size, struct tuple filter_size, int k )
{
	int i = blockIdx.x;
	int j = blockIdx.y;
	
	// This may look weird.
	// n is the location of the filter element corresponding to the element corresponding to this
	// thread, when the filter is flattened to a 1D array.
	// i.e. Assume that this thread is trying to access the image element corresponding to (2,2) of the
	// filter( Assume a 5x5 filter ) when it is centered in (i,j) of the image. n now is 2*5 + 2 = 12, since
	// (2,2) will come at 12th position when flattened.	
	int n = k*KERNEL_BLOCK_SIZE + threadIdx.x;

	// From n, we are calculating the coordinate i.e. (2,2) in the example in the above comment.
	// Why go through all this?
	// 	This implementation attempts to split a single convolution into a set of seperate vector
	// 	products and vector sums. But as the filter size grows, the number of values that has to be
	// 	copied grows. To ensure it works for scalable filter sizes, this splits any sized filter
	// 	into blocks of 64 elements each step. But 64 necessarily wont be aligned with the filter size,
	//	i.e. 64 may not end nicely at the end of row. So simulating a flattening operation, and then
	//	taking a chunk of 64 elements, looks like the only way out of this. 
	int I = n / filter_size.second;
	int J = n % filter_size.second;

	// I,J( after this computation ) will be the actual image coordinates, which will get multiplied with
	// ( I-filtersize/2, J-filtersize/2 ) of the filter.
	// filtersize/2 is subtracted from J, because 0,0 of the filter lies at the center. But 0,0 of filter
       	// array corresponds to the top left element of the filter.	
	I = i - ( I - filter_size.first / 2 );
	J = j - ( J - filter_size.second / 2 );	
	

	if( I < img_size.first && J < img_size.second && I >= 0 && J >= 0 )
		s_img[ threadIdx.x ] = img[ I*img_size.second + J ];
	else
		// Zero Padding
		s_img[ threadIdx.x ] = 0;

	__syncthreads();
}

/**
 * Copies a section of the filter.
 * 	
 * 	For detailed explanation look at copyflippedimage function
 *
 * Parameters:
 *	s_filter    - The shared memory space allocated for the filter inside the block
 *	img	    - The global memory copy of the filter
 *	img_size    - Image dimensions
 *	filter_size - Filter dimensions
 *	k 	    - Block number to be copied.
 */
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

/**
 * Element-wise multiplication of two arrays
 */
__device__ void outerproduct( float *img, float *flt, float *out )
{
	int i = threadIdx.x;
	out[ i ] = img[ i ]*flt[i];

	__syncthreads();
}

/**
 * Calculates the sum of an array
 */
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

/***
  * Parallel convolution kernel.
  *
  *	This kernel splits the job of the convolution into multiple parts by allocating seperate blocks for each
  *	and every element of the output.
  *	For each block, This kernel again splits the job into multiple loops of computing the elementwise product
  *	and the sum, for blocks of KERNEL_BLOCK_SIZE elements. Detailed explanation available in copyflippedwindow
  *	function.
  *
  * Parameters:
  *	image  - Global memory copy of the image
  *	filter - Global memory copy of the filter
  *	output - Global memory space for the output
  * 	img_size - Image size
  *	filter_size - Filter size
  */ 
__global__ void convolve2d( float *image, float *filter, float *output, struct tuple img_size, struct tuple filter_size )
{
	int i,j;
	__shared__ float s_filter[KERNEL_BLOCK_SIZE];
	__shared__ float s_image[KERNEL_BLOCK_SIZE];
	__shared__ float temp[KERNEL_BLOCK_SIZE];

	i = blockIdx.x;
	j = blockIdx.y;

	// Total no. of elements in the filter.
	int fsize = filter_size.first * filter_size.second;

	// No. of blocks required to complete one computation for one window.
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

void sconvolve2d( float *in, float *filter, float *__restrict__ out, int in_side, int filter_side )
{
	float accum;
	int index[2];
	
	// Lot to unpack here..
	// The first set of loops, loops through each index in the output image. ( Since input and output
	// has the same size. )
	
	for( int i=0; i<in_side; i++ ){
		for( int j=0; j<in_side; j++ ){
			
			out[ i*in_side + j ] = 0;
			accum = 0;
			
			// Since this is a convolution, the window is flipped. So the (-x/2,-x/2) of the filter
			// will be multiplied with ( x/2,x/2 ) of the image. (-x/2,-x/2) of the filter corresponds
			// to (0,0) in the filter array used here.	
			index[0] = i + filter_side/2;
			
			// This loop, loops through the filter's X axis.
			for( int fi=0; fi<filter_side; fi++ ){
				
				// Skip the entire inner loop if index is outside the bounds of the image. Since it
				// will be zero padded, the output is anyway a 0, and doesn't affect the computation.
				if( index[0] >= 0 && index[0] < in_side ){
					
					// Second index calculation for window.
					index[1] = j + filter_side/2;
					
					// This is the inner filter loop, which loops through the filter's Y axis
					for( int fj=0; fj<filter_side; fj++ ){
						
						// Skip computation, if index out of bounds.
						if( index[1] >= 0 && index[1] < in_side )
							accum += in[ index[0]*in_side + index[1]	] *
								filter[ fi*filter_side + fj ];
						
						// Index is getting decremented due to the fact that we have inverted the
						// window.
						index[1]--;
					}
				}

				index[0]--;
			}

			out[ i*in_side + j ] = accum;
		}
	}	
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
	//float filter[]={
	//	2,0,0,
	//	0,0,0,
	//	0,0,1
	//};
	float *filter;

	int size;		/* Size of the matrix */
	int filter_size;

	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	
	float pms = 0, sms = 0;	/* Parallel and sequential times */

	if( argc != 3 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix> <filter size>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	filter_size = atoi( argv[2] );
	
	if( size <= 12 ) do_print= true;

	A = (float *) malloc( sizeof(float)* size * size );
	filter = (float *) malloc( sizeof(float) * filter_size * filter_size );
	out = (float *) malloc( sizeof(float)* size * size );
	
	generate_notso_random_matrix( A, size );
	generate_notso_random_matrix( filter, filter_size );

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
	
	struct timespec seq_start,seq_end;

	/* clock_gettime gets the process specific time spent, as opposed to the system time expended
	 */
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_start );
	
	sconvolve2d( A, filter, out, size, filter_size );

	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_end );

	/*************************
	 * End of Sequential Stuff
	 ************************/

	if( do_print ){
		printf("Out=\n");
		print_matrix( out, size );
	}
	
	/* Getting time in milliseconds for comparability */
	sms = ( (float)seq_end.tv_sec - seq_start.tv_sec )*1000 + ( (float)seq_end.tv_nsec - seq_start.tv_nsec ) / 1000000;
	printf("%12s %12s %12s %12s\n","N","Parallel","Sequential","Speedup");
	printf("%12d % 12f % 12f % 12f\n",size,pms,sms,sms/pms);


	free(A);
	free(filter);
	free(out);
}





