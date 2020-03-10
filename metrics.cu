/************************************************
 * MATRIX TRANSPOSE CHECK between parallel
 * 	and sequential programs.
 *
 * Usage:
 * 	Compile using nvcc -lcudart transpose.cu -o transpose
 *	Run using ./mat <size of the matrix>
 *
 * Example:
 *	./mat 153
 *	The above will check whether for a random matrix, A = transpose(A)
 *
 ************************************************/ 

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef __CUDA_SAFE_CALL
cudaError_t __cuda_error;
#define __CUDA_SAFE_CALL(call) do { __cuda_error = call; if(__cuda_error != cudaSuccess) { fprintf(stderr,"CUDA Error: %s,%s, line %d\n",cudaGetErrorString(__cuda_error), __FILE__, __LINE__ ); return -1;} } while(0)
#endif

void generate_random_vector( float *A, int size )
{
	srand( time(NULL) );

	for( int i=0; i<size; i++ )
		A[i] = 1.2;
		//A[i] = ((float)rand())/100000;
}

#define THREADS_PER_BLOCK 1024
#define TPB THREADS_PER_BLOCK

__device__ void addfour( float *A, int id, int threads, float *B )
{
	B[id] = A[id] + A[threads+id] + A[2*threads + id] + A[3*threads+id];
}

__global__ void add4096( float *A, float *B )
{
	__shared__ float nums[4096];
	int id = blockIdx.x;
	int tid = threadIdx.x;
	int offset = 4096*id;
	int alive = 1024;

	nums[        tid ] = A[offset +        tid ];
	nums[ 1024 + tid ] = A[offset + 1024 + tid ];
	nums[ 2048 + tid ] = A[offset + 2048 + tid ];
	nums[ 3072 + tid ] = A[offset + 3072 + tid ];

	__syncthreads();

	while( 1 ){
		if( tid < alive ){
			addfour( (float *)&nums, tid, alive, (float *)&nums );
			printf("total alive=%d i=%d, %5.2f\n",alive,tid,nums[tid]);
		}
		if( alive == 1 )
			break;
		alive = alive>>2;
		__syncthreads();
	}

	B[id] = nums[0];
}

float additall( float* A, int size )
{
	double ans=0;
	for( int i=0; i< size; i++ ){
		ans += A[i];
		printf("%lf \n",ans);
	}
	return ans;
}

int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	float *A,*B;
	float *ga,*gb;

	int size;		/* Number of elements */
	int vector_size;	/* Physical size of the elements in the memory */
	
	int num_blocks;		
	
	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	
	float pms = 0,sms=0;	/* Parallel and sequential times */

	
	if( argc != 2 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	if( size % 4096 != 0 ){
		fprintf(stderr, "Please enter a size divisible by 4096\n");
		return -1;
	}

	vector_size = sizeof(float)* size;
	
	if( size <= 32 ) do_print= true;

	A = (float *) malloc( vector_size );
	B = (float *) malloc( vector_size / 4096 );

	generate_random_vector( A, size );

	if( do_print ){
		for( int i=0; i<size; i++ )
			printf("%5.2f ",A[i]);
	}


	/* Timers to time the parallel process */ 

	__CUDA_SAFE_CALL( cudaSetDevice(2) );

	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*********************
	  * Start of GPU run
	  *******************/
	cudaEventRecord(start);

	__CUDA_SAFE_CALL( cudaMalloc( &ga, vector_size      ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, vector_size/4096 ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, vector_size, cudaMemcpyHostToDevice ) );

	num_blocks = size / 4096 ;

	add4096<<<num_blocks,1024>>> (ga, gb);
	
	__CUDA_SAFE_CALL( cudaMemcpy( B, gb, vector_size/4096, cudaMemcpyDeviceToHost ) );

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	/*****************
	 * End of GPU code
	 ****************/
	
	cudaEventElapsedTime( &pms, start, stop );

	for( int i=0; i< size/4096; i++ )
		printf(" Partial sums are %5.2f\n",B[i]);

	cudaFree( ga );
	cudaFree( gb );
	/*********************
	 * Sequential Stuff
	 ********************/
	struct timespec seq_start,seq_end;
	float ans;

	/* clock_gettime gets the process specific time spent, as opposed to the system time expended
	 */
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_start );
	
	ans = additall( A, size );	

	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_end );

	/*************************
	 * End of Sequential Stuff
	 ************************/

	printf("Sum is %f\n",ans);
	//float a = 1.2;
	//printf("1.2 is %d\n",*(int *)&a);
	/* Getting time in milliseconds for comparability */
	sms = ( (float)seq_end.tv_sec - seq_start.tv_sec )*1000 + ( (float)seq_end.tv_nsec - seq_start.tv_nsec ) / 1000000;

	printf("%12d % 12f % 12f % 12f\n",size,pms,sms,sms/pms);

	free(A);
	free(B);
}





