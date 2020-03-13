/************************************************
 * MATRIX TRANSPOSE CHECK between parallel
 * 	and sequential programs.
 *
 * Usage:
 * 	Compile using nvcc -lcudart transpose.cu -o transpose
 *	Run using ./mat <size of the matrix>
 *
 * Notes:
 * 	Uncomment line number 157, if you try to run it in CSSC's computation
 *	server
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

void generate_assymmetric_matrix( float *A, int size, float assymmetricity )
{
	float r;
	/* Initializing random number generator */
	srand( time(NULL) );
	
	for( int i=0; i<size; i++ ){
		for( int j=0; j<=i; j++ ){
			A[i*size + j] = (float)rand() / 100000;
			r = (float)( (rand()%30000) * 10 ) / 30000 ;
		      	if ( r < assymmetricity ){
				A[ j*size + i ] = (float)rand() / 100000;
				//printf("%f %d %d\n",r,i,j);
			}
			else	
				A[j*size + i] = A[i*size + j];
		}
	}
}

void generate_symmetric_matrix( float *A, int size )
{
	/* Initializing random number generator */
	srand( time(NULL) );
	
	for( int i=0; i<size; i++ ){
		for( int j=0; j<=i; j++ ){
			A[i*size + j] = rand() / 100000;
			A[j*size + i] = A[i*size + j];
		}
	}
}
	
void generate_notso_random_matrix( float *A, int size )
{
	srand( time(NULL) );

	int r = rand()%100;
	if( r <= 50 )
		generate_assymmetric_matrix( A, size, 0.1 );
	else
		generate_symmetric_matrix( A, size );
}

#define MAX_BLOCK_SIZE (1<<4)

struct matrices{
	float *A,*B;
};

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
	//printf("%d %d %p %p\n",Ioffset+i,Koffset+j,i*MAX_BLOCK_SIZE + j,shared->B);
	if( Ioffset+i < side && Koffset+j < side )
		shared->A[ i*MAX_BLOCK_SIZE + j ] = g->A[ (Ioffset+i)*side + (Koffset+j) ];
	if( Koffset+i < side && Joffset+j < side )
		shared->B[ i*MAX_BLOCK_SIZE + j ] = g->B[ (Koffset+i)*side + (Joffset+j) ];
}

__global__ void blockmultiply(float *A, float *B, float *C, int side )
{
	__shared__ float blockA[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	__shared__ float blockB[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	__shared__ float blockC[MAX_BLOCK_SIZE*MAX_BLOCK_SIZE];
	
	int total_k = gridDim.x;
	
	int I = blockIdx.x;
	int J = blockIdx.y;

	int i = threadIdx.x;
	int j = threadIdx.y;
	struct matrices input = { A,B }, block={ (float *)&blockA,(float *)&blockB};
	//printf("%d %d %d\n",I,J,total_k);

	for( int k=0; k< total_k; k++ ){
		copyblock( &input, &block, k, side );
		//printf("Block %d,%d,k=%d\n",I,J,k); 
		printf("I=%d J=%d i=%d j=%d k=%d v=% 5.2f\n",I,J,i,j,k,blockA[i*MAX_BLOCK_SIZE + j]);
	}	
}

int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	float *A,*B,*C;
	float *ga,*gb,*gc;

	int size;		/* Size of the matrix */
	int matrix_size;	/* Physical size of the matrix in the memory */
	
	int num_blocks;		
	
	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	int dim_thread = 16;	/* Number of threads in each block */
	
	float pms = 0,sms=0;	/* Parallel and sequential times */

	
	if( argc != 2 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	matrix_size = sizeof(float)* size * size;
	
	if( size <= 12 ) do_print= true;

	A = (float *) malloc( sizeof(float)* size * size );
	B = (float *) malloc( sizeof(float)* size * size );
       	C = (float *) malloc( sizeof(float)* size * size );

	generate_notso_random_matrix( A, size );
	generate_notso_random_matrix( B, size );

	if( do_print ){
		printf("A=\n");
		for( int i=0; i<size; i++ ){
			for( int j=0; j<size; j++ ){
				printf("%5.2f ",A[i*size + j]);
			}
			printf("\n");
		}
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
	 * Start of GPU run
	 *******************/
	__CUDA_SAFE_CALL( cudaEventRecord(start) );


	__CUDA_SAFE_CALL( cudaMalloc( &ga, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, matrix_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gc, matrix_size ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, matrix_size, cudaMemcpyHostToDevice ) );	
	__CUDA_SAFE_CALL( cudaMemcpy( gb, B, matrix_size, cudaMemcpyHostToDevice ) );

	num_blocks = ( size - 1 )/dim_thread + 1;

	dim3 block( dim_thread, dim_thread );
	dim3 grid( num_blocks, num_blocks);
	
	blockmultiply<<<grid,block>>> (ga, gb, gc, size);
	
	//__CUDA_SAFE_CALL( cudaMemcpy( C, gc, matrix_size, cudaMemcpyDeviceToHost ) );

	__CUDA_SAFE_CALL( cudaEventRecord(stop) );
	__CUDA_SAFE_CALL( cudaEventSynchronize(stop) );
	/*****************
	 * End of GPU code
	 ****************/
	
	__CUDA_SAFE_CALL( cudaEventElapsedTime( &pms, start, stop ) );
	
	cudaFree( ga );
	cudaFree( gb );
	cudaFree( gc );



	/*********************
	 * Sequential Stuff
	 ********************/
	struct timespec seq_start,seq_end;
	
	/* clock_gettime gets the process specific time spent, as opposed to the system time expended
	 */
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_start );
	
	//seq_is_symmetric( A, &is_symmetric, size );

	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_end );

	/*************************
	 * End of Sequential Stuff
	 ************************/

	/* Getting time in milliseconds for comparability */
	sms = ( (float)seq_end.tv_sec - seq_start.tv_sec )*1000 + ( (float)seq_end.tv_nsec - seq_start.tv_nsec ) / 1000000;

	printf("%12d % 12f % 12f % 12f\n",size,pms,sms,sms/pms);

	free(A);
}





