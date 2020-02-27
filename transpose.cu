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

#define CHECK_STATUS(status)	if( status != cudaSuccess ){ fprintf(stderr,"Error at %d\n",__LINE__); return -1; }

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

__global__ void par_is_symmetric( float *A, bool* is_transpose, int size )
{
	if( !(*is_transpose) ){
		return;
	}
	else{
		register int c = blockIdx.x * blockDim.x + threadIdx.x;
		register int r = blockIdx.y * blockDim.y + threadIdx.y;
		
		if( c >= size || r >= size ) return;
		if( A[ r*size + c ] != A[ c*size + r ] )
			*is_transpose = false;
	}
}

/**
 * Sequential matrix symmetricity checking function
 */
void seq_is_symmetric( float *A, bool* is_transpose, int size )
{
	*is_transpose = true;
	for( int i=0; i<size; i++ ){
		for( int j=0; j<i; j++ ){
			if( A[i*size + j] != A[j*size + i] ){
			       *is_transpose = false;
			       return;
			}
		}
	}
}


int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	float *A;
	float *ga;

	int size;	/* Size of the matrix */
	int matrix_size;	/* Physical size of the matrix in the memory */
	
	int num_blocks;		
	
	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	int dim_thread = 32;	/* Number of threads in each block */
	
	float pms = 0,sms=0;	/* Parallel and sequential times */

	
	if( argc != 2 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );
	matrix_size = sizeof(float)* size * size;
	
	//if( size <= 12 ) do_print= true;

	A = (float *) malloc( sizeof(float)* size * size );
	
	generate_notso_random_matrix( A, size );

	if( do_print ){
		printf("A=\n");
		for( int i=0; i<size; i++ ){
			for( int j=0; j<size; j++ ){
				printf("%5.2f ",A[i*size + j]);
			}
			printf("\n");
		}
	}

	bool is_symmetric = true;
	bool *g_is_symmetric;
	cudaError_t status;

	/* Timers to time the parallel process */ 
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	/*********************
	  * Start of GPU run
	  *******************/
	cudaEventRecord(start);

	//size_t freev,totalv;
	status = cudaMalloc( &ga, sizeof(float)* size * size );
	//cudaMemGetInfo( &freev, &totalv );
	//printf("Free= %lu, total=%lu\n",freev,totalv);
	CHECK_STATUS(status);

	status = cudaMalloc( &g_is_symmetric, sizeof(bool) );
	CHECK_STATUS(status);

	status = cudaMemcpy( ga, A, matrix_size, cudaMemcpyHostToDevice );
	CHECK_STATUS(status);
	
	status = cudaMemcpy( g_is_symmetric, &is_symmetric, sizeof(bool), cudaMemcpyHostToDevice );
	CHECK_STATUS(status);

	num_blocks = ( size - 1 )/dim_thread + 1;

	dim3 grid( dim_thread, dim_thread );
	dim3 blocks( num_blocks, num_blocks);
	
	par_is_symmetric<<<blocks,grid>>> (ga, g_is_symmetric, size);
	
	status = cudaMemcpy( &is_symmetric, g_is_symmetric, sizeof(bool), cudaMemcpyDeviceToHost );
	CHECK_STATUS(status);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	/*****************
	 * End of GPU code
	 ****************/
	
	cudaEventElapsedTime( &pms, start, stop );

	char *msg;
	msg = is_symmetric ? (char *)"Symmetric":(char *)"Not Symmetric";
	printf("The given matrix is %s\n",msg);
	//printf("Time taken %5.7f milliseconds\n",milliseconds);
	
//	if( do_print ){	
//		printf("C=\n");
//		for( int i=0; i<size; i++ ){
//			for( int j=0; j<size; j++ ){
//				printf("%d ",checked[i*size + j]);
//			}
//			printf("\n");
//		}
//	}

	cudaFree( ga );
	
	/*********************
	 * Sequential Stuff
	 ********************/
	struct timespec seq_start,seq_end;
	
	/* clock_gettime gets the process specific time spent, as opposed to the system time expended
	 */
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_start );
	
	seq_is_symmetric( A, &is_symmetric, size );

	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_end );

	/*************************
	 * End of Sequential Stuff
	 ************************/

	msg = is_symmetric ? (char *)"Symmetric":(char *)"Not Symmetric";
	printf("The given matrix is %s\n",msg);

	/* Getting time in milliseconds for comparability */
	sms = ( (float)seq_end.tv_sec - seq_start.tv_sec )*1000 + ( (float)seq_end.tv_nsec - seq_start.tv_nsec ) / 1000000;

	printf("%12d % 12f % 12f % 12f\n",size,pms,sms,sms/pms);

	free(A);
}





