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
#include <math.h>

#ifndef __CUDA_SAFE_CALL
cudaError_t __cuda_error;
#define __CUDA_SAFE_CALL(call) do { __cuda_error = call; if(__cuda_error != cudaSuccess) { fprintf(stderr,"CUDA Error: %s,%s, line %d\n",cudaGetErrorString(__cuda_error), __FILE__, __LINE__ ); return -1;} } while(0)
#endif

void generate_random_vector( double *A, int size )
{
	srand( time(NULL) );

	for( int i=0; i<size; i++ )
		//A[i] = 1.2;
		A[i] = ((double)rand())/100000;
}

__device__ void addfour( volatile double *A, int id, int threads, int size, volatile double *B )
{
	int index=threads+id;
	for( int i=0; i<3; i++ ){
		if( index < size ){
			//printf("%d %d %d %d\n",index,id,threads,size);
			B[id] += A[index];
		}
		index += threads;
	}
}

__device__ void maxfour( volatile double *A, int id, int threads, int size, volatile double *B )
{
	int index=threads+id;
	for( int i=0; i<3; i++ ){
		if( index < size ){
			if( B[id] < A[index] )
				//printf("%d %d %d %d\n",index,id,threads,size);
				B[id] = A[index];
		}
		index += threads;
	}
}

__global__ void add4096( double *A, double *B, int size )
{
	__shared__ double nums[256];
	int id = blockIdx.x;
	int tid = threadIdx.x;
	int offset = 4096*id;
	int alive = blockDim.x;
	int this_block = ( size - offset >= 4096 )? 4096: size - offset;
	int intra_offset = 0;
	
	nums[tid] = 0;
	
	while( intra_offset < this_block ){
		if( intra_offset + tid < this_block )
			nums[tid] += A[ offset + intra_offset + tid ];
	       	intra_offset += alive;
	}	

	__syncthreads();
	
	alive = alive >> 2;
	this_block = ( this_block >= blockDim.x )? blockDim.x: this_block;

	while( 1 ){
		if( tid < alive ){
			addfour( (double *)&nums, tid, alive, this_block, (double *)&nums );
			//printf(" id=%d total alive=%d tid=%d, %5.2f\n",id,alive,tid,nums[tid]);
		}
		if( alive == 1 )
			break;
		this_block = ( this_block >= alive )? alive: this_block;
		alive = alive>>2;
		__syncthreads();
	}

	B[id] = nums[0];
}

__global__ void addsquared4096( double *A, double *B, int size )
{
	__shared__ double nums[256];
	int id = blockIdx.x;
	int tid = threadIdx.x;
	int offset = 4096*id;
	int alive = blockDim.x;
	int this_block = ( size - offset >= 4096 )? 4096: size - offset;
	int intra_offset = 0;
	
	nums[tid] = 0;
	
	while( intra_offset < this_block ){
		if( intra_offset + tid < this_block )
			nums[tid] += A[ offset + intra_offset + tid ] * A[ offset + intra_offset + tid ];
	       	intra_offset += alive;
	}	

	__syncthreads();
	
	alive = alive >> 2;
	this_block = ( this_block >= blockDim.x )? blockDim.x: this_block;
	while( 1 ){
		if( tid < alive ){
			addfour( (double *)&nums, tid, alive, this_block, (double *)&nums );
			//printf(" id=%d total alive=%d tid=%d, %5.2f\n",id,alive,tid,nums[tid]);
		}
		if( alive == 1 )
			break;
		this_block = ( this_block >= alive )? alive: this_block;
		alive = alive>>2;
		__syncthreads();
	}

	B[id] = nums[0];
}

__global__ void max4096( double *A, double *B, int size )
{
	__shared__ double nums[256];
	int id = blockIdx.x;
	int tid = threadIdx.x;
	int offset = 4096*id;
	int alive = blockDim.x;
	int this_block = ( size - offset >= 4096 )? 4096: size - offset;
	int intra_offset = 0;
	int tmp;

	nums[tid] = -1;
	
	while( intra_offset < this_block ){
		if( intra_offset + tid < this_block ){
			tmp = A[ offset + intra_offset + tid ];
			if( nums[tid] < tmp )
				nums[tid] = tmp;
		}
	       	intra_offset += alive;
	}	

	__syncthreads();
	
	alive = alive >> 2;
	this_block = ( this_block >= blockDim.x )? blockDim.x: this_block;
	while( 1 ){
		if( tid < alive ){
			maxfour( (double *)&nums, tid, alive, this_block, (double *)&nums );
			//printf(" id=%d total alive=%d tid=%d, %5.2f\n",id,alive,tid,nums[tid]);
		}
		if( alive == 1 )
			break;
		this_block = ( this_block >= alive )? alive: this_block;
		alive = alive>>2;
		__syncthreads();
	}

	B[id] = nums[0];
}

double padd(double *A, int size)
{
	double *ga,*gb;
	int vector_size = sizeof(double) * size;
	int num_blocks = ( ((size - 1) / 4096) + 1 );
	int out_vector = sizeof(double)* num_blocks;
	double ans;

	__CUDA_SAFE_CALL( cudaMalloc( &ga, vector_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, out_vector  ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, vector_size, cudaMemcpyHostToDevice ) );
	
	while( size > 1 ){
		add4096<<<num_blocks,256>>> (ga, gb, size);
		size = num_blocks;
		num_blocks = ( ((size - 1) / 4096) + 1 );
		ga = gb;
	}

	__CUDA_SAFE_CALL( cudaMemcpy( &ans, gb, sizeof(double) , cudaMemcpyDeviceToHost ) );
	
	cudaFree( ga );
	cudaFree( gb );

	return ans;
}

double psquareadd(double *A, int size )
{
	double *ga,*gb;
	int vector_size = sizeof(double) * size;
	int num_blocks = ( ((size - 1) / 4096) + 1 );
	int out_vector = sizeof(double)* num_blocks;
	double ans;

	__CUDA_SAFE_CALL( cudaMalloc( &ga, vector_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, out_vector  ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, vector_size, cudaMemcpyHostToDevice ) );
	
	while( size > 1 ){
		addsquared4096<<<num_blocks,256>>> (ga, gb, size);
		size = num_blocks;
		num_blocks = ( ((size - 1) / 4096) + 1 );
		ga = gb;
	}

	__CUDA_SAFE_CALL( cudaMemcpy( &ans, gb, sizeof(double) , cudaMemcpyDeviceToHost ) );
	
	cudaFree( ga );
	cudaFree( gb );

	return ans;
}

double pmax(double *A, int size )
{
	double *ga,*gb;
	int vector_size = sizeof(double) * size;
	int num_blocks = ( ((size - 1) / 4096) + 1 );
	int out_vector = sizeof(double)* num_blocks;
	double ans;

	__CUDA_SAFE_CALL( cudaMalloc( &ga, vector_size ) );
	__CUDA_SAFE_CALL( cudaMalloc( &gb, out_vector  ) );

	__CUDA_SAFE_CALL( cudaMemcpy( ga, A, vector_size, cudaMemcpyHostToDevice ) );
	
	while( size > 1 ){
		max4096<<<num_blocks,256>>> (ga, gb, size);
		size = num_blocks;
		num_blocks = ( ((size - 1) / 4096) + 1 );
		ga = gb;
	}

	__CUDA_SAFE_CALL( cudaMemcpy( &ans, gb, sizeof(double) , cudaMemcpyDeviceToHost ) );
	
	cudaFree( ga );
	cudaFree( gb );

	return ans;
}


double pmean( double *A, int size )
{
	double ans;
	
	ans = padd( A, size );
	return ans/size;
}

double pstd( double *A, int size )
{
	double mean, squaredsum;
	mean = pmean( A, size );
	squaredsum = psquareadd( A, size )/size;
	return sqrt(squaredsum-(mean*mean));
}

double sadd( double* A, int size )
{
	double ans=0;
	for( int i=0; i< size; i++ ){
		ans += A[i];
		//printf("%lf \n",ans);
	}
	return ans;
}

double smean( double *A, int size )
{
	double ans;
	ans = sadd( A, size );
	return ans/size;
}

int main( int argc, char* argv[] )
{
	/* Matrix container pointers */
	double *A;

	int size;		/* Number of elements */
	int vector_size;	/* Physical size of the elements in the memory */
	
	cudaEvent_t start,stop;
	
	bool do_print=false;	/* Debug flag to print matrices in case of small matrices */
	
	float pms = 0,sms=0;	/* Parallel and sequential times */

	double mean,std,max;
	
	if( argc != 2 ){
		fprintf(stderr, "Atleast one argument required. Usage: %s <Side of the matrix>",argv[0]);
		return -1;
	}
	
	/* Get size of the matrix from command line */
	size = atoi( argv[1] );

	vector_size = sizeof(double)* size;
		
	if( size <= 32 ) do_print= true;

	A = (double *) malloc( vector_size );
	//B = (double *) malloc( out_vector );

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
	cudaEventRecord(stop);
	
	
	mean = 0;
	std = 0;
	max = 0;
	mean = pmean( A, size );
	std = pstd( A, size );
	max = pmax( A, size );

	cudaEventSynchronize(stop);
	/*****************
	 * End of GPU code
	 ****************/
	
	cudaEventElapsedTime( &pms, start, stop );

	printf("Mean is %lf\n",mean);
	printf("Std is %lf\n",std);
	printf("Max is %lf\n",max);
	/*********************
	 * Sequential Stuff
	 ********************/
	struct timespec seq_start,seq_end;

	/* clock_gettime gets the process specific time spent, as opposed to the system time expended
	 */
	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_start );
	
	mean = smean( A, size );	

	clock_gettime( CLOCK_PROCESS_CPUTIME_ID, &seq_end );

	/*************************
	 * End of Sequential Stuff
	 ************************/

	printf("Mean is %lf\n",mean);
	
	/* Getting time in milliseconds for comparability */
	sms = ( (float)seq_end.tv_sec - seq_start.tv_sec )*1000 + ( (float)seq_end.tv_nsec - seq_start.tv_nsec ) / 1000000;
	printf("%12s %12s %12s %12s\n","N","Parallel","Sequential","Speedup");
	printf("%12d % 12f % 12f % 12f\n",size,pms,sms,sms/pms);
	/*
	printf("<html>\n\t<body>\n\t\t<table>\n");
	printf("<tr>\n");
	printf("\t<td> %12d </td>\n\t<td>% 12f</td>\n\t<td>% 12f</td>\n\t<td>% 12f</td>\n",size,pms,sms,sms/pms);
	printf("</tr>\n");
	printf("</table>\n</body>\n</html>\n");
	*/
	free(A);
}





