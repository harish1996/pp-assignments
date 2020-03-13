#include "random_gens.h"

/**
 * Random Matrix Generators
 */
void generate_assymmetric_matrix( float *A, int size, float assymmetricity )
{
	float r;
	/* Initializing random number generator */
	srand( time(NULL) );
	
	for( int i=0; i<size; i++ ){
		for( int j=0; j<=i; j++ ){
			A[i*size + j] = (float)rand() / 10;
			r = (float)( (rand()%30000) * 10 ) / 30000 ;
		      	if ( r < assymmetricity ){
				A[ j*size + i ] = (float)rand() / 10;
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
			A[i*size + j] = rand() / 10;
			A[j*size + i] = A[i*size + j];
		}
	}
}

void const_matrix( float *A, int size )
{
	for( int i=0; i<size; i++ ){
		for( int j=0; j<=i; j++ ){
			A[i*size + j] = 1;
			A[j*size + i] = 1;
		}
	}
}

void generate_notso_random_matrix( float *A, int size )
{
	srand( time(NULL) );

	int r = rand()%100;
	//const_matrix( A, size );

	if( r <= 50 )
		generate_assymmetric_matrix( A, size, 0.1 );
	else
		generate_symmetric_matrix( A, size );
}


