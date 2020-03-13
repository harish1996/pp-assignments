#ifndef __RANDOM_GENS_H
#define __RANDOM_GENS_H

#include <stdio.h>
#include <time.h>
#include <stdlib.h>

/**
 * Random Matrix Generators
 */
void generate_assymmetric_matrix( float *A, int size, float assymmetricity );

void generate_symmetric_matrix( float *A, int size );

void generate_notso_random_matrix( float *A, int size );

#endif
