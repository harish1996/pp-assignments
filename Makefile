all: problem1

problem1: matrixmul.cu matrix_kernels.cu random_gens.cu
	nvcc -lcudart $^ -o $@
