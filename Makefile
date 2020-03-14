all: problem1 problem4

problem1: matrixmul.cu matrix_kernels.cu random_gens.cu
	nvcc -lcudart $^ -o $@

problem4: convolution.cu random_gens.cu
	nvcc -lcudart $^ -o $@
