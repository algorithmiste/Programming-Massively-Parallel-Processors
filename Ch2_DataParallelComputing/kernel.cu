#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cuda.h"
#include <stdio.h>
#include <math.h>


#define PRINT_ERROR(err) {\
	if (err != cudaSuccess) {\
		printf("%s in %s at line %d\n", cudaGetErrorString(err), __FILE__, __LINE__ );\
		exit(EXIT_FAILURE);\
	}\
};

/* ERROR CHECKING AND HANDLING IN CUDA:
	It is important for a program to check and handle errors.
	CUDA API functions return flags that indicate whether an error has
		occurred when they served theh request. Most errors are due to 
		inappropriate argument values used in the call. See below examples.*/

		// Compute vector sum C = A+B
		// Each thread performs one pair-wise addition
__global__
void vecAddKernel(float* A, float* B, float* C, int n) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < n)
		C[i] = A[i] + B[i];

}

void vecAdd(float* h_A, float* h_B, float* h_C, int n) {
	int size = n * sizeof(float);
	float *d_A, *d_B, *d_C;
	// 1. Allocate device memory for A, B, and C
	//	  copy A and B to device memory
	cudaError_t err = cudaMalloc((void**)& d_A, size);
	PRINT_ERROR(err);

	err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
	PRINT_ERROR(err);

	err = cudaMalloc((void**)& d_B, size);
	PRINT_ERROR(err);
	err = cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
	PRINT_ERROR(err);
	err = cudaMalloc((void**)& d_C, size);
	PRINT_ERROR(err);
	// 2. Kernel launch code - to have the device perform the actual vector addition
	int nBlocks = ceil(n / 256.0);
	vecAddKernel<<<nBlocks, 256>>>(d_A, d_B, d_C, n);
	cudaDeviceSynchronize();
	
	// 3. copy C from the device memory 
	//	  free device vectors
	err = cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
	PRINT_ERROR(err);

	err = cudaFree(d_A);
	PRINT_ERROR(err);

	err = cudaFree(d_B);
	PRINT_ERROR(err);

	err = cudaFree(d_C);
	PRINT_ERROR(err);
}

