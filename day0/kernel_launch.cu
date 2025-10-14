#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void empty_kernel() {

}

int main() {
	empty_kernel<<<1, 1>>>();
	cudaDeviceSynchronize();
	return 0;
}
