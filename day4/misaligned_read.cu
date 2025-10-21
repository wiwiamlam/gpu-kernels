#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <cuda.h>
#include <time.h>
#include <assert.h>


__global__ void readoffset(float *A, float *B, float *C, const int n, const int offset) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int k = x + offset;
	if (k < n) {
		C[x] = A[k] + B[k];
	}
}

void sumArrayOnHost(float *A, float *B, float* C, const int n, int offset) {
	for (int i = 0, idx = offset; idx < n; i++, idx++) {
		C[i] = A[idx] + B[idx];
	}
}

void initialData(float *ip, int size) {
	for (int i = 0; i < size; i++) {
		ip[i] = (float)(rand() & 0xFF)/10.0f;
	}
}

void checkResult(float *hostRef, float *gpuRef, int nElem) {
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < nElem; i++) {
		if (abs(hostRef[i] - gpuRef[i]) > epsilon) {
			match = 0;
			printf("Arrays do not match!\n");
			printf("host %5.2f gpu %5.2f at current %d\n", hostRef[i], gpuRef[i], i);
			break;
		}
	}
	if (match) printf("Arrays match.\n\n");
	return;
}

int main(int argc, char **argv) {
	int dev = 0;
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, dev);
	printf("%s starting reduction at argc: %d\n", argv[0], argc);
	printf("device %d: %s \n", dev, deviceProp.name);
	cudaSetDevice(dev);
	srand(time(NULL));
	// set up array size
	//
	int nElem = 1<<30;
	printf(" with array size %d\n", nElem);
	size_t nBytes = nElem * sizeof(float);

	int blocksize = 512;
	int offset = 0;
	if (argc > 1) offset    = atoi(argv[1]);
	if (argc > 2) blocksize = atoi(argv[2]);
	printf("offset: %d\n", offset);
	dim3 block  (blocksize, 1);
	dim3 grid   ((nElem + block.x - 1) / block.x, 1);

	float *h_A = (float *) malloc(nBytes);
	float *h_B = (float *) malloc(nBytes);
	float *hostRef = (float *) malloc(nBytes);
	float *gpuRef = (float *) malloc(nBytes);


	initialData(h_A, nElem);
	initialData(h_B, nElem);
	sumArrayOnHost(h_A, h_B, hostRef, nElem, offset);

	float *d_A, *d_B, *d_C;
	cudaEvent_t start, end;
	float milliseconds;
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	cudaEventRecord(start);
	readoffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&milliseconds, start, end);

	printf("warmup<<<%4d,%4d>>> offset %4d took %f millisecons\n", grid.x, block.x, offset, milliseconds);

	printf("====================after warmup====================\n");

	cudaEventRecord(start);
	readoffset<<<grid, block>>>(d_A, d_B, d_C, nElem, offset);
	cudaEventRecord(end);
	cudaEventSynchronize(end);

	cudaEventElapsedTime(&milliseconds, start, end);

	printf("readoffset<<<%4d,%4d>>> offset %4d took %f millisecons\n", grid.x, block.x, offset, milliseconds);
	cudaMemcpy(gpuRef, d_C, nBytes, cudaMemcpyDeviceToHost);
	checkResult(hostRef, gpuRef, nElem - offset);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	free(h_A);
	free(h_B);

	return 0;
}
