#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>


/*
 * each SM has 4 warp scheduler and each warp contains 32 threads,
 * totaling 128 threads, so BLOCK_SIZE of 128 is typically used to
 * saturate the warp scheduler
 * */
/* Answer from Gemini
 * Too Small (e.g., 32 threads): A block with only one warp doesn't provide the scheduler with many options. To achieve high occupancy, the SM would need to be filled with many small blocks. This can sometimes be inefficient.

Too Large (e.g., 1024 threads): An SM has limited resources, such as registers and shared memory. A very large block consumes a significant chunk of these resources. For example, if a kernel uses many registers per thread, a single block of 1024 threads might consume all the available registers on an SM, preventing any other blocks from running alongside it. This limits the total number of active warps and can reduce occupancy.

The Sweet Spot (e.g., 128 or 256): A block size of 128 (4 warps) or 256 (8 warps) strikes a good balance.

It provides multiple warps per block for the scheduler to work with.

It's small enough that multiple blocks can typically fit on a single SM at the same time, increasing the total pool of active warps and boosting occupancy.

This balance allows the SM to effectively hide memory latency, which is often the biggest performance bottleneck.
 * */

#define THREADS_PER_BLOCK_X 2
#define THREADS_PER_BLOCK_Y 2
/*
formula for thread per block (2D case)
int nx, ny;
dim3 block(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y)
dim3 numOfBlock((nx + block.x - 1)/block.x, (ny + block.y - 1)/block.y)

1D case
int nx;
dim3 block(THREADS_PER_BLOCK_X);
dim3 numOfBLock((nx + block.x - 1)/block.x)
 * */

/* 
 * C = A * B
 *
 * A := M * K
 * B := K * N
 *
 * C := M * N
*/
// cpu version
void cpu_matmul(int *A, int *B, int *C, int M, int K, int N) {
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			for (int k = 0; k < K; k++) {
				C[i*N + j] += A[i*K + k] * B[k*N + j];
			}
		}
	}
	return;
}

__global__ void gpu_matmul(int *A, int *B, int *C, int M, int K, int N) {
	int x = blockDim.x * blockIdx.x + threadIdx.x;
	int y = blockDim.y * blockIdx.y + threadIdx.y;
	printf("x: %d y: %d\n", x, y);
	for (int k = 0; k < K; k++) {
		C[K*y + x] += A[K*y + k] * B[N*k + x];
	}
	return;
}

static void setup_val(int *matrix, int row, int col, int val) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			matrix[i*col + j] = rand() % 5;
		}
	}
}

static void print_matrix(int *matrix, int row, int col) {
	for (int i = 0; i < row; i++) {
		for (int j = 0; j < col; j++) {
			printf("%d,", matrix[i*col + j]);
		}
		printf("\n");
	}
}

int main(int argc, char **argv) {
	printf("argc: %d\n", argc);
	srand(time(NULL));
	for (int i = 0; i < argc; i++) {
		printf("argv[%d]: %s\n", i, argv[i]);
	}

	int m, k, n;
	m = atoi(argv[1]);
	k = atoi(argv[2]);
	n = atoi(argv[3]);

	int *A, *B, *C;
	int *d_A, *d_B, *d_C;

	A = (int *)malloc(m*k*sizeof(int));
	B = (int *)malloc(k*n*sizeof(int));
	C = (int *)malloc(m*n*sizeof(int));

	cudaMalloc(&d_A, m*k*sizeof(int));
	cudaMalloc(&d_B, k*n*sizeof(int));
	cudaMalloc(&d_C, m*n*sizeof(int));

	setup_val(A, m, k, 1);
	setup_val(B, k, n, 1);
	memset(C, 0, m*n*sizeof(int));
	// print A
	printf("========== printing A =========\n");
	//print_matrix(A, m, k);
	printf("========== printing B =========\n");
	//print_matrix(B, k, n);

	cpu_matmul(A, B, C, m, k, n);
	printf("========== printing C from cpu_matmul =========\n");
	print_matrix(C, m, n);

	cudaMemcpy(d_A, A, m*k*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B, k*n*sizeof(int), cudaMemcpyHostToDevice);

	dim3 threadsPerBlock(THREADS_PER_BLOCK_X, THREADS_PER_BLOCK_Y);
	dim3 numOfBlocks((n + threadsPerBlock.x - 1)/threadsPerBlock.x, (m + threadsPerBlock.y - 1)/threadsPerBlock.y);
	// the number of thread should match the dimension of C matrix (M*N)
	gpu_matmul<<<numOfBlocks, threadsPerBlock>>>(d_A, d_B, d_C, m, k, n);
	printf("========== printing C from gpu_matmul =========\n");
	cudaMemcpy(d_C, C, m*n*sizeof(int), cudaMemcpyDeviceToHost);

	print_matrix(C, m, n);

	free(A);
	free(B);
	free(C);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return 0;
}
