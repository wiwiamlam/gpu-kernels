# Learnt today
1. Thread per block matters
    - doesn't have to be squared sizing
    - optimal sizing would have some consideration on the memory load
    - 128/256 threads per block is normally optimal
2. Figure out the gpu matmul simple case formula
3. Implement the simple CUDA workflow
    - cudaMalloc
    - cudaMemcpy
