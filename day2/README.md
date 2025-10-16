# Learnt today
1. Add timing for CPU matmul and GPU matmul
    - use unistd.h and CUDA event
2. Bug fix for GPU matmul
    - use register for result accumulation first to avoid redundant global memory load/store
    - cudaMemcpy bug after kernel call (dst and src was wrong in the argument list)
