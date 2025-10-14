# Learnt today

1. Tried nsys profile to see kernel launch overhead
2. First kernel in the program is slow due to program warm up? (lots of ioctl call)
3. subsequent first kernel is also slightly slower, later same kernel is slightly faster probably because of caching
