# Learnt today
1. Fixed boundary in kernel
    - When checking boundary in kernel like `if (y < M && x < N)`, need to remember that index starts from 0 and end at M - 1. Also the direction of x and y

    ---------------> x
    |
    |
    |
    |
    |
    V
    y

2. Fixed the result loading memory location C[y*N + x]
