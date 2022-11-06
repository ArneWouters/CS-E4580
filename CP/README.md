# CP - correlated pairs
## CP1
Implement a simple **sequential** baseline solution. Do not try to use any form of parallelism yet;
try to make it work correctly first. Please do all arithmetic with **double-precision**
floating point numbers.

## CP2a
Parallelize your solution to CP1 by exploiting **instruction-level parallelism**.
Make sure that the performance-critical operations are pipelined efficiently.
Do not use any other form of parallelism yet in this exercise.
Please do all arithmetic with **double-precision** floating point numbers.

## CP2b
Parallelize your solution to CP1 with the help of **OpenMP and multithreading** so that you are
exploiting multiple CPU cores in parallel. Do not use any other form of parallelism yet in this
exercise. Please do all arithmetic with **double-precision** floating point numbers.

## CP2c
Parallelize your solution to CP1 with the help of **vector operations** so that you can perform
multiple useful arithmetic operations with one instruction. Do not use any other form of
parallelism yet in this exercise. Please do all arithmetic with **double-precision** floating
point numbers.

## CP3a
Using all resources that you have in the CPU, solve the task **as fast as possible**.
You are encouraged to exploit instruction-level parallelism, multithreading, and vector
instructions whenever possible, and also to optimize the memory access pattern.
Please do all arithmetic with **double-precision** floating point numbers.

## CP3b
Using all resources that you have in the CPU, solve the task **as fast as possible**.
You are encouraged to exploit instruction-level parallelism, multithreading, and vector
instructions whenever possible, and also to optimize the memory access pattern.
In this task, you are permitted to use **single-precision** floating point numbers.

## CP4
Implement a simple baseline solution for the **GPU**. Make sure it works correctly and that it is
reasonably efficient. Make sure that all performance-critical parts are executed on the GPU;
you can do some lightweight preprocessing and postprocessing also on the CPU.
In this task, you are permitted to use **single-precision** floating point numbers.

## CP5
Using all resources that you have in the **GPU**, solve the task **as fast as possible**.
In this task, you are permitted to use **single-precision** floating point numbers.

## CP9a

## Results

| Task | Points |     Time | Instr. ( $\times 10^9$ ) | Cycles ( $\times 10^9$ ) |  GHz | Threads |
|------|--------|----------|--------------------------|--------------------------|------|---------|
|  CP1 |    5/5 |     7.49 |                     80.2 |                     33.6 | 4.48 |    1.00 |
| CP2a |    3/3 |     4.35 |                     60.2 |                     19.5 | 4.47 |    1.00 |
| CP2b |    2/3 |     0.86 |                     24.3 |                     33.7 | 3.84 |    10.2 |
| CP2c |    3/3 |     2.93 |                     12.3 |                     13.1 | 4.46 |    1.00 |
| CP3a |    3/5 |     5.01 |                      309 |                      353 | 3.75 |    18.8 |
| CP3b |    4/5 |     1.95 |                      155 |                      134 | 3.73 |    18.4 |
|  CP4 |    5/5 |    0.158 |                     1.59 |                     0.70 | 4.40 |    1.00 |
|  CP5 |   9/10 |     0.69 |                     5.42 |                     2.79 | 4.05 |    1.00 |

