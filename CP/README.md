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
## CP3b
## CP4
## CP5
## CP9a

## Results

| Task | Points |     Time | Instr. ( $\times 10^9$ ) | Cycles ( $\times 10^9$ ) |  GHz | Threads |
|------|--------|----------|--------------------------|--------------------------|------|---------|
| CP1  |    5/5 |     7.49 |                     80.2 |                     33.6 | 4.48 |    1.00 |
| CP2a |    3/3 |     4.35 |                     60.2 |                     19.5 | 4.47 |    1.00 |
| CP2b |    2/3 |     0.86 |                     24.3 |                     33.7 | 3.84 |    10.2 |
| CP2c |    3/3 |     2.93 |                     12.3 |                     13.1 | 4.46 |    1.00 |

