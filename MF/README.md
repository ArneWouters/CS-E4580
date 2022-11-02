# MF - Median Filter
## MF1
Implement a simple **sequential** baseline solution. Make sure it works correctly.
Do not try to use any form of parallelism yet. You are expected to use a naive algorithm that
computes the median separately for each pixel, with a **linear-time median-finding algorithm**.

## MF2
Parallelize your solution to MF1 with the help of **OpenMP** so that you are exploiting multiple
CPU cores in parallel.

## MF9a

## Results
| Task | Points |     Time | Instr. ( $\times 10^9$ ) | Cycles ( $\times 10^9$ ) |  GHz | Threads |
|------|--------|----------|--------------------------|--------------------------|------|---------|
|  MF1 |    5/5 |     7.91 |                     30.4 |                     35.5 | 4.48 |    1.00 |
|  MF2 |    3/3 |    0.497 |                     17.9 |                     42.0 | 4.24 |    19.9 |

