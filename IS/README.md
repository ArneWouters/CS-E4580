# IS - Image Segmentation
## IS4
Using all resources that you have in the CPU, solve the task **as fast as possible**.
You are encouraged to exploit instruction-level parallelism, multithreading, and vector
instructions whenever possible, and also to optimize the memory access pattern.
Please do all arithmetic with **double-precision** floating point numbers.

## IS6a
In this task, the input is always a monochromatic image: each input pixel is either entirely
**white** with the RGB values (1,1,1) or entirely **black** with the RGB values (0,0,0).
Make your solution to IS4 faster by exploiting this property. It is now enough to find a solution
for only one color channel, and you will also have much less trouble with rounding errors.
In this task, you are permitted to use **single-precision** floating point numbers.

## IS6b
## IS9a

## Results

| Task | Points |     Time | Instr. ( $\times 10^9$ ) | Cycles ( $\times 10^9$ ) |  GHz | Threads |
|------|--------|----------|--------------------------|--------------------------|------|---------|
|  IS4 |    4/5 |     2.77 |                      168 |                      135 | 3.77 |    13.0 |
| IS6a |    4/5 |     1.32 |                     67.7 |                     54.4 | 3.69 |    11.2 |

