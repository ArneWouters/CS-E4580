#include <algorithm>
#include <cstdint>
#include <omp.h>

typedef unsigned long long data_t;

using namespace std;

static inline uint64_t prev_pow2(uint64_t x) {
    // https://jameshfisher.com/2018/03/30/round-up-power-2/
    return x == 1 ? 1 : 1 << ((64 - 1) - __builtin_clzl(x - 1));
}

void psort(int n, data_t *data) {
    int nThreads = (int) prev_pow2(omp_get_max_threads());

    if (nThreads < 2) return sort(data, data + n);

    int blockSize = (n + nThreads - 1) / nThreads;

    // split and sort subarrays
    #pragma omp parallel for
    for (int i = 0; i < nThreads; i++) {
        int start = min(i * blockSize, n);
        int end = min((i + 1) * blockSize, n);
        sort(data + start, data + end);
    }

    nThreads /= 2;

    // merge the sorted subarrays
    while (nThreads) {
        #pragma omp parallel for
        for (int i = 0; i < nThreads; i++) {
            int idx = 2 * i;
            int start = idx * blockSize;
            int middle = min((idx + 1) * blockSize, n);
            int end = min((idx + 2) * blockSize, n);

            inplace_merge(data + start, data + middle, data + end);
        }

        blockSize *= 2;
        nThreads /= 2;
    }
}

