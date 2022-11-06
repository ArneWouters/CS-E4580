#include <algorithm>
#include <random>
#include <iostream>

typedef unsigned long long data_t;

using namespace std;


data_t randint(const int a, const int b) {
    // random number in [a, b)
    random_device dev;
    static thread_local mt19937 rng(dev());
    uniform_int_distribution<mt19937::result_type> dist(a, b-1);
    return dist(rng);
}


void quicksort(data_t *data, data_t* left, data_t* right) {
    constexpr int threshold = 16;

    if (left >= right) return;
    else if (right - left < threshold) {
        sort(left, right);
        return;
    }

    data_t pivot = *(data + randint(left - data, right - data));
    data_t* middle1 = partition(left, right, [pivot](const auto &num) { return num < pivot; });
    data_t* middle2 = partition(middle1, right, [pivot](const auto &num) {
            return !(pivot < num);
    });

    #pragma omp task
    quicksort(data, left, middle1);

    #pragma omp task
    quicksort(data, middle2, right);
}

void psort(int n, data_t *data) {
    #pragma omp parallel
    #pragma omp single
    {
        quicksort(data, data, data + n);
    }
}

