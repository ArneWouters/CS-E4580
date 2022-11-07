#include <cuda_runtime.h>
#include <iostream>

using namespace std;


struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


static inline void check(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << "CUDA error: " << context << ": "
            << cudaGetErrorString(err) << std::endl;
        std::exit(EXIT_FAILURE);
    }
}

#define CHECK(x) check(x, #x)


static inline int divup(int a, int b) {
    return (a + b - 1)/b;
}


float* precompute_rectangles(const int &ny, const int &nx, const float* data) {
    int m = ny + 1;
    int n = nx + 1;

    float* sums = (float*) calloc(m * n, sizeof(float));

    for (int i = 1; i < m; i++) {
        for (int j = 1; j < n; j++) {
            sums[(i * n) + j] = data[(3 * (j - 1)) + (3 * nx * (i - 1))] + sums[((i - 1) * n) + j]
                + sums[(i * n) + (j - 1)] - sums[((i - 1) * n) + (j - 1)];
        }
    }

    return sums;
}


__global__ void mykernel(int ny, int nx, float* precomputed, float* data, Result* results) {
    int w = threadIdx.x + blockIdx.x * blockDim.x;
    int h = threadIdx.y + blockIdx.y * blockDim.y;

    if (w == 0 || h == 0 || w > nx || h > ny) return;

    int n = nx + 1;
    float nInside = h * w;  // amount of pixels inside
    float nOutside = (nx * ny) - nInside;  // amount of pixels outside
    float sumAll = precomputed[((ny) * n) + nx];
    float best = 0.0;

    // loop rectangle positions
    for (int y0 = 0; y0 < (ny + 1) - h; y0++) {
        for (int x0 = 0; x0 < (nx + 1) - w ; x0++) {
            int x1 = x0 + w;
            int y1 = y0 + h;

            float p1 = precomputed[(y1 * n) + x1];
            float p2 = precomputed[(y1 * n) + x0];
            float p3 = precomputed[(y0 * n) + x1];
            float p4 = precomputed[(y0 * n) + x0];

            float sumInside = p1 - p2 - p3 + p4;
            float sumOutside = sumAll - sumInside;

            float score = (sumInside * sumInside * (1.0 / nInside))
                + (sumOutside * sumOutside * (1.0 / nOutside));

            if (score > best) {
                best = score;
                int idx = ((h - 1) * nx) + (w - 1);
                results[idx].x0 = x0;
                results[idx].x1 = x1;
                results[idx].y0 = y0;
                results[idx].y1 = y1;
            }
        }
    }
}


Result find_best_result(const int &nx, const int &ny, float* precomputed, Result* results) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    int n = nx + 1;
    float sumAll = precomputed[(ny * n) + nx];
    float best_score = 0.0;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            Result res = results[(i * nx) + j];
            float nInside = (res.y1 - res.y0) * (res.x1 - res.x0);
            float nOutside = (nx * ny) - nInside;

            float p1 = precomputed[(res.y1 * n) + res.x1];
            float p2 = precomputed[(res.y1 * n) + res.x0];
            float p3 = precomputed[(res.y0 * n) + res.x1];
            float p4 = precomputed[(res.y0 * n) + res.x0];

            float sumInside = p1 - p2 - p3 + p4;
            float sumOutside = sumAll - sumInside;

            float score = (sumInside * sumInside * (1.0 / nInside))
                + (sumOutside * sumOutside * (1.0 / nOutside));

            if (score > best_score) {
                float innerValue = sumInside * (1.0 / nInside);
                float outerValue = sumOutside * (1.0 / nOutside);

                best_score = score;
                result.x0 = res.x0;
                result.x1 = res.x1;
                result.y0 = res.y0;
                result.y1 = res.y1;
                result.inner[0] = innerValue;
                result.inner[1] = innerValue;
                result.inner[2] = innerValue;
                result.outer[0] = outerValue;
                result.outer[1] = outerValue;
                result.outer[2] = outerValue;
            }
        }
    }

    return result;
}


/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    float* precomputed = precompute_rectangles(ny, nx, data);
    Result* results = (Result*) calloc(ny * nx, sizeof(Result));

    // Allocate memory & copy data to GPU
    float* pGPU = NULL;
    CHECK(cudaMalloc((void**)&pGPU, (ny + 1) * (nx + 1) * sizeof(float)));
    CHECK(cudaMemcpy(pGPU, precomputed, (ny + 1) * (nx + 1) * sizeof(float), cudaMemcpyHostToDevice));
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    Result* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * nx * sizeof(Result)));
    CHECK(cudaMemcpy(rGPU, results, ny * nx * sizeof(Result), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(nx, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, pGPU, dGPU, rGPU);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(results, rGPU, ny * nx * sizeof(Result), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(pGPU));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));

    return find_best_result(nx, ny, precomputed, results);
}

