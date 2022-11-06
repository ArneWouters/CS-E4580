#include <cstdlib>
#include <iostream>
#include <cuda_runtime.h>
#include <vector>

using namespace std;

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

static inline int roundup(int a, int b) {
    return divup(a, b) * b;
}


float* normalize(const int &ny, const int &nx, const float* data) {
    vector<double> mu (ny, 0.0);
    vector<double> std (ny, 0.0);

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            double val = (double) data[j + (i * nx)];
            mu[i] += val;
            std[i] += val * val;
        }

        mu[i] /= nx;
        std[i] = sqrt(std[i] - (nx * mu[i] * mu[i]));
    }

    float* normalized_data = (float*) malloc(ny * nx * sizeof(float));

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            normalized_data[(i * nx) + j] = (float) ((data[(i * nx) + j] - mu[i]) / std[i]);
        }
    }

    return normalized_data;
}


__global__ void mykernel(int ny, int nx, int n, int nn, float* normalized_data, float* result) {
    int ia = threadIdx.x;
    int ja = threadIdx.y;
    int ic = blockIdx.x;
    int jc = blockIdx.y;

    float sums[8][8];

    for (int ib = 0; ib < 8; ib++) {
        for (int jb = 0; jb < 8; jb++) {
            sums[ib][jb] = 0.0;
        }
    }

    for (int k = 0; k < n; ++k) {
        float x[8];
        float y[8];

        for (int ib = 0; ib < 8; ib++) {
            int i = (ic * 64) + (ib * 8) + ia;
            x[ib] = normalized_data[(nn * k) + i];
        }

        for (int jb = 0; jb < 8; jb++) {
            int j = (jc * 64) + (jb * 8) + ja;
            y[jb] = normalized_data[(nn * k) + j];
        }

        for (int ib = 0; ib < 8; ib++) {
            for (int jb = 0; jb < 8; jb++) {
                sums[ib][jb] += x[ib] * y[jb];
            }
        }
    }

    for (int ib = 0; ib < 8; ib++) {
        for (int jb = 0; jb < 8; jb++) {
            int i = (ic * 64) + (ib * 8) + ia;
            int j = (jc * 64) + (jb * 8) + ja;
            if (i < ny && j < ny) {
                result[(ny * i) + j] = sums[ib][jb];
            }
        }
    }
}


__global__ void preprocess_kernel(int ny, int nx, int nn, float* normalized_data, float* data) {
    int ja = threadIdx.x;
    int i = blockIdx.y;

    for (int jb = 0; jb < nn; jb += 64) {
        int j = jb + ja;
        float v = (i < ny && j < nx) ? normalized_data[(i * nx) + j] : 0.0;
        data[(nn * j) + i] = v;
    }
}


/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/
void correlate(int ny, int nx, const float *data, float *result) {
    int nn = (ny >= nx) ? roundup(ny, 64) : roundup(nx, 64);
    int n = (ny >= nx) ? ny : nx;
    float* normalized_data = normalize(ny, nx, data);

    // Allocate memory & copy data to GPU
    float* ndGPU = NULL;
    CHECK(cudaMalloc((void**)&ndGPU, ny * nx * sizeof(float)));
    CHECK(cudaMemcpy(ndGPU, normalized_data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, nn * nn * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));

    // Run kernel
    {
        dim3 dimBlock(64, 1);
        dim3 dimGrid(1, nn);
        preprocess_kernel<<<dimGrid, dimBlock>>>(ny, nx, nn, ndGPU, dGPU);
        CHECK(cudaGetLastError());
    }

    // Run kernel
    {
        dim3 dimBlock(8, 8);
        dim3 dimGrid(nn / 64, nn / 64);
        mykernel<<<dimGrid, dimBlock>>>(ny, nx, n, nn, dGPU, rGPU);
        CHECK(cudaGetLastError());
    }

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(ndGPU));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}

