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


pair<vector<double>, vector<double>> calculate_mu_std(const int &ny, const int &nx, const float* data) {
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

    return {mu, std};
}


__global__ void mykernel(int ny, int nx, float* normalized_data, float* result) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;

    if (i >= ny || j >= ny) return;

    float sum = 0.0;

    for (int k = 0; k < nx; k++) {
        sum += normalized_data[(i * nx) + k] * normalized_data[(j * nx) + k];
    }

    result[j + (i * ny)] = sum;
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
    auto temp = calculate_mu_std(ny, nx, data);
    vector<double> mu = temp.first;
    vector<double> std = temp.second;

    float* normalized_data = (float*) malloc(ny * nx * sizeof(float));

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            normalized_data[(i * nx) + j] = (float) ((data[j + (i * nx)] - mu[i]) / std[i]);
        }
    }

    // Allocate memory & copy data to GPU
    float* dGPU = NULL;
    CHECK(cudaMalloc((void**)&dGPU, ny * nx * sizeof(float)));
    float* rGPU = NULL;
    CHECK(cudaMalloc((void**)&rGPU, ny * ny * sizeof(float)));
    CHECK(cudaMemcpy(dGPU, normalized_data, ny * nx * sizeof(float), cudaMemcpyHostToDevice));

    // Run kernel
    dim3 dimBlock(16, 16);
    dim3 dimGrid(divup(ny, dimBlock.x), divup(ny, dimBlock.y));
    mykernel<<<dimGrid, dimBlock>>>(ny, nx, dGPU, rGPU);
    CHECK(cudaGetLastError());

    // Copy data back to CPU & release memory
    CHECK(cudaMemcpy(result, rGPU, ny * ny * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaFree(dGPU));
    CHECK(cudaFree(rGPU));
}

