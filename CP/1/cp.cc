/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <cmath>

void correlate(int ny, int nx, const float *data, float *result) {
    for (int i = 0; i < ny; i++) {
        result[i + (i * ny)] = 1.0;

        for (int j = i + 1; j < ny; j++) {
            double sum_i = 0.0;
            double sum_j = 0.0;
            double sum_ij = 0.0;
            double squareSum_i = 0.0;
            double squareSum_j = 0.0;

            for (int x = 0; x < nx; x++) {
                sum_i += (double) data[x + (i * nx)];
                sum_j += (double) data[x + (j * nx)];
                sum_ij += (double) data[x + (i * nx)] * (double) data[x + (j * nx)];
                squareSum_i += (double) data[x + (i * nx)] * (double) data[x + (i * nx)];
                squareSum_j += (double) data[x + (j * nx)] * (double) data[x + (j * nx)];
            }

            double res = (nx * sum_ij - sum_i * sum_j)
                / sqrt((nx * squareSum_i - sum_i * sum_i) * (nx * squareSum_j - sum_j * sum_j));
            result[j + (i * ny)] = (float) res;
        }
    }
}

