/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in data[x + y*nx]
- correlation between rows i and row j has to be stored in result[i + j*ny]
- only parts with 0 <= j <= i < ny need to be filled
*/

#include <numeric>
#include <cmath>
#include <vector>

using namespace std;

double mean(const float* data, const int &nx, const int &y) {
    double sum = 0.0;
    for (int i = 0; i < nx; i++) sum += (double) data[i + (y * nx)];
    return sum / nx;
}

double norm(const float* data, const int &nx, const int &y, const double &m) {
    double sum = 0.0;
    for (int i = 0; i < nx; i++) {
        sum += ((double) data[i + (y * nx)] - m) * ((double) data[i + (y * nx)] - m);
    }
    return sqrt(sum);
}


void correlate(int ny, int nx, const float *data, float *result) {
    vector<vector<double>> normalized_data (ny, vector<double> (nx));

    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double mean_i = mean(data, nx, i);
        double norm_i = norm(data, nx, i, mean_i);

        for (int k = 0; k < nx; k++) {
            double x = data[k + (i * nx)];
            normalized_data[i][k] = (x - mean_i) / norm_i;
        }
    }

    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        result[i + (i * ny)] = 1.0;

        for (int j = i + 1; j < ny; j++) {
            double sum = 0.0;
            for (int k = 0; k < nx; k++) sum += normalized_data[i][k] * normalized_data[j][k];
            result[j + (i * ny)] = (float) sum;
        }
    }
}

