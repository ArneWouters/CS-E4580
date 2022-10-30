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

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));

using namespace std;

static inline double hsum4(const double4_t &v) {
    double sum = 0.0;
    for (int i = 0; i < 4; i++) sum += v[i];
    return sum;
}

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
    constexpr int nb = 4;  // elements per vector
    int na = (nx + nb - 1) / nb;  // vectors per input row
    vector<vector<double>> normalized_data (ny, vector<double> (nx));

    for (int i = 0; i < ny; i++) {
        double mean_i = mean(data, nx, i);
        double norm_i = norm(data, nx, i, mean_i);

        for (int k = 0; k < nx; k++) {
            double x = data[k + (i * nx)];
            normalized_data[i][k] = (x - mean_i) / norm_i;
        }
    }

    // normalized data, padded, converted to vectors
    vector<vector<double4_t>> normalized_vdata (ny, vector<double4_t> (na));

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < na; j++) {
            for (int kb = 0; kb < nb; kb++) {
                int idx = (j * nb) + kb;
                normalized_vdata[i][j][kb] = idx < nx ? normalized_data[i][idx] : 0.0;
            }
        }
    }

    for (int i = 0; i < ny; i++) {
        result[i + (i * ny)] = 1.0;

        for (int j = i + 1; j < ny; j++) {
            double4_t sum = {0.0, 0.0, 0.0, 0.0};
            for (int k = 0; k < na; k++) sum += normalized_vdata[i][k] * normalized_vdata[j][k];
            result[j + (i * ny)] = (float) hsum4(sum);
        }

    }
}

