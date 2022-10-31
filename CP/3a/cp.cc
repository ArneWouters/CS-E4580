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
    constexpr double4_t zero = {0.0, 0.0, 0.0, 0.0};  // zero vector
    constexpr int nb = 4;  // elements per vector
    int na = (nx + nb - 1) / nb;  // vectors per input row
    int nc = (ny + nb - 1) / nb;
    vector<vector<double>> normalized_data (ny, vector<double> (nx, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        double mean_i = mean(data, nx, i);
        double norm_i = norm(data, nx, i, mean_i);

        for (int k = 0; k < nx; k++) {
            double x = data[k + (i * nx)];
            normalized_data[i][k] = (x - mean_i) / norm_i;
        }
    }

    // normalized data, padded, converted to vectors
    vector<vector<double4_t>> normalized_vdata (nc * nb, vector<double4_t> (na, zero));

    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < na; j++) {
            for (int kb = 0; kb < nb; kb++) {
                int idx = (j * nb) + kb;
                normalized_vdata[i][j][kb] = idx < nx ? normalized_data[i][idx] : 0.0;
            }
        }
    }

    #pragma omp parallel for schedule(dynamic, 1)
    for (int ic = 0; ic < nc; ic++) {
        for (int jc = ic; jc < nc; jc++) {
            vector<vector<double4_t>> sums (nb, vector<double4_t> (nb, zero));

            for (int k = 0; k < na; k++) {
                int i = (ic * nb);
                int j = (jc * nb);

                double4_t x0 = normalized_vdata[i][k];
                double4_t x1 = normalized_vdata[i + 1][k];
                double4_t x2 = normalized_vdata[i + 2][k];
                double4_t x3 = normalized_vdata[i + 3][k];

                double4_t y0 = normalized_vdata[j][k];
                double4_t y1 = normalized_vdata[j + 1][k];
                double4_t y2 = normalized_vdata[j + 2][k];
                double4_t y3 = normalized_vdata[j + 3][k];

                sums[0][0] += x0 * y0;
                sums[0][1] += x0 * y1;
                sums[0][2] += x0 * y2;
                sums[0][3] += x0 * y3;

                sums[1][0] += x1 * y0;
                sums[1][1] += x1 * y1;
                sums[1][2] += x1 * y2;
                sums[1][3] += x1 * y3;

                sums[2][0] += x2 * y0;
                sums[2][1] += x2 * y1;
                sums[2][2] += x2 * y2;
                sums[2][3] += x2 * y3;

                sums[3][0] += x3 * y0;
                sums[3][1] += x3 * y1;
                sums[3][2] += x3 * y2;
                sums[3][3] += x3 * y3;
            }

            for (int id = 0; id < nb; id++) {
                for (int jd = 0; jd < nb; jd++) {
                    int i = (ic * nb) + id;
                    int j = (jc * nb) + jd;

                    if (i < ny && j < ny) result[(ny * i) + j] = hsum4(sums[id][jd]);
                }
            }
        }
    }
}

