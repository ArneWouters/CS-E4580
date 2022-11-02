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

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));

using namespace std;

static inline float hsum8(const float8_t &v) {
    float sum = 0.0;
    for (int i = 0; i < 8; i++) sum += v[i];
    return sum;
}

float mean(const float* data, const int &nx, const int &y) {
    float sum = 0.0;
    for (int i = 0; i < nx; i++) sum += data[i + (y * nx)];
    return sum / nx;
}

float norm(const float* data, const int &nx, const int &y, const float &m) {
    float sum = 0.0;
    for (int i = 0; i < nx; i++) {
        sum += (data[i + (y * nx)] - m) * (data[i + (y * nx)] - m);
    }
    return sqrt(sum);
}


void correlate(int ny, int nx, const float *data, float *result) {
    constexpr float8_t zero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};  // zero vector
    constexpr int nb = 8;  // elements per vector
    int na = (nx + nb - 1) / nb;  // vectors per input row
    int nc = (ny + nb - 1) / nb;
    vector<vector<float>> normalized_data (ny, vector<float> (nx, 0.0));

    #pragma omp parallel for
    for (int i = 0; i < ny; i++) {
        float mean_i = mean(data, nx, i);
        float norm_i = norm(data, nx, i, mean_i);

        for (int k = 0; k < nx; k++) {
            float x = data[k + (i * nx)];
            normalized_data[i][k] = (x - mean_i) / norm_i;
        }
    }

    // normalized data, padded, converted to vectors
    vector<vector<float8_t>> normalized_vdata (nc * nb, vector<float8_t> (na, zero));

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
            vector<vector<float8_t>> sums (nb, vector<float8_t> (nb, zero));

            for (int k = 0; k < na; k++) {
                int i = (ic * nb);
                int j = (jc * nb);

                float8_t x0 = normalized_vdata[i][k];
                float8_t x1 = normalized_vdata[i + 1][k];
                float8_t x2 = normalized_vdata[i + 2][k];
                float8_t x3 = normalized_vdata[i + 3][k];
                float8_t x4 = normalized_vdata[i + 4][k];
                float8_t x5 = normalized_vdata[i + 5][k];
                float8_t x6 = normalized_vdata[i + 6][k];
                float8_t x7 = normalized_vdata[i + 7][k];

                float8_t y0 = normalized_vdata[j][k];
                float8_t y1 = normalized_vdata[j + 1][k];
                float8_t y2 = normalized_vdata[j + 2][k];
                float8_t y3 = normalized_vdata[j + 3][k];
                float8_t y4 = normalized_vdata[j + 4][k];
                float8_t y5 = normalized_vdata[j + 5][k];
                float8_t y6 = normalized_vdata[j + 6][k];
                float8_t y7 = normalized_vdata[j + 7][k];

                sums[0][0] += x0 * y0;
                sums[0][1] += x0 * y1;
                sums[0][2] += x0 * y2;
                sums[0][3] += x0 * y3;
                sums[0][4] += x0 * y4;
                sums[0][5] += x0 * y5;
                sums[0][6] += x0 * y6;
                sums[0][7] += x0 * y7;

                sums[1][0] += x1 * y0;
                sums[1][1] += x1 * y1;
                sums[1][2] += x1 * y2;
                sums[1][3] += x1 * y3;
                sums[1][4] += x1 * y4;
                sums[1][5] += x1 * y5;
                sums[1][6] += x1 * y6;
                sums[1][7] += x1 * y7;

                sums[2][0] += x2 * y0;
                sums[2][1] += x2 * y1;
                sums[2][2] += x2 * y2;
                sums[2][3] += x2 * y3;
                sums[2][4] += x2 * y4;
                sums[2][5] += x2 * y5;
                sums[2][6] += x2 * y6;
                sums[2][7] += x2 * y7;

                sums[3][0] += x3 * y0;
                sums[3][1] += x3 * y1;
                sums[3][2] += x3 * y2;
                sums[3][3] += x3 * y3;
                sums[3][4] += x3 * y4;
                sums[3][5] += x3 * y5;
                sums[3][6] += x3 * y6;
                sums[3][7] += x3 * y7;

                sums[4][0] += x4 * y0;
                sums[4][1] += x4 * y1;
                sums[4][2] += x4 * y2;
                sums[4][3] += x4 * y3;
                sums[4][4] += x4 * y4;
                sums[4][5] += x4 * y5;
                sums[4][6] += x4 * y6;
                sums[4][7] += x4 * y7;

                sums[5][0] += x5 * y0;
                sums[5][1] += x5 * y1;
                sums[5][2] += x5 * y2;
                sums[5][3] += x5 * y3;
                sums[5][4] += x5 * y4;
                sums[5][5] += x5 * y5;
                sums[5][6] += x5 * y6;
                sums[5][7] += x5 * y7;

                sums[6][0] += x6 * y0;
                sums[6][1] += x6 * y1;
                sums[6][2] += x6 * y2;
                sums[6][3] += x6 * y3;
                sums[6][4] += x6 * y4;
                sums[6][5] += x6 * y5;
                sums[6][6] += x6 * y6;
                sums[6][7] += x6 * y7;

                sums[7][0] += x7 * y0;
                sums[7][1] += x7 * y1;
                sums[7][2] += x7 * y2;
                sums[7][3] += x7 * y3;
                sums[7][4] += x7 * y4;
                sums[7][5] += x7 * y5;
                sums[7][6] += x7 * y6;
                sums[7][7] += x7 * y7;

            }

            for (int id = 0; id < nb; id++) {
                for (int jd = 0; jd < nb; jd++) {
                    int i = (ic * nb) + id;
                    int j = (jc * nb) + jd;

                    if (i < ny && j < ny) result[(ny * i) + j] = hsum8(sums[id][jd]);
                }
            }
        }
    }
}

