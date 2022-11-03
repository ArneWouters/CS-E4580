#include <utility>
#include <vector>
#include <iostream>

using namespace std;

typedef double double4_t __attribute__ ((vector_size (4 * sizeof(double))));
constexpr double4_t dzero = {0.0, 0.0, 0.0, 0.0};
constexpr int nb = 4;  // elements per vector

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


static inline double hsum4(const double4_t &v) {
    double sum = 0.0;
    for (int i = 0; i < nb; i++) sum += v[i];
    return sum;
}

vector<vector<double4_t>> vectorize_data(const int &ny, const int &nx, const float *data) {
    vector<vector<double4_t>> vdata (ny, vector<double4_t> (nx, dzero));

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            double4_t d = {
                data[(3 * j) + (3 * nx * i)],
                data[1 + (3 * j) + (3 * nx * i)],
                data[2 + (3 * j) + (3 * nx * i)],
                0.0
            };
            vdata[i][j] = d;
        }
    }

    return vdata;
}

vector<vector<double4_t>> precompute_rectangles(const int &ny, const int &nx,
        const vector<vector<double4_t>> &vdata) {
    vector<vector<double4_t>> sums (ny + 1, vector<double4_t> (nx + 1, dzero));

    for (int i = 1; i < ny + 1; i++) {
        for (int j = 1; j < nx + 1; j++) {
            sums[i][j] = vdata[i - 1][j - 1] + sums[i - 1][j] + sums[i][j - 1] - sums[i - 1][j - 1];
        }
    }

    return sums;
}

/*
This is the function you need to implement. Quick reference:
- x coordinates: 0 <= x < nx
- y coordinates: 0 <= y < ny
- color components: 0 <= c < 3
- input: data[c + 3 * x + 3 * nx * y]
*/
Result segment(int ny, int nx, const float *data) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    auto vdata = vectorize_data(ny, nx, data);
    auto precomputed = precompute_rectangles(ny, nx, vdata);
    double4_t sumAll = precomputed[ny][nx];

    vector<vector<pair<double, Result>>> scores (ny + 1,
            vector<pair<double, Result>> (nx + 1, make_pair(0.0, result)));

    // loop rectangle size
    #pragma omp parallel for collapse(2)
    for (int h = 1; h < ny + 1; h++) {
        for (int w = 1; w < nx + 1; w++) {
            double nInside = h * w;  // amount of pixels inside
            double nOutside = (nx * ny) - nInside;  // amount of pixels outside

            // loop rectangle pos
            for (int i = 0; i < ny - h + 1; i++) {
                for (int j = 0; j < nx - w + 1; j++) {
                    int y1 = i + h;
                    int x1 = j + w;

                    double4_t sumInside = precomputed[y1][x1] - precomputed[y1][j]
                        - precomputed[i][x1] + precomputed[i][j];
                    double4_t sumOutside = sumAll - sumInside;

                    double score = hsum4((sumInside * sumInside * (1.0 / nInside))
                        + (sumOutside * sumOutside * (1.0 / nOutside)));

                    if (score > scores[h][w].first) {
                        scores[h][w].first = score;
                        scores[h][w].second = {
                            i, j,
                            y1, x1,
                            {
                                (float) (sumOutside[0] / nOutside),
                                (float) (sumOutside[1] / nOutside),
                                (float) (sumOutside[2] / nOutside)
                            },
                            {
                                (float) (sumInside[0] / nInside),
                                (float) (sumInside[1] / nInside),
                                (float) (sumInside[2] / nInside)}
                        };
                    }
                }
            }
        }
    }

    double bestScore = 0.0;

    for (int i = 0; i < ny + 1; i++) {
        for (int j = 0; j < nx + 1; j++) {
            if (scores[i][j].first > bestScore) {
                bestScore = scores[i][j].first;
                result = scores[i][j].second;
            }
        }
    }

    return result;
}
