#include <utility>
#include <vector>
#include <iostream>
#include <limits>
#include <x86intrin.h>

using namespace std;

typedef float float8_t __attribute__ ((vector_size (8 * sizeof(float))));
constexpr float8_t fzero = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
constexpr int nb = 8;

struct Result {
    int y0;
    int x0;
    int y1;
    int x1;
    float outer[3];
    float inner[3];
};


static inline float8_t max8(const float8_t &x, const float8_t &y) {
    return x > y ? x : y;
}

static inline float hmax8(const float8_t &v) {
    float m = -numeric_limits<float>::infinity();
    for(int i = 0; i < nb; i++) m = max(v[i], m);
    return m;
}

vector<vector<float>> precompute_rectangles(const int &ny, const int &nx, const float* data) {
    vector<vector<float>> sums (ny + 1, vector<float> (nx + 1, 0.0));

    for (int i = 1; i < ny + 1; i++) {
        for (int j = 1; j < nx + 1; j++) {
            sums[i][j] = data[(3 * (j - 1)) + (3 * nx * (i - 1))] + sums[i - 1][j] + sums[i][j - 1]
                - sums[i - 1][j - 1];
        }
    }

    return sums;
}

pair<int, int> find_optimal_dimension(const int &nx, const int &ny,
        const vector<vector<float>> &precomputed) {
    pair<int, int> dim = {-1, -1};
    float sumAll = precomputed[ny][nx];
    float8_t sumAll8 = {sumAll, sumAll, sumAll, sumAll, sumAll, sumAll, sumAll, sumAll};
    float best_score = 0.0;

    // loop rectangle size
    #pragma omp parallel for collapse(2)
    for (int h = 1; h < ny + 1; h++) {
        for (int w = 1; w < nx + 1; w++) {
            double nInside = h * w;  // amount of pixels inside
            double nOutside = (nx * ny) - nInside;  // amount of pixels outside
            float8_t best8 = fzero;
            float best = 0.0;

            // loop rectangle pos in blocks
            for (int i = 0; i < ny - h + 1; i++) {
                int na = (nx - w + 1) / nb;  // number of blocks

                // iterate over blocks
                for (int k = 0; k < na; k++) {
                    int x0 = nb * k;
                    int x1 = x0 + w;
                    int y1 = i + h;

                    float8_t p1 = _mm256_loadu_ps(&precomputed[y1][x1]);
                    float8_t p2 = _mm256_loadu_ps(&precomputed[y1][x0]);
                    float8_t p3 = _mm256_loadu_ps(&precomputed[i][x1]);
                    float8_t p4 = _mm256_loadu_ps(&precomputed[i][x0]);

                    float8_t sumInside = p1 - p2 - p3 + p4;
                    float8_t sumOutside = sumAll8 - sumInside;

                    float8_t score = (sumInside * sumInside * (float) (1.0 / nInside))
                        + (sumOutside * sumOutside * (float) (1.0 / nOutside));

                    best8 = max8(score, best8);
                }

                // iterate over the remainder values that didn't fit inside a block
                for (int j = nb * na; j < nx - w + 1; j++) {
                    int y1 = i + h;
                    int x1 = j + w;

                    float sumInside = precomputed[y1][x1] - precomputed[y1][j]
                        - precomputed[i][x1] + precomputed[i][j];
                    float sumOutside = sumAll - sumInside;

                    float score = (sumInside * sumInside * (1.0 / nInside))
                        + (sumOutside * sumOutside * (1.0 / nOutside));

                    best = max(score, best);
                }

                // determine best possible score and update dimension if needed
                best = max(best, hmax8(best8));
            }

            #pragma omp critical
            {
                if (best > best_score) {
                    dim = {h, w};
                    best_score = best;
                }
            }
        }
    }

    return dim;
}

Result find_optimal_result(const int &nx, const int &ny, const int &h, const int &w,
        const vector<vector<float>> &precomputed) {
    Result result{0, 0, 0, 0, {0, 0, 0}, {0, 0, 0}};
    float sumAll = precomputed[ny][nx];
    double nInside = h * w;  // amount of pixels inside
    double nOutside = (nx * ny) - nInside;
    float best_score = 0.0;

    for (int i = 0; i < ny - h + 1; i++) {
        for (int j = 0; j < nx - w + 1; j++) {
            int y1 = i + h;
            int x1 = j + w;

            float sumInside = precomputed[y1][x1] - precomputed[y1][j]
                - precomputed[i][x1] + precomputed[i][j];
            float sumOutside = sumAll - sumInside;

            float score = (sumInside * sumInside * (1.0 / nInside))
                + (sumOutside * sumOutside * (1.0 / nOutside));

            if (score > best_score) {
                float innerValue = sumInside * (1.0 / nInside);
                float outerValue = sumOutside * (1.0 / nOutside);

                best_score = score;
                result.x0 = j;
                result.x1 = x1;
                result.y0 = i;
                result.y1 = y1;
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
        auto precomputed = precompute_rectangles(ny, nx, data);
    auto dim = find_optimal_dimension(nx, ny, precomputed);
    auto result = find_optimal_result(nx, ny, dim.first, dim.second, precomputed);
    return result;
}

