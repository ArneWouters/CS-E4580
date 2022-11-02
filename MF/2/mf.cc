/*
This is the function you need to implement. Quick reference:
- input rows: 0 <= y < ny
- input columns: 0 <= x < nx
- element at row y and column x is stored in in[x + y*nx]
- for each pixel (x, y), store the median of the pixels (a, b) which satisfy
  max(x-hx, 0) <= a < min(x+hx+1, nx), max(y-hy, 0) <= b < min(y+hy+1, ny)
  in out[x + y*nx].
*/

#include <vector>
#include <algorithm>

using namespace std;

void mf(int ny, int nx, int hy, int hx, const float *in, float *out) {
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            int lwy = max(i - hy, 0);
            int rwy = min(i + hy + 1, ny);
            int lwx = max(j - hx, 0);
            int rwx = min(j + hx + 1, nx);
            int window_size = (rwy - lwy) * (rwx - lwx);

            float v [window_size];

            for (int k = lwy; k < rwy; k++) {
                for (int l = lwx; l < rwx; l++) {
                    int idx = ((k - lwy) * (rwx - lwx)) + (l - lwx);
                    v[idx] = in[(k * nx) + l];
                }
            }

            int m = window_size / 2;
            nth_element(v, v + m, v + window_size);

            if (window_size % 2 == 0) {
                int m2 = (window_size / 2) - 1;
                nth_element(v, v + m2, v + window_size);
                out[(i * nx) + j] = (v[m] + v[m2]) / 2.0;

            } else {
                out[(i * nx) + j] = v[m];
            }
        }
    }
}

