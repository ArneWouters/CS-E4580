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
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            vector<float> v {};

            for (int k = max(i - hy, 0); k < min(i + hy + 1, ny); k++) {
                for (int l = max(j - hx, 0); l < min(j + hx + 1, nx); l++) {
                    v.push_back(in[(k * nx) + l]);
                }
            }

            int m = v.size() / 2;
            nth_element(v.begin(), v.begin() + m, v.end());

            if (v.size() % 2 == 0) {
                int m2 = (v.size() / 2) - 1;
                nth_element(v.begin(), v.begin() + m2, v.end());
                out[(i * nx) + j] = (v[m] + v[m2]) / 2.0;

            } else {
                out[(i * nx) + j] = v[m];
            }
        }
    }
}

