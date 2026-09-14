/* GHOST's checked piecewise polynomial evaluator. No third-party source.
   Coefficients are ascending powers of the normalized interval coordinate.
   Every call owns its output; immutable tables can be shared across threads. */
#include <math.h>
#include <stdint.h>
#if defined(_WIN32)
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

EXPORT int ghost_table_eval(int64_t count, const double *x, int intervals,
    const double *bounds, int degree, const double *coeff, int channel,
    double *output) {
    if (count < 0 || intervals < 1 || degree < 0 || degree > 32 ||
        channel < -1 || channel > 1) return 1;
    for (int64_t i = 0; i < count; ++i) {
        double r = x[i];
        int first = channel < 0 ? 0 : channel;
        int channels = channel < 0 ? 2 : 1;
        double *out = output + i * channels * 2;
        if (!isfinite(r) || r < bounds[0] || r > bounds[intervals]) {
            for (int c = 0; c < channels * 2; ++c) out[c] = NAN;
            continue;
        }
        int lo = 0, hi = intervals;
        while (lo + 1 < hi) {
            int mid = lo + (hi-lo)/2;
            if (r < bounds[mid]) hi = mid; else lo = mid;
        }
        double t = (r - bounds[lo]) / (bounds[lo+1] - bounds[lo]);
        const double *base = coeff + (int64_t)lo * (degree+1) * 4;
        for (int c = 0; c < channels; ++c) {
            int offset = 2*(first+c);
            double re = base[4*degree+offset], im = base[4*degree+offset+1];
            for (int j = degree-1; j >= 0; --j) {
                re = re*t + base[4*j+offset];
                im = im*t + base[4*j+offset+1];
            }
            out[2*c] = re; out[2*c+1] = im;
        }
    }
    return 0;
}
