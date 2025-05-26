
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cfloat>       

__constant__ float  d_eulerConstF = 0.5772156649015329f;
__constant__ float  d_epsilonF    = 1e-30f;
__constant__ int    d_maxIterF    = 2000000000;
__constant__ double d_eulerConstD = 0.5772156649015329;
__constant__ double d_epsilonD    = 1e-30;
__constant__ int    d_maxIterD    = 2000000000;

__global__
void expIntFloatKernel(int n,
                       const float* xs,
                       float*       out,
                       int          samples,
                       int          offset,
                       int          cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnt) return;
    int gIdx  = offset + idx;
    int order = gIdx / samples;
    int j     = gIdx % samples;
    float x   = xs[j];

    float ans = 0, h, b, c, d, fact, psi;
    int nm1 = order - 1;
    if (order == 0) {
        ans = expf(-x) / x;
    } else if (x > 1.0f) {
        b = x + order;
        c = FLT_MAX;
        d = 1.0f / b;
        h = d;
        for (int i = 1; i <= d_maxIterF; ++i) {
            float a = -i * (nm1 + i);
            b += 2.0f;
            d  = 1.0f / (a * d + b);
            c  = b + a / c;
            float del = c * d;
            h *= del;
            if (fabsf(del - 1.0f) <= d_epsilonF) break;
        }
        ans = h * expf(-x);
    } else {
        if (nm1 != 0) ans = 1.0f / nm1;
        else          ans = -logf(x) - d_eulerConstF;
        fact = 1.0f;
        for (int i = 1; i <= d_maxIterF; ++i) {
            fact *= -x / i;
            float del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -d_eulerConstF;
                for (int ii = 1; ii <= nm1; ++ii) psi += 1.0f / ii;
                del = fact * (-logf(x) + psi);
            }
            ans += del;
            if (fabsf(del) < fabsf(ans) * d_epsilonF) break;
        }
    }
    out[gIdx] = ans;
}

__global__
void expIntDoubleKernel(int n,
                        const double* xs,
                        double*       out,
                        int           samples,
                        int           offset,
                        int           cnt)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnt) return;
    int gIdx  = offset + idx;
    int order = gIdx / samples;
    int j     = gIdx % samples;
    double x  = xs[j];

    double ans = 0, h, b, c, d, fact, psi;
    int nm1 = order - 1;
    if (order == 0) {
        ans = exp(-x) / x;
    } else if (x > 1.0) {
        b = x + order;
        c = DBL_MAX;
        d = 1.0 / b;
        h = d;
        for (int i = 1; i <= d_maxIterD; ++i) {
            double a = -i * (nm1 + i);
            b += 2.0;
            d  = 1.0 / (a * d + b);
            c  = b + a / c;
            double del = c * d;
            h *= del;
            if (fabs(del - 1.0) <= d_epsilonD) break;
        }
        ans = h * exp(-x);
    } else {
        if (nm1 != 0) ans = 1.0 / nm1;
        else          ans = -log(x) - d_eulerConstD;
        fact = 1.0;
        for (int i = 1; i <= d_maxIterD; ++i) {
            fact *= -x / i;
            double del;
            if (i != nm1) del = -fact / (i - nm1);
            else {
                psi = -d_eulerConstD;
                for (int ii = 1; ii <= nm1; ++ii) psi += 1.0 / ii;
                del = fact * (-log(x) + psi);
            }
            ans += del;
            if (fabs(del) < fabs(ans) * d_epsilonD) break;
        }
    }
    out[gIdx] = ans;
}
