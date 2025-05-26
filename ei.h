#pragma once

float  exponentialIntegralFloat(int order, float x);
double exponentialIntegralDouble(int order, double x);

#ifdef __CUDACC__

__global__
void expIntFloatKernel(int n,
                       const float* xs,
                       float* out,
                       int samples,
                       int offset,
                       int cnt);

__global__
void expIntDoubleKernel(int n,
                        const double* xs,
                        double* out,
                        int samples,
                        int offset,
                        int cnt);
#endif
