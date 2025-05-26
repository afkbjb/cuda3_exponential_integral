#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "ei.h"

static double now_sec() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

void printUsage();

int main(int argc, char** argv) {
    // parse args
    unsigned n = 10, m = 10, blk = 256;     // <-- new blk var
    double a = 0.0, b = 10.0;
    bool timing = false, verbose = false;
    bool skipCPU = false, skipGPU = false;

    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-n")) n       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m")) m       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-a")) a       = atof(argv[++i]);
        else if (!strcmp(argv[i], "-b")) b       = atof(argv[++i]);
        else if (!strcmp(argv[i], "-B")) blk     = atoi(argv[++i]);  // <-- parse -B
        else if (!strcmp(argv[i], "-t")) timing  = true;
        else if (!strcmp(argv[i], "-v")) verbose = true;
        else if (!strcmp(argv[i], "-c")) skipCPU = true;
        else if (!strcmp(argv[i], "-g")) skipGPU = true;
        else if (!strcmp(argv[i], "-h")) { printUsage(); return 0; }
    }

    unsigned total = n * m;
    std::vector<float>  xsF(m),  cpuF(total), gpuF(total);
    std::vector<double> xsD(m), cpuD(total), gpuD(total);
    for (unsigned j = 0; j < m; ++j) {
        double x = a + (j + 1) * (b - a) / m;
        xsF[j] = (float)x;
        xsD[j] = x;
    }

    double cpuFsec = 0.0, cpuDsec = 0.0, gpuFsec = 0.0, gpuDsec = 0.0;

    // --- CPU float+double ---
    if (!skipCPU) {
        double t0 = now_sec();
        for (unsigned o = 0; o < n; ++o)
          for (unsigned j = 0; j < m; ++j)
            cpuF[o * m + j] = exponentialIntegralFloat(o + 1, xsF[j]);
        double t1 = now_sec();
        cpuFsec = t1 - t0;

        double t2 = now_sec();
        for (unsigned o = 0; o < n; ++o)
          for (unsigned j = 0; j < m; ++j)
            cpuD[o * m + j] = exponentialIntegralDouble(o + 1, xsD[j]);
        double t3 = now_sec();
        cpuDsec = t3 - t2;
    }

    if (!skipGPU) {
        // pre‐warm CUDA
        cudaFree(0);
        cudaDeviceSynchronize();

        // float
        double t4 = now_sec();
        float  *d_xf, *d_of;
        cudaMalloc(&d_xf, m     * sizeof(float));
        cudaMalloc(&d_of, total * sizeof(float));
        cudaMemcpy(d_xf, xsF.data(), m * sizeof(float), cudaMemcpyHostToDevice);
        {
            unsigned grid = (total + blk - 1) / blk;
            expIntFloatKernel<<<grid, blk>>>(n, d_xf, d_of, m, 0, total);
        }
        cudaMemcpy(gpuF.data(), d_of, total * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(d_xf);
        cudaFree(d_of);
        double t5 = now_sec();
        gpuFsec = t5 - t4;

        // double
        double t6 = now_sec();
        double *d_xd, *d_od;
        cudaMalloc(&d_xd, m     * sizeof(double));
        cudaMalloc(&d_od, total * sizeof(double));
        cudaMemcpy(d_xd, xsD.data(), m * sizeof(double), cudaMemcpyHostToDevice);
        {
            unsigned grid = (total + blk - 1) / blk;
            expIntDoubleKernel<<<grid, blk>>>(n, d_xd, d_od, m, 0, total);
        }
        cudaMemcpy(gpuD.data(), d_od, total * sizeof(double), cudaMemcpyDeviceToHost);
        cudaFree(d_xd);
        cudaFree(d_od);
        double t7 = now_sec();
        gpuDsec = t7 - t6;
    }

    if (timing) {
        printf("\n==== Timing (blk=%u) ====", blk);
        if (!skipCPU) {
            printf("\nCPU (float) : %.3f ms", cpuFsec * 1e3);
            printf("\nCPU (double): %.3f ms", cpuDsec * 1e3);
        }
        if (!skipGPU) {
            printf("\nGPU (float) : %.3f ms", gpuFsec * 1e3);
            printf("\nGPU (double): %.3f ms", gpuDsec * 1e3);
        }
        if (!skipCPU && !skipGPU) {
            printf("\nSpeedup (float) : %.2fx", cpuFsec / gpuFsec);
            printf("\nSpeedup (double): %.2fx", cpuDsec / gpuDsec);
        }
        printf("\n=================\n");
    }

    if (verbose && !skipCPU && !skipGPU) {
        printf("\n==== Comparison ====");
        for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < m; ++j) {
            unsigned idx = i * m + j;
            double x = a + (j + 1) * (b - a) / m;
            printf("\n(n=%2u,x=%.3f): f CPU=%.6g GPU=%.6g%s   d CPU=%.6g GPU=%.6g%s",
                   i+1, x,
                   cpuF[idx], gpuF[idx], (fabs(cpuF[idx]-gpuF[idx])>1e-5f?" ERRf":"" ),
                   cpuD[idx], gpuD[idx], (fabs(cpuD[idx]-gpuD[idx])>1e-5?   " ERRd":"" ));
        }
        printf("\n===========================\n");
    }

    return 0;
}

void printUsage () {
    printf("exponentialIntegral program\n");
    printf(" usage:\n");
    printf("  -n <orders>       (default 10)\n");
    printf("  -m <samples>      (default 10)\n");
    printf("  -a <start>        (default 0.0)\n");
    printf("  -b <end>          (default 10.0)\n");
    printf("  -B <blkSize>      threads per block (default 256)\n");
    printf("  -c                skip CPU\n");
    printf("  -g                skip GPU\n");
    printf("  -t                print timing\n");
    printf("  -v                verbose compare\n");
    printf("  -h                this help\n");
    printf("      -B           : threads per block (default 256)\n");
}
