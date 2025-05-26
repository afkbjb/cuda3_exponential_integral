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
    unsigned n = 10, m = 10, blk = 256;
    double a = 0.0, b = 10.0;
    bool timing = false, verbose = false;
    bool skipCPU = false, skipGPU = false;
    for (int i = 1; i < argc; ++i) {
        if      (!strcmp(argv[i], "-n")) n       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-m")) m       = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-a")) a       = atof(argv[++i]);
        else if (!strcmp(argv[i], "-b")) b       = atof(argv[++i]);
        else if (!strcmp(argv[i], "-B")) blk     = atoi(argv[++i]);
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

    double cpuFsec = 0.0, cpuDsec = 0.0;
    double gpuFsec = 0.0, gpuDsec = 0.0;

    // --- CPU float + double ---
    if (!skipCPU) {
        double t0 = now_sec();
        for (unsigned o = 0; o < n; ++o)
          for (unsigned j = 0; j < m; ++j)
            cpuF[o*m + j] = exponentialIntegralFloat(o+1, xsF[j]);
        double t1 = now_sec();
        cpuFsec = t1 - t0;

        double t2 = now_sec();
        for (unsigned o = 0; o < n; ++o)
          for (unsigned j = 0; j < m; ++j)
            cpuD[o*m + j] = exponentialIntegralDouble(o+1, xsD[j]);
        double t3 = now_sec();
        cpuDsec = t3 - t2;
    }

    // --- GPU float + double with streams and events ---
    if (!skipGPU) {
        cudaFree(0);

        cudaStream_t sF, sD;
        cudaStreamCreate(&sF);
        cudaStreamCreate(&sD);

        cudaEvent_t startF, stopF, startD, stopD;
        cudaEventCreate(&startF); cudaEventCreate(&stopF);
        cudaEventCreate(&startD); cudaEventCreate(&stopD);

        float  *d_xf, *d_of;
        double *d_xd, *d_od;
        cudaMalloc(&d_xf, m     * sizeof(float));
        cudaMalloc(&d_of, total * sizeof(float));
        cudaMalloc(&d_xd, m     * sizeof(double));
        cudaMalloc(&d_od, total * sizeof(double));

        cudaEventRecord(startF, sF);
        cudaMemcpyAsync(d_xf, xsF.data(), m * sizeof(float),
                        cudaMemcpyHostToDevice, sF);
        {
            unsigned grid = (total + blk - 1) / blk;
            expIntFloatKernel<<<grid, blk, 0, sF>>>(n, d_xf, d_of, m, 0, total);
        }
        cudaMemcpyAsync(gpuF.data(), d_of, total * sizeof(float),
                        cudaMemcpyDeviceToHost, sF);
        cudaEventRecord(stopF, sF);

        cudaEventRecord(startD, sD);
        cudaMemcpyAsync(d_xd, xsD.data(), m * sizeof(double),
                        cudaMemcpyHostToDevice, sD);
        {
            unsigned grid = (total + blk - 1) / blk;
            expIntDoubleKernel<<<grid, blk, 0, sD>>>(n, d_xd, d_od, m, 0, total);
        }
        cudaMemcpyAsync(gpuD.data(), d_od, total * sizeof(double),
                        cudaMemcpyDeviceToHost, sD);
        cudaEventRecord(stopD, sD);

        cudaStreamSynchronize(sF);
        cudaStreamSynchronize(sD);

        float msF = 0.0f, msD = 0.0f;
        cudaEventElapsedTime(&msF, startF, stopF);
        cudaEventElapsedTime(&msD, startD, stopD);
        gpuFsec = msF * 1e-3;
        gpuDsec = msD * 1e-3;

        cudaFree(d_xf); cudaFree(d_of);
        cudaFree(d_xd); cudaFree(d_od);
        cudaEventDestroy(startF); cudaEventDestroy(stopF);
        cudaEventDestroy(startD); cudaEventDestroy(stopD);
        cudaStreamDestroy(sF); cudaStreamDestroy(sD);
    }

    // --- print timing ---
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

    // --- verbose compare ---
    if (verbose && !skipCPU && !skipGPU) {
        printf("\n======= Comparison =======\n");
        for (unsigned i = 0; i < n; ++i)
        for (unsigned j = 0; j < m; ++j) {
            unsigned idx = i*m + j;
            double x = a + (j + 1) * (b - a) / m;
            printf("(n=%2u, x=%.1f):\n", i+1, x);
            printf("    float  CPU=%.6g  GPU=%.6g\n", cpuF[idx], gpuF[idx]);
            printf("    double CPU=%.6g  GPU=%.6g\n", cpuD[idx], gpuD[idx]);
        }
        printf("===========================\n");
    }

    return 0;
}

void printUsage () {
    printf("exponentialIntegral program\n");
    printf("by: Jose Mauricio Refojo <refojoj@tcd.ie>\n");
    printf("This program will calculate a number of exponential integrals\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("      -a   value   : will set the a value of the (a,b) interval in which the samples are taken to value (default: 0.0)\n");
    printf("      -b   value   : will set the b value of the (a,b) interval in which the samples are taken to value (default: 10.0)\n");
    printf("      -c           : will skip the CPU test\n");
    printf("      -g           : will skip the GPU test\n");
    printf("      -h           : will show this usage\n");
    printf("      -i   size    : will set the number of iterations to size (default: 2000000000)\n");
    printf("      -n   size    : will set the n (the order up to which we are calculating the exponential integrals) to size (default: 10)\n");
    printf("      -m   size    : will set the number of samples taken in the (a,b) interval to size (default: 10)\n");
    printf("      -t           : will output the amount of time that it took to generate each norm (default: no)\n");
    printf("      -v           : will activate the verbose mode  (default: no)\n");
    printf("      -B   value   : will set the number of threads per block to value (default: 256)\n");
    printf("     \n");
}
