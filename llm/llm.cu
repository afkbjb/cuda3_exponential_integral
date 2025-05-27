///// CUDA Version - Converted by Assistant 
//------------------------------------------------------------------------------
// File : llm.cu
//------------------------------------------------------------------------------

#include <time.h>
#include <iostream>
#include <limits>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include <time.h>
#include <unistd.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

bool verbose, timing, cpu, gpu;
int  maxIterations;
unsigned int n, numberOfSamples;
double a, b;

using namespace std;

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constants
__constant__ float d_eulerConstantFloat = 0.5772156649015329f;
__constant__ double d_eulerConstantDouble = 0.5772156649015329;
__constant__ int d_maxIterations;

// Device function for double precision exponential integral
__device__ double exponentialIntegralDeviceDouble(const int n, const double x) {
    double epsilon = 1.E-30;
    double bigDouble = 1e300; // Use a large but safe value
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0 || x < 0.0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
        return NAN; // Return NaN for invalid inputs instead of exiting
    }
    
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0) {
            // Continued fraction approach
            b = x + n;
            c = bigDouble;
            d = 1.0 / b;
            h = d;
            
            for (i = 1; i <= d_maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
        } else {
            // Series expansion
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - d_eulerConstantDouble);
            fact = 1.0;
            
            for (i = 1; i <= d_maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -d_eulerConstantDouble;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0 / ii;
                    }
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
        }
    }
    return ans;
}

// Device function for single precision exponential integral
__device__ float exponentialIntegralDeviceFloat(const int n, const float x) {
    float epsilon = 1.E-15f; // Adjusted for float precision
    float bigFloat = 1e30f;   // Use a large but safe value for float
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0f;

    if (n < 0 || x < 0.0f || (x == 0.0f && ((n == 0) || (n == 1)))) {
        return NAN; // Return NaN for invalid inputs
    }
    
    if (n == 0) {
        ans = expf(-x) / x;
    } else {
        if (x > 1.0f) {
            // Continued fraction approach
            b = x + n;
            c = bigFloat;
            d = 1.0f / b;
            h = d;
            
            for (i = 1; i <= d_maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0f;
                d = 1.0f / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabsf(del - 1.0f) <= epsilon) {
                    ans = h * expf(-x);
                    return ans;
                }
            }
            ans = h * expf(-x);
        } else {
            // Series expansion
            ans = (nm1 != 0 ? 1.0f / nm1 : -logf(x) - d_eulerConstantFloat);
            fact = 1.0f;
            
            for (i = 1; i <= d_maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -d_eulerConstantFloat;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0f / ii;
                    }
                    del = fact * (-logf(x) + psi);
                }
                ans += del;
                if (fabsf(del) < fabsf(ans) * epsilon) return ans;
            }
        }
    }
    return ans;
}

// CUDA kernel for computing exponential integrals
__global__ void exponentialIntegralKernel(
    float* resultsFloat, double* resultsDouble,
    unsigned int n, unsigned int numberOfSamples,
    double a, double b) {
    
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int totalThreads = gridDim.x * blockDim.x;
    
    double division = (b - a) / ((double)numberOfSamples);
    
    // Each thread processes multiple samples to ensure all work is covered
    for (int sample_idx = idx; sample_idx < numberOfSamples; sample_idx += totalThreads) {
        double x = a + (sample_idx + 1) * division;
        
        // Calculate for all orders n
        for (unsigned int order = 1; order <= n; order++) {
            int result_idx = (order - 1) * numberOfSamples + sample_idx;
            
            resultsFloat[result_idx] = exponentialIntegralDeviceFloat(order, (float)x);
            resultsDouble[result_idx] = exponentialIntegralDeviceDouble(order, x);
        }
    }
}

// CPU versions of exponential integral functions
double exponentialIntegralDouble(const int n, const double x) {
    static const double eulerConstant = 0.5772156649015329;
    double epsilon = 1.E-30;
    double bigDouble = std::numeric_limits<double>::max();
    int i, ii, nm1 = n - 1;
    double a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0.0 || x < 0.0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
        cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
        exit(1);
    }
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0) {
            b = x + n;
            c = bigDouble;
            d = 1.0 / b;
            h = d;
            for (i = 1; i <= ::maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
            return ans;
        } else { // Evaluate series
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant); // First term
            fact = 1.0;
            for (i = 1; i <= ::maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0 / ii;
                    }
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

float exponentialIntegralFloat(const int n, const float x) {
    static const float eulerConstant = 0.5772156649015329;
    float epsilon = 1.E-30;
    float bigfloat = std::numeric_limits<float>::max();
    int i, ii, nm1 = n - 1;
    float a, b, c, d, del, fact, h, psi, ans = 0.0;

    if (n < 0.0 || x < 0.0 || (x == 0.0 && ((n == 0) || (n == 1)))) {
        cout << "Bad arguments were passed to the exponentialIntegral function call" << endl;
        exit(1);
    }
    if (n == 0) {
        ans = exp(-x) / x;
    } else {
        if (x > 1.0) {
            b = x + n;
            c = bigfloat;
            d = 1.0 / b;
            h = d;
            for (i = 1; i <= ::maxIterations; i++) {
                a = -i * (nm1 + i);
                b += 2.0;
                d = 1.0 / (a * d + b);
                c = b + a / c;
                del = c * d;
                h *= del;
                if (fabs(del - 1.0) <= epsilon) {
                    ans = h * exp(-x);
                    return ans;
                }
            }
            ans = h * exp(-x);
            return ans;
        } else { // Evaluate series
            ans = (nm1 != 0 ? 1.0 / nm1 : -log(x) - eulerConstant); // First term
            fact = 1.0;
            for (i = 1; i <= ::maxIterations; i++) {
                fact *= -x / i;
                if (i != nm1) {
                    del = -fact / (i - nm1);
                } else {
                    psi = -eulerConstant;
                    for (ii = 1; ii <= nm1; ii++) {
                        psi += 1.0 / ii;
                    }
                    del = fact * (-log(x) + psi);
                }
                ans += del;
                if (fabs(del) < fabs(ans) * epsilon) return ans;
            }
            return ans;
        }
    }
    return ans;
}

void outputResultsCpu(const std::vector<std::vector<float>>& resultsFloatCpu,
                     const std::vector<std::vector<double>>& resultsDoubleCpu) {
    unsigned int ui, uj;
    double x, division = (::b - ::a) / ((double)(::numberOfSamples));

    for (ui = 1; ui <= ::n; ui++) {
        for (uj = 1; uj <= ::numberOfSamples; uj++) {
            x = ::a + uj * division;
            std::cout << "CPU==> exponentialIntegralDouble (" << ui << "," << x << ")=" << resultsDoubleCpu[ui - 1][uj - 1] << " ,";
            std::cout << "exponentialIntegralFloat  (" << ui << "," << x << ")=" << resultsFloatCpu[ui - 1][uj - 1] << endl;
        }
    }
}

// Function prototypes
float exponentialIntegralFloat(const int n, const float x);
double exponentialIntegralDouble(const int n, const double x);
void outputResultsGpu(const std::vector<float>& resultsFloatGpu, 
                     const std::vector<double>& resultsDoubleGpu,
                     unsigned int n, unsigned int numberOfSamples, double a, double b);
void outputResultsCpu(const std::vector<std::vector<float>>& resultsFloatCpu,
                     const std::vector<std::vector<double>>& resultsDoubleCpu);
int parseArguments(int argc, char **argv);
void printUsage(void);

int main(int argc, char *argv[]) {
    cpu = true;
    gpu = true;
    verbose = false;
    timing = false;
    n = 10;
    numberOfSamples = 10;
    a = 0.0;
    b = 10.0;
    maxIterations = 2000000000;

    struct timeval cpuStart, cpuEnd, gpuStart, gpuEnd;
    
    parseArguments(argc, argv);

    if (verbose) {
        cout << "n=" << n << endl;
        cout << "numberOfSamples=" << numberOfSamples << endl;
        cout << "a=" << a << endl;
        cout << "b=" << b << endl;
        cout << "timing=" << timing << endl;
        cout << "verbose=" << verbose << endl;
    }

    // Sanity checks
    if (a >= b) {
        cout << "Incorrect interval (" << a << "," << b << ") has been stated!" << endl;
        return 0;
    }
    if (n <= 0) {
        cout << "Incorrect orders (" << n << ") have been stated!" << endl;
        return 0;
    }
    if (numberOfSamples <= 0) {
        cout << "Incorrect number of samples (" << numberOfSamples << ") have been stated!" << endl;
        return 0;
    }

    // CPU computation
    std::vector<std::vector<float>> resultsFloatCpu;
    std::vector<std::vector<double>> resultsDoubleCpu;
    double timeTotalCpu = 0.0;

    if (cpu) {
        try {
            resultsFloatCpu.resize(n, vector<float>(numberOfSamples));
        } catch (std::bad_alloc const&) {
            cout << "resultsFloatCpu memory allocation fail!" << endl;
            exit(1);
        }
        try {
            resultsDoubleCpu.resize(n, vector<double>(numberOfSamples));
        } catch (std::bad_alloc const&) {
            cout << "resultsDoubleCpu memory allocation fail!" << endl;
            exit(1);
        }

        double x, division = (b - a) / ((double)(numberOfSamples));

        gettimeofday(&cpuStart, NULL);
        for (unsigned int ui = 1; ui <= n; ui++) {
            for (unsigned int uj = 1; uj <= numberOfSamples; uj++) {
                x = a + uj * division;
                resultsFloatCpu[ui - 1][uj - 1] = exponentialIntegralFloat(ui, x);
                resultsDoubleCpu[ui - 1][uj - 1] = exponentialIntegralDouble(ui, x);
            }
        }
        gettimeofday(&cpuEnd, NULL);
        timeTotalCpu = ((cpuEnd.tv_sec + cpuEnd.tv_usec * 0.000001) - 
                       (cpuStart.tv_sec + cpuStart.tv_usec * 0.000001));
    }

    // GPU computation
    double timeGpu = 0.0;
    if (gpu) {
        // Copy maxIterations to constant memory
        CUDA_CHECK(cudaMemcpyToSymbol(d_maxIterations, &maxIterations, sizeof(int)));
        
        // Calculate total result size
        size_t totalResults = n * numberOfSamples;
        
        // Allocate host memory for results
        std::vector<float> resultsFloatGpu(totalResults);
        std::vector<double> resultsDoubleGpu(totalResults);
        
        // Allocate device memory
        float* d_resultsFloat;
        double* d_resultsDouble;
        
        CUDA_CHECK(cudaMalloc(&d_resultsFloat, totalResults * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&d_resultsDouble, totalResults * sizeof(double)));
        
        // Configure kernel launch parameters
        int blockSize = 256;
        int numBlocks = min(65535, (int)((numberOfSamples + blockSize - 1) / blockSize));
        
        if (verbose) {
            cout << "CUDA Configuration:" << endl;
            cout << "Block size: " << blockSize << endl;
            cout << "Number of blocks: " << numBlocks << endl;
            cout << "Total threads: " << numBlocks * blockSize << endl;
        }
        
        // Launch kernel and measure time
        gettimeofday(&gpuStart, NULL);
        
        exponentialIntegralKernel<<<numBlocks, blockSize>>>(
            d_resultsFloat, d_resultsDouble, n, numberOfSamples, a, b);
        
        CUDA_CHECK(cudaDeviceSynchronize());
        gettimeofday(&gpuEnd, NULL);
        
        // Copy results back to host
        CUDA_CHECK(cudaMemcpy(resultsFloatGpu.data(), d_resultsFloat, 
                            totalResults * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(resultsDoubleGpu.data(), d_resultsDouble, 
                            totalResults * sizeof(double), cudaMemcpyDeviceToHost));
        
        timeGpu = ((gpuEnd.tv_sec + gpuEnd.tv_usec * 0.000001) - 
                         (gpuStart.tv_sec + gpuStart.tv_usec * 0.000001));
        
        if (timing) {
            printf("GPU computation took: %f seconds\n", timeGpu);
        }
        
        if (verbose) {
            outputResultsGpu(resultsFloatGpu, resultsDoubleGpu, n, numberOfSamples, a, b);
        }
        
        // Clean up device memory
        CUDA_CHECK(cudaFree(d_resultsFloat));
        CUDA_CHECK(cudaFree(d_resultsDouble));
    }
    
    // Output timing results
    if (timing) {
        if (cpu) {
            printf("CPU computation took: %f seconds\n", timeTotalCpu);
        }
        if (cpu && gpu) {
            printf("Speedup: %.2fx\n", timeTotalCpu / timeGpu);
        }
    }

    // Output verbose results
    if (verbose) {
        if (cpu) {
            outputResultsCpu(resultsFloatCpu, resultsDoubleCpu);
        }
    }
    
    return 0;
}

void outputResultsGpu(const std::vector<float>& resultsFloatGpu, 
                     const std::vector<double>& resultsDoubleGpu,
                     unsigned int n, unsigned int numberOfSamples, double a, double b) {
    double division = (b - a) / ((double)numberOfSamples);
    
    for (unsigned int order = 1; order <= n; order++) {
        for (unsigned int sample = 1; sample <= numberOfSamples; sample++) {
            double x = a + sample * division;
            int idx = (order - 1) * numberOfSamples + (sample - 1);
            
            std::cout << "GPU==> exponentialIntegralDouble (" << order << "," << x 
                     << ")=" << resultsDoubleGpu[idx] << " ,";
            std::cout << "exponentialIntegralFloat  (" << order << "," << x 
                     << ")=" << resultsFloatGpu[idx] << endl;
        }
    }
}

int parseArguments(int argc, char *argv[]) {
    int c;

    while ((c = getopt(argc, argv, "cghn:m:a:b:tvi:")) != -1) {
        switch(c) {
            case 'c':
                cpu = false; break;
            case 'g':
                gpu = false; break;
            case 'h':
                printUsage(); exit(0); break;
            case 'i':
                maxIterations = atoi(optarg); break;
            case 'n':
                n = atoi(optarg); break;
            case 'm':
                numberOfSamples = atoi(optarg); break;
            case 'a':
                a = atof(optarg); break;
            case 'b':
                b = atof(optarg); break;
            case 't':
                timing = true; break;
            case 'v':
                verbose = true; break;
            default:
                fprintf(stderr, "Invalid option given\n");
                printUsage();
                return -1;
        }
    }
    return 0;
}

void printUsage() {
    printf("CUDA exponentialIntegral program\n");
    printf("Converted from Jose Mauricio Refojo's CPU version\n");
    printf("This program will calculate exponential integrals using GPU acceleration\n");
    printf("usage:\n");
    printf("exponentialIntegral.out [options]\n");
    printf("      -a   value   : interval start value (default: 0.0)\n");
    printf("      -b   value   : interval end value (default: 10.0)\n");
    printf("      -c           : skip CPU test\n");
    printf("      -g           : skip GPU test\n");
    printf("      -h           : show this usage\n");
    printf("      -i   size    : number of iterations (default: 2000000000)\n");
    printf("      -n   size    : maximum order of exponential integrals (default: 10)\n");
    printf("      -m   size    : number of samples in interval (default: 10)\n");
    printf("      -t           : show timing information\n");
    printf("      -v           : verbose mode\n");
}