#include <iostream>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "../../kernels/blocktiled-2d.cuh" 

using namespace std;

#define cudaCheck(err) { \
    if (err != cudaSuccess) { \
        cerr << "CUDA error: " << cudaGetErrorString(err) << " at line " << __LINE__ << std::endl; \
        exit(1); \
    } \
}

void matrix_init(float *a, float *b, float *c, float *c_cublas, int K, int M, int N) {
    for (int i = 0; i < M * K; i++) {
        a[i] = (float)(rand() % 100) / 10.0f;
    }
    
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)(rand() % 100) / 10.0f;
    }

    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }

    for (int i = 0; i < M * N; i++) {
        c_cublas[i] = 0.0f;
    }

}

void benchmark_kernel(int SIZE) {
int M = SIZE, N = SIZE, K = SIZE;
    float alpha = 1.0f, beta = 0.0f;

    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);

    // allocate host memory
    float *h_a = (float*)malloc(bytes_A);
    float *h_b = (float*)malloc(bytes_B);
    float *h_c = (float*)malloc(bytes_C);
    float *h_c_cublas = (float*)malloc(bytes_C);

    // initialize matrices
    matrix_init(h_a, h_b, h_c, h_c_cublas, K, M, N);

    // allocate device memory
    float *d_a, *d_b, *d_c, *d_c_cublas;
    cudaCheck(cudaMalloc(&d_a, bytes_A));
    cudaCheck(cudaMalloc(&d_b, bytes_B));
    cudaCheck(cudaMalloc(&d_c, bytes_C));
    cudaCheck(cudaMalloc(&d_c_cublas, bytes_C));

    // transfer from host to device
    cudaCheck(cudaMemcpy(d_a, h_a, bytes_A, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_b, h_b, bytes_B, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_c, h_c, bytes_C, cudaMemcpyHostToDevice));
    cudaCheck(cudaMemcpy(d_c_cublas, h_c_cublas, bytes_C, cudaMemcpyHostToDevice));

    // set up timers
    int repeat = 100;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // cublas setup
    cublasHandle_t handle;
    cublasCreate(&handle);

    // test custom kernel
    dim3 gridDim((N + 127) / 128, (M + 127) / 128); 
    dim3 blockDim(256);

    // warmup
    for (int i = 0; i < 10; i++) {
        sgemm_2d_bt<<<gridDim, blockDim>>>(d_a, d_b, d_c, K, M, N, alpha, beta);
    }

    cudaDeviceSynchronize();

    // benchmark
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        sgemm_2d_bt<<<gridDim, blockDim>>>(d_a, d_b, d_c, K, M, N, alpha, beta);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_custom = 0;
    cudaEventElapsedTime(&ms_custom, start, stop);
    double tflops_custom = ((2.0 * M * N * K) / (ms_custom / repeat / 1000.0)) / 1e12;


    // warmup
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c_cublas, N);
    }

    cudaDeviceSynchronize();

    // benchmark
    cudaEventRecord(start);
    for (int i = 0; i < repeat; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha, d_b, N, d_a, K, &beta, d_c_cublas, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms_cublas = 0;
    cudaEventElapsedTime(&ms_cublas, start, stop);
    double tflops_cublas = ((2.0 * M * N * K) / (ms_cublas / repeat / 1000.0)) / 1e12;

    cout << "Size: " << SIZE << "x" << SIZE << std::endl;
    cout << "Custom: " << tflops_custom << " TFLOPS (" << (ms_custom/repeat) << " ms)" << endl;
    cout << "cuBLAS:    " << tflops_cublas << " TFLOPS (" << (ms_cublas/repeat) << " ms)" << endl;
    cout << "Achieved " << (tflops_custom / tflops_cublas) * 100.0f << "% of cuBLAS performance.\n" << endl;

    cublasDestroy(handle);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c); cudaFree(d_c_cublas);
    free(h_a); free(h_b); free(h_c); free(h_c_cublas);
    cudaEventDestroy(start); cudaEventDestroy(stop);
}

int main() {
    cout << "Starting Benchmarks..." << endl;
    for (int size = 256; size <= 4096; size *= 2) {
        benchmark_kernel(size);
    }
    return 0;
}