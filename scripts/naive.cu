#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

__global__ void sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // conditional prevents extra thread usage outside of matrix dimensions
    if (x < M && y < N) {
        float temp = 0.0;

        for (int i = 0; i < K; i++) {
            // row * width + column
            temp += a[x * K + i] * b[i * N + y];
        }

        // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
        c[x * N + y] = alpha * temp + beta * c[x * N + y];
    }
    
}

// matrix initialization
void matrix_init(float *a, float *b, float *c, int K, int M, int N) {
    for (int i = 0; i < M * K; i++) {
        a[i] = (float)(rand() % 100) / 10.0f;
    }
    
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)(rand() % 100) / 10.0f;
    }

    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }

}

int main() {
    // matrix dimensions
    int K = 4096;
    int M = 4096;
    int N = 4096;

    // sgemm scaling values
    float beta = 0.9f;
    float alpha = 1.0f;

    // matrix sizes
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);

    // host memory allocation
    float *h_a, *h_b, *h_c;

    h_a = (float*)malloc(a_bytes);
    h_b = (float*)malloc(b_bytes);
    h_c = (float*)malloc(c_bytes);

    // matrix initialization
    matrix_init(h_a, h_b, h_c, K, M, N);

    // device memory allocation (GPU)
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    // copy data to GPU
    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice);

    // grid and block dimensions
    dim3 gridDim(CEIL_DIV(M, 32), CEIL_DIV(N, 32), 1);
    dim3 blockDim(32, 32, 1);
    
    // launch kernel
    sgemm<<<gridDim, blockDim>>>(d_a, d_b, d_c, K, M, N, alpha, beta);

    cudaDeviceSynchronize();

    // fetch results
    cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}