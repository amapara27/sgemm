#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

__global__ void sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    int rows = blockIdx.y * blockDim.y + threadIdx.y;
    int cols = blockIdx.x * blockDim.x + threadIdx.x;

    // conditional prevents extra thread usage outside of matrix dimensions
    if (rows < M && cols < N) {
        float temp = 0.0;

        for (int i = 0; i < K; i++) {
            // row * width + column
            temp += a[rows * K + i] * b[i * N + cols];
        }

        // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
        c[rows * N + cols] = alpha * temp + beta * c[rows * N + cols];
    }
    
}