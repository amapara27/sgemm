#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

__global__ void sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // conditional prevents extra thread usage outside of matrix dimensions
    if (row < M && col < N) {
        float temp = 0.0;

        for (int i = 0; i < K; i++) {
            // row * width + column
            temp += a[row * K + i] * b[i * N + col];
        }

        // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
        c[row * N + col] = alpha * temp + beta * c[row * N + col];
    }
}