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