#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define SHMEM_SIZE 16 * 16 * 4

__global__ void tiled_sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta, int tile_size) {
    __shared__ float A[SHMEM_SIZE];
    __shared__ float B[SHMEM_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // calculate thread pos
    int row = by * tile_size + ty;
    int col = bx * tile_size + tx;

    // intermediate sum for element
    float temp = 0.0;

    int num_tiles = (K + tile_size - 1) / tile_size;

    for (int i = 0; i < num_tiles; i++) {
        // row invariant: row * K indexes row, i * tile_size indexes the subset of columns, + tx indexes the exact column
        A[(ty * tile_size) + tx] = a[row * K + (i * tile_size) + tx];

        // column invariant: i * tile_size * K indexes subset of rows, ty * n indexes the exact row, col indexes our global col
        B[(ty * tile_size) + tx] = b[(i * tile_size * N + ty * N) + col];

        // ensures every single thread within block has loaded data before moving on
        __syncthreads();

        // iterates through every element within tile and calculates temp val
        for (int j = 0; j < tile_size; j++) {
            temp += A[(ty * tile_size) + j] * B[(j * tile_size) + tx];
        }

        // ensures every single thread within block has computed vals before moving on
        __syncthreads();
    }

    // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
    c[row * N + col] = alpha * temp + beta * c[row * N + col];
}