#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define SHMEM_SIZE 16 * 16 * 4
#define TILE_SIZE 32

__global__ void tiled_sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    __shared__ float A[SHMEM_SIZE];
    __shared__ float B[SHMEM_SIZE];
    
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // calculate thread pos
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;

    // intermediate sum for element
    float temp = 0.0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < num_tiles; i++) {
        // row invariant: row * K indexes row, i * tile_size indexes the subset of columns, + tx indexes the exact column
        A[(ty * TILE_SIZE) + tx] = a[row * K + (i * TILE_SIZE) + tx];

        // column invariant: i * tile_size * K indexes subset of rows, ty * n indexes the exact row, col indexes our global col
        B[(ty * TILE_SIZE) + tx] = b[(i * TILE_SIZE * N + ty * N) + col];

        // ensures every single thread within block has loaded data before moving on
        __syncthreads();

        // iterates through every element within tile and calculates temp val
        for (int j = 0; j < TILE_SIZE; j++) {
            temp += A[(ty * TILE_SIZE) + j] * B[(j * TILE_SIZE) + tx];
        }

        // ensures every single thread within block has computed vals before moving on
        __syncthreads();
    }

    // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
    c[row * N + col] = alpha * temp + beta * c[row * N + col];
}