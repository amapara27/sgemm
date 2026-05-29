#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define SMEM_SIZE 16 * 16 * 4
#define TILE_SIZE 32

__global__ void tiled_sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    __shared__ float as[SMEM_SIZE];
    __shared__ float bs[SMEM_SIZE];
    
    // thread pos
    int tCol = threadIdx.x;
    int tRow = threadIdx.y;

    // block pos
    int bCol = blockIdx.x;
    int bRow = blockIdx.y;

    // calculate matrix pos
    int row = bRow * TILE_SIZE + tRow;
    int col = bCol * TILE_SIZE + tCol;

    // intermediate sum for element
    float temp = 0.0;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;

    for (int i = 0; i < num_tiles; i++) {
        // row invariant: row * K indexes row, i * tile_size indexes the subset of columns, + tx indexes the exact column
        as[(tRow * TILE_SIZE) + tCol] = a[row * K + (i * TILE_SIZE) + tCol];

        // column invariant: i * tile_size * K indexes subset of rows, ty * n indexes the exact row, col indexes our global col
        bs[(tRow * TILE_SIZE) + tCol] = b[(i * TILE_SIZE * N + tRow * N) + col];

        // ensures every single thread within block has loaded data before moving on
        __syncthreads();

        // iterates through every element within tile and calculates temp val
        for (int j = 0; j < TILE_SIZE; j++) {
            temp += as[(tRow * TILE_SIZE) + j] * bs[(j * TILE_SIZE) + tCol];
        }

        // ensures every single thread within block has computed vals before moving on
        __syncthreads();
    }
    // sgemm formula C = α * (A @ B) + β * C - accumulates change with weights, used for gradient descent
    c[row * N + col] = alpha * temp + beta * c[row * N + col];
}