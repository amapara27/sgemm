#include <iostream>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cmath>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define TILE_SIZE 32

__global__ void sgemm(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    // shared tiles
    __shared__ float as[TILE_SIZE * TILE_SIZE];
    __shared__ float bs[TILE_SIZE * TILE_SIZE];

    // thread pos
    int tCol = threadIdx.x;
    int tRow = threadIdx.y;

    // block pos
    int bCol = blockIdx.x;
    int bRow = blockIdx.y;

    // advance pointers to starting positions
    // target row * amt of columns
    a += bRow * TILE_SIZE * K;
    // target column: always starts at row 0
    b += bCol * TILE_SIZE;
    // row, col
    c += bRow * TILE_SIZE * N + bCol * TILE_SIZE;

    int num_tiles = (K + TILE_SIZE - 1) / TILE_SIZE;
    float temp = 0.0;

    for (int i = 0; i < num_tiles; i++) {
        // load values into shared memory
        as[tRow * TILE_SIZE + tCol] = a[tRow * K + tCol];
        bs[tRow * TILE_SIZE + tCol] = b[tRow * N + tCol];

        __syncthreads();

        for (int j = 0; j < TILE_SIZE; j++) {
            temp += as[tRow * TILE_SIZE + j] * bs[j * TILE_SIZE + tCol];
        }

        __syncthreads();

        // shift pointers
        a += TILE_SIZE;
        b += TILE_SIZE * N;
    }
    // write
    c[tRow * N + tCol] = alpha * temp + beta * c[tRow * N + tCol];
}