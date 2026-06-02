#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define BLOCK_SIZE 32

__global__ void sgemm_1d_bt(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    // block tile dims
    const int bk = 8; // step size along K
    const int bm = 64; // rows of C
    const int bn = 64; // cols of C
    const int tm = 8; // 8 elements per thread

    // c matrix locations
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // total results per blocktile and threads per block
    int totalPerBT = bm * bn; // each block tile is bm x bn
    int threadsPerBT = totalPerBT / tm; // total / amt per thread

    // ensure dimensions align: total results per blocktile / results per thread = threads per blocktile
    assert(threadsPerBT == blockDim.x); 

    // each warp calculates 32 * tm elements (32 threads per warp * 8 elements per thread):
    int tCol = threadIdx.x % bn; // cols: 0 - 63
    int tRow = threadIdx.x / bn; // rows 0 - 7

    // allocate shared memory for blocktiles
    __shared__ float as[bm * bk];
    __shared__ float bs[bk * bn];

    // warp level coalescing
    int iColA = threadIdx.x % bk; // 0 - 7
    int iRowA = threadIdx.x / bk; // 0 - 63
    int iColB = threadIdx.x % bn; // 0 - 63
    int iRowB = threadIdx.x / bn; // 0 - 7

    // matrix element pos
    int aRow = cRow * bm + iRowA; // block row * block dim + inner row
    int bCol = cCol * bn + iColB; // block col * block dim + inner col

    // allocate array for thread results (multiple entries)
    float res[tm] = {0.0f};

    // number of iterations per thread
    int num_bt = (K + bk - 1) / bk;

    for (int i = 0; i < num_bt; i++) {
        // populate SMEM caches
        as[iRowA * bk + iColA] = a[aRow * K + i * bk + iColA]; // slide to the right for matrix a
        bs[iRowB * bn + iColB] = b[i * bk * N + iRowB * N + bCol]; // slide down for matrix b

        __syncthreads();

        // calculate per-thread result
        for (int j = 0; j < bk; j++) {
            // load in b element to reuse for the dot product
            float tmp = bs[j * bn + tCol];

            // iterate over a elements tm times (tm elements per thread)
            for (int k = 0; k < tm; k++) {
                res[k] += as[(tRow * tm + k) * bk + j] * tmp;
            }
        }
        __syncthreads();
    }
    // write
    int resCol = cCol * bn + tCol; // block's col

    for (int r = 0; r < tm; r++) {
        int resRow = cRow * bm + (tRow * tm) + r; // block's absolute row
        
        c[resRow * N + resCol] = alpha * res[r] + beta * c[resRow * N + resCol];
    }
}