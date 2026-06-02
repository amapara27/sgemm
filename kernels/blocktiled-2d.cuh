#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))
#define BLOCK_SIZE 32

__global__ void sgemm_2d_bt(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    // block tile dims - increase to 128 to improve occupancy: 64 - 256 threads
    const int bm = 128;
    const int bn = 128;
    const int bk = 8; // chunks are 8 elements wide for a, 8 elements long for b

    // each thread calculates tm * tn elements
    const int tm = 8;
    const int tn = 8;

    // c matrix locations
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // total results per blocktile and threads per block
    int totalPerBT = bm * bn; // each block tile is bm x bn
    int threadsPerBT = totalPerBT / (tm * tn); // total / amt per thread

    // ensure dimensions align: total results per blocktile / results per thread = threads per blocktile
    assert(threadsPerBT == blockDim.x); 

    // anchor our threads - find the 8x8 batch they calculate
    // bn / tn are number of threads that span a column
    int tCol = threadIdx.x % (bn / tn); // cols: 0 - 15
    int tRow = threadIdx.x / (bn / tn); // rows 0 - 15

    // allocate shared memory for blocktiles
    __shared__ float as[bm * bk];
    __shared__ float bs[bk * bn];

    // warp level coalescing - indices loaded into smem
    int iColA = threadIdx.x % bk; // 0 - 7
    int iRowA = threadIdx.x / bk; // 0 - 31
    // number of rows of a that are block loads in - block jumps this value
    int strideA = threadsPerBT / bk;


    int iColB = threadIdx.x % bn; // 0 - 127
    int iRowB = threadIdx.x / bn; // 0 - 1
    // number of cols of b that block loads in - block jumps this value
    int strideB = threadsPerBT / bn;

    // matrix element pos
    int aRow = cRow * bm + iRowA; // block row * block dim + inner row
    int bCol = cCol * bn + iColB; // block col * block dim + inner col

    // allocate array for thread results (multiple entries of size tm * tn)
    float res[tm * tn] = {0.0f};

    // number of iterations per thread
    int num_bt = (K + bk - 1) / bk;

    // register caches for a and b
    float regM[tm] = {0.0f};
    float regN[tn] = {0.0f};

    // iterate over block tiles
    for (int i = 0; i < num_bt; i++) {
        // populate SMEM caches for a
        for (int l = 0; l < bm; l += strideA) {
            as[(iRowA + l) * bk + iColA] = a[(aRow + l) * K + i * bk + iColA]; // slide to the right for matrix a
        }
        
        // populate SMEM caches for b
        for (int l = 0; l < bk; l += strideB) {
            bs[(iRowB + l) * bn + iColB] = b[(i * bk + l) * N + iRowB * N + bCol]; // slide down for matrix b
        }

        __syncthreads();

        // calculate per-thread result
        for (int j = 0; j < bk; j++) {
            // block into registers
            for (int m = 0; m < tm; m++) {
                regM[m] = as[(tRow * tm + m) * bk + j];
            }

            for (int m = 0; m < tm; m++) {
                regN[m] = bs[(j * bn) + tCol * tn + m];
            }

            // iterate over a elements tm times (tm elements per thread)
            for (int resIdxM = 0; resIdxM < tm; resIdxM++) {
                for (int resIdxN = 0; resIdxN < tn; resIdxN++) {
                    res[resIdxM * tn + resIdxN] += regM[resIdxM] * regN[resIdxN];
                }
            }
        }
        __syncthreads();
    }
    
    // write
    for (int resIdxM = 0; resIdxM < tm; resIdxM++) {
        for (int resIdxN = 0; resIdxN < tn; resIdxN++) {
            int cGlobalRow = cRow * bm + tRow * tm + resIdxM;
            int cGlobalCol = cCol * bn + tCol * tn + resIdxN;

            c[cGlobalRow * N + cGlobalCol] = alpha * res[resIdxM * tn + resIdxN] + beta * c[cGlobalRow * N + cGlobalCol];
        }
    }
}