#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

__global__ void sgemm_2d_bt(const float *a, const float *b, float *c, int K, int M, int N, float alpha, float beta) {
    // block tile dims - increase to 128 to improve occupancy: 64 - 256 threads
    const int bm = 128;
    const int bn = 128;
    const int bk = 8; // 128 x 8 tile for a, 8 x 128 tile for b

    // each thread calculates tm * tn elements
    const int tm = 8;
    const int tn = 8;

    // locations for the 128 x 128 tile the block is calculating
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // total results per blocktile and threads per block
    int totalPerBT = bm * bn; // each block tile is bm x bn: 128 x 128 
    int threadsPerBT = totalPerBT / (tm * tn); // total / amt per thread: 128 x 128 / 64 = 256 threads per block

    // ensure dimensions align: total results per blocktile / results per thread = threads per blocktile
    assert(threadsPerBT == blockDim.x); 

    // anchor our threads - find the specific 8x8 batch they calculate within the block
    // bn / tn are number of threads that span a column
    int tCol = threadIdx.x % (bn / tn); // cols: 0 - 15 (16 cols x 8 elements per thread = 256 threads per block)
    int tRow = threadIdx.x / (bn / tn); // rows 0 - 15 (16 rows x 8 elements per thread = 256 threads per block)

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

    // number of iterations per thread: bk wide for matrix a, bk long for matrix b
    int num_bt = (K + bk - 1) / bk;

    // register caches for a and b
    float regM[tm] = {0.0f};
    float regN[tn] = {0.0f};

    // iterate over block tiles
    for (int bIdx = 0; bIdx < num_bt; bIdx++) {
        // populate SMEM caches for a
        for (int l = 0; l < bm; l += strideA) {
            as[(iRowA + l) * bk + iColA] = a[(aRow + l) * K + bIdx * bk + iColA]; // slide to the right for matrix a
        }
        
        // populate SMEM caches for b
        for (int l = 0; l < bk; l += strideB) {
            bs[(iRowB + l) * bn + iColB] = b[(bIdx * bk + l) * N + iRowB * N + bCol]; // slide down for matrix b
        }

        __syncthreads();

        // calculate per-thread result
        for (int dotIdx = 0; dotIdx < bk; dotIdx++) {
            // block into registers
            for (int i = 0; i < tm; i++) {
                regM[i] = as[(tRow * tm + i) * bk + dotIdx]; // 8 rows, after every dotIdx slide a col
            }

            for (int i = 0; i < tn; i++) {
                regN[i] = bs[(dotIdx * bn) + tCol * tn + i]; // 8 cols, after every dotIdx slide a row
            }

            // tm * tn elements
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