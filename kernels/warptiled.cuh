#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

const int WARPSIZE = 32;
const int NUM_THREADS = 128;

__global__ void sgemm_warptiled(float *a, float *b, float *c, int K, int M, int N, float alpha, float beta) {
    // block tile dims - increase to 128 to improve occupancy: 64 - 256 threads
    const int bn = 128;
    const int bm = 64; // reduced from 128 for better occupancy
    const int bk = 8; // 128 x 8 tile for a, 8 x 128 tile for b

    // warp tile dims - each warp assigned a 32 x 64 chunk
    const int wn = 64;
    const int wm = 32;
    const int wniter = 2; // iterates 2 rows at a time

    // thread tile dims - each thread calculates tm * tn elements
    const int tn = 4;
    const int tm = 4;

    // locations for the 128 x 64 tile the block is calculating
    int cRow = blockIdx.y;
    int cCol = blockIdx.x;

    // warp placement - each block is bn wide, each warp is wn threads wide
    int wIdx = threadIdx.x / WARPSIZE;
    int wCol = wIdx % (bn / wn); // block is bn wide, warp is wn wide
    int wRow = wIdx / (bn / wn); // block is bn wide, warp is wn wide

    // size of warp subtile, each subtile calculates a 16 x 32 chunk
    constexpr int wmiter = (wm * wn) / (WARPSIZE * tm * tn * wniter);
    constexpr int wsubm = wm / wmiter; // 32 / 2 = 16
    constexpr int wsubn = wn / wniter; // 64 / 2 = 32

    // thread pos within warp, warp placement - each warp subtile is wsubn wide, each thread is tn elements wide
    int tIdxW = threadIdx.x % 32;
    int tColW = tIdxW % (wsubn / tn); // cols: 0 - 7
    int tRowW = tIdxW / (wsubn / tn); // rows: 0 - 3

    // allocate shared memory for blocktiles
    __shared__ float as[bm * bk];
    __shared__ float bs[bk * bn];

    // warp level coalescing - indices loaded into smem, 4 elements per thread loads in all the elements in one pass
    int iColA = threadIdx.x % (bk / 4); // bk elements wide, each thread holds 4 elements, 2 cols
    int iRowA = threadIdx.x / (bk / 4); // bk elements wide, each thread holds 4 elements, 64 rows
    int strideA = NUM_THREADS * 4 / bk;  // 128 threads load 512, loads 8 x 64 in one pass

    int iColB = threadIdx.x % (bn / 4); // bn elements wide, each thread holds 4 elements, 32 cols
    int iRowB = threadIdx.x / (bn / 4); // bn elements wide, each thread holds 4 elements, 4 rows
    int strideB = NUM_THREADS / (bn / 4); // 128 threads load 512 elements, loads 4 x 128 in one pass, needs two passes

    // matrix element pos
    int aRow = cRow * bm + iRowA; // block row * block dim + inner row
    int bCol = cCol * bn + (iColB * 4); // different because of vectorization (horizontal width is less than previous kernel)

    // allocate array for thread results
    float res[wmiter * tm * wniter * tn] = {0.0f};

    // number of iterations per thread: bk wide for matrix a, bk long for matrix b
    int num_bt = (K + bk - 1) / bk;

    // register caches for a and b
    float regM[tm * wmiter] = {0.0f};
    float regN[tn * wniter] = {0.0f};

    // iterate over block tiles
    for (int bIdx = 0; bIdx < num_bt; ++bIdx) {
        // load in a - iterate over rows by strides
        for (int offset = 0; offset < bm; offset += strideA) { 
        // cast a to vector, transpose, and load into smem
            float4 tmp = reinterpret_cast<float4 *>(&a[(aRow + offset) * K + bIdx * bk + iColA * 4])[0];
            // transpose and store, * 4 steps down row by row, + iRowA gets to right col (old row)
            as[(iColA * 4 + 0) * bm + offset + iRowA] = tmp.x;
            as[(iColA * 4 + 1) * bm + offset + iRowA] = tmp.y;
            as[(iColA * 4 + 2) * bm + offset + iRowA] = tmp.z;
            as[(iColA * 4 + 3) * bm + offset + iRowA] = tmp.w;
        }
        
        // cast b to vector and load into smem
        for (int offset = 0; offset < bk; offset += strideB) {
            reinterpret_cast<float4 *>(&bs[(iRowB + offset) * bn + iColB * 4])[0] = 
                reinterpret_cast<float4 *>(&b[bIdx * bk * N + (iRowB + offset) * N + bCol])[0];
        }

        __syncthreads();

        // load into registers - iterates 8 times (bk)
        for (int dotIdx = 0; dotIdx < bk; ++dotIdx) {
            for (int wsRowIdx = 0; wsRowIdx < wmiter; ++wsRowIdx) {
                for (int i = 0; i < tm; ++i) {
                    // as is transposed: col means row in original a matrix
                    // dotIdx gets our col within the 64 x 8 original matrix, wRow gets the 32 rows assigned to the warp, wsRowIdx gets the 16 rows at a time, tRowW gets the 4 rows assigned to this thread, +i gets exact row of original
                    regM[wsRowIdx * tm + i] = as[(dotIdx * bm) + (wRow * wm) + (wsRowIdx * wsubm) + (tRowW * tm) + i];
                }
            }

            for (int wsColIdx = 0; wsColIdx < wniter; ++wsColIdx) {
                for (int i = 0; i < tn; ++i) {
                    // dotIdx gets our row within the 8 x 128  matrix, wCol gets the 64 cols assigned to the warp, wsColIdx gets the 32 cols at a time, tColW gets the 4 cols assigned to this thread, +i gets exact col
                    regN[wsColIdx * tn + i] = bs[(dotIdx * bn) + (wCol * wn) + (wsColIdx * wsubn) + (tColW * tn) + i];
                }
            }
        
            // warptile matmul
            for (int wsRowIdx = 0; wsRowIdx < wmiter; ++wsRowIdx) {
                for (int wsColIdx = 0; wsColIdx < wniter; ++wsColIdx) {
                    // per thread results
                    for (int resIdxM = 0; resIdxM < tm; ++resIdxM) {
                        for (int resIdxN = 0; resIdxN < tn; ++resIdxN) {
                            // row * width + col
                            int idx = (wsRowIdx * tm + resIdxM) * (wniter * tn) + (wsColIdx * tn + resIdxN);
                            res[idx] += regM[wsRowIdx * tm + resIdxM] * regN[wsColIdx * tn + resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();
    }
    
    // write - vectorize to increase write speed
    for (int wsRowIdx = 0; wsRowIdx < wmiter; ++wsRowIdx) {
        for (int wsColIdx = 0; wsColIdx < wniter; ++wsColIdx) {
            for (int resIdxM = 0; resIdxM < tm; ++resIdxM) {
                for (int resIdxN = 0; resIdxN < tn; resIdxN += 4) {
                    int gRow = (cRow * bm) + (wRow * wm) + (wsRowIdx * wsubm) + (tRowW * tm) + resIdxM;
                    int gCol = (cCol * bn) + (wCol * wn) + (wsColIdx * wsubn) + (tColW * tn) + resIdxN;

                    // finds specific 4 x 4 chunk - row ** width + col
                    int resIdx =  (wsRowIdx * tm + resIdxM) * (wniter * tn) + (wsColIdx * tn + resIdxN);

                    // row * width + col
                    int cIdx = gRow * N + gCol;

                    // vectorize
                    float4 tmp = reinterpret_cast<float4 *>(&c[cIdx])[0];

                    // gemm multiplication
                    tmp.x = alpha * res[resIdx] + beta * tmp.x;
                    tmp.y = alpha * res[resIdx + 1] + beta * tmp.y;
                    tmp.z = alpha * res[resIdx + 2] + beta * tmp.z;
                    tmp.w = alpha * res[resIdx + 3] + beta * tmp.w;

                    reinterpret_cast<float4 *>(&c[cIdx])[0] = tmp;
                }
            }
        }
    }
}