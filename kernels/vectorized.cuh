#include <cassert>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#define CEIL_DIV(A, B) (((A) + (B) - 1) / (B))

__global__ void sgemm_vectorized(float *a, float *b, float *c, int K, int M, int N, float alpha, float beta) {
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

    // warp level coalescing - indices loaded into smem, 4 elements per thread loads in all the elements in one pass
    int iColA = threadIdx.x % (bk / 4); // 0 - 1
    int iRowA = threadIdx.x / (bk / 4); // 0 - 127
    // number of rows of a that are block loads in - block jumps this value
    int strideA = threadsPerBT / (bk / 4); // 128 row jumps - 1 pass


    int iColB = threadIdx.x % (bn / 4); // 0 - 31
    int iRowB = threadIdx.x / (bn / 4); // 0 - 7
    // number of cols of b that block loads in - block jumps this value
    int strideB = threadsPerBT / (bn / 4); // 8 row jumps - 1 pass

    // matrix element pos
    int aRow = cRow * bm + iRowA; // block row * block dim + inner row
    int bCol = cCol * bn + (iColB * 4); // different because of vectorization (horizontal width is less than previous kernel)

    // allocate array for thread results (multiple entries of size tm * tn)
    float res[tm * tn] = {0.0f};

    // number of iterations per thread: bk wide for matrix a, bk long for matrix b
    int num_bt = (K + bk - 1) / bk;

    // register caches for a and b
    float regM[tm] = {0.0f};
    float regN[tn] = {0.0f};

    // iterate over block tiles
    for (int bIdx = 0; bIdx < num_bt; bIdx++) {
        // cast a to vector, transpose, and load into smem
        float4 tmp = reinterpret_cast<float4 *>(&a[aRow * K + bIdx * bk + iColA * 4])[0];
        // transpose. * 4 steps down row by row, + iRowA gets to right col (old row)
        as[(iColA * 4 + 0) * bm + iRowA] = tmp.x;
        as[(iColA * 4 + 1) * bm + iRowA] = tmp.y;
        as[(iColA * 4 + 2) * bm + iRowA] = tmp.z;
        as[(iColA * 4 + 3) * bm + iRowA] = tmp.w;
        
        // cast b to vector and load into smem
        reinterpret_cast<float4 *>(&bs[iRowB * bn + iColB * 4])[0] = reinterpret_cast<float4 *>(&b[bIdx * bk * N + iRowB * N + bCol])[0];

        __syncthreads();

        // calculate per-thread result
        for (int dotIdx = 0; dotIdx < bk; dotIdx++) {
            // block into registers
            for (int i = 0; i < tm; i++) {
                regM[i] = as[dotIdx * bm + tRow * tm + i]; // transposed a, so slide horizontally, vertically after every dotIdx
            }

            for (int i = 0; i < tn; i++) {
                regN[i] = bs[dotIdx * bn + tCol * tn + i]; // 8 cols, after every dotIdx slide a row
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
    
    // write - vectorize to increase write speed
    for (int resIdxM = 0; resIdxM < tm; resIdxM++) {
        for (int resIdxN = 0; resIdxN < tn; resIdxN += 4) {
            int cGlobalRow = cRow * bm + tRow * tm + resIdxM;
            int cGlobalCol = cCol * bn + tCol * tn + resIdxN;

            // grab existing elements from c
            float4 tmp = reinterpret_cast<float4*>(&c[cGlobalRow * N + cGlobalCol])[0];

            // update elements
            tmp.x = alpha * res[resIdxM * tn + resIdxN] +  beta * tmp.x;
            tmp.y = alpha * res[resIdxM * tn + resIdxN + 1] +  beta * tmp.y;
            tmp.z = alpha * res[resIdxM * tn + resIdxN + 2] +  beta * tmp.z;
            tmp.w = alpha * res[resIdxM * tn + resIdxN + 3] +  beta * tmp.w;

            // put elements back into c
            reinterpret_cast<float4*>(&c[cGlobalRow * N + cGlobalCol])[0] = tmp;
        }
    }
}