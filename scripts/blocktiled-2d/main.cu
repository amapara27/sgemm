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

// matrix initialization
void matrix_init(float *a, float *b, float *c, int K, int M, int N) {
    for (int i = 0; i < M * K; i++) {
        a[i] = (float)(rand() % 100) / 10.0f;
    }
    
    for (int i = 0; i < K * N; i++) {
        b[i] = (float)(rand() % 100) / 10.0f;
    }

    for (int i = 0; i < M * N; i++) {
        c[i] = 0.0f;
    }

}

// verifies the sgemm is correct - only use on smaller dimension matrices
void verify_result(const float *a, const float *b, const float *c_res, int K, int M, int N, float alpha, float beta) {
    std::cout << "Running CPU Verification... " << std::endl;
    
    bool passed = true;
    float t = 1e-2;

    // triple nested loop for CPU
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float temp = 0.0f;
            
            // slider
            for (int k = 0; k < K; k++) {
                temp += a[i * K + k] * b[k * N + j];
            }

            // apply sgemm
            float c_check = alpha * temp + beta * 0.0f; 

            // calculate error
            float e = std::abs(c_check - c_res[i * N + j]);
            
            // checks if error is higher than set tolerance
            if (e > t) {
                std::cout << "MISMATCH at [" << i << "][" << j << "]: " 
                          << "CPU = " << c_check << " | GPU = " << c_res[i * N + j] 
                          << " | Error = " << e << std::endl;
                passed = false;
                break;
            }
        }
        if (!passed) break; 
    }

    if (passed) {
        std::cout << "SUCCESS: GPU results perfectly match CPU results!" << std::endl;
    }
}

int main() {
    // matrix dimensions
    int K = 4096;
    int M = 4096;
    int N = 4096;

    // sgemm scaling values
    float beta = 0.9f;
    float alpha = 1.0f;

    // matrix sizes
    size_t a_bytes = M * K * sizeof(float);
    size_t b_bytes = K * N * sizeof(float);
    size_t c_bytes = M * N * sizeof(float);

    // host memory allocation
    float *h_a, *h_b, *h_c;

    h_a = (float*)malloc(a_bytes);
    h_b = (float*)malloc(b_bytes);
    h_c = (float*)malloc(c_bytes);

    // matrix initialization
    matrix_init(h_a, h_b, h_c, K, M, N);

    // device memory allocation (GPU)
    float *d_a, *d_b, *d_c;

    cudaMalloc(&d_a, a_bytes);
    cudaMalloc(&d_b, b_bytes);
    cudaMalloc(&d_c, c_bytes);

    // copy data to GPU
    cudaMemcpy(d_a, h_a, a_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, b_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_c, h_c, c_bytes, cudaMemcpyHostToDevice);

    // timer
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // grid and block dimensions
    dim3 gridDim(CEIL_DIV(N, 128), CEIL_DIV(M, 128), 1);
    dim3 blockDim(256);
    
    // launch kernel
    cudaEventRecord(start);
    sgemm_2d_bt<<<gridDim, blockDim>>>(d_a, d_b, d_c, K, M, N, alpha, beta);
    cudaEventRecord(stop);

    cudaDeviceSynchronize();

    // fetch results
    cudaMemcpy(h_c, d_c, c_bytes, cudaMemcpyDeviceToHost);

    // display metrics
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);

    // std::cout << "Computed value at C[2048][2048]: " << h_c[2048 * 4096 + 2048] << std::endl;
    // std::cout << "Kernel Execution Time: " << ms << " ms" << std::endl;

    // verify result with smaller matrices
    verify_result(h_a, h_b, h_c, K, M, N, alpha, beta);

    // Clean up timer memory
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // free memory
    free(h_a);
    free(h_b);
    free(h_c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    
    return 0;
}