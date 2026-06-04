<h1> Optimized SGEMM in CUDA </h1>
From Scratch

## Benchmarking - Multiplying Two 4096 x 4096 Matrices Together
### Relative to cuBLAS
Vectorized ~ 90.94%
2D Blocktiling ~ 79.27%  \
1D Blocktiling ~ 39.37%  \
Tiling ~ 12.66% \
Coalescing ~ 9.46% \
Naive ~ 1.29%

