<h1> Optimized SGEMM in CUDA </h1>
From Scratch

## Benchmarking - Multiplying Two 4096 x 4096 Matrices Together
### Relative to cuBLAS
2D Blocktiling (Vectorized) ~ 90.94% \
Warptiling (Vectorized) ~ 87.68% (most likely an occupancy issue, working on an autotuner to fix) \
2D Blocktiling ~ 79.27%  \
1D Blocktiling ~ 39.37%  \
Tiling ~ 12.66% \
Coalescing ~ 9.46% \
Naive ~ 1.29%

