#pragma once

#include <cstdint>
#include <cuda_runtime.h>

template <const uint32_t BLOCKSIZE>
__global__ void global_mem_coalesce_gemm(
    const float* A,
    const float* B,
    float* C,
    int M,
    int N,
    int K,
    float alpha,
    float beta)
{
    // x = column, y = row
    const int cCol = blockIdx.y * BLOCKSIZE + threadIdx.x;
    const int cRow = blockIdx.x * BLOCKSIZE + threadIdx.y;

    if (cRow < M && cCol < N) {
        float tmp = 0.0f;

        for (int i = 0; i < K; ++i) {
            tmp += A[cRow * K + i] * B[i * N + cCol];
        }

        C[cRow * N + cCol] =
            alpha * tmp + beta * C[cRow * N + cCol];
    }
}