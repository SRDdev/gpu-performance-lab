#pragma once

#include <cstdio>
#include <cstdlib>
#include <cublas_v2.h>
#include <cuda_runtime.h>

__global__ void naive_gemm(const float *A,
                         const float *B,
                         float *C,
                         int M, int N, int K, float alpha,
                         float beta){
    const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint32_t y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x < M && y < N) {   
        float tmp = 0.0f;
        for(int i=0; i<K; i++){
            tmp += A[x * K + i] * B[i * N + y];
        }
    C[x * N + y] = alpha * tmp + beta * C[x * N + y];
    }                     
}