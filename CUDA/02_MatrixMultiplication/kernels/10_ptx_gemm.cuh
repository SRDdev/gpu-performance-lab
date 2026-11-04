#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace warp_tiling_ptx
{
    constexpr int WARPSIZE = 32;

    // Explicit round-to-nearest fused multiply-add.
    __device__ __forceinline__
    float ptx_fma_rn(float a, float b, float acc)
    {
        asm volatile(
            "fma.rn.f32 %0, %1, %2, %0;"
            : "+f"(acc)
            : "f"(a), "f"(b)
        );
        return acc;
    }

    __device__ __forceinline__
    float4 load_float4(const float* ptr)
    {
        return *reinterpret_cast<const float4*>(ptr);
    }

    __device__ __forceinline__
    void store_float4(float* ptr, const float4& value)
    {
        *reinterpret_cast<float4*>(ptr) = value;
    }

    template <
        const int BM,
        const int BN,
        const int BK,
        const int WM,
        const int WN,
        const int WNITER,
        const int TM,
        const int TN,
        const int NUM_THREADS>
    __global__ void warp_tiling_ptx_gemm(
        const int M,
        const int N,
        const int K,
        const float alpha,
        const float* __restrict__ A,
        const float* __restrict__ B,
        const float beta,
        float* __restrict__ C)
    {
        constexpr int WMITER =
            (WM * WN) / (WARPSIZE * TM * TN * WNITER);

        static_assert(WMITER > 0, "PTX warp tile produces no M subtile");

        constexpr int WSUBM = WM / WMITER;
        constexpr int WSUBN = WN / WNITER;

        const int tid = threadIdx.x;
        const int warpId = tid / WARPSIZE;
        const int laneId = tid % WARPSIZE;

        const int warpM = warpId / (WN / WSUBN);
        const int warpN = warpId % (WN / WSUBN);

        const int threadM = laneId / (WSUBN / TN);
        const int threadN = laneId % (WSUBN / TN);

        const int blockRow = blockIdx.y * BM;
        const int blockCol = blockIdx.x * BN;

        __shared__ float As[BM * BK];
        __shared__ float Bs[BK * BN];

        float accum[WMITER * TM * WNITER * TN] = {0.0f};

        const int loadAElements = BM * BK;
        const int loadBElements = BK * BN;

        for (int kBase = 0; kBase < K; kBase += BK)
        {
            // Load A tile into shared memory.
            for (int idx = tid; idx < loadAElements; idx += NUM_THREADS)
            {
                const int row = idx / BK;
                const int col = idx % BK;

                const int globalRow = blockRow + row;
                const int globalCol = kBase + col;

                As[idx] =
                    (globalRow < M && globalCol < K)
                        ? A[globalRow * K + globalCol]
                        : 0.0f;
            }

            // Load B tile into shared memory.
            for (int idx = tid; idx < loadBElements; idx += NUM_THREADS)
            {
                const int row = idx / BN;
                const int col = idx % BN;

                const int globalRow = kBase + row;
                const int globalCol = blockCol + col;

                Bs[idx] =
                    (globalRow < K && globalCol < N)
                        ? B[globalRow * N + globalCol]
                        : 0.0f;
            }

            __syncthreads();

            // Each thread computes its TM x TN output tile.
            for (int k = 0; k < BK; ++k)
            {
                float aRegs[WMITER * TM];
                float bRegs[WNITER * TN];

                #pragma unroll
                for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
                {
                    for (int i = 0; i < TM; ++i)
                    {
                        const int row =
                            warpM * WM +
                            wSubRow * WSUBM +
                            threadM * TM + i;

                        aRegs[wSubRow * TM + i] =
                            As[row * BK + k];
                    }
                }

                #pragma unroll
                for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
                {
                    for (int j = 0; j < TN; ++j)
                    {
                        const int col =
                            warpN * WN +
                            wSubCol * WSUBN +
                            threadN * TN + j;

                        bRegs[wSubCol * TN + j] =
                            Bs[k * BN + col];
                    }
                }

                #pragma unroll
                for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
                {
                    for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
                    {
                        for (int i = 0; i < TM; ++i)
                        {
                            for (int j = 0; j < TN; ++j)
                            {
                                const int idx =
                                    (wSubRow * TM + i) * (WNITER * TN) +
                                    wSubCol * TN + j;

                                accum[idx] = ptx_fma_rn(
                                    aRegs[wSubRow * TM + i],
                                    bRegs[wSubCol * TN + j],
                                    accum[idx]);
                            }
                        }
                    }
                }
            }

            __syncthreads();
        }

        // Store output tile.
        #pragma unroll
        for (int wSubRow = 0; wSubRow < WMITER; ++wSubRow)
        {
            #pragma unroll
            for (int wSubCol = 0; wSubCol < WNITER; ++wSubCol)
            {
                for (int i = 0; i < TM; ++i)
                {
                    const int row =
                        blockRow +
                        warpM * WM +
                        wSubRow * WSUBM +
                        threadM * TM + i;

                    #pragma unroll
                    for (int j = 0; j < TN; ++j)
                    {
                        const int col =
                            blockCol +
                            warpN * WN +
                            wSubCol * WSUBN +
                            threadN * TN + j;

                        if (row < M && col < N)
                        {
                            const int idx =
                                (wSubRow * TM + i) * (WNITER * TN) +
                                wSubCol * TN + j;
                            const float oldC = C[row * N + col];

                            C[row * N + col] =
                                alpha * accum[idx] + beta * oldC;
                        }
                    }
                }
            }
        }
    }
}
