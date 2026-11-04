#pragma once

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

#include "utils.cuh"

struct CudaEventPair {
    cudaEvent_t start{};
    cudaEvent_t stop{};

    CudaEventPair() {
        CUDA_CHECK(cudaEventCreate(&start));
        try {
            CUDA_CHECK(cudaEventCreate(&stop));
        } catch (...) {
            cudaEventDestroy(start);
            throw;
        }
    }

    ~CudaEventPair() {
        if (start) cudaEventDestroy(start);
        if (stop) cudaEventDestroy(stop);
    }

    CudaEventPair(const CudaEventPair&) = delete;
    CudaEventPair& operator=(const CudaEventPair&) = delete;
};

template <typename LaunchFunction>
float benchmark_kernel(
    LaunchFunction&& launch,
    int warmup_iterations,
    int timed_iterations)
{
    for (int i = 0; i < warmup_iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CudaEventPair events;

    CUDA_CHECK(cudaEventRecord(events.start));
    for (int i = 0; i < timed_iterations; ++i) {
        launch();
    }
    CUDA_CHECK(cudaEventRecord(events.stop));
    CUDA_CHECK(cudaEventSynchronize(events.stop));

    float elapsed_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed_ms, events.start, events.stop));

    return elapsed_ms / static_cast<float>(timed_iterations);
}
