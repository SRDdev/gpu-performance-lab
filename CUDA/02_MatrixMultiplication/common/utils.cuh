#pragma once

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        const cudaError_t err__ = (call);                                       \
        if (err__ != cudaSuccess) {                                             \
            std::ostringstream oss__;                                           \
            oss__ << "CUDA error at " << __FILE__ << ":" << __LINE__            \
                  << " in " #call ": " << cudaGetErrorString(err__);             \
            throw std::runtime_error(oss__.str());                              \
        }                                                                       \
    } while (0)

struct GemmConfig {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    int threads_x = 16;
    int threads_y = 16;
    int warmup = 10;
    int iterations = 100;
    float alpha = 1.0f;
    float beta = 0.0f;
    unsigned int seed = 1234;
    std::string kernel = "naive";
};

inline size_t checked_num_elements(int rows, int cols, const char* name) {
    if (rows <= 0 || cols <= 0) {
        throw std::invalid_argument(std::string(name) + " dimensions must be positive");
    }
    const size_t r = static_cast<size_t>(rows);
    const size_t c = static_cast<size_t>(cols);
    if (r > std::numeric_limits<size_t>::max() / c) {
        throw std::overflow_error(std::string(name) + " element count overflow");
    }
    return r * c;
}

inline void fill_random(std::vector<float>& values, std::mt19937& rng) {
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (float& value : values) {
        value = dist(rng);
    }
}

inline void cpu_gemm_reference(
    const std::vector<float>& A,
    const std::vector<float>& B,
    const std::vector<float>& C_initial,
    std::vector<float>& C_reference,
    int M, int N, int K,
    float alpha, float beta)
{
    C_reference.resize(static_cast<size_t>(M) * N);

    for (int row = 0; row < M; ++row) {
        for (int col = 0; col < N; ++col) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[static_cast<size_t>(row) * K + k] *
                       B[static_cast<size_t>(k) * N + col];
            }

            const size_t idx = static_cast<size_t>(row) * N + col;
            C_reference[idx] = alpha * sum + beta * C_initial[idx];
        }
    }
}

struct CompareResult {
    double max_abs_error = 0.0;
    double max_rel_error = 0.0;
    size_t mismatches = 0;
};

inline CompareResult compare_vectors(
    const std::vector<float>& actual,
    const std::vector<float>& reference,
    double atol = 1e-4,
    double rtol = 1e-4)
{
    if (actual.size() != reference.size()) {
        throw std::invalid_argument("Comparison vectors have different sizes");
    }

    CompareResult result;

    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = actual[i];
        const double b = reference[i];
        const double abs_error = std::abs(a - b);
        const double rel_error = abs_error / std::max(std::abs(b), 1e-12);

        result.max_abs_error = std::max(result.max_abs_error, abs_error);
        result.max_rel_error = std::max(result.max_rel_error, rel_error);

        const double tolerance = atol + rtol * std::abs(b);
        if (!std::isfinite(a) || !std::isfinite(b) || abs_error > tolerance) {
            ++result.mismatches;
        }
    }

    return result;
}

inline void print_usage(const char* exe) {
    std::cout
        << "Usage: " << exe
        << " --M M --N N --K K [options]\n"
        << "Options:\n"
        << "  --kernel NAME       Kernel: all, naive, coalesced, shared_mem,\n"
        << "                      block_tile_1d, block_tile_2d, vectorized,\n"
        << "                      bank_conflicts, bank_extra_col, warp_tiling, ptx\n"
        << "  --threads-x X       Threads in block x dimension (default: 16)\n"
        << "  --threads-y Y       Threads in block y dimension (default: 16)\n"
        << "  --warmup W          Warmup launches (default: 10)\n"
        << "  --iters I           Timed launches (default: 100)\n"
        << "  --alpha A           GEMM alpha (default: 1)\n"
        << "  --beta B            GEMM beta (default: 0)\n"
        << "  --seed S            RNG seed (default: 1234)\n"
        << "  --help              Show this help\n";
}

inline int parse_int_arg(const char* text, const char* name) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (end == text || *end != '\0' ||
        value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::invalid_argument(std::string("Invalid integer for ") + name);
    }
    return static_cast<int>(value);
}

inline float parse_float_arg(const char* text, const char* name) {
    char* end = nullptr;
    const float value = std::strtof(text, &end);
    if (end == text || *end != '\0' || !std::isfinite(value)) {
        throw std::invalid_argument(std::string("Invalid float for ") + name);
    }
    return value;
}

inline GemmConfig parse_args(int argc, char** argv) {
    GemmConfig cfg;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help") {
            print_usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument("Missing value after " + arg);
        }

        const char* value = argv[++i];

        if (arg == "--M") cfg.M = parse_int_arg(value, "--M");
        else if (arg == "--N") cfg.N = parse_int_arg(value, "--N");
        else if (arg == "--K") cfg.K = parse_int_arg(value, "--K");
        else if (arg == "--threads-x") cfg.threads_x = parse_int_arg(value, "--threads-x");
        else if (arg == "--threads-y") cfg.threads_y = parse_int_arg(value, "--threads-y");
        else if (arg == "--warmup") cfg.warmup = parse_int_arg(value, "--warmup");
        else if (arg == "--iters") cfg.iterations = parse_int_arg(value, "--iters");
        else if (arg == "--alpha") cfg.alpha = parse_float_arg(value, "--alpha");
        else if (arg == "--beta") cfg.beta = parse_float_arg(value, "--beta");
        else if (arg == "--seed") {
            const int seed = parse_int_arg(value, "--seed");
            if (seed < 0) throw std::invalid_argument("--seed must be nonnegative");
            cfg.seed = static_cast<unsigned int>(seed);
        }
        else if (arg == "--kernel") cfg.kernel = value;
        else throw std::invalid_argument("Unknown argument: " + arg);
    }

    checked_num_elements(cfg.M, cfg.K, "A");
    checked_num_elements(cfg.K, cfg.N, "B");
    checked_num_elements(cfg.M, cfg.N, "C");

    if (cfg.threads_x <= 0 || cfg.threads_y <= 0) {
        throw std::invalid_argument("Thread dimensions must be positive");
    }
    if (static_cast<long long>(cfg.threads_x) * cfg.threads_y > 1024) {
        throw std::invalid_argument("threads-x * threads-y cannot exceed 1024");
    }
    if (cfg.warmup < 0) throw std::invalid_argument("--warmup must be nonnegative");
    if (cfg.iterations <= 0) throw std::invalid_argument("--iters must be positive");

    return cfg;
}
