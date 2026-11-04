// CUDA Vector Addition test harness with explicit input-shape arguments.
//
// Kernel expected in vector_add.cu:
// __global__ void vector_add(const float* A, const float* B, float* C, size_t N);
//
// Build:
//   nvcc -O3 -std=c++17 vector_add.cu vector_add_test.cu -o vector_add_test.exe
//
// Examples:
//   ./vector_add_test.exe --input0 1024 --input1 1024
//   ./vector_add_test.exe --input0 1000003 --input1 1000003 --threads 256
//   ./vector_add_test.exe --input0 1048576 --input1 1048576 --warmup 10 --iters 200
//
// Input tensors are 1D float32 vectors. Input lengths must match.

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>

__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    size_t N);

#define CUDA_CHECK(call)                                                     \
    do {                                                                     \
        cudaError_t err__ = (call);                                          \
        if (err__ != cudaSuccess) {                                          \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)         \
                      << " (" << static_cast<int>(err__) << ") at "          \
                      << __FILE__ << ":" << __LINE__ << '\n';                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

struct Options {
    size_t input0 = 0;
    size_t input1 = 0;
    int threads = 256;
    int warmup = 5;
    int iterations = 100;
};

static void usage(const char* exe) {
    std::cout
        << "Usage:\n"
        << "  " << exe
        << " --input0 N --input1 N [--threads T] [--warmup W] [--iters I]\n\n"
        << "Arguments:\n"
        << "  --input0 N   Length of input A (required)\n"
        << "  --input1 N   Length of input B (required)\n"
        << "  --threads T  Threads per block (default: 256)\n"
        << "  --warmup W   Warm-up kernel launches (default: 5)\n"
        << "  --iters I    Timed kernel launches (default: 100)\n\n"
        << "Example:\n"
        << "  " << exe << " --input0 1048576 --input1 1048576\n";
}

static size_t parse_size(const char* text, const char* arg_name) {
    try {
        const std::string s(text);
        size_t consumed = 0;
        const unsigned long long value = std::stoull(s, &consumed);
        if (consumed != s.size() || value == 0) {
            throw std::invalid_argument("invalid");
        }
        return static_cast<size_t>(value);
    } catch (...) {
        std::cerr << arg_name << " must be a positive integer; got: "
                  << text << '\n';
        std::exit(EXIT_FAILURE);
    }
}

static int parse_int(const char* text, const char* arg_name, int min_value) {
    try {
        const std::string s(text);
        size_t consumed = 0;
        const long value = std::stol(s, &consumed);
        if (consumed != s.size() || value < min_value) {
            throw std::invalid_argument("invalid");
        }
        return static_cast<int>(value);
    } catch (...) {
        std::cerr << arg_name << " must be an integer >= " << min_value
                  << "; got: " << text << '\n';
        std::exit(EXIT_FAILURE);
    }
}

static Options parse_args(int argc, char** argv) {
    Options opts;
    bool have_input0 = false;
    bool have_input1 = false;

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];

        if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            std::exit(EXIT_SUCCESS);
        }

        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << arg << '\n';
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }

        const char* value = argv[++i];

        if (arg == "--input0") {
            opts.input0 = parse_size(value, "--input0");
            have_input0 = true;
        } else if (arg == "--input1") {
            opts.input1 = parse_size(value, "--input1");
            have_input1 = true;
        } else if (arg == "--threads") {
            opts.threads = parse_int(value, "--threads", 1);
        } else if (arg == "--warmup") {
            opts.warmup = parse_int(value, "--warmup", 0);
        } else if (arg == "--iters") {
            opts.iterations = parse_int(value, "--iters", 1);
        } else {
            std::cerr << "Unknown argument: " << arg << '\n';
            usage(argv[0]);
            std::exit(EXIT_FAILURE);
        }
    }

    if (!have_input0 || !have_input1) {
        std::cerr << "Both --input0 and --input1 are required.\n";
        usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    if (opts.input0 != opts.input1) {
        std::cerr << "Vector addition requires matching shapes. "
                  << "input0=" << opts.input0
                  << ", input1=" << opts.input1 << '\n';
        std::exit(EXIT_FAILURE);
    }

    if (opts.threads > 1024) {
        std::cerr << "--threads cannot exceed 1024 for a CUDA block.\n";
        std::exit(EXIT_FAILURE);
    }

    return opts;
}

struct DeviceBuffers {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    explicit DeviceBuffers(size_t n) {
        if (n > std::numeric_limits<size_t>::max() / sizeof(float)) {
            std::cerr << "Input size overflows byte-size calculation.\n";
            std::exit(EXIT_FAILURE);
        }

        const size_t bytes = n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&A, bytes));
        CUDA_CHECK(cudaMalloc(&B, bytes));
        CUDA_CHECK(cudaMalloc(&C, bytes));
    }

    ~DeviceBuffers() {
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
    }

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;
};

int main(int argc, char** argv) {
    const Options opts = parse_args(argc, argv);
    const size_t N = opts.input0;
    const size_t bytes = N * sizeof(float);

    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    std::cout << "GPU: " << prop.name << '\n'
              << "Compute capability: " << prop.major << "." << prop.minor << '\n'
              << "Input0 shape: [" << opts.input0 << "] float32\n"
              << "Input1 shape: [" << opts.input1 << "] float32\n"
              << "Output shape: [" << N << "] float32\n"
              << "Threads per block: " << opts.threads << '\n'
              << "Warmup launches: " << opts.warmup << '\n'
              << "Timed launches: " << opts.iterations << "\n\n";

    std::vector<float> h_A(N);
    std::vector<float> h_B(N);
    std::vector<float> h_C(N, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> h_ref(N);

    std::mt19937 rng(2026);
    std::uniform_real_distribution<float> dist(-10.0f, 10.0f);

    for (size_t i = 0; i < N; ++i) {
        h_A[i] = dist(rng);
        h_B[i] = dist(rng);
        h_ref[i] = h_A[i] + h_B[i];
    }

    DeviceBuffers d(N);
    CUDA_CHECK(cudaMemcpy(d.A, h_A.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.B, h_B.data(), bytes, cudaMemcpyHostToDevice));

    const size_t blocks =
        (N + static_cast<size_t>(opts.threads) - 1) /
        static_cast<size_t>(opts.threads);

    if (blocks > prop.maxGridSize[0]) {
        std::cerr << "Grid size exceeds device limit for this 1D launch.\n";
        return EXIT_FAILURE;
    }

    for (int i = 0; i < opts.warmup; ++i) {
        vector_add<<<static_cast<unsigned int>(blocks), opts.threads>>>(
            d.A, d.B, d.C, N);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < opts.iterations; ++i) {
        vector_add<<<static_cast<unsigned int>(blocks), opts.threads>>>(
            d.A, d.B, d.C, N);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float total_ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&total_ms, start, stop));
    const double avg_ms = static_cast<double>(total_ms) / opts.iterations;

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    CUDA_CHECK(cudaMemcpy(h_C.data(), d.C, bytes, cudaMemcpyDeviceToHost));

    constexpr float ABS_TOL = 1.0e-6f;
    constexpr float REL_TOL = 1.0e-6f;
    size_t mismatches = 0;
    float max_abs_error = 0.0f;

    for (size_t i = 0; i < N; ++i) {
        const float got = h_C[i];
        const float expected = h_ref[i];
        const float abs_error = std::fabs(got - expected);
        const float allowed = ABS_TOL + REL_TOL * std::fabs(expected);
        max_abs_error = std::max(max_abs_error, abs_error);

        if (!std::isfinite(got) || abs_error > allowed) {
            if (mismatches < 5) {
                std::cerr << "Mismatch at index " << i
                          << ": GPU=" << got
                          << ", reference=" << expected
                          << ", abs_error=" << abs_error << '\n';
            }
            ++mismatches;
        }
    }

    // Requested traffic estimate: read A, read B, write C = 12 bytes/element.
    // This is effective bandwidth, not a direct measurement of DRAM traffic.
    const double requested_bytes = 3.0 * static_cast<double>(N) * sizeof(float);
    const double effective_gb_s = requested_bytes / (avg_ms * 1.0e6);

    std::cout << std::fixed << std::setprecision(3)
            << "Average kernel time: " << avg_ms * 1000.0
            << " us\n"
            << std::setprecision(2)
            << "Effective bandwidth: " << effective_gb_s << " GB/s\n"
            << std::scientific
            << "Max absolute error: " << max_abs_error << '\n'
            << (mismatches == 0 ? "TEST PASSED\n" : "TEST FAILED\n")
            << "Mismatches: " << mismatches << '\n';

    CUDA_CHECK(cudaFree(d.A));
    CUDA_CHECK(cudaFree(d.B));
    CUDA_CHECK(cudaFree(d.C));

    return mismatches == 0 ? EXIT_SUCCESS : EXIT_FAILURE;
}
