#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "benchmark.cuh"
#include "utils.cuh"
#include "terminal.cuh"

#include "../kernels/01_naive_gemm.cuh"
#include "../kernels/02_global_mem_coalesce_gemm.cuh"
#include "../kernels/03_shred_mem_block_gemm.cuh"
#include "../kernels/04_1D_block_tile_gemm.cuh"
#include "../kernels/05_2D_block_tile_gemm.cuh"
#include "../kernels/06_Vectorized_gemm.cuh"
#include "../kernels/07_Bank_conflits_gemm.cuh"
#include "../kernels/08_bank_extra_col_gemm.cuh"
#include "../kernels/09_wrap_tiling_gemm.cuh"

// ============================================================
// Device Buffers
// ============================================================

struct DeviceBuffers {
    float* A = nullptr;
    float* B = nullptr;
    float* C = nullptr;

    DeviceBuffers() = default;

    DeviceBuffers(const DeviceBuffers&) = delete;
    DeviceBuffers& operator=(const DeviceBuffers&) = delete;

    ~DeviceBuffers() {
        if (A) cudaFree(A);
        if (B) cudaFree(B);
        if (C) cudaFree(C);
    }
};

// ============================================================
// Kernel Result
// ============================================================

struct KernelResult {
    std::string name;

    float avg_ms = 0.0f;
    double gflops = 0.0;

    CompareResult compare{};
};

// ============================================================
// Main
// ============================================================

int main(int argc, char** argv) {
    terminal::enable_ansi_colors();

    try {
        // ----------------------------------------------------
        // Parse arguments
        // ----------------------------------------------------

        const GemmConfig cfg = parse_args(argc, argv);

        const std::vector<std::string> kernel_names = {
            "naive", "coalesced", "shared_mem", "block_tile_1d",
            "block_tile_2d", "vectorized", "bank_conflicts",
            "bank_extra_col", "warp_tiling"
        };
        if (cfg.kernel != "all" &&
            std::find(kernel_names.begin(), kernel_names.end(), cfg.kernel) ==
                kernel_names.end()) {
            throw std::runtime_error(
                "Invalid --kernel value. "
                "Use: all, naive, coalesced, shared_mem, block_tile_1d, "
                "block_tile_2d, vectorized, bank_conflicts, bank_extra_col, "
                "or warp_tiling"
            );
        }

        // ----------------------------------------------------
        // Device information
        // ----------------------------------------------------

        int device_id = 0;

        CUDA_CHECK(cudaGetDevice(&device_id));

        cudaDeviceProp prop{};

        CUDA_CHECK(
            cudaGetDeviceProperties(&prop, device_id)
        );

        // ----------------------------------------------------
        // Matrix sizes
        // ----------------------------------------------------

        const size_t a_count =
            checked_num_elements(cfg.M, cfg.K, "A");

        const size_t b_count =
            checked_num_elements(cfg.K, cfg.N, "B");

        const size_t c_count =
            checked_num_elements(cfg.M, cfg.N, "C");

        // ----------------------------------------------------
        // Host-side buffers
        // ----------------------------------------------------

        std::vector<float> A(a_count);
        std::vector<float> B(b_count);

        std::vector<float> C_initial(c_count);
        std::vector<float> C_output(c_count);

        std::vector<float> C_reference;

        std::mt19937 rng(cfg.seed);

        fill_random(A, rng);
        fill_random(B, rng);
        fill_random(C_initial, rng);

        // ----------------------------------------------------
        // CPU reference
        // ----------------------------------------------------

        cpu_gemm_reference(
            A,
            B,
            C_initial,
            C_reference,
            cfg.M,
            cfg.N,
            cfg.K,
            cfg.alpha,
            cfg.beta
        );

        // ----------------------------------------------------
        // Device memory allocation
        // ----------------------------------------------------

        DeviceBuffers d;

        CUDA_CHECK(
            cudaMalloc(
                &d.A,
                a_count * sizeof(float)
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d.B,
                b_count * sizeof(float)
            )
        );

        CUDA_CHECK(
            cudaMalloc(
                &d.C,
                c_count * sizeof(float)
            )
        );

        // ----------------------------------------------------
        // Host -> Device transfers
        // ----------------------------------------------------

        CUDA_CHECK(
            cudaMemcpy(
                d.A,
                A.data(),
                a_count * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                d.B,
                B.data(),
                b_count * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );

        CUDA_CHECK(
            cudaMemcpy(
                d.C,
                C_initial.data(),
                c_count * sizeof(float),
                cudaMemcpyHostToDevice
            )
        );

        // ----------------------------------------------------
        // Launch configurations
        // ----------------------------------------------------

        // Naive kernel launch configuration.
        const dim3 naive_block(
            static_cast<unsigned int>(cfg.threads_x),
            static_cast<unsigned int>(cfg.threads_y)
        );

        const dim3 naive_grid(
            static_cast<unsigned int>(
                (cfg.M + cfg.threads_x - 1) /
                cfg.threads_x
            ),
            static_cast<unsigned int>(
                (cfg.N + cfg.threads_y - 1) /
                cfg.threads_y
            )
        );

        // Coalesced kernel launch configuration.
        //
        // The template BLOCKSIZE must match both block
        // dimensions used in the launch.
        constexpr int BLOCKSIZE = 16;

        const dim3 coalesced_block(
            BLOCKSIZE,
            BLOCKSIZE
        );

        const dim3 coalesced_grid(
            static_cast<unsigned int>(
                (cfg.M + BLOCKSIZE - 1) /
                BLOCKSIZE
            ),
            static_cast<unsigned int>(
                (cfg.N + BLOCKSIZE - 1) /
                BLOCKSIZE
            )
        );

        // Shared-memory and tiled kernels use fixed 16x16 or 64x64 tiles.
        constexpr int TILE = 16;
        const dim3 shared_block(TILE * TILE);
        const dim3 shared_grid(
            static_cast<unsigned int>((cfg.M + TILE - 1) / TILE),
            static_cast<unsigned int>((cfg.N + TILE - 1) / TILE));

        constexpr int BM = 64;
        constexpr int BN = 64;
        constexpr int BK = 8;
        constexpr int TM = 8;
        constexpr int TN = 8;
        constexpr int TILE_THREADS_1D = BM * BK;
        constexpr int TILE_THREADS_2D = (BM * BN) / (TM * TN);
        constexpr int VECTOR_BM = 128;
        constexpr int VECTOR_BN = 128;
        constexpr int VECTOR_BK = 8;
        constexpr int VECTOR_TM = 8;
        constexpr int VECTOR_TN = 8;
        constexpr int TILE_THREADS_VECTOR =
            (VECTOR_BM * VECTOR_BN) / (VECTOR_TM * VECTOR_TN);
        constexpr int WARP_BM = 64;
        constexpr int WARP_BN = 64;
        constexpr int WARP_BK = 8;
        constexpr int WARP_WM = 32;
        constexpr int WARP_WN = 32;
        constexpr int WARP_WNITER = 1;
        constexpr int WARP_TM = 8;
        constexpr int WARP_TN = 4;
        constexpr int WARP_THREADS = 128;

        const dim3 tiled_1d_block(TILE_THREADS_1D);
        const dim3 tiled_1d_grid(
            static_cast<unsigned int>((cfg.N + BN - 1) / BN),
            static_cast<unsigned int>((cfg.M + BM - 1) / BM));
        const dim3 tiled_block(TILE_THREADS_2D);
        const dim3 vector_block(TILE_THREADS_VECTOR);
        const dim3 tiled_grid(
            static_cast<unsigned int>((cfg.N + BN - 1) / BN),
            static_cast<unsigned int>((cfg.M + BM - 1) / BM));
        const dim3 vector_grid(
            static_cast<unsigned int>((cfg.N + VECTOR_BN - 1) / VECTOR_BN),
            static_cast<unsigned int>((cfg.M + VECTOR_BM - 1) / VECTOR_BM));
        const dim3 warp_block(WARP_THREADS);
        const dim3 warp_grid(
            static_cast<unsigned int>((cfg.N + WARP_BN - 1) / WARP_BN),
            static_cast<unsigned int>((cfg.M + WARP_BM - 1) / WARP_BM));

        // ----------------------------------------------------
        // Validate grid dimensions
        // ----------------------------------------------------

        if (
            naive_grid.x >
                static_cast<unsigned int>(prop.maxGridSize[0]) ||
            naive_grid.y >
                static_cast<unsigned int>(prop.maxGridSize[1])
        ) {
            throw std::runtime_error(
                "Naive GEMM grid exceeds device limits"
            );
        }

        if (cfg.M % WARP_BM != 0 || cfg.N % WARP_BN != 0 ||
            cfg.K % BK != 0) {
            throw std::runtime_error(
                "Tiled kernels require M/N to be multiples of their tile "
                "size and K to be a multiple of 8");
        }

        if (
            coalesced_grid.x >
                static_cast<unsigned int>(prop.maxGridSize[0]) ||
            coalesced_grid.y >
                static_cast<unsigned int>(prop.maxGridSize[1])
        ) {
            throw std::runtime_error(
                "Coalesced GEMM grid exceeds device limits"
            );
        }

        // ----------------------------------------------------
        // Generic kernel runner
        // ----------------------------------------------------

        auto run_kernel = [&](const std::string& name, auto launch) {
            KernelResult result;

            result.name = name;

            // Reset C before benchmarking so the benchmark
            // starts from the same initial state.
            CUDA_CHECK(
                cudaMemcpy(
                    d.C,
                    C_initial.data(),
                    c_count * sizeof(float),
                    cudaMemcpyHostToDevice
                )
            );

            // Benchmark the selected kernel.
            result.avg_ms = benchmark_kernel(
                launch,
                cfg.warmup,
                cfg.iterations
            );

            // Restore C before correctness validation.
            //
            // This is important for beta != 0 because
            // repeated launches modify C.
            CUDA_CHECK(
                cudaMemcpy(
                    d.C,
                    C_initial.data(),
                    c_count * sizeof(float),
                    cudaMemcpyHostToDevice
                )
            );

            // Run once for correctness validation.
            launch();

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());

            // Device -> Host transfer.
            CUDA_CHECK(
                cudaMemcpy(
                    C_output.data(),
                    d.C,
                    c_count * sizeof(float),
                    cudaMemcpyDeviceToHost
                )
            );

            // Compare against CPU reference.
            result.compare = compare_vectors(
                C_output,
                C_reference
            );

            // ------------------------------------------------
            // Performance metrics
            // ------------------------------------------------

            const double flops =
                2.0 *
                static_cast<double>(cfg.M) *
                static_cast<double>(cfg.N) *
                static_cast<double>(cfg.K);

            result.gflops =
                flops /
                (
                    static_cast<double>(result.avg_ms) *
                    1.0e6
                );

            return result;
        };

        // ----------------------------------------------------
        // Run selected kernels
        // ----------------------------------------------------

        std::vector<KernelResult> results;

        // Run naive GEMM.
        if (cfg.kernel == "all" ||cfg.kernel == "naive") {
            auto naive_launch = [&]() {
                naive_gemm<<<naive_grid, naive_block>>>(
                    d.A,
                    d.B,
                    d.C,
                    cfg.M,
                    cfg.N,
                    cfg.K,
                    cfg.alpha,
                    cfg.beta
                );
            };

            results.push_back(
                run_kernel(
                    "Naive GEMM",
                    naive_launch
                )
            );
        }

        // Run coalesced GEMM.
        if (cfg.kernel == "all" ||cfg.kernel == "coalesced") {
            auto coalesced_launch = [&]() {
                global_mem_coalesce_gemm<BLOCKSIZE>
                    <<<coalesced_grid, coalesced_block>>>(
                        d.A,
                        d.B,
                        d.C,
                        cfg.M,
                        cfg.N,
                        cfg.K,
                        cfg.alpha,
                        cfg.beta
                    );
            };

            results.push_back(
                run_kernel(
                    "Coalesced GEMM",
                    coalesced_launch
                )
            );
        }

        if (cfg.kernel == "all" || cfg.kernel == "shared_mem") {
            auto shared_launch = [&]() {
                shared_mem_block_gemm<BLOCKSIZE>
                    <<<shared_grid, shared_block>>>(
                        d.A,
                        d.B,
                        d.C,
                        cfg.M,
                        cfg.N,
                        cfg.K,
                        cfg.alpha,
                        cfg.beta
                    );
            };

            results.push_back(
                run_kernel(
                    "Shared-memory GEMM",
                    shared_launch
                )
            );
        }

        auto add_tiled_kernel = [&](const std::string& id,
                                    const std::string& name,
                                    auto launch) {
            if (cfg.kernel == "all" || cfg.kernel == id) {
                results.push_back(run_kernel(name, launch));
            }
        };

        add_tiled_kernel("block_tile_1d", "1D block-tiled GEMM", [&]() {
            shared_1D_block_tile_gemm<BM, BN, BK, TM>
                <<<tiled_1d_grid, tiled_1d_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        add_tiled_kernel("block_tile_2d", "2D block-tiled GEMM", [&]() {
            shared_2D_block_tile_gemm<BM, BN, BK, TM, TN>
                <<<tiled_grid, tiled_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        add_tiled_kernel("vectorized", "Vectorized GEMM", [&]() {
            vectorized_gemm<VECTOR_BM, VECTOR_BN, VECTOR_BK, VECTOR_TM,
                            VECTOR_TN>
                <<<vector_grid, vector_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        add_tiled_kernel("bank_conflicts", "Bank-conflict-free GEMM", [&]() {
            bank_conflicts_gemm<VECTOR_BM, VECTOR_BN, VECTOR_BK, VECTOR_TM,
                                VECTOR_TN>
                <<<vector_grid, vector_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        add_tiled_kernel("bank_extra_col", "Bank-extra-column GEMM", [&]() {
            bank_extra_col_gemm<VECTOR_BM, VECTOR_BN, VECTOR_BK, VECTOR_TM,
                                VECTOR_TN>
                <<<vector_grid, vector_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        add_tiled_kernel("warp_tiling", "Warp-tiled GEMM", [&]() {
            warp_tiling_gemm<WARP_BM, WARP_BN, WARP_BK, WARP_WM, WARP_WN,
                            WARP_WNITER, WARP_TM, WARP_TN, WARP_THREADS>
                <<<warp_grid, warp_block>>>(
                    cfg.M, cfg.N, cfg.K, cfg.alpha, d.A, d.B, cfg.beta, d.C);
        });

        // ----------------------------------------------------
        // Terminal output: banner
        // ----------------------------------------------------

        terminal::print_banner(
            "GEMM KERNEL COMPARISON",
            "GPU KERNEL LAB"
        );

        // ----------------------------------------------------
        // Device information
        // ----------------------------------------------------

        terminal::section("DEVICE");

        terminal::key_value(
            "GPU",
            prop.name
        );

        terminal::key_value(
            "Compute Capability",
            std::to_string(prop.major) + "." +
            std::to_string(prop.minor)
        );

        // ----------------------------------------------------
        // Problem configuration
        // ----------------------------------------------------

        terminal::section("PROBLEM CONFIGURATION");

        terminal::print_table(
            {
                "Matrix",
                "Shape",
                "Data Type"
            },
            {
                {
                    "A",
                    terminal::format_shape(
                        cfg.M,
                        cfg.K
                    ),
                    "FP32"
                },
                {
                    "B",
                    terminal::format_shape(
                        cfg.K,
                        cfg.N
                    ),
                    "FP32"
                },
                {
                    "C",
                    terminal::format_shape(
                        cfg.M,
                        cfg.N
                    ),
                    "FP32"
                }
            }
        );

        terminal::key_value(
            "Kernel Selection",
            cfg.kernel
        );

        terminal::key_value(
            "Alpha",
            terminal::format_float(
                cfg.alpha,
                3
            )
        );

        terminal::key_value(
            "Beta",
            terminal::format_float(
                cfg.beta,
                3
            )
        );

        terminal::key_value(
            "Warmup Launches",
            std::to_string(cfg.warmup)
        );

        terminal::key_value(
            "Timed Iterations",
            std::to_string(cfg.iterations)
        );

        // ----------------------------------------------------
        // Launch configuration
        // ----------------------------------------------------

        terminal::section("LAUNCH CONFIGURATION");

        terminal::key_value(
            "Naive Block",
            terminal::format_shape(
                cfg.threads_x,
                cfg.threads_y
            )
        );

        terminal::key_value(
            "Coalesced Block",
            terminal::format_shape(
                BLOCKSIZE,
                BLOCKSIZE
            )
        );

        // ----------------------------------------------------
        // Combined results table
        // ----------------------------------------------------

        terminal::section("COMBINED RESULTS");

        std::vector<std::vector<std::string>> rows;

        for (const auto& result : results) {
            rows.push_back({
                result.name,

                terminal::format_float(
                    result.avg_ms * 1000.0,
                    3
                ) + " us",

                terminal::format_float(
                    result.gflops,
                    2
                ),

                terminal::format_float(
                    result.compare.max_abs_error,
                    3,
                    true
                ),

                std::to_string(
                    result.compare.mismatches
                ),

                result.compare.mismatches == 0
                    ? "PASS"
                    : "FAIL"
            });
        }

        terminal::print_table(
            {
                "Kernel",
                "Latency",
                "GFLOP/s",
                "Max Abs Error",
                "Mismatches",
                "Status"
            },
            rows
        );

        // ----------------------------------------------------
        // Final validation status
        // ----------------------------------------------------

        bool all_passed = true;

        for (const auto& result : results) {
            if (result.compare.mismatches != 0) {
                all_passed = false;
                break;
            }
        }

        terminal::section("VALIDATION");

        terminal::status(
            all_passed,
            "All selected kernels"
        );

        std::cout << '\n';

        return all_passed
            ? EXIT_SUCCESS
            : EXIT_FAILURE;

    } catch (const std::exception& e) {
        std::cerr
            << terminal::color::red
            << terminal::color::bold
            << "\n  [ ERROR ] "
            << terminal::color::reset
            << e.what()
            << '\n';

        return EXIT_FAILURE;
    }
}
