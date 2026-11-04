# CUDA Matrix Multiplication (GEMM)

This project is a guided progression through several CUDA implementations of
single-precision general matrix multiplication (GEMM). It is intended as a
learning exercise in GPU performance engineering: begin with a simple,
correct implementation, change one performance idea at a time, and use
benchmarking plus profiling to understand the result.

The project does not use cuBLAS for the measured kernels. Every implementation
in `kernels/` is a CUDA kernel written for this exercise and is validated
against a CPU reference implementation in `common/gemm_test.cu`.

## 1. The GEMM operation

The benchmark computes:

```text
C = alpha * A * B + beta * C
```

For matrix dimensions:

```text
A: M x K
B: K x N
C: M x N
```

the element at row `m` and column `n` is:

```text
C[m, n] = alpha * sum(A[m, k] * B[k, n], k = 0 .. K-1)
          + beta * C[m, n]
```

All matrices contain `float` values, so each element occupies four bytes.
Matrices use row-major indexing:

```cpp
A[m * K + k]
B[k * N + n]
C[m * N + n]
```

The operation performs approximately `2 * M * N * K` floating-point
operations. The benchmark reports:

```text
GFLOP/s = (2 * M * N * K) / (kernel_time_in_seconds * 1e9)
```

The `alpha` and `beta` terms matter for correctness. In particular, when
`beta != 0`, the initial contents of `C` must be restored before every timed
run and before the correctness run. The harness does this explicitly.

## 2. Repository layout

```text
02_MatrixMultiplication/
├── common/
│   ├── benchmark.cuh       Timing helper and CUDA error checks
│   ├── gemm_test.cu        Main benchmark and validation program
│   ├── terminal.cuh        Console formatting
│   └── utils.cuh           Arguments, initialization, CPU reference, checks
├── kernels/
│   ├── 01_naive_gemm.cuh
│   ├── 02_global_mem_coalesce_gemm.cuh
│   ├── 03_shred_mem_block_gemm.cuh
│   ├── 04_1D_block_tile_gemm.cuh
│   ├── 05_2D_block_tile_gemm.cuh
│   ├── 06_Vectorized_gemm.cuh
│   ├── 07_Bank_conflits_gemm.cuh
│   ├── 08_bank_extra_col_gemm.cuh
│   ├── 09_wrap_tiling_gemm.cuh
│   └── 10_ptx_gemm.cuh
└── Readme.md
```

Some filenames retain their original spelling, such as `shred` and
`conflits`. The CUDA kernel symbols and command-line identifiers use the
descriptive names listed below.

## 3. Kernel progression

| File | CUDA symbol | CLI identifier | Main idea |
|---|---|---|---|
| `01_naive_gemm.cuh` | `naive_gemm` | `naive` | One output element per thread |
| `02_global_mem_coalesce_gemm.cuh` | `global_mem_coalesce_gemm` | `coalesced` | Improve adjacent global-memory access |
| `03_shred_mem_block_gemm.cuh` | `shared_mem_block_gemm` | `shared_mem` | Reuse tiles through shared memory |
| `04_1D_block_tile_gemm.cuh` | `shared_1D_block_tile_gemm` | `block_tile_1d` | One thread computes a vertical strip |
| `05_2D_block_tile_gemm.cuh` | `shared_2D_block_tile_gemm` | `block_tile_2d` | One thread computes a 2D register tile |
| `06_Vectorized_gemm.cuh` | `vectorized_gemm` | `vectorized` | Use `float4` memory operations |
| `07_Bank_conflits_gemm.cuh` | `bank_conflicts_gemm` | `bank_conflicts` | Change shared-memory layout/access |
| `08_bank_extra_col_gemm.cuh` | `bank_extra_col_gemm` | `bank_extra_col` | Add shared-memory stride padding |
| `09_wrap_tiling_gemm.cuh` | `warp_tiling_gemm` | `warp_tiling` | Assign output subtiles to warps |
| `10_ptx_gemm.cuh` | `warp_tiling_ptx_gemm` | `ptx` | Use inline PTX for warp-tiled FP32 FMA |

Run all supported kernels with `--kernel all`, or select one using the CLI
identifier. The command-line name and the CUDA symbol are intentionally
separate: the former is a convenient user-facing label, while the latter is
the function instantiated by the CUDA compiler.

## 4. What changes from kernel to kernel?

### 4.1 Naive GEMM

`naive_gemm` assigns one matrix element `C[m, n]` to one thread. That thread
loops over all `K` values and computes one dot product.

This is the easiest implementation to understand and is the correctness
baseline. It is usually slow because neighboring threads repeatedly fetch the
same values of `A` and `B` from global memory. It also does not explicitly
reuse data through shared memory or registers beyond the current thread's
accumulator.

The default launch uses a 2D block of `16 x 16` threads. The grid covers the
`M x N` output matrix, and the kernel checks its row and column bounds.

### 4.2 Coalesced GEMM

`global_mem_coalesce_gemm` changes the mapping between thread coordinates and
matrix coordinates. Threads in a warp are arranged so neighboring lanes tend
to access neighboring columns of `B` and `C`.

Coalescing is about the addresses requested by a warp, not simply whether a
single thread accesses a contiguous array. When adjacent lanes access nearby
addresses, the GPU can satisfy the requests with fewer memory transactions.

This kernel uses a compile-time `16 x 16` block tile and checks matrix bounds.

### 4.3 Shared-memory GEMM

`shared_mem_block_gemm` cooperatively loads tiles of `A` and `B` into shared
memory. A thread block then reuses those values while computing a tile of `C`.

The basic sequence is:

1. Load one tile of `A` and one tile of `B`.
2. Synchronize with `__syncthreads()`.
3. Compute the partial dot products from shared memory.
4. Synchronize again before overwriting the shared tiles.
5. Advance to the next `K` tile.

The kernel uses a 16x16 tile and a one-dimensional 256-thread block. The
one-dimensional block is important because the implementation derives its
shared-memory row and column from the linear `threadIdx.x` value.

### 4.4 1D block tiling

`shared_1D_block_tile_gemm` gives each thread several output values in one
matrix dimension. Its compile-time parameters are currently:

```text
BM = 64, BN = 64, BK = 8, TM = 8
```

Here `BM` and `BN` describe the output tile, `BK` is the depth of a loaded
tile, and `TM` is the number of output rows handled by a thread. Partial sums
are kept in registers, reducing repeated work and improving arithmetic
intensity.

### 4.5 2D block tiling

`shared_2D_block_tile_gemm` extends register tiling in both output dimensions:

```text
BM = 64, BN = 64, BK = 8, TM = 8, TN = 8
```

Each thread computes a `TM x TN` result tile. The block therefore performs
more work per thread while reusing the shared-memory tile and keeping partial
results in registers.

The number of threads is derived from:

```text
threads_per_block = (BM * BN) / (TM * TN)
```

For the current parameters this is 64 threads.

### 4.6 Vectorized GEMM

`vectorized_gemm` uses `float4` loads and stores. One vector operation transfers
four FP32 values, which can reduce instruction overhead and improve memory
throughput when addresses are aligned and the access pattern is suitable.

The current harness uses a 128x128 tile with 8-wide register tiles. This
configuration gives the vectorized loader enough threads to cover all required
`float4` values in both input tiles.

Vectorization has requirements:

- The address must be suitably aligned.
- The vector must remain inside the matrix row.
- The matrix dimensions and tile boundaries must be compatible with the
  vector width.
- The extra work must be worthwhile; vectorization is not automatically faster.

### 4.7 Bank-conflict reduction

Shared memory is divided into banks. A bank conflict occurs when threads in a
warp request different addresses that map to the same bank during a shared
memory instruction. The requests may then be serialized.

`bank_conflicts_gemm` changes the layout used for the `B` tile. This is more
subtle than simply adding padding: the load layout, the compute layout, and the
register indexing must all agree. The current implementation uses a larger
shared-memory allocation because its remapped layout occupies more storage
than a simple `BK x BN` array.

### 4.8 Extra-column padding

`bank_extra_col_gemm` adds five extra columns to the shared `B` tile:

```cpp
__shared__ float Bs[BK * (BN + extraCols)];
```

Changing the row stride changes which banks successive rows map to. Padding
can remove a conflict for one access pattern, but it can also increase shared
memory usage or have no benefit for another pattern. This is why padding
should be verified with Nsight Compute rather than assumed to help.

### 4.9 Warp tiling

`warp_tiling_gemm` divides a block tile into warp-sized output tiles. Threads
within a warp cooperate on a smaller region, and values are reused through
registers and shared memory.

The current harness uses:

```text
BM = 64, BN = 64, BK = 8
WM = 32, WN = 32
WNITER = 1, TM = 8, TN = 4
threads_per_block = 128
```

The relationship between these parameters is important. In particular, the
kernel derives `WMITER` at compile time. Invalid combinations can produce
`WMITER = 0`, which leads to zero-sized arrays or division-by-zero compiler
errors.

## 5. Building

Build from a Visual Studio Developer Command Prompt, or another shell where
both `nvcc` and the MSVC `cl.exe` compiler are available:

```powershell
nvcc -O3 -lineinfo -std=c++17 common/gemm_test.cu -o gemm_test.exe
```

On Windows, `nvcc` normally uses MSVC as its host compiler. If the build says
`Cannot find compiler 'cl.exe' in PATH`, open the appropriate **x64 Native
Tools Command Prompt for Visual Studio** or initialize the Visual Studio build
environment before invoking `nvcc`.

The flags mean:

- `-O3`: enable aggressive device/compiler optimization.
- `-lineinfo`: preserve source line information for profiling without full
  debug instrumentation.
- `-std=c++17`: compile the host-side C++ code as C++17.
- `-o gemm_test.exe`: name the output executable.

## 6. Running the benchmark

Run every kernel:

```powershell
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel all
```

Run one kernel at a time:

```powershell
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel naive
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel coalesced
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel shared_mem
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel block_tile_1d
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel block_tile_2d
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel vectorized
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel bank_conflicts
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel bank_extra_col
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel warp_tiling
```

For fast correctness checks, reduce the timing work:

```powershell
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel vectorized --warmup 0 --iters 1
```

For a smaller learning example:

```powershell
.\gemm_test.exe --M 256 --N 256 --K 256 --kernel all --warmup 2 --iters 5
```

The command-line options are:

| Option | Meaning | Default |
|---|---|---:|
| `--M` | Rows of `A` and `C` | 1024 |
| `--N` | Columns of `B` and `C` | 1024 |
| `--K` | Columns of `A`, rows of `B` | 1024 |
| `--kernel` | Kernel identifier or `all` | `naive` |
| `--threads-x` | Naive block x dimension | 16 |
| `--threads-y` | Naive block y dimension | 16 |
| `--warmup` | Untimed launches before measurement | 10 |
| `--iters` | Timed launches | 100 |
| `--alpha` | GEMM scale for `AB` | 1.0 |
| `--beta` | GEMM scale for original `C` | 0.0 |
| `--seed` | Random input seed | 1234 |

Example with non-default GEMM coefficients:

```powershell
.\gemm_test.exe --M 512 --N 512 --K 1024 --kernel block_tile_2d --alpha 1.5 --beta 0.25
```

## 7. How the harness validates results

Before launching a CUDA kernel, the program:

1. Parses and validates the matrix dimensions and options.
2. Allocates host and device matrices.
3. Fills `A`, `B`, and the initial `C` with deterministic pseudo-random
   FP32 values.
4. Computes `C_reference` using a CPU GEMM implementation.
5. Copies the inputs to the GPU.

For each selected kernel, it then:

1. Restores the original `C` on the device.
2. Runs warmup launches.
3. Measures the requested timed launches using CUDA events.
4. Restores `C` again.
5. Runs one launch for correctness.
6. Synchronizes and checks the last CUDA error.
7. Copies the result back to the host.
8. Compares every output element against the CPU reference.

The result table reports:

- **Latency**: average kernel time, displayed in microseconds.
- **GFLOP/s**: estimated floating-point throughput.
- **Max Abs Error**: largest `abs(actual - reference)`.
- **Mismatches**: number of values outside the absolute/relative tolerance.
- **Status**: `PASS` only when no mismatches are found.

Small floating-point differences are expected because floating-point addition
is not associative and optimized kernels may accumulate values in a different
order. A small nonzero error can be acceptable; a large error, many mismatches,
or an illegal memory access indicates a real indexing, synchronization,
alignment, or launch-configuration problem.

## 8. Choosing matrix sizes safely

The naive and coalesced kernels include bounds checks. Several advanced
kernels are educational fixed-tile implementations and assume that dimensions
fit their tile and vectorization requirements. For reliable comparisons, begin
with dimensions divisible by the relevant tile sizes. The common comparison
shape is:

```text
M = 256, N = 2048, K = 2048
```

which is divisible by 64, 128, and 8.

When experimenting with arbitrary dimensions, inspect every kernel's bounds
handling. A grid rounded up with `ceil((size + tile - 1) / tile)` is not enough
by itself: every thread that can land in a partial tile must avoid reading or
writing beyond the matrix. Vectorized `float4` operations require additional
care because a single instruction accesses four consecutive values.

## 9. Reading benchmark results correctly

Compare kernels using the same:

- GPU and power/clock state.
- Matrix dimensions.
- `alpha`, `beta`, and random seed.
- Warmup count and timed iteration count.
- Compiler version and optimization flags.

Lower latency and higher GFLOP/s are generally better, but a faster result is
not useful if validation fails. Always inspect the `Status` column first.

The first run can be affected by startup overhead, clock ramping, and cache
state. Warmup launches reduce these effects. More iterations reduce timing
noise, while fewer iterations are useful during correctness debugging.

The benchmark's GFLOP/s number is an estimate based on the mathematical GEMM
work. It does not mean that every instruction executed by the kernel is a
floating-point multiply or add, and it does not include host-side setup in the
reported kernel latency.

## 10. Profiling with NVIDIA tools

### Nsight Systems

Nsight Systems provides a timeline. Use it to understand kernel launches,
memory transfers, synchronization, CPU/GPU gaps, and overall application
behavior.

```powershell
nsys profile --trace=cuda --output=gemm_system `
  .\gemm_test.exe --M 1024 --N 1024 --K 1024 --kernel block_tile_2d --warmup 1 --iters 1
```

Summarize CUDA kernel execution:

```powershell
nsys stats --report cuda_gpu_kern_sum .\gemm_system.nsys-rep
```

### Nsight Compute

Nsight Compute provides kernel-level hardware metrics. It is useful for
investigating:

- Achieved occupancy and active warps.
- Register count and local-memory spills.
- Global-load and global-store efficiency.
- L1/L2 cache behavior.
- Shared-memory throughput and bank conflicts.
- Warp execution efficiency.
- Compute and memory-pipeline utilization.

Profile one launch to keep the report manageable:

```powershell
ncu --set basic -c 1 -o gemm_compute `
  .\gemm_test.exe --M 1024 --N 1024 --K 1024 --kernel block_tile_2d --warmup 1 --iters 1
```

Use the actual CUDA symbol when filtering a kernel with `-k`, for example:

```powershell
ncu -k vectorized_gemm -c 1 `
  .\gemm_test.exe --M 1024 --N 1024 --K 1024 --kernel vectorized --warmup 1 --iters 1
```

Systems and Compute answer different questions: Nsight Systems explains when
work runs; Nsight Compute helps explain why a particular kernel runs at its
observed speed.

## 11. Common problems and debugging workflow

### `Cannot find compiler 'cl.exe'`

`nvcc` is available, but its Windows host compiler is not on `PATH`. Start a
Visual Studio Developer Command Prompt or run the Visual Studio environment
setup script before building.

### Illegal memory access

Run one kernel at a time:

```powershell
.\gemm_test.exe --M 256 --N 2048 --K 2048 --kernel vectorized --warmup 0 --iters 1
```

Then check, in order:

1. Grid dimensions and block dimensions.
2. Whether `blockIdx` x/y represents row/column in that kernel.
3. Bounds for every global-memory load and store.
4. `float4` alignment and four-element row bounds.
5. Shared-memory array sizes and index ranges.
6. `__syncthreads()` placement and whether all threads reach each barrier.
7. The first CUDA API call that reports the error; later errors may be delayed
   symptoms of the original fault.

### Correct launch but incorrect output

Check whether the kernel uses a one-dimensional or two-dimensional block. Some
educational kernels derive a logical row and column from linear `threadIdx.x`
and therefore must not be launched as a 2D block. Also verify that the grid's
x/y order matches the kernel's interpretation of `blockIdx.x` and `blockIdx.y`.

### Compiler errors involving template constants

Advanced kernels derive array sizes and warp subtiles at compile time. Verify
that all tile parameters produce positive integer values. For example, a warp
configuration that makes `WMITER` equal to zero cannot compile because it
creates zero-sized arrays.

### Unexpected performance

Confirm that you are comparing the same dimensions and timing settings. Then
profile. A kernel can be mathematically more sophisticated but slower because
of register pressure, low occupancy, synchronization overhead, uncoalesced
loads, or shared-memory bank conflicts.

## 12. Suggested learning path

1. Run `naive` and inspect its row/column indexing.
2. Run `coalesced` and compare global-memory behavior.
3. Study the two `__syncthreads()` calls in `shared_mem_block_gemm`.
4. Change `BM`, `BN`, `BK`, `TM`, and `TN` one at a time.
5. Compare the 1D and 2D register-tile layouts.
6. Use Nsight Compute to verify whether `float4` operations are generated for
   `vectorized`.
7. Compare shared-memory metrics for `bank_conflicts` and `bank_extra_col`.
8. Draw the warp-to-output mapping for `warp_tiling_gemm` on paper.
9. Test square and rectangular matrices separately.
10. Only after correctness is stable, optimize for throughput.

## 13. Extensions to try

- Add explicit bounds handling to every advanced kernel so arbitrary matrix
  dimensions are safe.
- Allocate matrices with pitched memory and compare the indexing changes.
- Add a transpose-based or pre-transposed-B implementation.
- Compare against `cublasSgemm` as a production-quality reference.
- Add half-precision and Tensor Core implementations.
- Use CUDA streams to overlap transfers and computation.
- Add command-line selection of tile parameters where compile-time templates
  permit it.
- Record results for multiple matrix shapes instead of relying on one case.
- Export benchmark results as CSV for plotting.

## 14. Main lessons

1. Correct indexing is the foundation of every optimization.
2. Coalesced access improves how a warp uses global-memory transactions.
3. Shared memory is valuable because it enables explicit data reuse.
4. Register tiling increases work per thread and can improve arithmetic
   intensity, but may increase register pressure.
5. Vectorization requires alignment and dimension assumptions.
6. Shared-memory bank behavior depends on the complete access pattern.
7. Warp tiling is powerful but makes the mapping and template constraints more
   difficult to reason about.
8. A benchmark result is meaningful only when the kernel passes correctness.
9. Timing shows what is faster; profiling helps explain why.
10. The best configuration depends on the GPU, matrix shape, compiler, and
    workload - not just the kernel label.

