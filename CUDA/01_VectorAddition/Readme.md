# CUDA Vector Addition

A first CUDA kernel: element-wise vector addition, implemented in CUDA C++ and evaluated with a small command-line test and benchmark harness.

This project is part of `gpu-kernel-lab`. The goal is not just to write a working GPU kernel, but to understand how a CUDA kernel maps work to threads, how to validate its output, and how to measure its performance.

## 1. What are we computing?

Given two vectors of equal length:

\[
A = [a_0, a_1, \ldots, a_{N-1}]
\]

\[
B = [b_0, b_1, \ldots, b_{N-1}]
\]

The output is:

\[
C_i = A_i + B_i,\quad 0 \le i < N
\]

Each output element is independent. This makes vector addition a natural example of a data-parallel GPU workload.

For `N = 4`:

```text
A = [1, 2, 3, 4]
B = [5, 6, 7, 8]

C = [6, 8, 10, 12]
```

The kernel assigns each element to a CUDA thread.

## 2. Project layout

```text
01_VectorAddition/
├── vector_add.cu
├── vector_add_test.cu
└── README.md
```

- `vector_add.cu` — the CUDA kernel implementation.
- `vector_add_test.cu` — command-line interface, memory management, correctness checks, and performance measurement.
- `README.md` — explanation and usage guide.

## 3. Reading the CUDA kernel

The kernel has the following interface:

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    size_t N
);
```

### Understanding the function signature

| Component | Meaning |
|---|---|
| `__global__` | Declares a CUDA kernel callable from the host and executed on the GPU. |
| `A`, `B` | Input vectors stored in device memory. |
| `C` | Output vector stored in device memory. |
| `const float*` | The kernel reads these input values but does not modify them. |
| `size_t N` | Number of elements in each vector. |

The kernel uses `size_t` for the vector length and index. This is a suitable unsigned type for sizes and avoids limiting the interface to a 32-bit `int`.

### Kernel implementation

```cpp
__global__ void vector_add(
    const float* A,
    const float* B,
    float* C,
    size_t N
) {
    size_t idx =
        blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### How does a thread find its element?

CUDA organizes threads into blocks, and blocks into a grid.

```text
Grid
 ├── Block 0: threads 0 ... 255
 ├── Block 1: threads 0 ... 255
 ├── Block 2: threads 0 ... 255
 └── ...
```

For a one-dimensional launch, a thread's global index is:

```cpp
idx = blockIdx.x * blockDim.x + threadIdx.x;
```

- `blockIdx.x` — which block the thread belongs to.
- `blockDim.x` — number of threads in each block.
- `threadIdx.x` — thread's index within its block.

For example, with 256 threads per block:

```text
Block 0, Thread 0   -> idx 0
Block 0, Thread 255 -> idx 255
Block 1, Thread 0   -> idx 256
Block 1, Thread 1   -> idx 257
```

Each valid thread adds one pair of values and writes one output value.

### Why is the bounds check necessary?

```cpp
if (idx < N)
```

The number of elements may not be divisible by the block size. The launch therefore may create some threads whose indices are outside the input range.

For example, if `N = 1024` and the block size is 256, exactly four blocks are required. But for `N = 1000`, the launch still creates 1024 threads. The final 24 threads must not access memory beyond the vectors.

The bounds check prevents those out-of-range accesses.

## 4. Understanding the test harness

The test harness is responsible for more than launching the kernel. It provides a repeatable way to test different vector sizes and collect performance measurements.

Its workflow is:

1. Parse command-line arguments.
2. Query the CUDA device.
3. Allocate device buffers.
4. Generate input data on the CPU.
5. Compute a CPU reference result.
6. Copy inputs to the GPU.
7. Warm up the kernel.
8. Measure repeated kernel launches with CUDA events.
9. Copy the output back to the CPU.
10. Compare GPU output against the reference.
11. Report correctness and performance.

### Input arguments

The harness accepts the vector sizes explicitly:

```cmd
vector_add_test.exe --input0 1024 --input1 1024
```

`--input0` and `--input1` represent the lengths of the two input vectors. Both must have the same length for this element-wise operation.

Additional options control the benchmark:

| Argument | Purpose |
|---|---|
| `--threads` | Threads per CUDA block. |
| `--warmup` | Number of warmup kernel launches. |
| `--iters` | Number of timed kernel launches. |

For example:

```cmd
vector_add_test.exe --input0 1048576 --input1 1048576 --threads 256 --warmup 10 --iters 200
```

The default block size is 256 threads. The warmup launches are excluded from the reported average.

## 5. Build the project

Run the following command from the project directory:

```cmd
nvcc -O3 -std=c++17 vector_add.cu vector_add_test.cu -o vector_add_test.exe
```

What the options mean:

- `nvcc` — NVIDIA's CUDA compiler driver.
- `-O3` — enables a high level of compiler optimization.
- `-std=c++17` — compiles using the C++17 language standard.
- `-o vector_add_test.exe` — names the output executable.

The kernel definition and its declaration in the test harness must have matching signatures. In particular, both must use the same type for `N`.

## 6. Run the tests

### Small vector

```cmd
./vector_add_test.exe --input0 1024 --input1 1024
```

This tests a vector containing 1,024 float32 elements.

### Large vector

```cmd
./vector_add_test.exe --input0 1048576 --input1 1048576 --warmup 10 --iters 200
```

This tests a vector containing 1,048,576 float32 elements, or \(2^{20}\) elements.

The larger workload is useful for observing throughput when the kernel performs enough work to make memory bandwidth more visible.

## 7. Correctness validation

The harness computes a CPU reference result and compares it with the GPU output.

The maximum absolute error is:

\[
E_{\max} = \max_i |C_i^{GPU} - C_i^{CPU}|
\]

The mismatch count is the number of output elements that fail the harness's comparison criteria.

For vector addition with ordinary float32 values, the expected result is exact in these tests. A result such as:

```text
Max absolute error: 0.00e+00
TEST PASSED
Mismatches: 0
```

means the output passed the harness's correctness check.

A passing result validates the tested input size and values. It does not by itself prove correctness for every possible input or vector length.

## 8. Measuring kernel time

The harness uses CUDA events to measure GPU execution time.

The average kernel time is computed from the elapsed time over the timed launches:

\[
t_{\text{avg}} = \frac{t_{\text{total}}}{\text{iterations}}
\]

The reported time is the average time per kernel launch. Host-side input generation, allocation, and host-to-device or device-to-host copies are not included in this kernel timing.

This distinction matters: kernel execution time is not the same as end-to-end application latency.

## 9. Effective memory bandwidth

Vector addition performs very little arithmetic relative to the amount of data it reads and writes. It is therefore a useful example of a bandwidth-sensitive kernel.

For each element, the kernel performs:

- One float32 load from `A`: 4 bytes.
- One float32 load from `B`: 4 bytes.
- One float32 store to `C`: 4 bytes.

The nominal data movement is 12 bytes per element.

For `N` elements:

\[
\text{Bytes} = 3 \times N \times \text{sizeof(float)}
\]

The harness calculates effective bandwidth as:

\[
BW_{\text{effective}} =
\frac{\text{Bytes transferred}}{\text{Kernel time}}
\]

When using bytes and seconds, divide by \(10^9\) to report GB/s.

This is an *effective* bandwidth estimate based on the algorithm's nominal memory traffic. It is not a direct measurement of physical DRAM traffic. Cache behavior and other hardware effects can influence the result.

## 10. Benchmark results

The following results were collected on an NVIDIA GeForce RTX 3060 Laptop GPU, compute capability 8.6.

| Metric | 1,024 elements | 1,048,576 elements |
|---|---:|---:|
| Threads per block | 256 | 256 |
| Warmup launches | 5 | 10 |
| Timed launches | 100 | 200 |
| Average kernel time | 7.588 µs | 43.837 µs |
| Effective bandwidth | 1.62 GB/s | 287.04 GB/s |
| Max absolute error | 0.00e+00 | 0.00e+00 |
| Mismatches | 0 | 0 |
| Result | PASS | PASS |

### What do these results tell us?

**The small vector is dominated by overheads.**

For only 1,024 elements, the kernel performs a small amount of work. The measured time includes the cost of launching and scheduling the GPU work, so the effective bandwidth is not representative of the GPU's peak memory bandwidth.

**The larger vector exposes more of the memory-throughput behavior.**

For 1,048,576 elements, the kernel processes approximately 12 MiB of nominal data:

```text
Input A: 4 MiB
Input B: 4 MiB
Output C: 4 MiB
Total:   12 MiB
```

The measured 287.04 GB/s is the effective bandwidth calculated from that nominal traffic and the measured kernel time.

The two workloads demonstrate why kernel performance should be evaluated across multiple input sizes. A single small test can be useful for correctness, but it is not enough to characterize throughput.

## 11. Experiment with block size

Try different thread counts while keeping the vector size and iteration count constant:

```cmd
vector_add_test.exe --input0 1048576 --input1 1048576 --threads 128 --warmup 10 --iters 200
vector_add_test.exe --input0 1048576 --input1 1048576 --threads 256 --warmup 10 --iters 200
vector_add_test.exe --input0 1048576 --input1 1048576 --threads 512 --warmup 10 --iters 200
```

Record the average time and effective bandwidth.

Do not assume that a larger block size is automatically faster. Performance depends on the GPU, workload, compiler, and execution characteristics. Repeat runs under comparable conditions before drawing conclusions.

## 12. What this example teaches

This project introduces several fundamentals of CUDA kernel development:

- Mapping data-parallel work to CUDA threads.
- Computing global thread indices.
- Handling non-multiple vector sizes safely.
- Managing device memory and data transfers.
- Validating GPU output against a CPU reference.
- Measuring GPU execution with CUDA events.
- Interpreting effective bandwidth.
- Comparing performance across workload sizes and launch configurations.

Vector addition is intentionally simple. The same workflow—implement, validate, benchmark, and investigate—can be reused for more complex kernels such as reductions, matrix multiplication, quantization, and attention.

## Notes

These measurements are specific to the tested GPU, driver, runtime, launch configuration, and system state. They are useful as a baseline for this implementation, not as a universal performance guarantee.

The benchmark is a learning and experimentation harness. It is not intended to replace a production benchmarking framework or a comprehensive CUDA test suite.
