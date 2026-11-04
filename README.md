# GPU Performace Lab

A collection of CUDA and Triton kernel implementations, GPU optimisation experiments, and hands-on explorations of GPU performance engineering.

This repository serves as both the **code companion to my technical blogs** and a playground for experimenting with GPU kernels, optimisation techniques, and low-level GPU programming.

Some directories contain implementations and benchmarks referenced in published articles. Others are independent experiments exploring new ideas, kernel designs, and performance optimisations.

The focus is on understanding GPU execution, implementing kernels from first principles, and investigating how algorithmic and hardware-level decisions affect performance.

## Areas of Exploration

- **CUDA Programming** — kernel development, execution model, memory hierarchy, and synchronisation
- **GPU Optimisations** — memory coalescing, shared-memory bank conflicts, alignment, tiling, and occupancy
- **Matrix Multiplication** — GEMM implementations, tiling strategies, and Tensor Core computation
- **Attention** — attention kernels, FlashAttention, and memory-efficient attention algorithms
- **Quantisation** — dynamic and static activation quantisation, weight quantisation, and low-precision GEMM
- **LLM Inference** — prefill, decode, KV cache, and inference-oriented kernel optimisations
- **Triton** — GPU kernel implementations and performance tuning
- **Performance Analysis** — benchmarking, profiling, and understanding GPU bottlenecks

## Repository Structure

```text
gpu-performace-lab/
│
├── blog-experiments/
│   ├── prefill-decode/
│   ├── bank-conflicts/
│   ├── activation-quantization/
│   └── ...
│
├── cuda/
│   ├── matmul/
│   ├── attention/
│   ├── flash-attention/
│   ├── reductions/
│   └── ...
│
├── triton/
│   ├── matmul/
│   ├── attention/
│   ├── flash-attention/
│   └── ...
│
├── benchmarks/
│
└── README.md
```

The structure may evolve as the collection grows. Each experiment is organised around the kernel or optimisation being explored.

## Blog Companion Experiments

Some implementations are directly associated with my technical articles.

These directories contain the code, benchmarks, and experiments referenced in the corresponding blog posts. The full explanations and discussions are published separately.

## Independent Kernel Experiments

This repository is also a space to explore ideas beyond the scope of published articles.

Experiments may include:

- Implementing Attention and FlashAttention from scratch
- Exploring different GEMM tiling strategies
- Optimising memory access patterns
- Investigating shared-memory layouts and bank conflicts
- Implementing quantised kernels
- Comparing CUDA and Triton implementations
- Exploring new algorithms and kernel optimisations

Not every experiment is intended to be a polished or production-ready implementation. Some are exploratory, serving as a starting point for further optimisation and analysis.

## Performance and Correctness

Where applicable, experiments include:

- Reference implementations
- Correctness validation
- Benchmarking scripts
- Performance comparisons
- Profiling results
- Notes on optimisation strategies and observed behaviour

Performance is workload- and architecture-dependent. The goal is to understand why a kernel performs the way it does, rather than simply report a speedup.

## About

GPU Kernel Lab is an ongoing exploration of GPU programming, kernel optimisation, and high-performance machine learning computation.

**Implement. Experiment. Profile. Optimise.**
