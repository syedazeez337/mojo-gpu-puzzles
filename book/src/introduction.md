# CUDA GPU Puzzles

<div class="social-buttons">
  <a href="https://github.com/syedazeez337/mojo-gpu-puzzles" target="_blank">
    <span class="title">Puzzles Repo</span>
    <span class="icon icon-github"></span>
  </a>
  <a href="https://docs.nvidia.com/cuda/cuda-c-programming-guide/" target="_blank">
    <span class="title">CUDA C++ Programming Guide</span>
    <span class="icon icon-book"></span>
  </a>
</div>

> _"For the things we have to learn before we can do them, we learn by doing
> them."_ Aristotle (Nicomachean Ethics)

Welcome to a hands-on guide to GPU programming using **CUDA C++** — NVIDIA's
platform for writing code that runs on the GPU. You'll learn by writing real
kernels, compiling them with `nvcc`, and watching the results come back from the
hardware.

> 📌 **About this edition.** This course is a CUDA C++ port of Modular's
> excellent [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles).
> **Part I (Puzzles 1–8) has been fully converted to modern CUDA C++** — source,
> solutions, and chapters. Later parts are being ported puzzle-by-puzzle; until
> then their chapters may still show the original Mojo. The concepts transfer
> directly: the GPU is the same, only the language changes.

## Why GPU programming?

GPU programming has evolved from a specialized skill into fundamental
infrastructure for modern computing. From large language models processing
billions of parameters to computer vision systems analyzing real-time video
streams, GPU acceleration drives the computational breakthroughs we see today.
Scientific advances in climate modeling, drug discovery, and quantum simulation
depend on the massive parallel processing capabilities that GPUs uniquely
provide. Financial institutions rely on GPU computing for real-time risk
analysis and algorithmic trading, while autonomous vehicles process sensor data
through GPU-accelerated neural networks for critical decision-making.

The economic implications are substantial. Organizations that effectively
leverage GPU computing achieve significant competitive advantages: accelerated
development cycles, reduced computational costs, and the capacity to address
previously intractable computational challenges. In an era where computational
capability directly correlates with business value, GPU programming skills
represent a strategic differentiator for engineers, researchers, and
organizations.

## Why CUDA?

The computing industry has reached a critical point. CPU performance no longer
increases through higher clock speeds due to power and heat constraints.
Hardware manufacturers have shifted toward increasing physical cores. This
multi-core approach reaches its peak in modern GPUs, which contain thousands of
cores operating in parallel. The NVIDIA H100, for example, can run 16,896
threads simultaneously in a single clock cycle, with hundreds of thousands of
threads queued for execution.

**CUDA** (Compute Unified Device Architecture) is NVIDIA's parallel computing
platform and programming model. It remains the most direct, mature, and widely
deployed way to program NVIDIA GPUs:

- **Direct hardware access** to the GPU's full feature set — threads, blocks,
  warps, shared memory, tensor cores, and more
- **A mature C++ toolchain** (`nvcc`, `compute-sanitizer`, Nsight Systems and
  Nsight Compute) for building, debugging, and profiling
- **An enormous ecosystem** of libraries (cuBLAS, cuDNN, CUB, Thrust, CUTLASS)
  built on the same primitives you'll learn here
- **Modern C++ in device code**: with CUDA 12/13 and a recent `nvcc`, kernels can
  use C++17/20 features, and the standard library is available on the device via
  [libcu++](https://nvidia.github.io/cccl/libcudacxx/) (`cuda::std::`)
- **Portability of skills**: the thread/block/grid model you master here is the
  foundation under PyTorch, TensorFlow, JAX, and every major GPU framework

> **Learning CUDA gives you the mental model that underlies all GPU computing.**
> Once you understand how threads map to data, how memory hierarchies work, and
> how to coordinate thousands of threads, every higher-level GPU tool becomes
> easier to reason about.

## Why learn through puzzles?

Most GPU programming resources start with extensive theory before practical
implementation. This can overwhelm newcomers with abstract concepts that only
become clear through direct application.

This book uses a different approach: immediate engagement with practical
problems that progressively introduce concepts through guided discovery.

**Advantages of puzzle-based learning:**

- **Direct experience**: immediate execution on GPU hardware provides concrete
  feedback
- **Incremental complexity**: each challenge builds on previously established
  concepts
- **Applied focus**: problems mirror real-world computational scenarios
- **Diagnostic skills**: systematic debugging practice develops troubleshooting
  capabilities
- **Knowledge retention**: active problem-solving reinforces understanding more
  effectively than passive consumption

The methodology emphasizes discovery over memorization. Concepts emerge
naturally through experimentation, creating deeper understanding and practical
competency.

> **Acknowledgement**: This course is adapted from Modular's
> [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles), whose Parts I
> and III are themselves heavily inspired by
> [GPU Puzzles](https://github.com/srush/GPU-Puzzles), an interactive GPU
> learning project by Sasha Rush. This edition reimplements the concepts in
> modern CUDA C++.

## The GPU programming mindset

Effective GPU programming requires a fundamental shift in how we think about
computation. Here are some key mental models that will guide your journey:

### From sequential to parallel: Eliminating loops with threads

In traditional CPU programming, we process data sequentially through loops:

```cpp
// CPU approach
for (int i = 0; i < data_size; ++i) {
    result[i] = process(data[i]);
}
```

GPU programming inverts this paradigm completely. Rather than iterating
sequentially through data, we assign thousands of parallel threads to process
data elements simultaneously:

```cpp
// GPU approach (a CUDA kernel)
__global__ void kernel(float* result, const float* data, int data_size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < data_size) {
        result[i] = process(data[i]);
    }
}
```

Each thread handles a single data element, replacing explicit iteration with
massive parallelism. This fundamental reframing — from sequential processing to
concurrent execution across all data elements — represents the core conceptual
shift in GPU programming.

### Fitting a mesh of compute over data

Consider your data as a structured grid, with GPU threads forming a
corresponding computational grid that maps onto it. Effective GPU programming
involves designing this thread organization to optimally cover your data space:

- **Threads**: individual processing units, each responsible for specific data
  elements
- **Blocks**: coordinated thread groups with shared memory access and
  synchronization capabilities (`__syncthreads()`)
- **Grid**: the complete thread hierarchy spanning the entire computational
  problem

Successful GPU programming requires balancing this thread organization to
maximize parallel efficiency while managing memory access patterns and
synchronization requirements.

### Data movement vs. computation

In GPU programming, data movement is often more expensive than computation:

- Moving data between CPU and GPU (`cudaMemcpy`) is slow
- Moving data between global and shared memory is faster
- Operating on data already in registers or shared memory is extremely fast

This inverts another common assumption in programming: computation is no longer
the bottleneck — data movement is.

Through the puzzles in this book, you'll develop an intuitive understanding of
these principles, transforming how you approach computational problems.

## What you will learn

This book takes you on a journey from first principles to advanced GPU
programming techniques. Rather than treating the GPU as a mysterious black box,
the content builds understanding layer by layer — starting with how individual
threads operate and culminating in sophisticated parallel algorithms.

### Your current learning path

| Essential Skill            | Status                | Puzzles      |
|----------------------------|-----------------------|--------------|
| Thread/Block basics        | ✅ **CUDA (ported)**  | Part I (1-8) |
| Debugging GPU Programs      | 🚧 Port in progress   | Part II      |
| Core algorithms            | 🚧 Port in progress   | Part III     |
| Advanced topics            | 🚧 Port in progress   | Parts IV–XII |

### Detailed learning objectives

**Part I: GPU fundamentals (Puzzles 1-8) ✅**

- Learn thread indexing and block organization (`threadIdx`, `blockIdx`,
  `blockDim`)
- Understand memory access patterns and bounds guards
- Work with both raw pointers and a small `TensorView` abstraction
- Move data between host and device with `cudaMalloc` / `cudaMemcpy`
- Learn shared memory basics for inter-thread communication
  (`__shared__`, `__syncthreads()`)

The rest of the parts — debugging with `compute-sanitizer`, core algorithms
(reduction, scan, convolution, matmul), warp- and block-level programming,
profiling, tensor cores, and more — follow the same puzzle structure and are
being ported from the original Mojo course.

This book uniquely challenges the status quo by first building understanding with
low-level memory manipulation, then layering on lightweight abstractions. This
gives you both a deep understanding of GPU memory patterns and practical
knowledge of modern, readable kernel code.

## Ready to get started?

You now understand why GPU programming matters, why CUDA is a great way to learn
it, and how puzzle-based learning works. You're prepared to begin.

**Next step**: Head to [How to Use This Book](howto.md) for setup instructions,
system requirements, and guidance on running your first puzzle.
