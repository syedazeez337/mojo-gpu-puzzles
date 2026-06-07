<p align="center">
  <img src="book/src/puzzles_images/puzzle-mark.svg" alt="CUDA GPU Puzzles Logo" width="150">
</p>

<p align="center">
  <h1 align="center">CUDA GPU Puzzles</h1>
</p>

<p align="center">
  <h3 align="center">Learn GPU Programming in CUDA C++ Through Interactive Puzzles 🧩</h3>
</p>

<p align="center">
  <a href="#overview"><strong>Overview</strong></a> •
  <a href="#why-cuda"><strong>Why CUDA</strong></a> •
  <a href="#getting-started"><strong>Getting Started</strong></a> •
  <a href="#development"><strong>Development</strong></a> •
  <a href="#acknowledgments"><strong>Acknowledgments</strong></a>
</p>

## Overview

> _"For the things we have to learn before we can do them, we learn by doing
> them."_ — Aristotle, (Nicomachean Ethics)

Welcome to **CUDA GPU Puzzles** — an interactive approach to learning GPU
programming through hands-on puzzle solving. Instead of traditional textbook
learning, you'll immediately dive into writing real CUDA C++ kernels, compiling
them with `nvcc`, and seeing the results come back from the hardware.

> 📌 **Port status.** This is a CUDA C++ port of Modular's
> [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles). **Part I
> (Puzzles 1–8) is fully converted** — problem stubs, solutions, and book
> chapters. The remaining parts are being ported puzzle-by-puzzle; until then,
> their chapters and source may still be the original Mojo. The GPU concepts are
> identical — only the language changes.

## Why CUDA

CUDA is NVIDIA's parallel computing platform and the most direct, mature way to
program NVIDIA GPUs. Learning it gives you the mental model that underlies every
higher-level GPU tool (PyTorch, cuBLAS, cuDNN, CUTLASS, …):

- ⚡ **Direct hardware access** to threads, blocks, warps, shared memory, and
  tensor cores
- 🧰 **A mature toolchain**: `nvcc`, `compute-sanitizer`, `cuda-gdb`, and Nsight
- 🧠 **Modern C++ on the device**: C++17/20 in kernels, plus the standard library
  via [libcu++](https://nvidia.github.io/cccl/libcudacxx/) (`cuda::std::`)
- 📚 **A huge ecosystem** built on the very primitives these puzzles teach
- 🎯 **Transferable skills**: the thread/block/grid model is the foundation under
  all GPU frameworks

## Getting Started

### Prerequisites

- An **NVIDIA GPU** (compute capability `sm_70`+; the `Makefile` defaults to
  `sm_86`)
- The **CUDA Toolkit** (provides `nvcc`; CUDA 12.x or 13.x)
- A compatible **C++ host compiler** and **`make`**

Verify your setup:

```bash
nvcc --version
nvidia-smi
```

### Run your first puzzle

```bash
# Clone the repository
git clone https://github.com/syedazeez337/mojo-gpu-puzzles
cd mojo-gpu-puzzles

# Run puzzle 1 (it fails until you implement it — that's the point!)
make p01
```

Edit `problems/p01/p01.cu`, fill in the kernel, and run `make p01` again until
you see `Puzzle 01 complete ✅`.

### Common commands

```bash
make p01             # build & run your implementation of a puzzle
make test-p01        # build & run the reference solution for one puzzle
make test            # build & run every reference solution
make build-all       # compile everything without running
make clean           # remove build artifacts

# Override the target GPU or C++ standard
make p01 ARCH=sm_90  # build for Hopper
make test STD=c++17
```

If your GPU isn't Ampere, set `ARCH` to match your card (see the
[CUDA GPUs list](https://developer.nvidia.com/cuda-gpus)).

## Project structure

```
problems/    # puzzle stubs with `// FILL ME IN` — this is where you work
solutions/   # reference solutions
include/     # shared headers: puzzles.cuh (CUDA_CHECK + helpers),
             #                 tensor_view.cuh (the TensorView abstraction)
book/        # the mdBook source for the written guide
Makefile     # per-puzzle build & run targets
```

## Development

The written guide is built with [mdBook](https://rust-lang.github.io/mdBook/).

```bash
# Serve the book locally with live reload
cd book && mdbook serve --open

# Run the full CUDA test suite
make test

# Debug a kernel for memory/race errors
compute-sanitizer ./build/sol_p03
```

## Contributing

Contributions are welcome — improving explanations, fixing bugs, or porting more
puzzles to CUDA. Feel free to fork, branch, and open a pull request.

## Acknowledgments

- This project is a CUDA C++ port of Modular's
  [Mojo GPU Puzzles](https://github.com/modular/mojo-gpu-puzzles).
- The initial puzzles are heavily inspired by
  [GPU Puzzles](https://github.com/srush/GPU-Puzzles) by Sasha Rush.
- Built with [mdBook](https://rust-lang.github.io/mdBook/).

## License

This project is licensed under the LLVM License — see the [LICENSE](LICENSE)
file for details.
