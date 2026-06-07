## How to Use This Book

Each puzzle maintains a consistent structure to support systematic skill
development:

- **Overview**: problem definition and key concepts for each challenge
- **Configuration**: technical setup and memory organization details
- **Code to Complete**: an implementation framework in `problems/pXX/` with a
  clearly marked `// FILL ME IN` section
- **Tips**: strategic hints available when needed, without revealing the full
  solution
- **Solution**: a complete implementation with explanation

The puzzles increase in complexity systematically, building new concepts on
established foundations. Working through them sequentially is recommended, as
later puzzles assume familiarity with concepts from earlier ones.

## Prerequisites

### Hardware

You need an **NVIDIA GPU**. The puzzles target compute capability `sm_70`
(Volta) and newer; the defaults in the `Makefile` build for `sm_86` (Ampere,
e.g. RTX 30-series / A-series). Set `ARCH` to match your card — find your
card's value in the
[CUDA GPUs list](https://developer.nvidia.com/cuda-gpus).

### Software

- **The CUDA Toolkit** (which provides `nvcc`). Part I was developed and tested
  against CUDA 13.x, but CUDA 12.x works fine too. Install it from the
  [CUDA Toolkit downloads](https://developer.nvidia.com/cuda-downloads) or your
  distribution's packages.
- **A C++ host compiler** compatible with your `nvcc` (recent GCC or Clang on
  Linux). The puzzles use C++20 by default.
- **`make`** to drive the builds.

Verify your setup:

```bash
nvcc --version     # should print your CUDA version
nvidia-smi         # should list your GPU
```

## Operating system notes

### Windows (WSL2)

To use an NVIDIA GPU from the Windows Subsystem for Linux, follow NVIDIA's
[CUDA on WSL guide](https://docs.nvidia.com/cuda/wsl-user-guide/index.html). The
key point: install the **NVIDIA Windows driver** on the Windows host (it exposes
the GPU to WSL as `libcuda.so`), then install the **CUDA Toolkit** *inside* WSL —
do **not** install a Linux GPU driver inside WSL.

Verify from inside WSL:

```bash
nvidia-smi
nvcc --version
```

### Linux (native)

Install a recent NVIDIA driver and the CUDA Toolkit:

```bash
# Check your GPU and OS version
lspci | grep -i nvidia
lsb_release -a

# Install the driver (Ubuntu example)
sudo ubuntu-drivers autoinstall
sudo reboot
```

Then install the CUDA Toolkit from the
[downloads page](https://developer.nvidia.com/cuda-downloads) and make sure
`nvcc` is on your `PATH`.

## Programming knowledge

Basic familiarity with:

- Programming fundamentals (variables, loops, conditionals, functions)
- **C/C++ basics** — pointers, arrays, and `const` in particular
- Parallel computing concepts (threads, synchronization, race conditions) are
  helpful but not required

No prior GPU programming experience is necessary — we'll build that knowledge
through the puzzles.

## Setting up your environment

Clone the repository and navigate into it:

```bash
git clone https://github.com/syedazeez337/mojo-gpu-puzzles
cd mojo-gpu-puzzles
```

Run your first puzzle (it fails until you implement it — that's expected):

```bash
make p01
```

## Working with puzzles

### Project structure

- **`problems/`** — where you implement your solutions (this is where you work!)
- **`solutions/`** — reference solutions used throughout the book
- **`include/`** — shared headers: `puzzles.cuh` (the test-harness helpers and
  the `CUDA_CHECK` macro) and `tensor_view.cuh` (the `TensorView` abstraction)
- **`Makefile`** — per-puzzle build and run targets

### Workflow

1. Open `problems/pXX/pXX.cu` and find the `// FILL ME IN` marker
2. Implement the kernel
3. Build and run it: `make pXX`
4. Compare against `solutions/pXX/pXX.cu` (or `make test-pXX`)

### Essential commands

```bash
# Run your implementation of a puzzle (fails until solved)
make p01

# Run the reference solution for one puzzle
make test-p01

# Run every reference solution (the full test suite)
make test

# Compile everything without running (quick "does it build?" check)
make build-all

# Remove build artifacts
make clean

# Override the GPU architecture or C++ standard
make p01 ARCH=sm_90      # build for Hopper
make test STD=c++17
```

Puzzle 4 has an extra variant and a standalone demo:

```bash
make p04                 # raw-pointer 2D map
make p04_tensor_view     # the TensorView version
make tensor_view_intro   # the TensorView walkthrough program
```

## Debugging and profiling

The CUDA Toolkit ships tools you'll use in later parts:

- **`compute-sanitizer`** — catches out-of-bounds accesses, race conditions, and
  uninitialized memory:

  ```bash
  compute-sanitizer ./build/sol_p03
  ```

- **`cuda-gdb`** — a source-level debugger for kernels
- **Nsight Systems / Nsight Compute** — timeline and kernel-level profilers

## Free cloud GPUs

If you don't have a local NVIDIA GPU, free options include **Google Colab** and
**Kaggle Notebooks**, which provide CUDA-capable GPUs (typically a Tesla T4).
Both can compile and run these puzzles with `nvcc` — install/verify the toolkit
in the notebook, then use the same `make` commands.
