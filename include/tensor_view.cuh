// CUDA GPU Puzzles — a tiny multidimensional view abstraction.
//
// Raw CUDA kernels address memory with hand-written index math:
//
//     output[row * width + col] = a[row * width + col] + 10.0f;
//
// That works, but it becomes error prone the moment shapes, strides, or the
// number of dimensions change. This header introduces a minimal `TensorView`
// so you can instead write:
//
//     output(row, col) = a(row, col) + 10.0f;
//
// while compiling down to the exact same arithmetic. There is no hidden cost:
// a TensorView is just a pointer plus a couple of integers, passed to the
// kernel by value.
//
// This is the CUDA analogue of the higher-level "tensor"/"layout" types you
// will meet in real GPU libraries. For production code, the C++23 standard
// library offers `std::mdspan` — usable in device code as `cuda::std::mdspan`
// from <cuda/std/mdspan>. `TensorView` here is a deliberately stripped-down
// teaching version so the indexing has no magic left to hide.
#pragma once

// 1D row-major view over a flat buffer.
template <typename T>
struct TensorView1D {
    T* data;
    int n;

    __host__ __device__ TensorView1D(T* ptr, int size) : data(ptr), n(size) {}

    __host__ __device__ T& operator()(int i) const { return data[i]; }
    __host__ __device__ int size() const { return n; }
};

// 2D row-major view: element (r, c) lives at data[r * cols + c].
//
// A shape of (1, N) behaves as a row vector and (N, 1) as a column vector, so
// the same type doubles as a broadcastable vector — see Puzzle 5, where a
// (1, N) input and an (N, 1) input combine into an (N, N) output.
template <typename T>
struct TensorView2D {
    T* data;
    int rows;
    int cols;

    __host__ __device__ TensorView2D(T* ptr, int r, int c)
        : data(ptr), rows(r), cols(c) {}

    __host__ __device__ T& operator()(int r, int c) const {
        return data[r * cols + c];
    }
};
