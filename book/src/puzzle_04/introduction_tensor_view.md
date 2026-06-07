# Introduction to TensorView

Let's take a quick break from solving puzzles to introduce a small abstraction
that will make our GPU programming journey more pleasant: the **`TensorView`**.

> 💡 _This is a motivational overview. Don't worry about understanding every
> detail now — we'll use these features as we progress through the puzzles._

## The challenge: Growing complexity

Let's look at the challenges we've faced so far:

```cpp
// Puzzle 1: Simple indexing
output[i] = a[i] + 10.0f;

// Puzzle 2: Multiple array management
output[i] = a[i] + b[i];

// Puzzle 3: Bounds checking
if (i < size) {
    output[i] = a[i] + 10.0f;
}
```

As dimensions grow, the index math grows with them:

```cpp
// Traditional 2D indexing for a row-major matrix
int idx = row * WIDTH + col;
if (row < height && col < width) {
    output[idx] = a[idx] + 10.0f;
}
```

## The solution: A peek at TensorView

`TensorView` packages a pointer together with its shape and exposes an
`operator()` that does the index arithmetic for you:

1. **Natural indexing**: write `tensor(i, j)` instead of `data[i * width + j]`
2. **Shape travels with the data**: the view knows its own rows and columns
3. **Zero overhead**: it compiles to the same row-major arithmetic you'd write
   by hand

The full (tiny) implementation lives in
[`include/tensor_view.cuh`]({{#include ../_includes/repo_url.md}}/blob/main/include/tensor_view.cuh):

```cpp
template <typename T>
struct TensorView2D {
    T* data;
    int rows, cols;

    __host__ __device__ TensorView2D(T* ptr, int r, int c)
        : data(ptr), rows(r), cols(c) {}

    __host__ __device__ T& operator()(int r, int c) const {
        return data[r * cols + c];  // the row-major math, written once
    }
};
```

> 💡 **Real-world note**: production CUDA code increasingly uses
> [`cuda::std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) (the
> C++23 `std::mdspan` made available for device code in `<cuda/std/mdspan>`).
> Our `TensorView` is a deliberately minimal stand-in so nothing is hidden.

## Basic usage

```cpp
#include "tensor_view.cuh"

constexpr int HEIGHT = 2;
constexpr int WIDTH = 3;

// Wrap a flat device buffer in a 2 x 3 view
TensorView2D<float> tensor(buffer, HEIGHT, WIDTH);

// Access elements naturally (inside a kernel)
tensor(0, 0) = 1.0f;  // first element
tensor(1, 2) = 2.0f;  // last element
```

## Quick example

Let's put everything together with a small program that creates a `2 x 3` view,
prints it, bumps one element, and prints it again:

```cpp
{{#include ./intro.cu}}
```

Run it with:

```bash
make tensor_view_intro
```

```txt
Before:
0 0 0
0 0 0
After:
1 0 0
0 0 0
```

Let's break down what's happening:

1. We create a `2 x 3` view over a zero-filled device buffer
2. The kernel prints it (a device buffer can't be read from the host directly)
3. Using natural indexing, we modify a single element with `t(0, 0) += 1`
4. The change is reflected when we print again

This simple example demonstrates the key benefits:

- Clean syntax for tensor access
- The shape stays attached to the data
- Natural multi-dimensional indexing with no runtime cost

While this example is straightforward, the same pattern scales to the more
complex GPU operations in upcoming puzzles: multi-block grids, shared-memory
tiling, and beyond.

💡 **Tip**: Keep this example in mind as we progress — we'll build on these
fundamentals to write increasingly sophisticated GPU programs.
