# TensorView Version

## Overview

Implement a CUDA kernel that adds 10 to each position of a 2D `TensorView` `a`
and stores it in a 2D `TensorView` `output`.

**Note:** _You have more threads than positions_.

## Key concepts

In this puzzle, you'll learn about:

- Using `TensorView` for 2D array access
- Direct 2D indexing with `tensor(i, j)`
- Handling bounds checking with `TensorView`

The key insight is that `TensorView` provides a natural 2D indexing interface,
abstracting away the row-major math while still requiring bounds checking.

- **2D access**: Natural \\((row,col)\\) indexing with `tensor(row, col)`
- **Memory abstraction**: No manual `row * size + col` calculation needed
- **Guard condition**: Still need bounds checking in both dimensions
- **Thread bounds**: More threads \\((3 \times 3)\\) than tensor elements \\((2
  \times 2)\\)

The view is created on the host from the raw device pointer and passed to the
kernel **by value** (it's just a pointer plus two ints):

```cpp
TensorView2D<float> out_view(d_out, SIZE, SIZE);
TensorView2D<const float> a_view(d_a, SIZE, SIZE);
add_10_2d<<<1, dim3(3, 3)>>>(out_view, a_view, SIZE);
```

## Code to complete

```cpp
{{#include ../../../problems/p04/p04_tensor_view.cu:add_10_2d_tensor_view}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04_tensor_view.cu" class="filename">View full file: problems/p04/p04_tensor_view.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Get 2D indices: `row = threadIdx.y`, `col = threadIdx.x`
2. Add the guard: `if (row < size && col < size)`
3. Inside the guard, add 10 to `a(row, col)`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p04_tensor_view
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, 0]
expected: [10, 11, 12, 13]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p04/p04_tensor_view.cu:add_10_2d_tensor_view_solution}}
```

<div class="solution-explanation">

This solution:

- Gets 2D thread indices with `row = threadIdx.y`, `col = threadIdx.x`
- Guards against out-of-bounds access with `if (row < size && col < size)`
- Uses `TensorView`'s 2D indexing: `output(row, col) = a(row, col) + 10.0f`

Compare this with the [raw version](./raw.md): the kernel body is the same shape,
but `output(row, col)` reads more like the math than `output[row * size + col]`.
Both compile to identical instructions.

</div>
</details>
