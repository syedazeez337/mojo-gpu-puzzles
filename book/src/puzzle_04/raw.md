## Overview

Implement a CUDA kernel that adds 10 to each position of a 2D square matrix `a`
and stores it in a 2D square matrix `output`.

**Note:** _You have more threads than positions_.

## Key concepts

In this puzzle, you'll learn about:

- Working with 2D thread indices (`threadIdx.x`, `threadIdx.y`)
- Converting 2D coordinates to 1D memory indices
- Handling boundary checks in two dimensions

The key insight is understanding how to map from 2D thread coordinates
\\((row,col)\\) to elements in a row-major matrix of size \\(n \times n\\), while
ensuring thread indices are within bounds.

- **2D indexing**: Each thread has a unique \\((row,col)\\) position
- **Memory layout**: Row-major ordering maps 2D to 1D memory: `row * size + col`
- **Guard condition**: We need bounds checking in both dimensions
- **Thread bounds**: More threads \\((3 \times 3)\\) than matrix elements \\((2
  \times 2)\\)

### Launching a 2D block

CUDA lets you shape blocks and grids in up to three dimensions with `dim3`. Here
we use a single block of `3 x 3` threads:

```cpp
dim3 threads_per_block(3, 3);  // .x = columns, .y = rows
add_10_2d<<<1, threads_per_block>>>(d_out, d_a, SIZE);
```

## Code to complete

```cpp
{{#include ../../../problems/p04/p04.cu:add_10_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p04/p04.cu" class="filename">View full file: problems/p04/p04.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Get 2D indices: `row = threadIdx.y`, `col = threadIdx.x`
2. Add the guard: `if (row < size && col < size)`
3. Inside the guard, add 10 the row-major way: `output[row * size + col] = ...`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p04
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
{{#include ../../../solutions/p04/p04.cu:add_10_2d_solution}}
```

<div class="solution-explanation">

This solution:

1. Gets 2D indices: `row = threadIdx.y`, `col = threadIdx.x`
2. Adds the guard: `if (row < size && col < size)`
3. Inside the guard: `output[row * size + col] = a[row * size + col] + 10.0f`

The `row * size + col` expression is the row-major flattening: row 0 occupies
the first `size` slots, row 1 the next `size`, and so on. Keep this formula in
mind — the next chapter hides it behind a cleaner interface.

</div>
</details>
