# Puzzle 5: Broadcast

## Overview

Implement a CUDA kernel that broadcast-adds a 1D row vector `a` and a 1D column
vector `b` and stores the result in a 2D matrix `output`.

**Broadcasting** in parallel programming refers to the operation where
lower-dimensional arrays are automatically expanded to match the shape of
higher-dimensional arrays during element-wise operations. Instead of physically
replicating data in memory, values are logically repeated across the additional
dimensions. For example, adding a row vector to each row of a matrix applies the
same vector elements repeatedly without creating multiple copies.

**Note:** _You have more threads than positions._

<img src="./media/05.png" alt="Broadcast visualization" class="light-mode-img">
<img src="./media/05d.png" alt="Broadcast visualization" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- Broadcasting 1D vectors across different dimensions with `TensorView`
- Using 2D thread indices to map GPU threads to a 2D output matrix
- Working with different tensor shapes for mixed-dimension operations
- Handling boundary conditions in broadcast patterns

The key insight is that a row-major `TensorView2D` of shape \\((1, n)\\) reads
the same element for every row, and a shape of \\((n, 1)\\) reads the same
element for every column — exactly what broadcasting needs.

- **Tensor shapes**: Input vectors have shapes \\((1, n)\\) and \\((n, 1)\\)
- **Broadcasting**: Each element of `a` combines with each element of `b`; the
  output expands both dimensions to \\((n,n)\\)
- **Access patterns**: `a(0, col)` broadcasts horizontally across rows;
  `b(row, 0)` broadcasts vertically across columns
- **Guard condition**: Still need bounds checking for the output size
- **Thread bounds**: More threads \\((3 \times 3)\\) than tensor elements \\((2
  \times 2)\\)

> 💡 **Why this works**: a `TensorView2D(d_a, 1, SIZE)` computes the address as
> `0 * SIZE + col = col`, so every row sees the same row vector. A
> `TensorView2D(d_b, SIZE, 1)` computes `row * 1 + 0 = row`, so every column sees
> the same column vector. No data is duplicated.

## Code to complete

```cpp
{{#include ../../../problems/p05/p05.cu:broadcast_add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p05/p05.cu" class="filename">View full file: problems/p05/p05.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Get 2D indices: `row = threadIdx.y`, `col = threadIdx.x`
2. Add the guard: `if (row < size && col < size)`
3. Inside the guard, broadcast: combine `a(0, col)` with `b(row, 0)`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p05
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, 0]
expected: [1, 2, 11, 12]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p05/p05.cu:broadcast_add_solution}}
```

<div class="solution-explanation">

This solution demonstrates the key concepts of broadcasting and GPU thread
mapping:

1. **Thread to matrix mapping**

   - Uses `threadIdx.y` for the row and `threadIdx.x` for the column
   - Natural 2D indexing matches the output matrix structure
   - Excess threads (3×3 grid) are handled by bounds checking

2. **Broadcasting mechanics**
   - Input `a` has shape `(1, n)`: `a(0, col)` broadcasts across rows
   - Input `b` has shape `(n, 1)`: `b(row, 0)` broadcasts across columns
   - Output has shape `(n, n)`: each element is the sum of the corresponding
     broadcasts

   ```txt
   [ a0 a1 ]  +  [ b0 ]  =  [ a0+b0  a1+b0 ]
                 [ b1 ]     [ a0+b1  a1+b1 ]
   ```

3. **Bounds checking**
   - The guard `row < size && col < size` prevents out-of-bounds access
   - Handles both the matrix bounds and the excess threads efficiently

This pattern forms the foundation for more complex tensor operations we'll
explore in later puzzles.

</div>
</details>
