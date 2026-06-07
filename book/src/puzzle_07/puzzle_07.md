# Puzzle 7: 2D Blocks

## Overview

Implement a CUDA kernel that adds 10 to each position of a 2D `TensorView` `a`
and stores it in a 2D `TensorView` `output`.

**Note:**
_You have fewer threads per block than the size of `a` in both directions._

<img src="./media/07.png" alt="Blocks 2D visualization" class="light-mode-img">
<img src="./media/07d.png" alt="Blocks 2D visualization" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- Using `TensorView` with multiple blocks
- Handling large matrices with 2D block organization
- Combining block indexing with `TensorView` access

The key insight is that `TensorView` simplifies 2D indexing while you still need
proper block coordination to cover a matrix larger than one block.

> 🔑 **2D thread indexing convention**
>
> We extend the block-based indexing from [Puzzle 6](../puzzle_06/puzzle_06.md)
> to 2D:
>
> ```txt
> Global position calculation:
> row = blockDim.y * blockIdx.y + threadIdx.y
> col = blockDim.x * blockIdx.x + threadIdx.x
> ```
>
> For example, with 2×2 blocks in a 4×4 grid:
>
> ```txt
> Block (0,0):   Block (1,0):
> [0,0  0,1]     [0,2  0,3]
> [1,0  1,1]     [1,2  1,3]
>
> Block (0,1):   Block (1,1):
> [2,0  2,1]     [2,2  2,3]
> [3,0  3,1]     [3,2  3,3]
> ```
>
> Each position shows (row, col) for that thread's global index.
> The block dimensions and indices work together to ensure:
>
> - Continuous coverage of the 2D space
> - No overlap between blocks
> - Efficient memory access patterns

## Configuration

- **Matrix size**: \\(5 \times 5\\) elements
- **Block coordination**: a \\(2 \times 2\\) grid of \\(3 \times 3\\) blocks
- **2D indexing**: natural \\((row,col)\\) access with bounds checking
- **Total threads**: \\(36\\) for \\(25\\) elements
- **Thread mapping**: each thread processes one matrix element

## Code to complete

```cpp
{{#include ../../../problems/p07/p07.cu:add_10_blocks_2d}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p07/p07.cu" class="filename">View full file: problems/p07/p07.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Global indices: `row = blockDim.y * blockIdx.y + threadIdx.y`,
   `col = blockDim.x * blockIdx.x + threadIdx.x`
2. Add the guard: `if (row < size && col < size)`
3. Inside the guard, add 10 to `a(row, col)`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p07
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, ... , 0]
expected: [10, 11, 12, ... , 34]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p07/p07.cu:add_10_blocks_2d_solution}}
```

<div class="solution-explanation">

This solution shows how `TensorView` simplifies 2D block-based processing:

1. **2D thread indexing**
   - Global row: `blockDim.y * blockIdx.y + threadIdx.y`
   - Global col: `blockDim.x * blockIdx.x + threadIdx.x`
   - Maps the thread grid onto the tensor elements:

     ```txt
     5×5 tensor with 3×3 blocks:

     Block (0,0)         Block (1,0)
     [(0,0) (0,1) (0,2)] [(0,3) (0,4)    *  ]
     [(1,0) (1,1) (1,2)] [(1,3) (1,4)    *  ]
     [(2,0) (2,1) (2,2)] [(2,3) (2,4)    *  ]

     Block (0,1)         Block (1,1)
     [(3,0) (3,1) (3,2)] [(3,3) (3,4)    *  ]
     [(4,0) (4,1) (4,2)] [(4,3) (4,4)    *  ]
     [  *     *     *  ] [  *     *      *  ]
     ```

     (* = thread exists but is outside the tensor bounds)

2. **TensorView benefits**
   - Natural 2D indexing: `tensor(row, col)` instead of manual offset
     calculation
   - The shape travels with the data, so the kernel signature documents the
     intent

3. **Bounds checking**
   - The guard `row < size && col < size` handles:
     - Excess threads in partial blocks
     - Edge cases at the tensor boundaries
     - 36 threads (2×2 blocks of 3×3) for 25 elements

4. **Block coordination**
   - Each 3×3 block processes part of the 5×5 tensor
   - The grid + block dimensions together tile the whole matrix without gaps or
     overlaps

This pattern shows how `TensorView` keeps 2D block processing readable while you
manage the thread/block coordination explicitly.

</div>
</details>
