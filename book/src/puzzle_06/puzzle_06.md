# Puzzle 6: Blocks

## Overview

Implement a CUDA kernel that adds 10 to each position of vector `a` and stores
it in `output`.

A **thread block** (or just **block**) is a group of threads that execute
together on a single GPU streaming multiprocessor (SM). All threads in a block
share the same shared memory and can synchronize with each other. When the data
is larger than one block can handle, the GPU schedules multiple blocks — each
block independently processes its portion of the data. The global position of a
thread is computed from both its position within the block (`threadIdx.x`) and
which block it belongs to (`blockIdx.x`):

```txt
global_i = blockDim.x * blockIdx.x + threadIdx.x
```

**Note:** _You have fewer threads per block than the size of `a`._

<img src="./media/06.png" alt="Blocks visualization" class="light-mode-img">
<img src="./media/06d.png" alt="Blocks visualization" class="dark-mode-img">

## Key concepts

This puzzle covers:

- Processing data larger than a single thread block
- Coordinating multiple blocks of threads
- Computing global thread positions

The key insight is understanding how blocks of threads work together to process
data that's larger than a single block's capacity, while maintaining a correct
element-to-thread mapping.

## Code to complete

```cpp
{{#include ../../../problems/p06/p06.cu:add_10_blocks}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p06/p06.cu" class="filename">View full file: problems/p06/p06.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Calculate the global index: `i = blockDim.x * blockIdx.x + threadIdx.x`
2. Add the guard: `if (i < size)`
3. Inside the guard: `output[i] = a[i] + 10.0f`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p06
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, 0, 0, 0, 0, 0, 0]
expected: [10, 11, 12, 13, 14, 15, 16, 17, 18]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p06/p06.cu:add_10_blocks_solution}}
```

<div class="solution-explanation">

This solution covers the key concepts of block-based GPU processing:

1. **Global thread indexing**
   - Combines block and thread indices:
     `blockDim.x * blockIdx.x + threadIdx.x`
   - Maps each thread to a unique global position
   - Example for 4 threads per block:

     ```txt
     Block 0: [0 1 2 3]
     Block 1: [4 5 6 7]
     Block 2: [8 ...   ]
     ```

2. **Block coordination**
   - Each block processes a contiguous chunk of data
   - Block size (4) < data size (9), so we need multiple blocks
   - Automatic work distribution across blocks:

     ```txt
     Data:    [0 1 2 3 4 5 6 7 8]
     Block 0: [0 1 2 3]
     Block 1:         [4 5 6 7]
     Block 2:                 [8]
     ```

3. **Bounds checking**
   - The guard `i < size` handles edge cases
   - Prevents out-of-bounds access when `size` isn't a multiple of the block
     size (here 3 blocks × 4 threads = 12 threads for 9 elements)
   - Essential for handling the partial block at the end of the data

4. **Memory access pattern**
   - Coalesced memory access: threads in a block read contiguous addresses,
     which the hardware can service in a single wide memory transaction
   - Each thread processes one element: `output[i] = a[i] + 10.0f`

This pattern forms the foundation for processing large datasets that exceed the
size of a single thread block.

</div>
</details>
