# Puzzle 2: Zip

## Overview

Implement a CUDA kernel that adds together each position of vector `a` and
vector `b` and stores it in `output`.

**Note:** _You have 1 thread per position._

<img src="./media/02.png" alt="Zip" class="light-mode-img">
<img src="./media/02d.png" alt="Zip" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- Processing multiple input arrays in parallel
- Element-wise operations with multiple inputs
- Thread-to-data mapping across arrays
- Memory access patterns with multiple arrays

For each thread \\(i\\): \\[\Large output[i] = a[i] + b[i]\\]

### Memory access pattern

```txt
Thread 0:  a[0] + b[0] → output[0]
Thread 1:  a[1] + b[1] → output[1]
Thread 2:  a[2] + b[2] → output[2]
...
```

💡 **Note**: Notice how we're now managing three arrays (`a`, `b`, `output`) in
our kernel. Each gets its own `cudaMalloc` / `cudaMemcpy` on the host side, but
inside the kernel the indexing stays just as simple as Puzzle 1.

## Code to complete

```cpp
{{#include ../../../problems/p02/p02.cu:add}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p02/p02.cu" class="filename">View full file: problems/p02/p02.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `threadIdx.x` in `i`
2. Add `a[i]` and `b[i]`
3. Store the result in `output[i]`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p02
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, 0]
expected: [0, 2, 4, 6]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p02/p02.cu:add_solution}}
```

<div class="solution-explanation">

This solution:

- Gets the thread index with `int i = threadIdx.x`
- Adds values from both arrays: `output[i] = a[i] + b[i]`

</div>
</details>

### Looking ahead

While this direct indexing works for simple element-wise operations, consider:

- What if arrays have different shapes (different number of rows/columns)?
- What if we need to broadcast one array against another?
- How do we keep multi-dimensional index math readable?

These questions are addressed when we
[introduce TensorView in Puzzle 4](../puzzle_04/introduction_tensor_view.md).
