## Key concepts

In this puzzle, you'll learn about:

- Basic CUDA kernel structure
- Thread indexing with `threadIdx.x`
- Simple parallel operations

- **Parallelism**: Each thread executes the kernel independently
- **Thread indexing**: Access element at position `i = threadIdx.x`
- **Memory access**: Read from `a[i]` and write to `output[i]`
- **Data independence**: Each output depends only on its corresponding input

## Anatomy of a CUDA program

Every puzzle is a normal C++ program compiled with `nvcc`. The GPU-specific
pieces are:

- A **kernel**: a function marked `__global__` that runs on the GPU. Every
  thread runs the same kernel body, distinguished only by its built-in indices
  (`threadIdx`, `blockIdx`, `blockDim`).
- A **launch**: the `kernel<<<blocks, threads>>>(args)` syntax that asks the GPU
  to run the kernel across a grid of threads.
- **Explicit memory movement**: the GPU has its own memory. We `cudaMalloc`
  device buffers, `cudaMemcpy` inputs from host (CPU) to device (GPU), launch,
  then `cudaMemcpy` the results back.

## Code to complete

```cpp
{{#include ../../../problems/p01/p01.cu:add_10}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p01/p01.cu" class="filename">View full file: problems/p01/p01.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `threadIdx.x` in `i`
2. Add 10 to `a[i]`
3. Store the result in `output[i]`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p01
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
{{#include ../../../solutions/p01/p01.cu:add_10_solution}}
```

<div class="solution-explanation">

This solution:

- Gets the thread index with `int i = threadIdx.x`
- Adds 10 to the input value: `output[i] = a[i] + 10.0f`

Because we launch exactly `SIZE` threads in a single block, thread `i` owns
element `i` — no loop required. The `f` suffix on `10.0f` keeps the arithmetic
in single precision, matching the `float` data.

</div>
</details>
