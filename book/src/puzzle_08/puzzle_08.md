# Puzzle 8: Shared Memory

## Overview

Implement a CUDA kernel that adds 10 to each position of vector `a` and stores
it in `output`, routing the data through **shared memory**.

**Shared memory** is fast, on-chip storage that is visible to all threads within
the same block. Unlike global memory (which all blocks can access but is slow),
shared memory has latency closer to a register/L1 cache. Each block gets its own
private shared memory region — threads in one block cannot see the shared memory
of another block. Because threads can read and write the same shared locations,
coordination via `__syncthreads()` is required to prevent one thread from
reading a value before another thread has finished writing it.

**Note:** _You have fewer threads per block than the size of `a`._

<img src="./media/08.png" alt="Shared memory visualization" class="light-mode-img">
<img src="./media/08d.png" alt="Shared memory visualization" class="dark-mode-img">

## Key concepts

In this puzzle, you'll learn about:

- Declaring shared memory with `__shared__`
- Thread synchronization with `__syncthreads()`
- The load → sync → compute pattern for block-local data

## Configuration

- Array size: `SIZE = 8` elements
- Threads per block: `TPB = 4`
- Number of blocks: 2
- Shared memory: `TPB` elements per block

> **Warning**: A statically-declared shared array (`__shared__ float s[TPB];`)
> needs a **compile-time constant** size, not a runtime variable. After writing
> to shared memory you must call
> [`__syncthreads()`](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#synchronization-functions)
> before another thread reads those values. (For runtime-sized shared memory,
> CUDA offers `extern __shared__` plus a third launch parameter — we'll meet that
> later.)

**Educational note**: In this specific puzzle the `__syncthreads()` isn't
strictly necessary since each thread only accesses its own shared memory slot.
However, it's included to teach the proper synchronization pattern for the more
complex scenarios — coming up soon — where threads need to read data their
neighbors wrote.

## Code to complete

```cpp
{{#include ../../../problems/p08/p08.cu:add_10_shared}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p08/p08.cu" class="filename">View full file: problems/p08/p08.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. The shared array and the load + `__syncthreads()` are already written for you
2. After the barrier, guard against out-of-bounds: `if (global_i < size)`
3. Inside the guard, read from shared memory and write the result:
   `output[global_i] = shared[local_i] + 10.0f`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p08
```

Your output will look like this if the puzzle isn't solved yet:

```txt
out: [0, 0, 0, 0, 0, 0, 0, 0]
expected: [11, 11, 11, 11, 11, 11, 11, 11]
```

## Solution

<details class="solution-details">
<summary></summary>

```cpp
{{#include ../../../solutions/p08/p08.cu:add_10_shared_solution}}
```

<div class="solution-explanation">

This solution demonstrates the canonical shared-memory workflow:

1. **Memory hierarchy**
   - Global memory: `a` and `output` (large, slow, visible to all blocks)
   - Shared memory: `shared` (small, fast, private to each block)
   - Example for 8 elements with 4 threads per block:

     ```txt
     Global array a: [1 1 1 1 | 1 1 1 1]  # input: all ones

     Block 0:           Block 1:
     shared[0..3]       shared[0..3]
     [1 1 1 1]          [1 1 1 1]
     ```

2. **Thread coordination**
   - Load phase:

     ```txt
     Thread 0: shared[0] = a[0]=1    Thread 2: shared[2] = a[2]=1
     Thread 1: shared[1] = a[1]=1    Thread 3: shared[3] = a[3]=1
     __syncthreads()   # wait for all loads to finish
     ```

   - Compute phase: each thread adds 10 to its shared value
   - Result: `output[global_i] = shared[local_i] + 10 = 11`

**Note**: In this specific case the `__syncthreads()` isn't strictly necessary
since each thread only writes and reads its own slot (`shared[local_i]`). It's
included to demonstrate the pattern that *is* essential when threads read each
other's data.

3. **Declaring shared memory**

   ```cpp
   __shared__ float shared[TPB];  // TPB must be a compile-time constant
   ```

4. **Memory access pattern**
   - Load: global → shared
   - Sync: `__syncthreads()`
   - Compute: add 10 to the shared values
   - Store: shared → global

This load/sync/compute structure is the backbone of nearly every high-performance
GPU kernel you'll write — from reductions and convolutions to tiled matrix
multiplication.

</div>
</details>
