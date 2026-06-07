# Puzzle 3: Guards

## Overview

Implement a CUDA kernel that adds 10 to each position of vector `a` and stores
it in vector `output`.

**Note**: _You have more threads than positions. This means you need to protect
against out-of-bounds memory access._

<img src="./media/03.png" alt="Guard" class="light-mode-img">
<img src="./media/03d.png" alt="Guard" class="dark-mode-img">

## Key concepts

This puzzle covers:

- Handling thread/data size mismatches
- Preventing out-of-bounds memory access
- Using conditional execution in GPU kernels
- Safe memory access patterns

### Mathematical description

For each thread \\(i\\):
\\[\Large \text{if}\\ i < \text{size}: output[i] = a[i] + 10\\]

### Memory safety pattern

```txt
Thread 0 (i=0):  if 0 < size:  output[0] = a[0] + 10  ✓ Valid
Thread 1 (i=1):  if 1 < size:  output[1] = a[1] + 10  ✓ Valid
Thread 2 (i=2):  if 2 < size:  output[2] = a[2] + 10  ✓ Valid
Thread 3 (i=3):  if 3 < size:  output[3] = a[3] + 10  ✓ Valid
Thread 4 (i=4):  if 4 < size:  ❌ Skip (out of bounds)
Thread 5 (i=5):  if 5 < size:  ❌ Skip (out of bounds)
```

In CUDA you almost always launch a "round" number of threads (a multiple of the
warp size, 32) even when your data isn't a multiple of that. The bounds check is
what makes the extra threads harmless.

## Code to complete

```cpp
{{#include ../../../problems/p03/p03.cu:add_10_guard}}
```

<a href="{{#include ../_includes/repo_url.md}}/blob/main/problems/p03/p03.cu" class="filename">View full file: problems/p03/p03.cu</a>

<details>
<summary><strong>Tips</strong></summary>

<div class="solution-tips">

1. Store `threadIdx.x` in `i`
2. Add the guard: `if (i < size)`
3. Inside the guard: `output[i] = a[i] + 10.0f`

</div>
</details>

## Running the code

To test your solution, run the following command in your terminal:

```bash
make p03
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
{{#include ../../../solutions/p03/p03.cu:add_10_guard_solution}}
```

<div class="solution-explanation">

This solution:

- Gets the thread index with `int i = threadIdx.x`
- Guards against out-of-bounds access with `if (i < size)`
- Inside the guard: adds 10 to the input value

> You might wonder why a kernel can sometimes pass the test even *without* the
> bounds check. Passing the test doesn't mean the code is sound and free of
> undefined behavior — an out-of-bounds write may land on memory you don't own
> and silently corrupt it. In [Puzzle 10](../puzzle_10/puzzle_10.md) we'll use
> `compute-sanitizer` to catch exactly these soundness bugs.

</div>
</details>

### Looking ahead

While simple boundary checks work here, consider these challenges:

- What about 2D/3D array boundaries?
- How do we handle different shapes efficiently?
- What if we need padding or edge handling?

Example of growing complexity:

```cpp
// Current: 1D bounds check
if (i < size) { ... }

// Coming soon: 2D bounds check
if (row < height && col < width) { ... }

// Later: 3D with padding
if (row < height && col < width && depth < d &&
    row >= pad && col >= pad) { ... }
```

These boundary-handling patterns stay manageable when we
[learn about TensorView in Puzzle 4](../puzzle_04/introduction_tensor_view.md),
which keeps the shape alongside the data.
