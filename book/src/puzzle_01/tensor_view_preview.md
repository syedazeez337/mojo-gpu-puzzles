## Why consider a TensorView?

Looking at our raw implementation below, you might notice some potential issues:

### Current approach

```cpp
int i = threadIdx.x;
output[i] = a[i] + 10.0f;
```

This works for 1D arrays, but what happens when we need to:

- Handle 2D or 3D data?
- Deal with different memory layouts?
- Keep the index math readable and hard to get wrong?

### Preview of future challenges

As we progress through the puzzles, array indexing becomes more complex:

```cpp
// 2D indexing coming in later puzzles
int idx = row * WIDTH + col;

// 3D indexing
int idx = (batch * HEIGHT + row) * WIDTH + col;

// With padding
int idx = (batch * padded_height + row) * padded_width + col;
```

### TensorView preview

A small `TensorView` helper (see [`include/tensor_view.cuh`](
{{#include ../_includes/repo_url.md}}/blob/main/include/tensor_view.cuh)) lets us
write these cases more elegantly:

```cpp
// Future preview - don't worry about this syntax yet!
output(i, j) = a(i, j) + 10.0f;        // 2D indexing
output(b, i, j) = a(b, i, j) + 10.0f;  // 3D indexing
```

It is just a pointer plus a shape, with an `operator()` that does the row-major
math for you, so there is no runtime cost. In real-world CUDA code the C++23
standard library provides the same idea as
[`std::mdspan`](https://en.cppreference.com/w/cpp/container/mdspan) (usable in
device code via `cuda::std::mdspan`); our `TensorView` is a tiny teaching
version of that.

We'll learn about `TensorView` in detail in Puzzle 4, where these concepts
become essential. For now, focus on understanding:

- Basic thread indexing
- Simple memory access patterns
- One-to-one mapping of threads to data

💡 **Key Takeaway**: While direct indexing works for simple cases, we'll soon
want a more structured tool for complex GPU programming patterns.
