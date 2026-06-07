// CUDA GPU Puzzles - TensorView introduction (book walkthrough)
//
// A minimal program showing how a TensorView gives a flat device buffer a
// natural 2D shape. We launch a single thread that prints the matrix, bumps
// one element with `t(0, 0) += 1`, and prints it again.
//
//   make tensor_view_intro
#include "puzzles.cuh"
#include "tensor_view.cuh"

constexpr int HEIGHT = 2;
constexpr int WIDTH = 3;

__global__ void demo(TensorView2D<float> t) {
    printf("Before:\n");
    for (int r = 0; r < t.rows; ++r) {
        for (int c = 0; c < t.cols; ++c) printf("%g ", t(r, c));
        printf("\n");
    }

    t(0, 0) += 1.0f;  // natural 2D indexing, no row * width + col by hand

    printf("After:\n");
    for (int r = 0; r < t.rows; ++r) {
        for (int c = 0; c < t.cols; ++c) printf("%g ", t(r, c));
        printf("\n");
    }
}

int main() {
    constexpr size_t bytes = HEIGHT * WIDTH * sizeof(float);

    float* d_buf = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buf, bytes));
    CUDA_CHECK(cudaMemset(d_buf, 0, bytes));  // start at all zeros

    TensorView2D<float> t(d_buf, HEIGHT, WIDTH);

    // A device tensor can't be printed from the host directly, so the work and
    // the printing both happen inside the kernel.
    demo<<<1, 1>>>(t);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaFree(d_buf));
    return 0;
}
