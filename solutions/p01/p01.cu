// CUDA GPU Puzzles - Puzzle 1: Map (reference solution)
//
//   make test-p01   build & run this solution
#include "puzzles.cuh"

constexpr int SIZE = 4;
constexpr int THREADS_PER_BLOCK = SIZE;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: add_10_solution
__global__ void add_10(float* output, const float* a) {
    int i = threadIdx.x;
    output[i] = a[i] + 10.0f;
}
// ANCHOR_END: add_10_solution

int main() {
    constexpr size_t bytes = SIZE * sizeof(float);

    float h_a[SIZE], h_expected[SIZE], h_out[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = static_cast<float>(i);
        h_expected[i] = h_a[i] + 10.0f;
    }

    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    add_10<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_out, d_a);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, SIZE);
    puzzles::print_vector("expected", h_expected, SIZE);
    puzzles::expect_equal(h_out, h_expected, SIZE);
    std::printf("Puzzle 01 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
