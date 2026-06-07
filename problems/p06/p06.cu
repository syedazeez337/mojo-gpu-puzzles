// CUDA GPU Puzzles - Puzzle 6: Blocks
//
// Add 10 to each element of `a` - but the data (SIZE = 9) is bigger than a
// single block (4 threads). The GPU launches multiple blocks, and each thread
// must compute its GLOBAL index from both its block and its position inside the
// block:
//     i = blockDim.x * blockIdx.x + threadIdx.x
//
// With more total threads than elements, you still need the bounds guard.
//
//   make p06        build & run this file (fails until you fill it in)
//   make test-p06   build & run the reference solution
#include "puzzles.cuh"

constexpr int SIZE = 9;
constexpr int THREADS_PER_BLOCK = 4;
constexpr int BLOCKS_PER_GRID = 3;  // 3 * 4 = 12 threads cover 9 elements

// ANCHOR: add_10_blocks
__global__ void add_10_blocks(float* output, const float* a, int size) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    // FILL ME IN (roughly 2 lines)
}
// ANCHOR_END: add_10_blocks

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

    add_10_blocks<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_out, d_a, SIZE);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, SIZE);
    puzzles::print_vector("expected", h_expected, SIZE);
    puzzles::expect_equal(h_out, h_expected, SIZE);
    std::printf("Puzzle 06 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
