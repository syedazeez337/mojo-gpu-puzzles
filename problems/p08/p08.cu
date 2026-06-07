// CUDA GPU Puzzles - Puzzle 8: Shared Memory
//
// Add 10 to each element of `a`, but route the data through SHARED MEMORY first.
// Shared memory is fast, on-chip storage private to each block. The pattern is:
//   1. each thread copies one element from global memory into shared memory,
//   2. __syncthreads() so every thread in the block has finished loading,
//   3. each thread reads from shared memory and writes the result back out.
//
// Educational note: in THIS puzzle the barrier is not strictly required (each
// thread only touches its own slot), but loading-into-shared then syncing is
// the canonical pattern you will reuse constantly, so we practice it here.
//
//   make p08        build & run this file (fails until you fill it in)
//   make test-p08   build & run the reference solution
#include "puzzles.cuh"

constexpr int TPB = 4;   // threads per block (and shared-memory size)
constexpr int SIZE = 8;  // total elements
constexpr int BLOCKS_PER_GRID = 2;

// ANCHOR: add_10_shared
__global__ void add_10_shared(float* output, const float* a, int size) {
    // Statically-sized shared memory, one slot per thread in the block.
    __shared__ float shared[TPB];

    int global_i = blockDim.x * blockIdx.x + threadIdx.x;
    int local_i = threadIdx.x;

    if (global_i < size) {
        shared[local_i] = a[global_i];
    }

    // Wait until every thread in this block has finished its load above.
    __syncthreads();

    // FILL ME IN (roughly 2 lines)
}
// ANCHOR_END: add_10_shared

int main() {
    constexpr size_t bytes = SIZE * sizeof(float);

    float h_a[SIZE], h_expected[SIZE], h_out[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = 1.0f;
        h_expected[i] = 11.0f;
    }

    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    add_10_shared<<<BLOCKS_PER_GRID, TPB>>>(d_out, d_a, SIZE);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, SIZE);
    puzzles::print_vector("expected", h_expected, SIZE);
    puzzles::expect_equal(h_out, h_expected, SIZE);
    std::printf("Puzzle 08 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
