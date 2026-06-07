// CUDA GPU Puzzles - Puzzle 1: Map
//
// Add 10 to every element of vector `a`, writing the result into `output`.
// You get one thread per element. This is the "hello world" of GPU
// programming: the loop you would write on a CPU disappears, replaced by many
// threads each handling a single index.
//
//   make p01        build & run this file (fails until you fill it in)
//   make test-p01   build & run the reference solution
#include "puzzles.cuh"

constexpr int SIZE = 4;
constexpr int THREADS_PER_BLOCK = SIZE;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: add_10
__global__ void add_10(float* output, const float* a) {
    int i = threadIdx.x;
    // FILL ME IN (roughly 1 line)
}
// ANCHOR_END: add_10

int main() {
    constexpr size_t bytes = SIZE * sizeof(float);

    // Host data: input and the answer we expect to get back.
    float h_a[SIZE], h_expected[SIZE], h_out[SIZE];
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = static_cast<float>(i);
        h_expected[i] = h_a[i] + 10.0f;
    }

    // Device buffers.
    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    // Copy the input from host to device.
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    // Launch one thread per element.
    add_10<<<BLOCKS_PER_GRID, THREADS_PER_BLOCK>>>(d_out, d_a);
    CUDA_CHECK_KERNEL();

    // Copy the result back from device to host.
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, SIZE);
    puzzles::print_vector("expected", h_expected, SIZE);
    puzzles::expect_equal(h_out, h_expected, SIZE);
    std::printf("Puzzle 01 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
