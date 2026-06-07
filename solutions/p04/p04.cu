// CUDA GPU Puzzles - Puzzle 4: 2D Map (raw memory, reference solution)
//
//   make test-p04   build & run this solution
#include "puzzles.cuh"

constexpr int SIZE = 2;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: add_10_2d_solution
__global__ void add_10_2d(float* output, const float* a, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < size && col < size) {
        output[row * size + col] = a[row * size + col] + 10.0f;
    }
}
// ANCHOR_END: add_10_2d_solution

int main() {
    constexpr int N = SIZE * SIZE;
    constexpr size_t bytes = N * sizeof(float);

    float h_a[N], h_expected[N], h_out[N];
    for (int i = 0; i < SIZE; ++i) {
        for (int j = 0; j < SIZE; ++j) {
            h_a[i * SIZE + j] = static_cast<float>(i * SIZE + j);
            h_expected[i * SIZE + j] = h_a[i * SIZE + j] + 10.0f;
        }
    }

    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    dim3 threads_per_block(3, 3);
    add_10_2d<<<BLOCKS_PER_GRID, threads_per_block>>>(d_out, d_a, SIZE);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, N);
    puzzles::print_vector("expected", h_expected, N);
    puzzles::expect_equal(h_out, h_expected, N);
    std::printf("Puzzle 04 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
