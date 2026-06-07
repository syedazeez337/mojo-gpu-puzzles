// CUDA GPU Puzzles - Puzzle 4: 2D Map (raw memory)
//
// Add 10 to each element of a SIZE x SIZE matrix. Threads are now arranged in a
// 2D block (3 x 3), and you have more threads than matrix elements (2 x 2), so
// you need a guard in BOTH dimensions.
//
// Convention: threadIdx.y is the row, threadIdx.x is the column. The matrix is
// stored row-major in a flat array, so element (row, col) lives at
// a[row * size + col].
//
//   make p04        build & run this file (fails until you fill it in)
//   make test-p04   build & run the reference solution
#include "puzzles.cuh"

constexpr int SIZE = 2;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: add_10_2d
__global__ void add_10_2d(float* output, const float* a, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    // FILL ME IN (roughly 2 lines)
}
// ANCHOR_END: add_10_2d

int main() {
    constexpr int N = SIZE * SIZE;
    constexpr size_t bytes = N * sizeof(float);

    float h_a[N], h_expected[N], h_out[N];
    for (int i = 0; i < SIZE; ++i) {       // row
        for (int j = 0; j < SIZE; ++j) {   // col
            h_a[i * SIZE + j] = static_cast<float>(i * SIZE + j);
            h_expected[i * SIZE + j] = h_a[i * SIZE + j] + 10.0f;
        }
    }

    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    dim3 threads_per_block(3, 3);  // (x = cols, y = rows)
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
