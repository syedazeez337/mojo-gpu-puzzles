// CUDA GPU Puzzles - Puzzle 5: Broadcast (reference solution)
//
//   make test-p05   build & run this solution
#include "puzzles.cuh"
#include "tensor_view.cuh"

constexpr int SIZE = 2;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: broadcast_add_solution
__global__ void broadcast_add(TensorView2D<float> output,
                              TensorView2D<const float> a,
                              TensorView2D<const float> b, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < size && col < size) {
        output(row, col) = a(0, col) + b(row, 0);
    }
}
// ANCHOR_END: broadcast_add_solution

int main() {
    constexpr int N = SIZE * SIZE;

    float h_a[SIZE], h_b[SIZE];
    float h_expected[N], h_out[N];
    for (int i = 0; i < SIZE; ++i) {
        h_a[i] = static_cast<float>(i + 1);   // [1, 2]
        h_b[i] = static_cast<float>(i * 10);  // [0, 10]
    }
    for (int row = 0; row < SIZE; ++row) {
        for (int col = 0; col < SIZE; ++col) {
            h_expected[row * SIZE + col] = h_a[col] + h_b[row];
        }
    }

    float *d_a = nullptr, *d_b = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, N * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_out, 0, N * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, SIZE * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, SIZE * sizeof(float), cudaMemcpyHostToDevice));

    TensorView2D<float> out_view(d_out, SIZE, SIZE);
    TensorView2D<const float> a_view(d_a, 1, SIZE);
    TensorView2D<const float> b_view(d_b, SIZE, 1);

    dim3 threads_per_block(3, 3);
    broadcast_add<<<BLOCKS_PER_GRID, threads_per_block>>>(out_view, a_view,
                                                          b_view, SIZE);
    CUDA_CHECK_KERNEL();

    CUDA_CHECK(cudaMemcpy(h_out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost));

    puzzles::print_vector("out", h_out, N);
    puzzles::print_vector("expected", h_expected, N);
    puzzles::expect_equal(h_out, h_expected, N);
    std::printf("Puzzle 05 complete \xE2\x9C\x85\n");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_out));
    return 0;
}
