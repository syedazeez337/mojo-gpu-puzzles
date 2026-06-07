// CUDA GPU Puzzles - Puzzle 4: 2D Map (TensorView version, reference solution)
//
//   make test-p04_tensor_view   build & run this solution
#include "puzzles.cuh"
#include "tensor_view.cuh"

constexpr int SIZE = 2;
constexpr int BLOCKS_PER_GRID = 1;

// ANCHOR: add_10_2d_tensor_view_solution
__global__ void add_10_2d(TensorView2D<float> output,
                          TensorView2D<const float> a, int size) {
    int row = threadIdx.y;
    int col = threadIdx.x;
    if (row < size && col < size) {
        output(row, col) = a(row, col) + 10.0f;
    }
}
// ANCHOR_END: add_10_2d_tensor_view_solution

int main() {
    constexpr int N = SIZE * SIZE;
    constexpr size_t bytes = N * sizeof(float);

    float h_a[N], h_expected[N], h_out[N];
    for (int i = 0; i < N; ++i) {
        h_a[i] = static_cast<float>(i);
        h_expected[i] = h_a[i] + 10.0f;
    }

    float *d_a = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));
    CUDA_CHECK(cudaMemset(d_out, 0, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));

    TensorView2D<float> out_view(d_out, SIZE, SIZE);
    TensorView2D<const float> a_view(d_a, SIZE, SIZE);

    dim3 threads_per_block(3, 3);
    add_10_2d<<<BLOCKS_PER_GRID, threads_per_block>>>(out_view, a_view, SIZE);
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
