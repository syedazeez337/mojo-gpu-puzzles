// CUDA GPU Puzzles — shared host-side helpers.
//
// Every puzzle is a small, self-contained program: it sets up some data,
// launches a kernel, copies the result back, and checks it. The few utilities
// that this test harness needs live here so each puzzle file can stay focused
// on the GPU concept it teaches.
//
// Nothing in this header is required to *write* a kernel — it is only the
// scaffolding around it.
#pragma once

#include <cmath>
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>

// Wrap every CUDA runtime call so failures are caught immediately instead of
// silently corrupting later results. On error it reports the file, line, and
// human-readable reason, then exits.
//
//   CUDA_CHECK(cudaMalloc(&ptr, bytes));
//
// Checking *every* call is a habit worth forming: CUDA errors are sticky and
// often surface far from where they were actually caused.
#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t _err = (call);                                           \
        if (_err != cudaSuccess) {                                           \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__,      \
                         __LINE__, cudaGetErrorString(_err));                \
            std::exit(EXIT_FAILURE);                                         \
        }                                                                    \
    } while (0)

// Call this right after a kernel launch. Kernel launches are asynchronous and
// do not return an error code directly, so we check two things:
//   1. cudaGetLastError()    — was the launch itself rejected (bad config)?
//   2. cudaDeviceSynchronize — did the kernel hit an error while running?
//
//   my_kernel<<<grid, block>>>(...);
//   CUDA_CHECK_KERNEL();
#define CUDA_CHECK_KERNEL()                                                   \
    do {                                                                     \
        CUDA_CHECK(cudaGetLastError());                                      \
        CUDA_CHECK(cudaDeviceSynchronize());                                 \
    } while (0)

namespace puzzles {

// Print a labelled 1D float array, e.g.  out: [10, 11, 12, 13]
inline void print_vector(const char* label, const float* data, int n) {
    std::printf("%s: [", label);
    for (int i = 0; i < n; ++i) {
        std::printf("%s%g", i ? ", " : "", static_cast<double>(data[i]));
    }
    std::printf("]\n");
}

// Exit with a clear message unless every element matches within tolerance.
// Returns normally on success so the caller can print its "complete" banner.
inline void expect_equal(const float* got, const float* expected, int n,
                         float tol = 1e-5f) {
    for (int i = 0; i < n; ++i) {
        if (std::fabs(got[i] - expected[i]) > tol) {
            std::fprintf(stderr,
                         "\xE2\x9D\x8C Mismatch at index %d: got %g, "
                         "expected %g\n",
                         i, static_cast<double>(got[i]),
                         static_cast<double>(expected[i]));
            std::exit(EXIT_FAILURE);
        }
    }
}

}  // namespace puzzles
