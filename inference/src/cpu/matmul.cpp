#include <cpu/matmul.hpp>
#include <metal/backend.hpp>
#include <cstddef>
#include <format>
#include <stdexcept>

namespace inference {

inline std::size_t matrix_index(std::size_t a, std::size_t b,
                                std::size_t width) {
  return a * width + b;
}

// I used the wrong notation my bad. MxN -> NxM here lol
// Convention is supposed to be MxK • KxN -> MxN
float *matmul(std::vector<std::vector<float>> const X,
              std::vector<std::vector<float>> const Y) {
  std::size_t N = X.size();
  std::size_t M = X[0].size();
  std::size_t P = Y[0].size();
  if (Y.size() != M)
    throw std::invalid_argument(std::format(
        "Mismatching dimensions for matrix multiplication: {} versus {}", M,
        Y.size()));

  // (N x M) x (M x P) -> (N x P)
  float *output = static_cast<float *>(calloc(N * P, sizeof(float)));
  for (std::size_t oidx = 0; oidx < P; oidx++) {
    for (std::size_t row = 0; row < N; row++) {
      for (std::size_t col = 0; col < M; col++) {
        output[matrix_index(row, oidx, P)] += X[row][col] * Y[col][oidx];
      }
    }
  }

  return output;
}

} // namespace inference
