#include <cpu/matmul.hpp>

#include <cstddef>
#include <cstdlib>
#include <new>
#include <stdexcept>

namespace inference {

inline std::size_t matrix_index(std::size_t a, std::size_t b,
                                std::size_t width) {
  return a * width + b;
}

float *matmul(const float *A, MatMulFlag flagA, const float *B,
              MatMulFlag flagB, std::size_t M, std::size_t K,
              std::size_t N) {
  if (A == nullptr || B == nullptr) {
    throw std::invalid_argument("Matmul inputs cannot be null");
  }
  if (M == 0 || K == 0 || N == 0) {
    throw std::invalid_argument("Matmul dimensions must be positive");
  }

  bool transposeA = flagA == MatMulFlag::TRANSPOSE;
  bool transposeB = flagB == MatMulFlag::TRANSPOSE;
  std::size_t A_width = transposeA ? M : K;
  std::size_t B_width = transposeB ? K : N;

  float *output = static_cast<float *>(std::calloc(M * N, sizeof(float)));
  if (output == nullptr) {
    throw std::bad_alloc();
  }

  for (std::size_t row = 0; row < M; row++) {
    for (std::size_t column = 0; column < N; column++) {
      for (std::size_t k = 0; k < K; k++) {
        std::size_t A_row = transposeA ? k : row;
        std::size_t A_column = transposeA ? row : k;
        std::size_t B_row = transposeB ? column : k;
        std::size_t B_column = transposeB ? k : column;
        output[matrix_index(row, column, N)] +=
            A[matrix_index(A_row, A_column, A_width)] *
            B[matrix_index(B_row, B_column, B_width)];
      }
    }
  }

  return output;
}

} // namespace inference
