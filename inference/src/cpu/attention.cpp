#include <cpu/attention.hpp>
#include <cpu/matmul.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>

namespace inference {

// Standalone row-wise Softmax function for a 2D matrix stored as flat/vector
// data
void softmax(float *matrix, std::size_t rows, std::size_t columns) {
  for (std::size_t row = 0; row < rows; row++) {
    std::size_t row_start = row * columns;

    // Find max value in the row for numerical stability
    float max_val = matrix[row_start];
    for (std::size_t column = 1; column < columns; column++) {
      if (matrix[row_start + column] > max_val) {
        max_val = matrix[row_start + column];
      }
    }

    // Compute exponentials and sum
    float sum_exp = 0.0f;
    for (std::size_t column = 0; column < columns; column++) {
      matrix[row_start + column] =
          std::exp(matrix[row_start + column] - max_val);
      sum_exp += matrix[row_start + column];
    }

    // Normalize row
    for (std::size_t column = 0; column < columns; column++) {
      matrix[row_start + column] /= sum_exp;
    }
  }
}

// Scaled Dot-Product Attention using the modularized functions
void scaled_dot_product_attention(const float *Q, const float *K,
                                  const float *V, float *&output,
                                  std::size_t seq_len,
                                  std::size_t head_size) {
  float scale = 1.0f / std::sqrt(static_cast<float>(head_size));

  // Attention matrix NxN | N = # of tokens
  // Q * K^T and scale
  float *scores = matmul(Q, MatMulFlag::NO_TRANSPOSE, K,
                         MatMulFlag::TRANSPOSE, seq_len, head_size, seq_len);
  for (std::size_t index = 0; index < seq_len * seq_len; index++) {
    scores[index] *= scale;
  }

  // Apply Softmax row-wise
  softmax(scores, seq_len, seq_len);

  // Compute Attention Weights * V -> output [seq_len, head_size]
  output = matmul(scores, MatMulFlag::NO_TRANSPOSE, V, MatMulFlag::NO_TRANSPOSE, seq_len, seq_len, head_size);
  std::free(scores);
}

} // namespace inference
