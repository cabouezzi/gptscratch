#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <vector>

#include <cpu/attention.hpp>
#include <metal/backend.hpp>

namespace {

using MatrixData = std::vector<std::vector<float>>;

struct Matrix {
  std::size_t rows;
  std::size_t columns;
  std::vector<float> values;
};

bool operator==(const Matrix &left, const Matrix &right) {
  if (left.rows != right.rows || left.columns != right.columns ||
      left.values.size() != right.values.size()) {
    return false;
  }

  for (std::size_t index = 0; index < left.values.size(); index++) {
    if (!std::isfinite(left.values[index]) ||
        std::fabs(left.values[index] - right.values[index]) > 0.00001F) {
      return false;
    }
  }

  return true;
}

std::ostream &operator<<(std::ostream &output, const Matrix &matrix) {
  output << '\n';
  for (std::size_t row = 0; row < matrix.rows; row++) {
    output << "  [";
    for (std::size_t column = 0; column < matrix.columns; column++) {
      if (column > 0) {
        output << ", ";
      }
      output << matrix.values[row * matrix.columns + column];
    }
    output << ']';
    if (row + 1 < matrix.rows) {
      output << '\n';
    }
  }
  return output;
}

Matrix multiply_metal(const MatrixData &X, inference::MatMulFlag flagX,
                      const MatrixData &Y, inference::MatMulFlag flagY,
                      bool tile) {
  float *output = inference::matmul_metal(
      X, flagX, Y, flagY, tile);
  REQUIRE(output != nullptr);

  std::size_t rows = flagX == inference::MatMulFlag::TRANSPOSE ? X[0].size()
                                                                : X.size();
  std::size_t columns =
      flagY == inference::MatMulFlag::TRANSPOSE ? Y.size() : Y[0].size();
  Matrix result{rows, columns, {}};
  result.values.assign(output, output + result.rows * result.columns);
  std::free(output);
  return result;
}

Matrix multiply_metal(const MatrixData &X, const MatrixData &Y, bool tile) {
  return multiply_metal(X, inference::MatMulFlag::NO_TRANSPOSE, Y,
                        inference::MatMulFlag::NO_TRANSPOSE, tile);
}

MatrixData make_matrix(std::size_t rows, std::size_t columns, int seed) {
  MatrixData matrix(rows, std::vector<float>(columns));
  for (std::size_t row = 0; row < rows; row++) {
    for (std::size_t column = 0; column < columns; column++) {
      int value = static_cast<int>((row * 17 + column * 31 + seed) % 19) - 9;
      matrix[row][column] = static_cast<float>(value) / 8.0F;
    }
  }
  return matrix;
}

Matrix multiply_reference(const MatrixData &X, const MatrixData &Y) {
  Matrix result{X.size(), Y[0].size(), {}};
  result.values.resize(result.rows * result.columns);

  for (std::size_t row = 0; row < result.rows; row++) {
    for (std::size_t column = 0; column < result.columns; column++) {
      for (std::size_t k = 0; k < Y.size(); k++) {
        result.values[row * result.columns + column] += X[row][k] * Y[k][column];
      }
    }
  }
  return result;
}

MatrixData transpose(const MatrixData &matrix) {
  MatrixData result(matrix[0].size(), std::vector<float>(matrix.size()));

  for (std::size_t row = 0; row < matrix.size(); row++) {
    for (std::size_t column = 0; column < matrix[row].size(); column++) {
      result[column][row] = matrix[row][column];
    }
  }

  return result;
}

std::vector<float> make_attention_input(std::size_t sequence_length,
                                        std::size_t head_size, int seed,
                                        float divisor) {
  std::vector<float> values(sequence_length * head_size);
  for (std::size_t row = 0; row < sequence_length; row++) {
    for (std::size_t column = 0; column < head_size; column++) {
      int value =
          static_cast<int>((row * 17 + column * 31 + seed) % 29) - 14;
      values[row * head_size + column] =
          static_cast<float>(value) / divisor;
    }
  }
  return values;
}

std::vector<float> run_cpu_attention(const std::vector<float> &Q,
                                     const std::vector<float> &K,
                                     const std::vector<float> &V,
                                     std::size_t sequence_length,
                                     std::size_t head_size) {
  float *output = nullptr;
  inference::scaled_dot_product_attention(Q.data(), K.data(), V.data(), output,
                                          sequence_length, head_size);
  REQUIRE(output != nullptr);
  std::vector<float> result(output,
                            output + sequence_length * head_size);
  std::free(output);
  return result;
}

std::vector<float> run_metal_attention(const std::vector<float> &Q,
                                       const std::vector<float> &K,
                                       const std::vector<float> &V,
                                       std::size_t sequence_length,
                                       std::size_t head_size,
                                       bool is_causal) {
  float *output = inference::scaled_dot_product_attention_metal(
      Q.data(), K.data(), V.data(), sequence_length, head_size, is_causal);
  REQUIRE(output != nullptr);
  std::vector<float> result(output,
                            output + sequence_length * head_size);
  std::free(output);
  return result;
}

std::vector<float> attention_reference(const std::vector<float> &Q,
                                       const std::vector<float> &K,
                                       const std::vector<float> &V,
                                       std::size_t sequence_length,
                                       std::size_t head_size,
                                       bool is_causal) {
  std::vector<float> output(sequence_length * head_size, 0.0F);
  std::vector<float> scores(sequence_length);
  const float scale = 1.0F / std::sqrt(static_cast<float>(head_size));

  for (std::size_t query = 0; query < sequence_length; query++) {
    const std::size_t visible_keys = is_causal ? query + 1 : sequence_length;
    float maximum = -INFINITY;

    for (std::size_t key = 0; key < visible_keys; key++) {
      float score = 0.0F;
      for (std::size_t column = 0; column < head_size; column++) {
        score += Q[query * head_size + column] *
                 K[key * head_size + column];
      }
      scores[key] = score * scale;
      maximum = std::fmax(maximum, scores[key]);
    }

    float denominator = 0.0F;
    for (std::size_t key = 0; key < visible_keys; key++) {
      scores[key] = std::exp(scores[key] - maximum);
      denominator += scores[key];
    }

    for (std::size_t column = 0; column < head_size; column++) {
      float weighted_value = 0.0F;
      for (std::size_t key = 0; key < visible_keys; key++) {
        weighted_value += scores[key] * V[key * head_size + column];
      }
      output[query * head_size + column] = weighted_value / denominator;
    }
  }

  return output;
}

void require_attention_close(const std::vector<float> &received,
                             const std::vector<float> &expected,
                             std::size_t head_size) {
  REQUIRE(received.size() == expected.size());
  for (std::size_t index = 0; index < expected.size(); index++) {
    if (!std::isfinite(received[index]) ||
        std::fabs(received[index] - expected[index]) > 0.001F) {
      INFO("row: " << index / head_size);
      INFO("column: " << index % head_size);
      INFO("received: " << received[index]);
      INFO("expected: " << expected[index]);
      FAIL("Metal attention differs from expected attention");
    }
  }
}

} // namespace

TEST_CASE("Metal rectangular matrices multiply", "[metal][matmul]") {
  MatrixData X{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData Y{{7.0F, 8.0F}, {9.0F, 10.0F}, {11.0F, 12.0F}};
  Matrix expected{2, 2, {58.0F, 64.0F, 139.0F, 154.0F}};

  CHECK(multiply_metal(X, Y, false) == expected);
}

TEST_CASE("Metal signed values multiply", "[metal][matmul]") {
  MatrixData X{{-1.0F, 2.0F}, {3.0F, -4.0F}};
  MatrixData Y{{5.0F}, {-6.0F}};
  Matrix expected{2, 1, {-17.0F, 39.0F}};

  CHECK(multiply_metal(X, Y, false) == expected);
}

TEST_CASE("Metal identity matrix preserves its input", "[metal][matmul]") {
  MatrixData X{{3.0F, 4.0F}, {5.0F, 6.0F}};
  MatrixData identity{{1.0F, 0.0F}, {0.0F, 1.0F}};
  Matrix expected{2, 2, {3.0F, 4.0F, 5.0F, 6.0F}};

  CHECK(multiply_metal(X, identity, false) == expected);
}

TEST_CASE("Metal naïve one by one matrices multiply",
          "[metal][matmul][edge]") {
  MatrixData X{{2.5F}};
  MatrixData Y{{-4.0F}};
  Matrix expected{1, 1, {-10.0F}};

  CHECK(multiply_metal(X, Y, false) == expected);
}

TEST_CASE("Metal naïve handles zero and fractional values",
          "[metal][matmul][edge]") {
  MatrixData X{{0.0F, 0.5F, -1.5F}, {2.0F, -0.25F, 0.0F}};
  MatrixData Y{{4.0F, 0.0F}, {-2.0F, 8.0F}, {0.0F, -2.0F}};
  Matrix expected{2, 2, {-1.0F, 7.0F, 8.5F, -2.0F}};

  CHECK(multiply_metal(X, Y, false) == expected);
}

TEST_CASE("Metal matmul rejects incompatible shapes",
          "[metal][matmul][validation]") {
  MatrixData X{{1.0F, 2.0F, 3.0F}};
  MatrixData Y{{4.0F, 5.0F}, {6.0F, 7.0F}};

  SECTION("naïve") {
    CHECK_THROWS(multiply_metal(X, Y, false));
  }

  SECTION("tiled") {
    CHECK_THROWS(multiply_metal(X, Y, true));
  }
}

TEST_CASE("scaled attention uses Metal naïve matmul",
          "[metal][matmul][attention]") {
  FAIL("Not implemented: Metal naïve attention");
}

TEST_CASE("scaled attention uses Metal tiled kernel",
          "[metal][matmul][tile][attention]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q(sequence_length * head_size, 0.0F);
  std::vector<float> K(sequence_length * head_size, 0.0F);
  std::vector<float> V(sequence_length * head_size);

  for (std::size_t row = 0; row < sequence_length; row++) {
    for (std::size_t column = 0; column < head_size; column++) {
      V[row * head_size + column] =
          static_cast<float>(row) * 0.25F +
          static_cast<float>(column) * 0.015625F;
    }
  }

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, false),
      run_cpu_attention(Q, K, V, sequence_length, head_size), head_size);
}

TEST_CASE("Metal attention handles asymmetric signed fractional inputs",
          "[metal][attention]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q =
      make_attention_input(sequence_length, head_size, 1, 16.0F);
  std::vector<float> K =
      make_attention_input(sequence_length, head_size, 7, 12.0F);
  std::vector<float> V =
      make_attention_input(sequence_length, head_size, 13, 8.0F);

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, false),
      run_cpu_attention(Q, K, V, sequence_length, head_size), head_size);
}

TEST_CASE("Metal attention handles two key-value tiles",
          "[metal][attention][tile]") {
  constexpr std::size_t sequence_length = 128;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q =
      make_attention_input(sequence_length, head_size, 2, 20.0F);
  std::vector<float> K =
      make_attention_input(sequence_length, head_size, 9, 18.0F);
  std::vector<float> V =
      make_attention_input(sequence_length, head_size, 21, 10.0F);

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, false),
      run_cpu_attention(Q, K, V, sequence_length, head_size), head_size);
}

TEST_CASE("Metal attention softmax remains stable for extreme scores",
          "[metal][attention][softmax]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q(sequence_length * head_size, 0.0F);
  std::vector<float> K(sequence_length * head_size, 0.0F);
  std::vector<float> V =
      make_attention_input(sequence_length, head_size, 5, 4.0F);

  for (std::size_t row = 0; row < sequence_length; row++) {
    std::size_t column = row % head_size;
    Q[row * head_size + column] = row % 2 == 0 ? 1000.0F : -1000.0F;
    K[row * head_size + column] = row % 3 == 0 ? 1000.0F : -1000.0F;
  }

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, false),
      run_cpu_attention(Q, K, V, sequence_length, head_size), head_size);
}

TEST_CASE("Metal causal attention averages only the visible prefix",
          "[metal][attention][causal]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q(sequence_length * head_size, 0.0F);
  std::vector<float> K(sequence_length * head_size, 0.0F);
  std::vector<float> V(sequence_length * head_size);
  std::vector<float> expected(sequence_length * head_size);

  for (std::size_t row = 0; row < sequence_length; row++) {
    for (std::size_t column = 0; column < head_size; column++) {
      V[row * head_size + column] =
          static_cast<float>(row) * 0.5F +
          static_cast<float>(column) * 0.01F;
      expected[row * head_size + column] =
          static_cast<float>(row) * 0.25F +
          static_cast<float>(column) * 0.01F;
    }
  }

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, true), expected,
      head_size);
}

TEST_CASE("Metal causal attention ignores dominant future keys",
          "[metal][attention][causal]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q(sequence_length * head_size, 0.0F);
  std::vector<float> K(sequence_length * head_size, 0.0F);
  std::vector<float> V =
      make_attention_input(sequence_length, head_size, 19, 7.0F);

  for (std::size_t row = 0; row < sequence_length; row++) {
    Q[row * head_size] = 1.0F;
    K[row * head_size] = static_cast<float>(row) * 10.0F;
  }

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, true),
      attention_reference(Q, K, V, sequence_length, head_size, true),
      head_size);
}

TEST_CASE("Metal causal attention handles asymmetric signed fractional inputs",
          "[metal][attention][causal]") {
  constexpr std::size_t sequence_length = 64;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q =
      make_attention_input(sequence_length, head_size, 3, 13.0F);
  std::vector<float> K =
      make_attention_input(sequence_length, head_size, 11, 17.0F);
  std::vector<float> V =
      make_attention_input(sequence_length, head_size, 23, 9.0F);

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, true),
      attention_reference(Q, K, V, sequence_length, head_size, true),
      head_size);
}

TEST_CASE("Metal causal attention masks across key-value tile boundaries",
          "[metal][attention][causal][tile]") {
  constexpr std::size_t sequence_length = 128;
  constexpr std::size_t head_size = 64;
  std::vector<float> Q(sequence_length * head_size, 0.0F);
  std::vector<float> K(sequence_length * head_size, 0.0F);
  std::vector<float> V(sequence_length * head_size);
  std::vector<float> expected(sequence_length * head_size);

  for (std::size_t row = 0; row < sequence_length; row++) {
    for (std::size_t column = 0; column < head_size; column++) {
      V[row * head_size + column] = static_cast<float>(row);
      expected[row * head_size + column] = static_cast<float>(row) * 0.5F;
    }
  }

  require_attention_close(
      run_metal_attention(Q, K, V, sequence_length, head_size, true), expected,
      head_size);
}

TEST_CASE("Metal attention validates inputs", "[metal][attention][validation]") {
  float value = 0.0F;

  SECTION("Q is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        nullptr, &value, &value, 64, 64, false),
                    std::invalid_argument);
  }

  SECTION("K is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        &value, nullptr, &value, 64, 64, false),
                    std::invalid_argument);
  }

  SECTION("V is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        &value, &value, nullptr, 64, 64, false),
                    std::invalid_argument);
  }

  SECTION("sequence length is zero") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        &value, &value, &value, 0, 64, false),
                    std::invalid_argument);
  }

  SECTION("head size is zero") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        &value, &value, &value, 64, 0, false),
                    std::invalid_argument);
  }

  SECTION("sequence length is not a complete tile") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention_metal(
                        &value, &value, &value, 65, 64, false),
                    std::invalid_argument);
  }
}

TEST_CASE("Metal naïve matmul transposes A", "[metal][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData B{{7.0F, 8.0F}, {9.0F, 10.0F}};
  Matrix expected{3, 2, {43.0F, 48.0F, 59.0F, 66.0F, 75.0F, 84.0F}};

  CHECK(multiply_metal(A, inference::MatMulFlag::TRANSPOSE, B,
                        inference::MatMulFlag::NO_TRANSPOSE, false) ==
        expected);
}

TEST_CASE("Metal naïve matmul transposes B", "[metal][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F}, {3.0F, 4.0F}};
  MatrixData B{{5.0F, 6.0F}, {7.0F, 8.0F}, {9.0F, 10.0F}};
  Matrix expected{2, 3, {17.0F, 23.0F, 29.0F, 39.0F, 53.0F, 67.0F}};

  CHECK(multiply_metal(A, inference::MatMulFlag::NO_TRANSPOSE, B,
                        inference::MatMulFlag::TRANSPOSE, false) ==
        expected);
}

TEST_CASE("Metal naïve matmul transposes both matrices",
          "[metal][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData B{{7.0F, 8.0F}, {9.0F, 10.0F},
               {11.0F, 12.0F}, {13.0F, 14.0F}};
  Matrix expected{3, 4,
                  {39.0F, 49.0F, 59.0F, 69.0F,
                   54.0F, 68.0F, 82.0F, 96.0F,
                   69.0F, 87.0F, 105.0F, 123.0F}};

  CHECK(multiply_metal(A, inference::MatMulFlag::TRANSPOSE, B,
                        inference::MatMulFlag::TRANSPOSE, false) ==
        expected);
}

TEST_CASE("Metal tiled matmul matches the reference", "[metal][matmul][tile]") {
  MatrixData X = make_matrix(32, 64, 1);
  MatrixData Y = make_matrix(64, 32, 2);

  CHECK(multiply_metal(X, Y, true) == multiply_reference(X, Y));
}

TEST_CASE("Metal tiled matmul supports one complete tile",
          "[metal][matmul][tile][edge]") {
  MatrixData X = make_matrix(32, 32, 3);
  MatrixData Y = make_matrix(32, 32, 4);

  CHECK(multiply_metal(X, Y, true) == multiply_reference(X, Y));
}

TEST_CASE("Metal tiled matmul reports an incomplete inner tile",
          "[metal][matmul][tile][edge][unsupported]") {
  MatrixData X(32, std::vector<float>(33, 1.0F));
  MatrixData Y(33, std::vector<float>(32, 1.0F));
  Matrix expected{32, 32, std::vector<float>(32 * 32, 33.0F)};

  CHECK(multiply_metal(X, Y, true) == expected);
}

TEST_CASE("Metal tiled matmul reports a partial output tile",
          "[metal][matmul][tile][edge][unsupported]") {
  MatrixData X(1, std::vector<float>(32, 1.0F));
  MatrixData Y(32, std::vector<float>(1, 1.0F));
  Matrix expected{1, 1, {32.0F}};

  CHECK(multiply_metal(X, Y, true) == expected);
}

TEST_CASE("Metal tiled matmul transposes A", "[metal][matmul][tile][transpose]") {
  MatrixData A = make_matrix(64, 32, 1);
  MatrixData B = make_matrix(64, 32, 2);

  CHECK(multiply_metal(A, inference::MatMulFlag::TRANSPOSE, B,
                        inference::MatMulFlag::NO_TRANSPOSE, true) ==
        multiply_reference(transpose(A), B));
}

TEST_CASE("Metal tiled matmul transposes B", "[metal][matmul][tile][transpose]") {
  MatrixData A = make_matrix(32, 64, 1);
  MatrixData B = make_matrix(32, 64, 2);

  CHECK(multiply_metal(A, inference::MatMulFlag::NO_TRANSPOSE, B,
                        inference::MatMulFlag::TRANSPOSE, true) ==
        multiply_reference(A, transpose(B)));
}

TEST_CASE("Metal tiled matmul transposes both matrices",
          "[metal][matmul][tile][transpose]") {
  MatrixData A = make_matrix(64, 32, 1);
  MatrixData B = make_matrix(32, 64, 2);

  CHECK(multiply_metal(A, inference::MatMulFlag::TRANSPOSE, B,
                        inference::MatMulFlag::TRANSPOSE, true) ==
        multiply_reference(transpose(A), transpose(B)));
}
