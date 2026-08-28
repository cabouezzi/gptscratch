#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <stdexcept>
#include <vector>

namespace {

using MatrixData = std::vector<std::vector<float>>;

struct Matrix {
  std::size_t rows;
  std::size_t columns;
  std::vector<float> values;
};

bool operator==(const Matrix& left, const Matrix& right) {
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

std::ostream& operator<<(std::ostream& output, const Matrix& matrix) {
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

} // namespace

#include <catch2/catch_test_macros.hpp>
#include <cpu/attention.hpp>
#include <cpu/matmul.hpp>

namespace {

std::vector<float> flatten(const MatrixData& matrix) {
  std::vector<float> values;
  values.reserve(matrix.size() * matrix[0].size());
  for (const std::vector<float>& row : matrix) {
    values.insert(values.end(), row.begin(), row.end());
  }
  return values;
}

Matrix multiply(const MatrixData& X, const MatrixData& Y) {
  std::vector<float> flat_X = flatten(X);
  std::vector<float> flat_Y = flatten(Y);
  float* output = inference::matmul(
      flat_X.data(), inference::MatMulFlag::NO_TRANSPOSE, flat_Y.data(),
      inference::MatMulFlag::NO_TRANSPOSE, X.size(), X[0].size(),
      Y[0].size());
  REQUIRE(output != nullptr);

  Matrix result{X.size(), Y[0].size(), {}};
  result.values.assign(output, output + result.rows * result.columns);
  std::free(output);
  return result;
}

Matrix multiply(const MatrixData& X, inference::MatMulFlag flagX,
                const MatrixData& Y, inference::MatMulFlag flagY,
                std::size_t output_rows, std::size_t output_columns) {
  std::vector<float> flat_X = flatten(X);
  std::vector<float> flat_Y = flatten(Y);
  std::size_t inner_dimension =
      flagX == inference::MatMulFlag::TRANSPOSE ? X.size() : X[0].size();
  float* output =
      inference::matmul(flat_X.data(), flagX, flat_Y.data(), flagY,
                        output_rows, inner_dimension, output_columns);
  REQUIRE(output != nullptr);

  Matrix result{output_rows, output_columns, {}};
  result.values.assign(output, output + result.rows * result.columns);
  std::free(output);
  return result;
}

MatrixData identity_matrix(std::size_t size) {
  MatrixData matrix(size, std::vector<float>(size, 0.0F));
  for (std::size_t index = 0; index < size; index++) {
    matrix[index][index] = 1.0F;
  }
  return matrix;
}

MatrixData shifted_identity_matrix(std::size_t size) {
  MatrixData matrix(size, std::vector<float>(size, 0.0F));
  for (std::size_t row = 0; row < size; row++) {
    matrix[row][(row + 1) % size] = 1.0F;
  }
  return matrix;
}

MatrixData weighted_identity_matrix(std::size_t size) {
  MatrixData matrix(size, std::vector<float>(size, 0.0F));
  for (std::size_t index = 0; index < size; index++) {
    matrix[index][index] = static_cast<float>(index + 1);
  }
  return matrix;
}

Matrix expected_shifted_attention(std::size_t size) {
  float exponential = std::exp(1.0F / std::sqrt(static_cast<float>(size)));
  float denominator = exponential + static_cast<float>(size - 1);
  float low = 1.0F / denominator;
  float high = exponential / denominator;
  Matrix expected{size, size, std::vector<float>(size * size)};

  for (std::size_t row = 0; row < size; row++) {
    std::size_t high_column = (row + size - 1) % size;
    for (std::size_t column = 0; column < size; column++) {
      float attention_weight = column == high_column ? high : low;
      expected.values[row * size + column] =
          attention_weight * static_cast<float>(column + 1);
    }
  }
  return expected;
}

} // namespace

TEST_CASE("rectangular matrices multiply", "[cpu][matmul]") {
  MatrixData X{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData Y{{7.0F, 8.0F}, {9.0F, 10.0F}, {11.0F, 12.0F}};
  Matrix expected{2, 2, {58.0F, 64.0F, 139.0F, 154.0F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("identity matrix preserves its input", "[cpu][matmul]") {
  MatrixData X{{3.0F, 4.0F}, {5.0F, 6.0F}};
  MatrixData identity{{1.0F, 0.0F}, {0.0F, 1.0F}};
  Matrix expected{2, 2, {3.0F, 4.0F, 5.0F, 6.0F}};

  CHECK(multiply(X, identity) == expected);
}

TEST_CASE("signed values multiply", "[cpu][matmul]") {
  MatrixData X{{-1.0F, 2.0F}, {3.0F, -4.0F}};
  MatrixData Y{{5.0F}, {-6.0F}};
  Matrix expected{2, 1, {-17.0F, 39.0F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("one by one matrices multiply", "[cpu][matmul]") {
  MatrixData X{{2.5F}};
  MatrixData Y{{-4.0F}};
  Matrix expected{1, 1, {-10.0F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("row vector multiplies a wide matrix", "[cpu][matmul]") {
  MatrixData X{{1.0F, -2.0F, 3.0F}};
  MatrixData Y{{4.0F, 0.0F, 1.0F, 2.0F},
               {5.0F, 1.0F, -1.0F, 0.0F},
               {2.0F, 3.0F, 4.0F, -2.0F}};
  Matrix expected{1, 4, {0.0F, 7.0F, 15.0F, -4.0F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("tall matrix multiplies a wide matrix", "[cpu][matmul]") {
  MatrixData X{{1.0F, 2.0F}, {3.0F, 4.0F}, {5.0F, 6.0F}};
  MatrixData Y{{7.0F, 8.0F, 9.0F}, {10.0F, 11.0F, 12.0F}};
  Matrix expected{3, 3,
                  {27.0F, 30.0F, 33.0F,
                   61.0F, 68.0F, 75.0F,
                   95.0F, 106.0F, 117.0F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("zero matrix produces zeros", "[cpu][matmul]") {
  MatrixData X{{1.0F, 2.0F}, {3.0F, 4.0F}};
  MatrixData zero{{0.0F, 0.0F, 0.0F}, {0.0F, 0.0F, 0.0F}};
  Matrix expected{2, 3, {0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F}};

  CHECK(multiply(X, zero) == expected);
}

TEST_CASE("fractional values multiply", "[cpu][matmul]") {
  MatrixData X{{0.5F, 1.5F}, {-2.0F, 0.25F}};
  MatrixData Y{{2.0F, -1.0F}, {4.0F, 3.0F}};
  Matrix expected{2, 2, {7.0F, 4.0F, -3.0F, 2.75F}};

  CHECK(multiply(X, Y) == expected);
}

TEST_CASE("scaled attention uses CPU matmul", "[cpu][matmul][attention]") {
  constexpr std::size_t size = 32;
  MatrixData Q = identity_matrix(size);
  MatrixData K = shifted_identity_matrix(size);
  MatrixData V = weighted_identity_matrix(size);
  std::vector<float> flat_Q = flatten(Q);
  std::vector<float> flat_K = flatten(K);
  std::vector<float> flat_V = flatten(V);
  float* output = nullptr;

  inference::scaled_dot_product_attention(
      flat_Q.data(), flat_K.data(), flat_V.data(), output, size, size);
  REQUIRE(output != nullptr);

  Matrix actual{size, size, {}};
  actual.values.assign(output, output + size * size);
  std::free(output);

  CHECK(actual == expected_shifted_attention(size));
}

TEST_CASE("transpose A before multiplying", "[cpu][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData B{{7.0F, 8.0F}, {9.0F, 10.0F}};
  Matrix expected{3, 2, {43.0F, 48.0F, 59.0F, 66.0F, 75.0F, 84.0F}};

  CHECK(multiply(A, inference::MatMulFlag::TRANSPOSE, B,
                 inference::MatMulFlag::NO_TRANSPOSE, 3, 2) == expected);
}

TEST_CASE("transpose B before multiplying", "[cpu][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F}, {3.0F, 4.0F}};
  MatrixData B{{5.0F, 6.0F}, {7.0F, 8.0F}, {9.0F, 10.0F}};
  Matrix expected{2, 3, {17.0F, 23.0F, 29.0F, 39.0F, 53.0F, 67.0F}};

  CHECK(multiply(A, inference::MatMulFlag::NO_TRANSPOSE, B,
                 inference::MatMulFlag::TRANSPOSE, 2, 3) == expected);
}

TEST_CASE("transpose both matrices before multiplying",
          "[cpu][matmul][transpose]") {
  MatrixData A{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData B{{7.0F, 8.0F}, {9.0F, 10.0F},
               {11.0F, 12.0F}, {13.0F, 14.0F}};
  Matrix expected{3, 4,
                  {39.0F, 49.0F, 59.0F, 69.0F,
                   54.0F, 68.0F, 82.0F, 96.0F,
                   69.0F, 87.0F, 105.0F, 123.0F}};

  CHECK(multiply(A, inference::MatMulFlag::TRANSPOSE, B,
                 inference::MatMulFlag::TRANSPOSE, 3, 4) == expected);
}

TEST_CASE("null matmul input throws", "[cpu][matmul]") {
  float matrix[]{1.0F};

  SECTION("A is null") {
    CHECK_THROWS_AS(inference::matmul(
                        nullptr, inference::MatMulFlag::NO_TRANSPOSE, matrix,
                        inference::MatMulFlag::NO_TRANSPOSE, 1, 1, 1),
                    std::invalid_argument);
  }

  SECTION("B is null") {
    CHECK_THROWS_AS(inference::matmul(
                        matrix, inference::MatMulFlag::NO_TRANSPOSE, nullptr,
                        inference::MatMulFlag::NO_TRANSPOSE, 1, 1, 1),
                    std::invalid_argument);
  }
}

TEST_CASE("zero matmul dimensions throw", "[cpu][matmul][validation]") {
  float matrix[]{1.0F};

  SECTION("M is zero") {
    CHECK_THROWS_AS(inference::matmul(
                        matrix, inference::MatMulFlag::NO_TRANSPOSE, matrix,
                        inference::MatMulFlag::NO_TRANSPOSE, 0, 1, 1),
                    std::invalid_argument);
  }

  SECTION("K is zero") {
    CHECK_THROWS_AS(inference::matmul(
                        matrix, inference::MatMulFlag::NO_TRANSPOSE, matrix,
                        inference::MatMulFlag::NO_TRANSPOSE, 1, 0, 1),
                    std::invalid_argument);
  }

  SECTION("N is zero") {
    CHECK_THROWS_AS(inference::matmul(
                        matrix, inference::MatMulFlag::NO_TRANSPOSE, matrix,
                        inference::MatMulFlag::NO_TRANSPOSE, 1, 1, 0),
                    std::invalid_argument);
  }
}

TEST_CASE("single-token attention returns its value",
          "[cpu][attention][edge]") {
  float Q[]{2.0F};
  float K[]{-3.0F};
  float V[]{7.5F};
  float* output = nullptr;

  inference::scaled_dot_product_attention(Q, K, V, output, 1, 1);
  REQUIRE(output != nullptr);

  Matrix actual{1, 1, {output[0]}};
  std::free(output);
  Matrix expected{1, 1, {7.5F}};
  CHECK(actual == expected);
}

TEST_CASE("attention softmax remains stable for extreme scores",
          "[cpu][attention][edge][stability]") {
  float Q[]{1000.0F, -1000.0F};
  float K[]{1000.0F, -1000.0F};
  float V[]{3.0F, -5.0F};
  float* output = nullptr;

  inference::scaled_dot_product_attention(Q, K, V, output, 2, 1);
  REQUIRE(output != nullptr);

  Matrix actual{2, 1, {output[0], output[1]}};
  std::free(output);
  Matrix expected{2, 1, {3.0F, -5.0F}};
  CHECK(actual == expected);
}

TEST_CASE("attention rejects null inputs", "[cpu][attention][validation]") {
  float matrix[]{1.0F};
  float* output = nullptr;

  SECTION("Q is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention(
                        nullptr, matrix, matrix, output, 1, 1),
                    std::invalid_argument);
  }

  SECTION("K is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention(
                        matrix, nullptr, matrix, output, 1, 1),
                    std::invalid_argument);
  }

  SECTION("V is null") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention(
                        matrix, matrix, nullptr, output, 1, 1),
                    std::invalid_argument);
    std::free(output);
  }
}

TEST_CASE("attention rejects zero dimensions",
          "[cpu][attention][validation]") {
  float matrix[]{1.0F};
  float* output = nullptr;

  SECTION("sequence length is zero") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention(
                        matrix, matrix, matrix, output, 0, 1),
                    std::invalid_argument);
  }

  SECTION("head size is zero") {
    CHECK_THROWS_AS(inference::scaled_dot_product_attention(
                        matrix, matrix, matrix, output, 1, 0),
                    std::invalid_argument);
  }
}

TEST_CASE("4096 by 4096 outer product", "[cpu][matmul][.large]") {
  constexpr std::size_t size = 4096;
  MatrixData X(size, std::vector<float>(1));
  MatrixData Y(1, std::vector<float>(size));

  for (std::size_t row = 0; row < size; row++) {
    X[row][0] = static_cast<float>(static_cast<int>(row % 13) - 6);
  }
  for (std::size_t column = 0; column < size; column++) {
    Y[0][column] = static_cast<float>(static_cast<int>(column % 11) - 5);
  }

  std::vector<float> flat_X = flatten(X);
  std::vector<float> flat_Y = flatten(Y);
  float* output = inference::matmul(
      flat_X.data(), inference::MatMulFlag::NO_TRANSPOSE, flat_Y.data(),
      inference::MatMulFlag::NO_TRANSPOSE, size, 1, size);
  REQUIRE(output != nullptr);

  for (std::size_t row = 0; row < size; row++) {
    for (std::size_t column = 0; column < size; column++) {
      float expected = X[row][0] * Y[0][column];
      float received = output[row * size + column];
      if (received != expected) {
        std::free(output);
        INFO("row: " << row);
        INFO("column: " << column);
        INFO("received: " << received);
        INFO("expected: " << expected);
        FAIL("4096 by 4096 output contains an incorrect value");
      }
    }
  }

  std::free(output);
}
