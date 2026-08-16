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
#include <cpu/matmul.hpp>

namespace {

Matrix multiply(const MatrixData& X, const MatrixData& Y) {
  float* output = inference::matmul(X, Y);
  REQUIRE(output != nullptr);

  Matrix result{X.size(), Y[0].size(), {}};
  result.values.assign(output, output + result.rows * result.columns);
  std::free(output);
  return result;
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

TEST_CASE("incompatible dimensions throw", "[cpu][matmul]") {
  MatrixData X{{1.0F, 2.0F, 3.0F}};
  MatrixData Y{{4.0F, 5.0F}, {6.0F, 7.0F}};

  CHECK_THROWS_AS(inference::matmul(X, Y), std::invalid_argument);
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

  float* output = inference::matmul(X, Y);
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
