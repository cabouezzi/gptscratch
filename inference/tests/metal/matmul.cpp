#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <ostream>
#include <vector>

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

Matrix multiply_metal(const MatrixData &X, const MatrixData &Y) {
  float *output = matmul_metal(X, Y);
  REQUIRE(output != nullptr);

  Matrix result{X.size(), Y[0].size(), {}};
  result.values.assign(output, output + result.rows * result.columns);
  std::free(output);
  return result;
}

} // namespace

TEST_CASE("Metal rectangular matrices multiply", "[metal][matmul]") {
  MatrixData X{{1.0F, 2.0F, 3.0F}, {4.0F, 5.0F, 6.0F}};
  MatrixData Y{{7.0F, 8.0F}, {9.0F, 10.0F}, {11.0F, 12.0F}};
  Matrix expected{2, 2, {58.0F, 64.0F, 139.0F, 154.0F}};

  CHECK(multiply_metal(X, Y) == expected);
}

TEST_CASE("Metal signed values multiply", "[metal][matmul]") {
  MatrixData X{{-1.0F, 2.0F}, {3.0F, -4.0F}};
  MatrixData Y{{5.0F}, {-6.0F}};
  Matrix expected{2, 1, {-17.0F, 39.0F}};

  CHECK(multiply_metal(X, Y) == expected);
}

TEST_CASE("Metal identity matrix preserves its input", "[metal][matmul]") {
  MatrixData X{{3.0F, 4.0F}, {5.0F, 6.0F}};
  MatrixData identity{{1.0F, 0.0F}, {0.0F, 1.0F}};
  Matrix expected{2, 2, {3.0F, 4.0F, 5.0F, 6.0F}};

  CHECK(multiply_metal(X, identity) == expected);
}
