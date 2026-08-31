#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <metal/backend.hpp>

TEST_CASE("Metal ReLU clamps negative values", "[metal][relu]") {
  const std::vector<float> input{-9.0F, -0.25F, 0.0F, 0.5F, 12.0F};
  const std::vector<float> expected{0.0F, 0.0F, 0.0F, 0.5F, 12.0F};

  float *output = inference::relu_metal(input.data(), input.size());
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < expected.size(); index++) {
    CAPTURE(index);
    CHECK(std::fabs(output[index] - expected[index]) < 0.00001F);
  }

  std::free(output);
}

TEST_CASE("Metal ReLU handles multiple threadgroups",
          "[metal][relu][edge]") {
  constexpr std::size_t elementCount = 513;
  std::vector<float> input(elementCount);

  for (std::size_t index = 0; index < elementCount; index++) {
    input[index] = static_cast<float>(index) - 256.0F;
  }

  float *output = inference::relu_metal(input.data(), input.size());
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < elementCount; index++) {
    float expected = std::fmax(input[index], 0.0F);
    CAPTURE(index);
    CHECK(std::fabs(output[index] - expected) < 0.00001F);
  }

  std::free(output);
}
