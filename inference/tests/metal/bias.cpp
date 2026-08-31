#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <metal/backend.hpp>

TEST_CASE("Metal bias addition broadcasts across rows", "[metal][bias]") {
  const std::vector<float> input{1.0F, 2.0F, 3.0F,
                                 4.0F, 5.0F, 6.0F};
  const std::vector<float> bias{0.5F, -2.0F, 10.0F};
  const std::vector<float> expected{1.5F, 0.0F, 13.0F,
                                    4.5F, 3.0F, 16.0F};

  float *output = inference::add_bias_metal(
      input.data(), bias.data(), input.size(), bias.size());
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < expected.size(); index++) {
    CAPTURE(index);
    CHECK(std::fabs(output[index] - expected[index]) < 0.00001F);
  }

  std::free(output);
}

TEST_CASE("Metal bias addition handles multiple threadgroups",
          "[metal][bias][edge]") {
  constexpr std::size_t outputWidth = 3;
  constexpr std::size_t elementCount = 513;
  const std::vector<float> bias{1.0F, -2.0F, 3.0F};
  std::vector<float> input(elementCount);

  for (std::size_t index = 0; index < elementCount; index++) {
    input[index] = static_cast<float>(index) * 0.25F;
  }

  float *output = inference::add_bias_metal(
      input.data(), bias.data(), elementCount, outputWidth);
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < elementCount; index++) {
    float expected = input[index] + bias[index % outputWidth];
    CAPTURE(index);
    CHECK(std::fabs(output[index] - expected) < 0.00001F);
  }

  std::free(output);
}
