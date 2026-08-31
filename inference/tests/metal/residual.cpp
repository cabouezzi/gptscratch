#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <cstdlib>
#include <vector>

#include <metal/backend.hpp>

TEST_CASE("Metal residual addition adds every element",
          "[metal][residual]") {
  const std::vector<float> residual{1.0F, -2.0F, 3.5F, 0.0F, 9.0F,
                                    -4.25F, 8.0F};
  const std::vector<float> input{2.0F, 5.0F, -1.5F, 0.25F, -9.0F,
                                 4.0F, 1.0F};
  const std::vector<float> expected{3.0F, 3.0F, 2.0F, 0.25F, 0.0F,
                                    -0.25F, 9.0F};

  float *output = inference::residual_add_metal(
      residual.data(), input.data(), residual.size());
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < expected.size(); index++) {
    INFO("index: " << index);
    INFO("received: " << output[index]);
    INFO("expected: " << expected[index]);
    CHECK(std::fabs(output[index] - expected[index]) < 0.00001F);
  }

  std::free(output);
}

TEST_CASE("Metal residual addition handles multiple threadgroups",
          "[metal][residual][edge]") {
  constexpr std::size_t elementCount = 513;
  std::vector<float> residual(elementCount);
  std::vector<float> input(elementCount);

  for (std::size_t index = 0; index < elementCount; index++) {
    residual[index] = static_cast<float>(index) * 0.25F;
    input[index] = -static_cast<float>(index) * 0.125F;
  }

  float *output = inference::residual_add_metal(
      residual.data(), input.data(), elementCount);
  REQUIRE(output != nullptr);

  for (std::size_t index = 0; index < elementCount; index++) {
    CAPTURE(index);
    CHECK(std::fabs(output[index] - static_cast<float>(index) * 0.125F) <
          0.00001F);
  }

  std::free(output);
}
