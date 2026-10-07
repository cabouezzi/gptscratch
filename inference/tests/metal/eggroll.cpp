#include <catch2/catch_test_macros.hpp>

#include <eggroll/backend.hpp>

#include <cmath>
#include <cstdlib>
#include <numbers>
#include <stdexcept>
#include <vector>

TEST_CASE("Metal EGGROLL applies a rank-one perturbation", "[metal][eggroll]") {
  inference::eggroll::EGGROLLPerturbation perturbation(3, 2, 1);
  perturbation.A[0] = 2.0F;
  perturbation.A[1] = -1.0F;
  perturbation.B[0] = 1.0F;
  perturbation.B[1] = 0.5F;
  perturbation.B[2] = -2.0F;

  const std::vector<float> X{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<float> output{10.0F, 20.0F, 30.0F, 40.0F};
  const std::vector<float> expected{8.0F, 21.0F, 27.25F, 41.375F};

  float *received = inference::eggroll::applyPerturbationMetal(
      X.data(), output.data(), perturbation, 2, 0.25F);
  REQUIRE(received != nullptr);
  for (std::size_t index = 0; index < expected.size(); index++) {
    CAPTURE(index);
    CHECK(std::fabs(received[index] - expected[index]) < 0.00001F);
  }
  std::free(received);
}

TEST_CASE("Metal EGGROLL supports ranks greater than one", "[metal][eggroll]") {
  inference::eggroll::EGGROLLPerturbation perturbation(2, 3, 2);
  const float A[]{1.0F, 0.0F, 0.0F, 1.0F, 2.0F, -1.0F};
  const float B[]{1.0F, 2.0F, 3.0F, 4.0F};
  for (std::size_t index = 0; index < 6; index++) {
    perturbation.A[index] = A[index];
  }
  for (std::size_t index = 0; index < 4; index++) {
    perturbation.B[index] = B[index];
  }

  const std::vector<float> X{1.0F, 2.0F, -1.0F, 3.0F};
  const std::vector<float> output{0.5F, 1.0F, 1.5F, 2.0F, 2.5F, 3.0F};
  const std::vector<float> expected{7.5F, 11.0F, 5.5F, 10.0F, 12.5F, 9.0F};

  float *received = inference::eggroll::applyPerturbationMetal(
      X.data(), output.data(), perturbation, 2, std::sqrt(2.0F));
  REQUIRE(received != nullptr);
  for (std::size_t index = 0; index < expected.size(); index++) {
    CAPTURE(index);
    CHECK(std::fabs(received[index] - expected[index]) < 0.00001F);
  }
  std::free(received);
}

TEST_CASE("Metal EGGROLL handles negative epsilon and partial grids",
          "[metal][eggroll][edge]") {
  inference::eggroll::EGGROLLPerturbation perturbation(2, 5, 1);
  for (std::size_t index = 0; index < 5; index++) {
    perturbation.A[index] = static_cast<float>(index + 1);
  }
  perturbation.B[0] = 2.0F;
  perturbation.B[1] = -1.0F;

  const std::vector<float> X{3.0F, 1.0F};
  const std::vector<float> output{10.0F, 20.0F, 30.0F, 40.0F, 50.0F};
  const std::vector<float> expected{7.5F, 15.0F, 22.5F, 30.0F, 37.5F};

  float *received = inference::eggroll::applyPerturbationMetal(
      X.data(), output.data(), perturbation, 1, -0.5F);
  REQUIRE(received != nullptr);
  for (std::size_t index = 0; index < expected.size(); index++) {
    CAPTURE(index);
    CHECK(std::fabs(received[index] - expected[index]) < 0.00001F);
  }
  std::free(received);
}

TEST_CASE("Metal EGGROLL validates CPU inputs",
          "[metal][eggroll][validation]") {
  inference::eggroll::EGGROLLPerturbation perturbation(2, 3, 1);
  const float values[]{1.0F, 2.0F, 3.0F};

  CHECK_THROWS_AS(inference::eggroll::applyPerturbationMetal(
                      nullptr, values, perturbation, 1, 1.0F),
                  std::invalid_argument);
  CHECK_THROWS_AS(inference::eggroll::applyPerturbationMetal(
                      values, nullptr, perturbation, 1, 1.0F),
                  std::invalid_argument);
  CHECK_THROWS_AS(inference::eggroll::applyPerturbationMetal(
                      values, values, perturbation, 0, 1.0F),
                  std::invalid_argument);
}

TEST_CASE("Metal EGGROLL computes negative cross-entropy fitness",
          "[metal][eggroll][fitness]") {
  const float logits[]{0.0F, 0.0F, 0.0F, std::log(3.0F)};
  const int targets[]{0, 1};

  float received = inference::eggroll::fitnessMetal(logits, targets, 2, 2);
  float expected = -(std::log(2.0F) + std::log(4.0F / 3.0F)) / 2.0F;
  CHECK(std::fabs(received - expected) < 0.00001F);
}

TEST_CASE("Metal EGGROLL fitness is stable for large logits",
          "[metal][eggroll][fitness][edge]") {
  const float logits[]{10000.0F, 9999.0F, -10000.0F};
  const int targets[]{0};

  float received = inference::eggroll::fitnessMetal(logits, targets, 1, 3);
  float expected = -std::log(1.0F + std::exp(-1.0F));
  CHECK(std::isfinite(received));
  CHECK(std::fabs(received - expected) < 0.0005F);
}

TEST_CASE("Metal EGGROLL fitness validates targets",
          "[metal][eggroll][fitness][validation]") {
  const float logits[]{1.0F, 2.0F};
  const int negativeTarget[]{-1};
  const int largeTarget[]{2};
  CHECK_THROWS_AS(
      inference::eggroll::fitnessMetal(logits, negativeTarget, 1, 2),
      std::out_of_range);
  CHECK_THROWS_AS(inference::eggroll::fitnessMetal(logits, largeTarget, 1, 2),
                  std::out_of_range);
}

TEST_CASE("Metal EGGROLL permanently updates a weight matrix",
          "[metal][eggroll][update]") {
  inference::eggroll::EGGROLLPerturbation perturbation(3, 2, 1);
  perturbation.A[0] = 2.0F;
  perturbation.A[1] = -1.0F;
  perturbation.B[0] = 1.0F;
  perturbation.B[1] = 0.5F;
  perturbation.B[2] = -2.0F;
  const float weights[]{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const std::vector<const inference::eggroll::EGGROLLPerturbation *>
      perturbations{&perturbation};
  const std::vector<float> fitnesses{1.0F};
  const float expected[]{1.5F, 2.25F, 2.0F, 3.75F, 4.875F, 6.5F};

  float *received = inference::eggroll::updateWeightsMetal(
      weights, perturbations, fitnesses, 0.25F);
  REQUIRE(received != nullptr);
  for (std::size_t index = 0; index < 6; index++) {
    CAPTURE(index);
    CHECK(std::fabs(received[index] - expected[index]) < 0.00001F);
  }
  std::free(received);
}

TEST_CASE("Metal EGGROLL averages a population update",
          "[metal][eggroll][update][population]") {
  inference::eggroll::EGGROLLPerturbation first(2, 2, 1);
  first.A[0] = 1.0F;
  first.A[1] = 2.0F;
  first.B[0] = 3.0F;
  first.B[1] = 4.0F;
  inference::eggroll::EGGROLLPerturbation second(2, 2, 1);
  second.A[0] = 2.0F;
  second.A[1] = -1.0F;
  second.B[0] = 1.0F;
  second.B[1] = 5.0F;

  const float weights[]{0.0F, 0.0F, 0.0F, 0.0F};
  const std::vector<const inference::eggroll::EGGROLLPerturbation *>
      perturbations{&first, &second};
  const std::vector<float> fitnesses{1.0F, -1.0F};
  const float expected[]{0.1F, -0.6F, 0.7F, 1.3F};

  float *received = inference::eggroll::updateWeightsMetal(
      weights, perturbations, fitnesses, 0.2F);
  REQUIRE(received != nullptr);
  for (std::size_t index = 0; index < 4; index++) {
    CAPTURE(index);
    CHECK(std::fabs(received[index] - expected[index]) < 0.00001F);
  }
  std::free(received);
}
