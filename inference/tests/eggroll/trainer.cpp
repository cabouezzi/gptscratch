#include <catch2/catch_test_macros.hpp>

#include <eggroll/trainer.hpp>
#include <model.hpp>
#include <tokenizer.hpp>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <vector>

TEST_CASE("EGGROLL trainer evaluates antithetic directions and updates weights",
          "[eggroll][trainer][.integration]") {
  const std::filesystem::path modelPath =
      std::filesystem::path(INFERENCE_SOURCE_ROOT) / "resources/model.gguf";
  inference::Model model(modelPath);
  model.load();
  inference::Tokenizer tokenizer;
  std::vector<int> tokens = tokenizer.encode("ROMEO:\n");
  std::vector<int> targets(tokens.size());
  for (std::size_t index = 0; index + 1 < tokens.size(); index++) {
    targets[index] = tokens[index + 1];
  }
  targets.back() = 0;

  float *before =
      model.forward(tokens.data(), static_cast<unsigned int>(tokens.size()));
  std::vector<inference::eggroll::EGGROLLMatrix> matrices =
      model.linearMatrices();
  REQUIRE_FALSE(matrices.empty());
  for (const inference::eggroll::EGGROLLMatrix &matrix : matrices) {
    CHECK(matrix.M > 0);
    CHECK(matrix.N > 0);
  }

  inference::eggroll::EGGROLLTrainer trainer(model, 1, 1, 0.0001F, 0.00001F,
                                             23);
  inference::eggroll::EGGROLLStepResult result =
      trainer.train(tokens.data(), targets.data(), tokens.size());
  REQUIRE(result.positiveFitness.size() == 1);
  REQUIRE(result.negativeFitness.size() == 1);
  REQUIRE(result.shapedFitness.size() == 1);
  CHECK(std::isfinite(result.baseFitness));
  CHECK(std::isfinite(result.positiveFitness[0]));
  CHECK(std::isfinite(result.negativeFitness[0]));
  CHECK((result.shapedFitness[0] == -1.0F || result.shapedFitness[0] == 0.0F ||
         result.shapedFitness[0] == 1.0F));

  float *after =
      model.forward(tokens.data(), static_cast<unsigned int>(tokens.size()));
  bool changed = false;
  std::size_t elementCount = tokens.size() * model.vocabularySize();
  for (std::size_t index = 0; index < elementCount; index++) {
    if (std::fabs(before[index] - after[index]) > 0.000001F) {
      changed = true;
      break;
    }
  }
  if (result.shapedFitness[0] != 0.0F) {
    CHECK(changed);
  }

  std::free(before);
  std::free(after);
  model.release();
}
