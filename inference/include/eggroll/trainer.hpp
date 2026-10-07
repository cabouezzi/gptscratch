#pragma once

#include "eggroll/perturbation.hpp"
#include "export.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace inference {
class Model;
}

namespace inference::eggroll {

struct INFERENCE_PUBLIC EGGROLLStepResult {
  float baseFitness;
  std::vector<float> positiveFitness;
  std::vector<float> negativeFitness;
  std::vector<float> shapedFitness;
};

class INFERENCE_PUBLIC EGGROLLTrainer {

public:
  EGGROLLTrainer(Model &model, std::size_t populationSize, std::size_t rank,
                 float sigma, float learningRate, std::uint64_t seed);

  EGGROLLStepResult train(const int *tokens, const int *targets,
                          std::size_t sequenceLength);

  EGGROLLStepResult trainBatch(const int *tokens, const int *targets,
                               std::size_t batchSize,
                               std::size_t sequenceLength);

private:
  Model *model;
  std::size_t populationSize;
  std::size_t rank;
  float sigma;
  float learningRate;
  std::uint64_t nextSeed;
};

} // namespace inference::eggroll
