#include <eggroll/trainer.hpp>

#include <model.hpp>

#include <cmath>
#include <cstdlib>
#include <limits>
#include <stdexcept>

namespace inference::eggroll {

EGGROLLTrainer::EGGROLLTrainer(Model &model, std::size_t populationSize,
                               std::size_t rank, float sigma,
                               float learningRate, std::uint64_t seed) {
  if (populationSize == 0 || rank == 0 || !std::isfinite(sigma) ||
      sigma <= 0.0F || !std::isfinite(learningRate) || learningRate <= 0.0F) {
    throw std::invalid_argument("EGGROLL trainer settings are invalid");
  }
  this->model = &model;
  this->populationSize = populationSize;
  this->rank = rank;
  this->sigma = sigma;
  this->learningRate = learningRate;
  this->nextSeed = seed;
}

EGGROLLStepResult EGGROLLTrainer::train(const int *tokens, const int *targets, std::size_t sequenceLength) {
  return this->trainBatch(tokens, targets, 1, sequenceLength);
}

EGGROLLStepResult EGGROLLTrainer::trainBatch(const int *tokens, const int *targets, std::size_t batchSize, std::size_t sequenceLength) {
  if (tokens == nullptr || targets == nullptr || batchSize == 0 || sequenceLength == 0) {
    throw std::invalid_argument("EGGROLL training inputs are invalid");
  }
  if (sequenceLength > std::numeric_limits<unsigned int>::max()) {
    throw std::invalid_argument("EGGROLL sequence length is too large");
  }

  std::size_t rowCount = batchSize * sequenceLength;
  float *baseLogits = this->model->forwardBatch(tokens, batchSize, sequenceLength);
  float baseFitness = 0.0F;
  try {
    baseFitness = this->model->fitness(baseLogits, targets, rowCount);
    std::free(baseLogits);
  } catch (...) {
    std::free(baseLogits);
    throw;
  }

  // EGGROLLMatrix is just MxN deltas
  std::vector<EGGROLLMatrix> matrices = this->model->linearMatrices();
  if (matrices.empty()) {
    throw std::logic_error("The model did not execute any linear matrices");
  }
  // Allocate/create candidates
  // list of perturbations for every matrix in the model
  std::vector<EGGROLLCandidate> population;
  population.reserve(this->populationSize);
  for (std::size_t candidate = 0; candidate < this->populationSize; candidate++) {
    population.push_back(generateCandidate(matrices, this->rank, this->nextSeed++));
  }
  // Allocate fitnesses
  EGGROLLStepResult result;
  result.baseFitness = baseFitness;
  result.positiveFitness.reserve(this->populationSize);
  result.negativeFitness.reserve(this->populationSize);
  result.shapedFitness.reserve(this->populationSize);

  for (const EGGROLLCandidate &candidate : population) {
    float *positiveLogits = this->model->forwardPerturbedBatch(tokens, batchSize, sequenceLength, candidate, this->sigma);
    float positiveFitness = 0.0F;
    try {
      positiveFitness = this->model->fitness(positiveLogits, targets, rowCount);
      std::free(positiveLogits);
    } catch (...) {
      std::free(positiveLogits);
      throw;
    }
    float *negativeLogits = this->model->forwardPerturbedBatch(tokens, batchSize, sequenceLength, candidate, -this->sigma);
    float negativeFitness = 0.0F;
    try {
      negativeFitness = this->model->fitness(negativeLogits, targets, rowCount);
      std::free(negativeLogits);
    } catch (...) {
      std::free(negativeLogits);
      throw;
    }
    // determine sign, used in W += (alpha / P) sum_i(s_i E_i)
    float shapedFitness = 0.0F;
    if (positiveFitness > negativeFitness) {
      shapedFitness = 1.0F;
    } else if (positiveFitness < negativeFitness) {
      shapedFitness = -1.0F;
    }
    result.positiveFitness.push_back(positiveFitness);
    result.negativeFitness.push_back(negativeFitness);
    result.shapedFitness.push_back(shapedFitness);
  }

  this->model->applyEGGROLLUpdate(population, result.shapedFitness, this->learningRate);
  return result;
}

} // namespace inference::eggroll
