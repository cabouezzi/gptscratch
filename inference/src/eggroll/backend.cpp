#include <eggroll/backend.hpp>

#include <matmulflags.hpp>
#include <metal/backend.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <vector>

namespace inference::eggroll {

namespace {

std::size_t checkedProduct(std::size_t left, std::size_t right) {
  if (left != 0 && right > std::numeric_limits<std::size_t>::max() / left) {
    throw std::overflow_error("EGGROLL buffer size overflows");
  }
  return left * right;
}

} // namespace

float *applyPerturbationMetal(const float *input, const float *output,
                              const EGGROLLPerturbation &perturbation,
                              std::size_t sequenceLength, float epsilon) {
  MetalContext context;
  return applyPerturbationMetal(context, input, output, perturbation,
                                sequenceLength, epsilon);
}

float *applyPerturbationMetal(MetalContext &context, const float *input,
                              const float *output,
                              const EGGROLLPerturbation &perturbation,
                              std::size_t sequenceLength, float epsilon) {
  if (input == nullptr || output == nullptr) {
    throw std::invalid_argument("EGGROLL input and output cannot be null");
  }
  if (sequenceLength == 0) {
    throw std::invalid_argument("EGGROLL sequence length must be positive");
  }

  std::size_t inputCount = checkedProduct(sequenceLength, perturbation.M);
  std::size_t outputCount = checkedProduct(sequenceLength, perturbation.N);
  std::size_t inputByteCount = checkedProduct(inputCount, sizeof(float));
  std::size_t outputByteCount = checkedProduct(outputCount, sizeof(float));
  if (inputByteCount >
      std::numeric_limits<std::size_t>::max() - outputByteCount) {
    throw std::overflow_error("EGGROLL buffer size overflows");
  }

  MTL::Buffer *buffer =
      allocate_metal_buffer(context, inputByteCount + outputByteCount);
  try {
    write_metal_buffer(buffer, 0, input, inputCount);
    write_metal_buffer(buffer, inputByteCount, output, outputCount);
    applyPerturbationMetal(context, buffer, 0, buffer, inputByteCount,
                           perturbation, sequenceLength, epsilon);
    float *result = read_metal_buffer(buffer, inputByteCount, outputCount);
    release_metal_buffer(buffer);
    return result;
  } catch (...) {
    release_metal_buffer(buffer);
    throw;
  }
}

void applyPerturbationMetal(MetalContext &context, MTL::Buffer *inputBuffer,
                            std::size_t inputOffset, MTL::Buffer *outputBuffer,
                            std::size_t outputOffset,
                            const EGGROLLPerturbation &perturbation,
                            std::size_t sequenceLength, float epsilon) {
  if (sequenceLength == 0) {
    throw std::invalid_argument("EGGROLL sequence length must be positive");
  }

  std::size_t aCount = checkedProduct(perturbation.N, perturbation.r);
  std::size_t bCount = checkedProduct(perturbation.M, perturbation.r);
  std::size_t xbCount = checkedProduct(sequenceLength, perturbation.r);
  std::size_t aByteCount = checkedProduct(aCount, sizeof(float));
  std::size_t bByteCount = checkedProduct(bCount, sizeof(float));
  std::size_t xbByteCount = checkedProduct(xbCount, sizeof(float));
  if (aByteCount > std::numeric_limits<std::size_t>::max() - bByteCount ||
      aByteCount + bByteCount >
          std::numeric_limits<std::size_t>::max() - xbByteCount) {
    throw std::overflow_error("EGGROLL buffer size overflows");
  }

  std::size_t aOffset = 0;
  std::size_t bOffset = aByteCount;
  std::size_t xbOffset = aByteCount + bByteCount;
  MTL::Buffer *perturbationBuffer =
      allocate_metal_buffer(context, aByteCount + bByteCount + xbByteCount);

  try {
    write_metal_buffer(perturbationBuffer, aOffset, perturbation.A, aCount);
    write_metal_buffer(perturbationBuffer, bOffset, perturbation.B, bCount);

    MetalCommandBatch batch(context);
    batch.matmul(inputBuffer, inputOffset, MatMulFlag::NO_TRANSPOSE,
                 perturbationBuffer, bOffset, MatMulFlag::NO_TRANSPOSE,
                 perturbationBuffer, xbOffset, sequenceLength, perturbation.M,
                 perturbation.r, true);
    batch.eggrollOutputAdd(outputBuffer, outputOffset, perturbationBuffer,
                           xbOffset, perturbationBuffer, aOffset,
                           sequenceLength, perturbation.N, perturbation.r,
                           epsilon);
    batch.commitAndWait();
    release_metal_buffer(perturbationBuffer);
  } catch (...) {
    release_metal_buffer(perturbationBuffer);
    throw;
  }
}

float fitnessMetal(const float *logits, const int *targets,
                   std::size_t sequenceLength, std::size_t vocabularySize) {
  MetalContext context;
  return fitnessMetal(context, logits, targets, sequenceLength, vocabularySize);
}

float fitnessMetal(MetalContext &context, const float *logits,
                   const int *targets, std::size_t sequenceLength,
                   std::size_t vocabularySize) {
  if (logits == nullptr || targets == nullptr) {
    throw std::invalid_argument("EGGROLL fitness inputs cannot be null");
  }
  if (sequenceLength == 0 || vocabularySize == 0) {
    throw std::invalid_argument("EGGROLL fitness dimensions must be positive");
  }

  std::size_t logitsCount = checkedProduct(sequenceLength, vocabularySize);
  std::size_t logitsByteCount = checkedProduct(logitsCount, sizeof(float));
  std::size_t vectorByteCount = checkedProduct(sequenceLength, sizeof(float));
  if (logitsByteCount >
          std::numeric_limits<std::size_t>::max() - vectorByteCount ||
      logitsByteCount + vectorByteCount >
          std::numeric_limits<std::size_t>::max() - vectorByteCount ||
      logitsByteCount + vectorByteCount * 2 >
          std::numeric_limits<std::size_t>::max() - sizeof(float)) {
    throw std::overflow_error("EGGROLL fitness buffer size overflows");
  }

  std::vector<float> metalTargets(sequenceLength);
  for (std::size_t row = 0; row < sequenceLength; row++) {
    if (targets[row] < 0 ||
        static_cast<std::size_t>(targets[row]) >= vocabularySize) {
      throw std::out_of_range("EGGROLL fitness target exceeds vocabulary");
    }
    metalTargets[row] = static_cast<float>(targets[row]);
  }

  std::size_t logitsOffset = 0;
  std::size_t targetsOffset = logitsByteCount;
  std::size_t lossesOffset = targetsOffset + vectorByteCount;
  std::size_t fitnessOffset = lossesOffset + vectorByteCount;
  MTL::Buffer *buffer =
      allocate_metal_buffer(context, fitnessOffset + sizeof(float));
  try {
    write_metal_buffer(buffer, logitsOffset, logits, logitsCount);
    write_metal_buffer(buffer, targetsOffset, metalTargets.data(),
                       sequenceLength);
    MetalCommandBatch batch(context);
    batch.eggrollFitness(buffer, logitsOffset, buffer, targetsOffset, buffer,
                         lossesOffset, fitnessOffset, sequenceLength,
                         vocabularySize);
    batch.commitAndWait();
    float *result = read_metal_buffer(buffer, fitnessOffset, 1);
    float fitness = result[0];
    std::free(result);
    release_metal_buffer(buffer);
    return fitness;
  } catch (...) {
    release_metal_buffer(buffer);
    throw;
  }
}

float *updateWeightsMetal(
    const float *weights,
    const std::vector<const EGGROLLPerturbation *> &perturbations,
    const std::vector<float> &fitnesses, float learningRate) {
  if (weights == nullptr || perturbations.empty() ||
      perturbations.front() == nullptr) {
    throw std::invalid_argument("EGGROLL update inputs cannot be empty");
  }

  const EGGROLLPerturbation &first = *perturbations.front();
  std::size_t weightCount = checkedProduct(first.M, first.N);
  MetalContext context;
  MTL::Buffer *buffer = allocate_metal_buffer(
      context, checkedProduct(weightCount, sizeof(float)));
  try {
    write_metal_buffer(buffer, 0, weights, weightCount);
    updateWeightsMetal(context, buffer, 0, perturbations, fitnesses,
                       learningRate);
    float *result = read_metal_buffer(buffer, 0, weightCount);
    release_metal_buffer(buffer);
    return result;
  } catch (...) {
    release_metal_buffer(buffer);
    throw;
  }
}

void updateWeightsMetal(
    MetalContext &context, MTL::Buffer *weightBuffer, std::size_t weightOffset,
    const std::vector<const EGGROLLPerturbation *> &perturbations,
    const std::vector<float> &fitnesses, float learningRate) {
  if (perturbations.empty() || fitnesses.size() != perturbations.size() ||
      perturbations.front() == nullptr || !std::isfinite(learningRate)) {
    throw std::invalid_argument("EGGROLL update inputs are invalid");
  }

  const EGGROLLPerturbation &first = *perturbations.front();
  std::size_t populationSize = perturbations.size();
  std::size_t aCount =
      checkedProduct(populationSize, checkedProduct(first.N, first.r));
  std::size_t bCount =
      checkedProduct(populationSize, checkedProduct(first.M, first.r));
  std::size_t aByteCount = checkedProduct(aCount, sizeof(float));
  std::size_t bByteCount = checkedProduct(bCount, sizeof(float));
  std::size_t fitnessByteCount = checkedProduct(populationSize, sizeof(float));
  if (aByteCount > std::numeric_limits<std::size_t>::max() - bByteCount ||
      aByteCount + bByteCount >
          std::numeric_limits<std::size_t>::max() - fitnessByteCount) {
    throw std::overflow_error("EGGROLL update buffer size overflows");
  }

  std::vector<float> packedA(aCount);
  std::vector<float> packedB(bCount);
  std::size_t aStride = first.N * first.r;
  std::size_t bStride = first.M * first.r;
  for (std::size_t candidate = 0; candidate < populationSize; candidate++) {
    const EGGROLLPerturbation *perturbation = perturbations[candidate];
    if (perturbation == nullptr || perturbation->M != first.M ||
        perturbation->N != first.N || perturbation->r != first.r ||
        !std::isfinite(fitnesses[candidate])) {
      throw std::invalid_argument(
          "EGGROLL population perturbations must have matching dimensions");
    }
    std::copy_n(perturbation->A, aStride, packedA.data() + candidate * aStride);
    std::copy_n(perturbation->B, bStride, packedB.data() + candidate * bStride);
  }

  std::size_t aOffset = 0;
  std::size_t bOffset = aByteCount;
  std::size_t fitnessOffset = aByteCount + bByteCount;
  MTL::Buffer *perturbationBuffer = allocate_metal_buffer(
      context, aByteCount + bByteCount + fitnessByteCount);
  try {
    write_metal_buffer(perturbationBuffer, aOffset, packedA.data(), aCount);
    write_metal_buffer(perturbationBuffer, bOffset, packedB.data(), bCount);
    write_metal_buffer(perturbationBuffer, fitnessOffset, fitnesses.data(),
                       populationSize);
    MetalCommandBatch batch(context);
    batch.eggrollWeightUpdate(
        weightBuffer, weightOffset, perturbationBuffer, aOffset, bOffset,
        fitnessOffset, first.M, first.N, first.r, populationSize,
        learningRate / static_cast<float>(populationSize));
    batch.commitAndWait();
    release_metal_buffer(perturbationBuffer);
  } catch (...) {
    release_metal_buffer(perturbationBuffer);
    throw;
  }
}

} // namespace inference::eggroll
