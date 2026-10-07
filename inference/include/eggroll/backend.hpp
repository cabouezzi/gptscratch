#pragma once

#include "eggroll/perturbation.hpp"
#include "export.hpp"

#include <cstddef>
#include <vector>

namespace MTL {
class Buffer;
}

namespace inference {
class MetalContext;
}

namespace inference::eggroll {

INFERENCE_PUBLIC float *
applyPerturbationMetal(const float *input, const float *output,
                       const EGGROLLPerturbation &perturbation,
                       std::size_t sequenceLength, float epsilon);

INFERENCE_PUBLIC float *
applyPerturbationMetal(MetalContext &context, const float *input,
                       const float *output,
                       const EGGROLLPerturbation &perturbation,
                       std::size_t sequenceLength, float epsilon);

INFERENCE_PUBLIC void
applyPerturbationMetal(MetalContext &context, MTL::Buffer *inputBuffer,
                       std::size_t inputOffset, MTL::Buffer *outputBuffer,
                       std::size_t outputOffset,
                       const EGGROLLPerturbation &perturbation,
                       std::size_t sequenceLength, float epsilon);

INFERENCE_PUBLIC float fitnessMetal(const float *logits, const int *targets,
                                    std::size_t sequenceLength,
                                    std::size_t vocabularySize);

INFERENCE_PUBLIC float fitnessMetal(MetalContext &context, const float *logits,
                                    const int *targets,
                                    std::size_t sequenceLength,
                                    std::size_t vocabularySize);

INFERENCE_PUBLIC float *updateWeightsMetal(
    const float *weights,
    const std::vector<const EGGROLLPerturbation *> &perturbations,
    const std::vector<float> &fitnesses, float learningRate);

INFERENCE_PUBLIC void updateWeightsMetal(
    MetalContext &context, MTL::Buffer *weightBuffer, std::size_t weightOffset,
    const std::vector<const EGGROLLPerturbation *> &perturbations,
    const std::vector<float> &fitnesses, float learningRate);

} // namespace inference::eggroll
