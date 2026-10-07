#pragma once

#include "export.hpp"
#include "eggroll/perturbation.hpp"
#include "metal/backend.hpp"
#include "metal/kv_cache.hpp"
#include "model_loader.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace inference {

class INFERENCE_PUBLIC Model {

public:
  explicit Model(const std::filesystem::path &path);
  Model(const std::filesystem::path &path, std::size_t weightShardByteLimit);
  ~Model();
  void load();
  void release();
  void save(const std::filesystem::path &path);
  bool loaded() const;
  float *embed(const int *tokens, std::size_t sequenceLength) const;
  float *forward(const int *tokens, unsigned int sequenceLength);
  float *forwardBatch(const int *tokens, std::size_t batchSize,
                      std::size_t sequenceLength);
  float *forwardPerturbed(
      const int *tokens, std::size_t sequenceLength,
      std::size_t targetWeightOffset,
      const eggroll::EGGROLLPerturbation &perturbation,
      float epsilon);
  float *forwardPerturbed(
      const int *tokens, std::size_t sequenceLength,
      const eggroll::EGGROLLCandidate &candidate, float epsilon);
  float *forwardPerturbedBatch(
      const int *tokens, std::size_t batchSize, std::size_t sequenceLength,
      const eggroll::EGGROLLCandidate &candidate, float epsilon);
  float fitness(const float *logits, const int *targets,
                std::size_t sequenceLength);
  std::vector<eggroll::EGGROLLMatrix> linearMatrices() const;
  void applyEGGROLLUpdate(
      const std::vector<eggroll::EGGROLLCandidate> &population,
      const std::vector<float> &fitnesses, float learningRate);
  float *prefill(const int *tokens, std::size_t sequenceLength);
  float *decode(int token);
  float *decodeSingleCommand(int token);
  float *decodeSingleCommandParallelHeads(int token);
  void resetCache();
  std::size_t cacheLength() const;
  std::size_t contextSize() const;
  std::size_t vocabularySize() const;
  std::size_t weightOffset(const std::string &name) const;
  std::size_t weightShardCount() const;

private:
  struct WeightLocation {
    std::size_t shardIndex;
    std::size_t localOffset;
    std::size_t byteCount;
  };

  struct WeightBinding {
    MTL::Buffer *buffer;
    std::size_t offset;
  };

  struct Workspace {
    MTL::Buffer *buffer;
    std::size_t scratchAOffset;
    std::size_t scratchBOffset;
    std::size_t residualOffset;
    std::size_t queryOffset;
    std::size_t keyOffset;
    std::size_t valueOffset;
  };

  float *embedAt(const int *tokens, std::size_t sequenceLength,
                 std::size_t startPosition) const;
  float *embedBatch(const int *tokens, std::size_t batchSize,
                    std::size_t sequenceLength) const;
  Workspace allocateWorkspace(std::size_t sequenceLength);
  float *execute(const int *tokens, std::size_t sequenceLength,
                 std::size_t startPosition, KVCache *cache,
                 std::size_t cachePosition,
                 bool singleCommand = false,
                 bool parallelHeads = false,
                 std::size_t batchSize = 1);
  std::size_t forwardBlock(Workspace &workspace,
                           std::size_t inputOffset,
                           std::size_t sequenceLength,
                           std::size_t blockIndex,
                           KVCache *cache = nullptr,
                           std::size_t cachePosition = 0,
                           MetalCommandBatch *batch = nullptr,
                           bool parallelHeads = false,
                           std::size_t batchSize = 1);
  std::size_t finish(Workspace &workspace, std::size_t inputOffset,
                     std::size_t sequenceLength,
                     MetalCommandBatch *batch = nullptr);
  void linear(Workspace &workspace, std::size_t inputOffset,
              std::size_t weightOffset, std::size_t outputOffset,
              std::size_t sequenceLength, std::size_t inputSize,
              std::size_t outputSize,
              MetalCommandBatch *batch = nullptr,
              MTL::Buffer *outputBuffer = nullptr);
  void ensureCache();
  float *decodeCached(int token, bool singleCommand, bool parallelHeads);
  float *executePerturbed(const int *tokens, std::size_t sequenceLength);
  float *executePerturbedBatch(const int *tokens, std::size_t batchSize,
                               std::size_t sequenceLength);
  void initialize(const std::filesystem::path &path,
                  std::size_t weightShardByteLimit);
  WeightBinding resolveWeight(std::size_t globalOffset) const;
  MetalContext metal_context;
  std::unique_ptr<KVCache> kv_cache;
  std::vector<MTL::Buffer *> model_buffers;
  std::vector<std::size_t> model_buffer_byte_counts;
  std::unordered_map<std::string, std::size_t> weight_offsets;
  std::unordered_map<std::size_t, WeightLocation> weight_locations;
  std::size_t weight_shard_byte_limit;
  ModelParameters parameters;
  std::size_t input_size;
  std::size_t output_size;
  std::size_t context_size;
  std::unordered_map<std::size_t, const eggroll::EGGROLLPerturbation *>
      active_perturbations;
  std::unordered_set<std::size_t> applied_perturbations;
  std::unordered_map<std::size_t, eggroll::EGGROLLMatrix>
      linear_matrices;
  float active_epsilon;
};

} // namespace inference
