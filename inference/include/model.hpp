#pragma once

#include "export.hpp"
#include "metal/backend.hpp"
#include "metal/kv_cache.hpp"
#include "model_loader.hpp"

#include <cstddef>
#include <filesystem>
#include <string>
#include <unordered_map>

namespace inference {

class INFERENCE_PUBLIC Model {

public:
  explicit Model(const std::filesystem::path &path);
  ~Model();
  void load();
  void release();
  bool loaded() const;
  float *embed(const int *tokens, std::size_t sequenceLength) const;
  float *forward(const int *tokens, unsigned int sequenceLength);
  float *prefill(const int *tokens, std::size_t sequenceLength);
  float *decode(int token);
  float *decodeSingleCommand(int token);
  float *decodeSingleCommandParallelHeads(int token);
  void resetCache();
  std::size_t cacheLength() const;
  std::size_t contextSize() const;
  std::size_t vocabularySize() const;

private:
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
  Workspace allocateWorkspace(std::size_t sequenceLength);
  float *execute(const int *tokens, std::size_t sequenceLength,
                 std::size_t startPosition, KVCache *cache,
                 std::size_t cachePosition,
                 bool singleCommand = false,
                 bool parallelHeads = false);
  std::size_t forwardBlock(Workspace &workspace,
                           std::size_t inputOffset,
                           std::size_t sequenceLength,
                           std::size_t blockIndex,
                           KVCache *cache = nullptr,
                           std::size_t cachePosition = 0,
                           MetalCommandBatch *batch = nullptr,
                           bool parallelHeads = false);
  std::size_t finish(Workspace &workspace, std::size_t inputOffset,
                     std::size_t sequenceLength,
                     MetalCommandBatch *batch = nullptr);
  void linear(Workspace &workspace, std::size_t inputOffset,
              const std::string &weightName, std::size_t outputOffset,
              std::size_t sequenceLength, std::size_t inputSize,
              std::size_t outputSize,
              MetalCommandBatch *batch = nullptr,
              MTL::Buffer *outputBuffer = nullptr);
  void ensureCache();
  float *decodeCached(int token, bool singleCommand, bool parallelHeads);
  std::size_t weightOffset(const std::string &name) const;

  MetalContext metal_context;
  std::unique_ptr<KVCache> kv_cache;
  MTL::Buffer *model_buffer = nullptr;
  std::unordered_map<std::string, std::size_t> weight_offsets;
  ModelParameters parameters;
  std::size_t input_size;
  std::size_t output_size;
  std::size_t context_size;
};

} // namespace inference
