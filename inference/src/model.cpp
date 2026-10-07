#include <model.hpp>

#include <eggroll/backend.hpp>
#include <metal/backend.hpp>

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <format>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <vector>

namespace inference {

namespace {

float *allocateFloats(std::size_t count) {
  float *buffer = static_cast<float *>(std::malloc(count * sizeof(float)));
  if (buffer == nullptr) {
    throw std::bad_alloc();
  }
  return buffer;
} // namespace

std::size_t alignBytes(std::size_t value, std::size_t alignment) {
  std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
}

}

Model::Model(const std::filesystem::path &path) {
  this->initialize(path, 512ULL * 1024ULL * 1024ULL);
}

Model::Model(const std::filesystem::path &path,
             std::size_t weightShardByteLimit) {
  this->initialize(path, weightShardByteLimit);
}

void Model::initialize(const std::filesystem::path &path,
                       std::size_t weightShardByteLimit) {
  if (weightShardByteLimit == 0) {
    throw std::invalid_argument("Model weight shard size must be positive");
  }
  this->parameters = ModelParameters::loadGGUF(path);

  const GGUFTensor &tokenEmbedding =
      this->parameters.tensor("token_embedding_table.weight");
  const GGUFTensor &positionEmbedding =
      this->parameters.tensor("position_embedding_table.weight");

  this->input_size = static_cast<std::size_t>(tokenEmbedding.shape[1]);
  this->output_size = static_cast<std::size_t>(tokenEmbedding.shape[0]);
  this->context_size = static_cast<std::size_t>(positionEmbedding.shape[0]);
  this->weight_shard_byte_limit = weightShardByteLimit;
  this->active_epsilon = 0.0F;
}

Model::~Model() { this->release(); }

void Model::load() {
  if (this->loaded()) {
    return;
  }

  constexpr std::size_t weightAlignment = 256;
  std::vector<std::string> names;
  names.reserve(this->parameters.allTensors().size());
  for (const auto &[name, tensor] : this->parameters.allTensors()) {
    names.push_back(name);
  }
  std::sort(names.begin(), names.end());

  std::unordered_map<std::string, std::size_t> offsets;
  std::unordered_map<std::size_t, WeightLocation> locations;
  std::vector<std::size_t> shardByteCounts;
  std::size_t globalOffset = 0;
  std::size_t shardOffset = 0;
  std::size_t shardIndex = 0;
  for (const std::string &name : names) {
    const GGUFTensor &tensor = this->parameters.tensor(name);
    std::size_t tensorByteCount = static_cast<std::size_t>(tensor.elementCount) * sizeof(float);
    if (tensorByteCount > this->weight_shard_byte_limit) {
      throw std::runtime_error("Model tensor exceeds the configured GPU weight shard size: " + name);
    }
    globalOffset = alignBytes(globalOffset, weightAlignment);
    shardOffset = alignBytes(shardOffset, weightAlignment);
    if (shardOffset != 0 && (shardOffset > this->weight_shard_byte_limit || tensorByteCount > this->weight_shard_byte_limit - shardOffset)) {
      shardByteCounts.push_back(shardOffset);
      shardIndex++;
      shardOffset = 0;
    }
    offsets.emplace(name, globalOffset);
    locations.emplace(globalOffset, WeightLocation{.shardIndex = shardIndex, .localOffset = shardOffset, .byteCount = tensorByteCount});
    globalOffset += tensorByteCount;
    shardOffset += tensorByteCount;
  }
  if (!names.empty()) {
    shardByteCounts.push_back(shardOffset);
  }

  std::vector<MTL::Buffer *> gpuWeights;
  gpuWeights.reserve(shardByteCounts.size());
  try {
    for (std::size_t currentShard = 0; currentShard < shardByteCounts.size(); currentShard++) {
      MTL::Buffer *staging = allocate_metal_buffer(this->metal_context, shardByteCounts[currentShard]);
      MTL::Buffer *gpuShard = nullptr;
      try {
        for (const std::string &name : names) {
          const WeightLocation &location = locations.at(offsets.at(name));
          if (location.shardIndex != currentShard) {
            continue;
          }
          const GGUFTensor &tensor = this->parameters.tensor(name);
          write_metal_buffer(staging, location.localOffset, this->parameters.data(name), static_cast<std::size_t>(tensor.elementCount));
        }
        gpuShard = allocate_private_metal_buffer(this->metal_context, shardByteCounts[currentShard]);
        copy_metal_buffer(this->metal_context, staging, 0, gpuShard, 0, shardByteCounts[currentShard]);
        release_metal_buffer(staging);
        gpuWeights.push_back(gpuShard);
      } catch (...) {
        release_metal_buffer(staging);
        release_metal_buffer(gpuShard);
        throw;
      }
    }
    this->model_buffers = std::move(gpuWeights);
    this->model_buffer_byte_counts = std::move(shardByteCounts);
    this->weight_offsets = std::move(offsets);
    this->weight_locations = std::move(locations);
  } catch (...) {
    for (MTL::Buffer *buffer : gpuWeights) {
      release_metal_buffer(buffer);
    }
    throw;
  }
}

void Model::release() {
  this->kv_cache.reset();
  for (MTL::Buffer *buffer : this->model_buffers) {
    release_metal_buffer(buffer);
  }
  this->model_buffers.clear();
  this->model_buffer_byte_counts.clear();
  this->weight_offsets.clear();
  this->weight_locations.clear();
}

void Model::save(const std::filesystem::path &path) {
  if (!this->loaded()) {
    throw std::runtime_error("Load model weights before saving");
  }
  this->parameters.saveGGUF(path, [this](const std::string &name, const GGUFTensor &tensor) {
    WeightBinding weight = this->resolveWeight(this->weightOffset(name));
    std::size_t byteCount = static_cast<std::size_t>(tensor.elementCount) * sizeof(float);
    MTL::Buffer *staging = allocate_metal_buffer(this->metal_context, byteCount);
    float *values = nullptr;
    try {
      copy_metal_buffer(this->metal_context, weight.buffer, weight.offset, staging, 0, byteCount);
      values = read_metal_buffer(staging, 0, static_cast<std::size_t>(tensor.elementCount));
      std::vector<float> result(values, values + tensor.elementCount);
      std::free(values);
      release_metal_buffer(staging);
      return result;
    } catch (...) {
      std::free(values);
      release_metal_buffer(staging);
      throw;
    }
  });
}

bool Model::loaded() const { return !this->model_buffers.empty(); }

std::size_t Model::weightOffset(const std::string &name) const {
  if (!this->loaded()) {
    throw std::runtime_error("Model weights are not loaded into GPU memory");
  }
  auto offset = this->weight_offsets.find(name);
  if (offset == this->weight_offsets.end()) {
    throw std::out_of_range("GPU model tensor not found: " + name);
  }
  return offset->second;
}

std::size_t Model::weightShardCount() const { return this->model_buffers.size(); }

Model::WeightBinding Model::resolveWeight(std::size_t globalOffset) const {
  auto location = this->weight_locations.find(globalOffset);
  if (location == this->weight_locations.end() || location->second.shardIndex >= this->model_buffers.size()) {
    throw std::out_of_range("GPU model weight offset was not found");
  }
  return WeightBinding{.buffer = this->model_buffers[location->second.shardIndex], .offset = location->second.localOffset};
}

float *Model::embed(const int *tokens, std::size_t sequenceLength) const {
  return this->embedAt(tokens, sequenceLength, 0);
}

float *Model::embedAt(const int *tokens, std::size_t sequenceLength,
                      std::size_t startPosition) const {
  if (startPosition > this->context_size ||
      sequenceLength > this->context_size - startPosition) {
    throw std::invalid_argument("Sequence exceeds the model context length");
  }

  const float *tokenEmbeddingWeights =
      this->parameters.data("token_embedding_table.weight");
  const float *positionEmbeddingWeights =
      this->parameters.data("position_embedding_table.weight");

  float *embeddings =
      allocateFloats(sequenceLength * this->input_size);
  for (std::size_t position = 0; position < sequenceLength; position++) {
    int token = tokens[position];
    if (token < 0 || static_cast<std::size_t>(token) >= this->output_size) {
      std::free(embeddings);
      throw std::out_of_range("Token ID exceeds the model vocabulary");
    }

    for (std::size_t channel = 0; channel < this->input_size; channel++) {
      std::size_t embeddingIndex =
          position * this->input_size + channel;
      std::size_t tokenEmbeddingIndex =
          static_cast<std::size_t>(token) * this->input_size + channel;
      std::size_t positionEmbeddingIndex =
          (startPosition + position) * this->input_size + channel;
      embeddings[embeddingIndex] =
          tokenEmbeddingWeights[tokenEmbeddingIndex] +
          positionEmbeddingWeights[positionEmbeddingIndex];
    }
  }
  return embeddings;
}

float *Model::embedBatch(const int *tokens, std::size_t batchSize, std::size_t sequenceLength) const {
  if (batchSize == 0 || sequenceLength == 0 || sequenceLength > this->context_size) {
    throw std::invalid_argument("Model batch dimensions are invalid");
  }
  const float *tokenEmbeddingWeights = this->parameters.data("token_embedding_table.weight");
  const float *positionEmbeddingWeights = this->parameters.data("position_embedding_table.weight");
  std::size_t rowCount = batchSize * sequenceLength;
  float *embeddings = allocateFloats(rowCount * this->input_size);
  for (std::size_t batch = 0; batch < batchSize; batch++) {
    for (std::size_t position = 0; position < sequenceLength; position++) {
      std::size_t row = batch * sequenceLength + position;
      int token = tokens[row];
      if (token < 0 || static_cast<std::size_t>(token) >= this->output_size) {
        std::free(embeddings);
        throw std::out_of_range("Token ID exceeds the model vocabulary");
      }
      for (std::size_t channel = 0; channel < this->input_size; channel++) {
        std::size_t embeddingIndex = row * this->input_size + channel;
        std::size_t tokenEmbeddingIndex = static_cast<std::size_t>(token) * this->input_size + channel;
        std::size_t positionEmbeddingIndex = position * this->input_size + channel;
        embeddings[embeddingIndex] = tokenEmbeddingWeights[tokenEmbeddingIndex] + positionEmbeddingWeights[positionEmbeddingIndex];
      }
    }
  }
  return embeddings;
}

Model::Workspace Model::allocateWorkspace(std::size_t sequenceLength) {
  std::size_t feedForwardSize = this->input_size * 4;
  std::size_t scratchWidth =
      std::max({this->input_size, feedForwardSize, this->output_size});
  std::size_t scratchByteCount =
      sequenceLength * scratchWidth * sizeof(float);
  std::size_t activationByteCount =
      sequenceLength * this->input_size * sizeof(float);

  Workspace workspace = {
      .buffer = nullptr,
      .scratchAOffset = 0,
      .scratchBOffset = scratchByteCount,
      .residualOffset = scratchByteCount * 2,
      .queryOffset = scratchByteCount * 2 + activationByteCount,
      .keyOffset = scratchByteCount * 2 + activationByteCount * 2,
      .valueOffset = scratchByteCount * 2 + activationByteCount * 3,
  };
  std::size_t workspaceByteCount =
      workspace.valueOffset + activationByteCount;
  workspace.buffer =
      allocate_metal_buffer(this->metal_context, workspaceByteCount);
  return workspace;
}

void Model::linear(Workspace &workspace, std::size_t inputOffset,
                   std::size_t weightOffset,
                   std::size_t outputOffset,
                   std::size_t sequenceLength, std::size_t inputSize,
                   std::size_t outputSize, MetalCommandBatch *batch,
                   MTL::Buffer *outputBuffer) {
  if (outputBuffer == nullptr) {
    outputBuffer = workspace.buffer;
  }
  auto [matrix, inserted] = this->linear_matrices.emplace(
      weightOffset, eggroll::EGGROLLMatrix{
                        .weightOffset = weightOffset,
                        .M = inputSize,
                        .N = outputSize,
                    });
  if (!inserted &&
      (matrix->second.M != inputSize || matrix->second.N != outputSize)) {
    throw std::logic_error(
        "A model weight offset was used with inconsistent dimensions");
  }
  WeightBinding weight = this->resolveWeight(weightOffset);
  if (sequenceLength == 1) {
    if (batch != nullptr) {
      batch->matvecmul(workspace.buffer, inputOffset, weight.buffer, weight.offset, outputBuffer, outputOffset, inputSize, outputSize);
    } else {
      matvecmul_metal(this->metal_context, workspace.buffer, inputOffset, weight.buffer, weight.offset, outputBuffer, outputOffset, inputSize, outputSize);
    }
  } else {
    matmul_metal(this->metal_context, workspace.buffer, inputOffset, MatMulFlag::NO_TRANSPOSE, weight.buffer, weight.offset, MatMulFlag::TRANSPOSE, outputBuffer, outputOffset, sequenceLength, inputSize, outputSize, true);
  }

  auto active = this->active_perturbations.find(weightOffset);
  if (active != this->active_perturbations.end()) {
    if (batch != nullptr) {
      throw std::logic_error(
          "Perturbed forward does not support command batching");
    }
    const eggroll::EGGROLLPerturbation &perturbation = *active->second;
    if (perturbation.M != inputSize || perturbation.N != outputSize) {
      throw std::invalid_argument(
          "EGGROLL perturbation dimensions do not match the matrix");
    }
    eggroll::applyPerturbationMetal(
        this->metal_context, workspace.buffer, inputOffset,
        outputBuffer, outputOffset, perturbation,
        sequenceLength, this->active_epsilon);
    this->applied_perturbations.insert(weightOffset);
  }
}

std::size_t Model::forwardBlock(Workspace &workspace,
                                std::size_t inputOffset,
                                std::size_t sequenceLength,
                                std::size_t blockIndex, KVCache *cache,
                                std::size_t cachePosition,
                                MetalCommandBatch *batch,
                                bool parallelHeads,
                                std::size_t batchSize) {
  std::string block = std::format("blocks.{}", blockIndex);
  std::size_t numHeads = std::get<std::uint64_t>(
      this->parameters.metadata("gptscratch.attention.head_count"));
  std::size_t headSize = this->input_size / numHeads;
  std::size_t rowCount = batchSize * sequenceLength;
  std::size_t elementCount = rowCount * this->input_size;
  std::size_t elementsPerHead = rowCount * headSize;
  WeightBinding ln1Gamma = this->resolveWeight(this->weightOffset(block + ".ln1.weight"));
  WeightBinding ln1Beta = this->resolveWeight(this->weightOffset(block + ".ln1.bias"));

  if (batch == nullptr) {
    layer_norm_metal(this->metal_context, workspace.buffer, inputOffset, ln1Gamma.buffer, ln1Gamma.offset, ln1Beta.buffer, ln1Beta.offset, workspace.buffer, workspace.residualOffset, rowCount, this->input_size);
  } else {
    batch->layerNorm(workspace.buffer, inputOffset, ln1Gamma.buffer, ln1Gamma.offset, ln1Beta.buffer, ln1Beta.offset, workspace.buffer, workspace.residualOffset, rowCount, this->input_size);
  }

  if (parallelHeads) {
    if (batch == nullptr || cache == nullptr || sequenceLength != 1 || batchSize != 1) {
      throw std::invalid_argument(
          "Parallel heads require one cached command batch");
    }
    std::vector<std::size_t> queryWeightOffsets;
    std::vector<std::size_t> keyWeightOffsets;
    std::vector<std::size_t> valueWeightOffsets;
    std::vector<MTL::Buffer *> queryWeightBuffers;
    std::vector<MTL::Buffer *> keyWeightBuffers;
    std::vector<MTL::Buffer *> valueWeightBuffers;
    std::vector<std::size_t> queryOutputOffsets;
    std::vector<std::size_t> keyOutputOffsets;
    std::vector<std::size_t> valueOutputOffsets;
    queryWeightOffsets.reserve(numHeads);
    keyWeightOffsets.reserve(numHeads);
    valueWeightOffsets.reserve(numHeads);
    queryWeightBuffers.reserve(numHeads);
    keyWeightBuffers.reserve(numHeads);
    valueWeightBuffers.reserve(numHeads);
    queryOutputOffsets.reserve(numHeads);
    keyOutputOffsets.reserve(numHeads);
    valueOutputOffsets.reserve(numHeads);
    for (std::size_t head = 0; head < numHeads; head++) {
      std::string headPrefix =
          std::format("{}.sa.heads.{}.", block, head);
      WeightBinding queryWeight = this->resolveWeight(this->weightOffset(headPrefix + "query.weight"));
      WeightBinding keyWeight = this->resolveWeight(this->weightOffset(headPrefix + "key.weight"));
      WeightBinding valueWeight = this->resolveWeight(this->weightOffset(headPrefix + "value.weight"));
      queryWeightBuffers.push_back(queryWeight.buffer);
      keyWeightBuffers.push_back(keyWeight.buffer);
      valueWeightBuffers.push_back(valueWeight.buffer);
      queryWeightOffsets.push_back(queryWeight.offset);
      keyWeightOffsets.push_back(keyWeight.offset);
      valueWeightOffsets.push_back(valueWeight.offset);
      queryOutputOffsets.push_back(
          workspace.queryOffset + head * elementsPerHead * sizeof(float));
      std::size_t cacheOffset =
          cache->byteOffset(blockIndex, head, cachePosition);
      keyOutputOffsets.push_back(cacheOffset);
      valueOutputOffsets.push_back(cacheOffset);
    }
    MTL::Buffer *sharedWeightBuffer = queryWeightBuffers.front();
    bool weightsShareBuffer = true;
    for (std::size_t head = 0; head < numHeads; head++) {
      weightsShareBuffer = weightsShareBuffer && queryWeightBuffers[head] == sharedWeightBuffer && keyWeightBuffers[head] == sharedWeightBuffer && valueWeightBuffers[head] == sharedWeightBuffer;
    }
    if (weightsShareBuffer) {
      batch->qkvMatvecmulHeads(workspace.buffer, workspace.residualOffset, sharedWeightBuffer, queryWeightOffsets, keyWeightOffsets, valueWeightOffsets, workspace.buffer, queryOutputOffsets, cache->keyBuffer(), keyOutputOffsets, cache->valueBuffer(), valueOutputOffsets, this->input_size, headSize);
    } else {
      for (std::size_t head = 0; head < numHeads; head++) {
        batch->matvecmul(workspace.buffer, workspace.residualOffset, queryWeightBuffers[head], queryWeightOffsets[head], workspace.buffer, queryOutputOffsets[head], this->input_size, headSize);
        batch->matvecmul(workspace.buffer, workspace.residualOffset, keyWeightBuffers[head], keyWeightOffsets[head], cache->keyBuffer(), keyOutputOffsets[head], this->input_size, headSize);
        batch->matvecmul(workspace.buffer, workspace.residualOffset, valueWeightBuffers[head], valueWeightOffsets[head], cache->valueBuffer(), valueOutputOffsets[head], this->input_size, headSize);
      }
    }
  } else {
    for (std::size_t head = 0; head < numHeads; head++) {
      std::string headPrefix =
          std::format("{}.sa.heads.{}.", block, head);
      std::size_t headOutputOffset =
          head * elementsPerHead * sizeof(float);
      this->linear(workspace, workspace.residualOffset,
                   this->weightOffset(headPrefix + "query.weight"),
                   workspace.queryOffset + headOutputOffset, rowCount,
                   this->input_size, headSize, batch);

      MTL::Buffer *keyOutputBuffer = workspace.buffer;
      MTL::Buffer *valueOutputBuffer = workspace.buffer;
      std::size_t keyOutputOffset = workspace.keyOffset + headOutputOffset;
      std::size_t valueOutputOffset = workspace.valueOffset + headOutputOffset;
      if (batch != nullptr && cache != nullptr) {
        keyOutputBuffer = cache->keyBuffer();
        valueOutputBuffer = cache->valueBuffer();
        keyOutputOffset = cache->byteOffset(blockIndex, head, cachePosition);
        valueOutputOffset = keyOutputOffset;
      }

      this->linear(workspace, workspace.residualOffset,
                   this->weightOffset(headPrefix + "key.weight"), keyOutputOffset,
                   rowCount, this->input_size, headSize, batch,
                   keyOutputBuffer);

      this->linear(workspace, workspace.residualOffset,
                   this->weightOffset(headPrefix + "value.weight"), valueOutputOffset,
                   rowCount, this->input_size, headSize, batch,
                   valueOutputBuffer);
    }
  }

  if (cache != nullptr && batch == nullptr) {
    cache->writeLayer(blockIndex, cachePosition, workspace.buffer,
                      workspace.keyOffset, workspace.valueOffset,
                      sequenceLength);
  }

  std::size_t readOffset = inputOffset;
  std::size_t writeOffset =
      inputOffset == workspace.scratchAOffset
          ? workspace.scratchBOffset
          : workspace.scratchAOffset;
  if (cache == nullptr) {
    scaled_dot_product_attention_metal(
        this->metal_context, workspace.buffer, workspace.queryOffset,
        workspace.keyOffset, workspace.valueOffset, writeOffset,
        sequenceLength, headSize, numHeads, true, this->input_size, batchSize);
  } else {
    if (batch == nullptr) {
      scaled_dot_product_attention_cached_metal(
          this->metal_context, *cache, blockIndex, workspace.buffer,
          workspace.queryOffset, writeOffset, sequenceLength,
          cachePosition + sequenceLength, cachePosition, true,
          this->input_size);
    } else {
      batch->cachedAttention(
          *cache, blockIndex, workspace.buffer, workspace.queryOffset,
          writeOffset, sequenceLength, cachePosition + sequenceLength,
          cachePosition, true, this->input_size);
    }
  }
  std::swap(readOffset, writeOffset);

  this->linear(workspace, readOffset,
               this->weightOffset(block + ".sa.proj.weight"),
               writeOffset, rowCount, this->input_size,
               this->input_size, batch);
  std::swap(readOffset, writeOffset);
  WeightBinding projectionBias = this->resolveWeight(this->weightOffset(block + ".sa.proj.bias"));

  if (batch == nullptr) {
    add_bias_metal(this->metal_context, workspace.buffer, readOffset, projectionBias.buffer, projectionBias.offset, workspace.buffer, writeOffset, elementCount, this->input_size);
  } else {
    batch->addBias(workspace.buffer, readOffset, projectionBias.buffer, projectionBias.offset, workspace.buffer, writeOffset, elementCount, this->input_size);
  }
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    residual_add_metal(this->metal_context, workspace.buffer,
                       workspace.residualOffset, readOffset, writeOffset,
                       elementCount);
  } else {
    batch->residualAdd(workspace.buffer, workspace.residualOffset,
                       readOffset, writeOffset, elementCount);
  }
  std::swap(readOffset, writeOffset);
  WeightBinding ln2Gamma = this->resolveWeight(this->weightOffset(block + ".ln2.weight"));
  WeightBinding ln2Beta = this->resolveWeight(this->weightOffset(block + ".ln2.bias"));

  if (batch == nullptr) {
    layer_norm_metal(this->metal_context, workspace.buffer, readOffset, ln2Gamma.buffer, ln2Gamma.offset, ln2Beta.buffer, ln2Beta.offset, workspace.buffer, workspace.residualOffset, rowCount, this->input_size);
  } else {
    batch->layerNorm(workspace.buffer, readOffset, ln2Gamma.buffer, ln2Gamma.offset, ln2Beta.buffer, ln2Beta.offset, workspace.buffer, workspace.residualOffset, rowCount, this->input_size);
  }

  std::size_t feedForwardSize = this->input_size * 4;
  std::size_t hiddenElementCount = rowCount * feedForwardSize;
  this->linear(workspace, workspace.residualOffset,
               this->weightOffset(block + ".ffwd.net.0.weight"), writeOffset,
               rowCount, this->input_size, feedForwardSize, batch);
  std::swap(readOffset, writeOffset);
  WeightBinding feedForwardInputBias = this->resolveWeight(this->weightOffset(block + ".ffwd.net.0.bias"));

  if (batch == nullptr) {
    add_bias_metal(this->metal_context, workspace.buffer, readOffset, feedForwardInputBias.buffer, feedForwardInputBias.offset, workspace.buffer, writeOffset, hiddenElementCount, feedForwardSize);
  } else {
    batch->addBias(workspace.buffer, readOffset, feedForwardInputBias.buffer, feedForwardInputBias.offset, workspace.buffer, writeOffset, hiddenElementCount, feedForwardSize);
  }
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    relu_metal(this->metal_context, workspace.buffer, readOffset,
               writeOffset, hiddenElementCount);
  } else {
    batch->relu(workspace.buffer, readOffset, writeOffset,
                hiddenElementCount);
  }
  std::swap(readOffset, writeOffset);

  this->linear(workspace, readOffset,
               this->weightOffset(block + ".ffwd.net.2.weight"),
               writeOffset, rowCount, feedForwardSize,
               this->input_size, batch);
  std::swap(readOffset, writeOffset);
  WeightBinding feedForwardOutputBias = this->resolveWeight(this->weightOffset(block + ".ffwd.net.2.bias"));

  if (batch == nullptr) {
    add_bias_metal(this->metal_context, workspace.buffer, readOffset, feedForwardOutputBias.buffer, feedForwardOutputBias.offset, workspace.buffer, writeOffset, elementCount, this->input_size);
  } else {
    batch->addBias(workspace.buffer, readOffset, feedForwardOutputBias.buffer, feedForwardOutputBias.offset, workspace.buffer, writeOffset, elementCount, this->input_size);
  }
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    residual_add_metal(this->metal_context, workspace.buffer,
                       workspace.residualOffset, readOffset, writeOffset,
                       elementCount);
  } else {
    batch->residualAdd(workspace.buffer, workspace.residualOffset,
                       readOffset, writeOffset, elementCount);
  }
  std::swap(readOffset, writeOffset);
  return readOffset;
}

std::size_t Model::finish(Workspace &workspace, std::size_t inputOffset,
                          std::size_t sequenceLength,
                          MetalCommandBatch *batch) {
  std::size_t blockCount = std::get<std::uint64_t>(
      this->parameters.metadata("gptscratch.block_count"));
  std::string finalLayerNorm = std::format("blocks.{}", blockCount);
  std::size_t readOffset = inputOffset;
  std::size_t writeOffset =
      inputOffset == workspace.scratchAOffset
          ? workspace.scratchBOffset
          : workspace.scratchAOffset;
  WeightBinding finalGamma = this->resolveWeight(this->weightOffset(finalLayerNorm + ".weight"));
  WeightBinding finalBeta = this->resolveWeight(this->weightOffset(finalLayerNorm + ".bias"));

  if (batch == nullptr) {
    layer_norm_metal(this->metal_context, workspace.buffer, readOffset, finalGamma.buffer, finalGamma.offset, finalBeta.buffer, finalBeta.offset, workspace.buffer, writeOffset, sequenceLength, this->input_size);
  } else {
    batch->layerNorm(workspace.buffer, readOffset, finalGamma.buffer, finalGamma.offset, finalBeta.buffer, finalBeta.offset, workspace.buffer, writeOffset, sequenceLength, this->input_size);
  }
  std::swap(readOffset, writeOffset);

  this->linear(workspace, readOffset, this->weightOffset("lm_head.weight"), writeOffset,
               sequenceLength, this->input_size, this->output_size, batch);
  std::swap(readOffset, writeOffset);
  WeightBinding languageModelBias = this->resolveWeight(this->weightOffset("lm_head.bias"));

  if (batch == nullptr) {
    add_bias_metal(this->metal_context, workspace.buffer, readOffset, languageModelBias.buffer, languageModelBias.offset, workspace.buffer, writeOffset, sequenceLength * this->output_size, this->output_size);
  } else {
    batch->addBias(workspace.buffer, readOffset, languageModelBias.buffer, languageModelBias.offset, workspace.buffer, writeOffset, sequenceLength * this->output_size, this->output_size);
  }
  std::swap(readOffset, writeOffset);

  return readOffset;
}

float *Model::execute(const int *tokens, std::size_t sequenceLength,
                      std::size_t startPosition, KVCache *cache,
                      std::size_t cachePosition, bool singleCommand,
                      bool parallelHeads, std::size_t batchSize) {
  if (!this->loaded()) {
    throw std::runtime_error("Load model weights before inference");
  }
  if (batchSize == 0 || sequenceLength == 0 || (batchSize > 1 && (startPosition != 0 || cache != nullptr))) {
    throw std::invalid_argument("Model batch dimensions are invalid");
  }
  std::size_t rowCount = batchSize * sequenceLength;
  Workspace workspace = this->allocateWorkspace(rowCount);
  float *embeddings = nullptr;
  try {
    embeddings = batchSize == 1 ? this->embedAt(tokens, sequenceLength, startPosition) : this->embedBatch(tokens, batchSize, sequenceLength);
    write_metal_buffer(workspace.buffer, workspace.scratchAOffset,
                       embeddings, rowCount * this->input_size);
    std::free(embeddings);
    embeddings = nullptr;

    if (singleCommand && (sequenceLength != 1 || cache == nullptr || batchSize != 1)) {
      throw std::invalid_argument(
          "Single-command execution requires one cached token");
    }
    if (parallelHeads && !singleCommand) {
      throw std::invalid_argument(
          "Parallel heads require single-command execution");
    }
    std::unique_ptr<MetalCommandBatch> batch;
    if (singleCommand) {
      batch = std::make_unique<MetalCommandBatch>(this->metal_context);
    }

    std::size_t outputOffset = workspace.scratchAOffset;
    std::size_t blockCount = std::get<std::uint64_t>(
        this->parameters.metadata("gptscratch.block_count"));
    for (std::size_t block = 0; block < blockCount; block++) {
      outputOffset = this->forwardBlock(
          workspace, outputOffset, sequenceLength, block, cache,
          cachePosition, batch.get(), parallelHeads, batchSize);
    }

    outputOffset = this->finish(workspace, outputOffset, rowCount,
                                batch.get());
    if (batch != nullptr) {
      batch->commitAndWait();
    }
    float *result = read_metal_buffer(
        workspace.buffer, outputOffset,
        rowCount * this->output_size);
    release_metal_buffer(workspace.buffer);
    return result;
  } catch (...) {
    std::free(embeddings);
    release_metal_buffer(workspace.buffer);
    throw;
  }
}

float *Model::forward(const int *tokens, unsigned int sequenceLength) {
  return this->execute(tokens, sequenceLength, 0, nullptr, 0);
}

float *Model::forwardBatch(const int *tokens, std::size_t batchSize, std::size_t sequenceLength) {
  return this->execute(tokens, sequenceLength, 0, nullptr, 0, false, false, batchSize);
}

float *Model::forwardPerturbed(
    const int *tokens, std::size_t sequenceLength,
    std::size_t targetWeightOffset,
    const eggroll::EGGROLLPerturbation &perturbation,
    float epsilon) {
  if (!this->active_perturbations.empty()) {
    throw std::logic_error("A perturbed forward pass is already active");
  }
  this->active_perturbations.emplace(targetWeightOffset, &perturbation);
  this->active_epsilon = epsilon;
  return this->executePerturbed(tokens, sequenceLength);
}

float *Model::forwardPerturbed(
    const int *tokens, std::size_t sequenceLength,
    const eggroll::EGGROLLCandidate &candidate, float epsilon) {
  if (!this->active_perturbations.empty()) {
    throw std::logic_error("A perturbed forward pass is already active");
  }
  if (candidate.empty()) {
    throw std::invalid_argument("An EGGROLL candidate cannot be empty");
  }
  for (const eggroll::EGGROLLTarget &target : candidate) {
    if (!this->active_perturbations
             .emplace(target.weightOffset, &target.perturbation)
             .second) {
      this->active_perturbations.clear();
      throw std::invalid_argument(
          "An EGGROLL candidate contains a duplicate weight offset");
    }
  }
  this->active_epsilon = epsilon;
  return this->executePerturbed(tokens, sequenceLength);
}

float *Model::forwardPerturbedBatch(const int *tokens, std::size_t batchSize, std::size_t sequenceLength, const eggroll::EGGROLLCandidate &candidate, float epsilon) {
  if (!this->active_perturbations.empty()) {
    throw std::logic_error("A perturbed forward pass is already active");
  }
  if (candidate.empty()) {
    throw std::invalid_argument("An EGGROLL candidate cannot be empty");
  }
  for (const eggroll::EGGROLLTarget &target : candidate) {
    if (!this->active_perturbations.emplace(target.weightOffset, &target.perturbation).second) {
      this->active_perturbations.clear();
      throw std::invalid_argument("An EGGROLL candidate contains a duplicate weight offset");
    }
  }
  this->active_epsilon = epsilon;
  return this->executePerturbedBatch(tokens, batchSize, sequenceLength);
}

float *Model::executePerturbed(const int *tokens,
                               std::size_t sequenceLength) {
  return this->executePerturbedBatch(tokens, 1, sequenceLength);
}

float *Model::executePerturbedBatch(const int *tokens, std::size_t batchSize, std::size_t sequenceLength) {
  this->applied_perturbations.clear();

  float *result = nullptr;
  try {
    result = this->execute(tokens, sequenceLength, 0, nullptr, 0, false, false, batchSize);
  } catch (...) {
    this->active_perturbations.clear();
    this->applied_perturbations.clear();
    this->active_epsilon = 0.0F;
    throw;
  }

  bool applied = this->applied_perturbations.size() ==
                 this->active_perturbations.size();
  this->active_perturbations.clear();
  this->applied_perturbations.clear();
  this->active_epsilon = 0.0F;
  if (!applied) {
    std::free(result);
    throw std::invalid_argument(
        "An EGGROLL target offset is not used by a linear matrix");
  }
  return result;
}

float Model::fitness(const float *logits, const int *targets,
                     std::size_t sequenceLength) {
  return eggroll::fitnessMetal(this->metal_context, logits, targets,
                               sequenceLength, this->output_size);
}

std::vector<eggroll::EGGROLLMatrix> Model::linearMatrices() const {
  std::vector<eggroll::EGGROLLMatrix> matrices;
  matrices.reserve(this->linear_matrices.size());
  for (const auto &[offset, matrix] : this->linear_matrices) {
    matrices.push_back(matrix);
  }
  std::sort(matrices.begin(), matrices.end(),
            [](const eggroll::EGGROLLMatrix &left,
               const eggroll::EGGROLLMatrix &right) {
              return left.weightOffset < right.weightOffset;
            });
  return matrices;
}

void Model::applyEGGROLLUpdate(
    const std::vector<eggroll::EGGROLLCandidate> &population,
    const std::vector<float> &fitnesses, float learningRate) {
  if (!this->loaded()) {
    throw std::runtime_error("Load model weights before an EGGROLL update");
  }
  if (population.empty() || population.front().empty() ||
      population.size() != fitnesses.size()) {
    throw std::invalid_argument("EGGROLL update population is invalid");
  }

  const eggroll::EGGROLLCandidate &first = population.front();
  std::unordered_set<std::size_t> updatedOffsets;
  for (const eggroll::EGGROLLTarget &firstTarget : first) {
    if (!updatedOffsets.insert(firstTarget.weightOffset).second) {
      throw std::invalid_argument(
          "An EGGROLL candidate contains a duplicate weight offset");
    }
    auto matrix = this->linear_matrices.find(firstTarget.weightOffset);
    if (matrix == this->linear_matrices.end() ||
        matrix->second.M != firstTarget.perturbation.M ||
        matrix->second.N != firstTarget.perturbation.N) {
      throw std::invalid_argument(
          "An EGGROLL update does not match a model matrix");
    }
    std::vector<const eggroll::EGGROLLPerturbation *> perturbations;
    perturbations.reserve(population.size());
    for (const eggroll::EGGROLLCandidate &candidate : population) {
      if (candidate.size() != first.size()) {
        throw std::invalid_argument(
            "EGGROLL candidates target different weight offsets");
      }
      auto target = std::find_if(
          candidate.begin(), candidate.end(),
          [&firstTarget](const eggroll::EGGROLLTarget &value) {
            return value.weightOffset == firstTarget.weightOffset;
          });
      if (target == candidate.end()) {
        throw std::invalid_argument(
            "EGGROLL candidates target different weight offsets");
      }
      perturbations.push_back(&target->perturbation);
    }
    WeightBinding weight = this->resolveWeight(firstTarget.weightOffset);
    eggroll::updateWeightsMetal(this->metal_context, weight.buffer, weight.offset, perturbations, fitnesses, learningRate);
  }
  this->resetCache();
}

void Model::ensureCache() {
  if (this->kv_cache != nullptr) {
    return;
  }

  std::size_t blockCount = std::get<std::uint64_t>(
      this->parameters.metadata("gptscratch.block_count"));
  std::size_t headCount = std::get<std::uint64_t>(
      this->parameters.metadata("gptscratch.attention.head_count"));
  this->kv_cache = std::make_unique<KVCache>(
      this->metal_context, blockCount, headCount, this->context_size,
      this->input_size / headCount);
}

float *Model::prefill(const int *tokens, std::size_t sequenceLength) {
  this->ensureCache();
  this->kv_cache->reset();

  float *result = this->execute(tokens, sequenceLength, 0,
                                this->kv_cache.get(), 0);
  this->kv_cache->setLength(sequenceLength);
  return result;
}

float *Model::decode(int token) {
  return this->decodeCached(token, false, false);
}

float *Model::decodeSingleCommand(int token) {
  return this->decodeCached(token, true, false);
}

float *Model::decodeSingleCommandParallelHeads(int token) {
  return this->decodeCached(token, true, true);
}

float *Model::decodeCached(int token, bool singleCommand,
                           bool parallelHeads) {
  this->ensureCache();
  std::size_t position = this->kv_cache->length();
  if (position == this->kv_cache->capacity()) {
    this->kv_cache->reset();
    position = 0;
  }

  float *result = this->execute(&token, 1, position,
                                this->kv_cache.get(), position,
                                singleCommand, parallelHeads);
  this->kv_cache->setLength(position + 1);
  return result;
}

void Model::resetCache() {
  if (this->kv_cache != nullptr) {
    this->kv_cache->reset();
  }
}

std::size_t Model::cacheLength() const {
  return this->kv_cache == nullptr ? 0 : this->kv_cache->length();
}

std::size_t Model::contextSize() const { return this->context_size; }

std::size_t Model::vocabularySize() const { return this->output_size; }

} // namespace inference
