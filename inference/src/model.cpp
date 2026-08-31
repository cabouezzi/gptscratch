#include <model.hpp>

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
  this->parameters = ModelParameters::loadGGUF(path);

  const GGUFTensor &tokenEmbedding =
      this->parameters.tensor("token_embedding_table.weight");
  const GGUFTensor &positionEmbedding =
      this->parameters.tensor("position_embedding_table.weight");

  this->input_size = static_cast<std::size_t>(tokenEmbedding.shape[1]);
  this->output_size = static_cast<std::size_t>(tokenEmbedding.shape[0]);
  this->context_size = static_cast<std::size_t>(positionEmbedding.shape[0]);
}

Model::~Model() { this->release(); }

void Model::load() {
  if (this->model_buffer != nullptr) {
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
  std::size_t byteCount = 0;
  for (const std::string &name : names) {
    byteCount = alignBytes(byteCount, weightAlignment);
    offsets.emplace(name, byteCount);
    const GGUFTensor &tensor = this->parameters.tensor(name);
    byteCount +=
        static_cast<std::size_t>(tensor.elementCount) * sizeof(float);
  }

  MTL::Buffer *staging =
      allocate_metal_buffer(this->metal_context, byteCount);
  MTL::Buffer *gpuWeights = nullptr;
  try {
    for (const std::string &name : names) {
      const GGUFTensor &tensor = this->parameters.tensor(name);
      write_metal_buffer(
          staging, offsets.at(name), this->parameters.data(name),
          static_cast<std::size_t>(tensor.elementCount));
    }

    gpuWeights =
        allocate_private_metal_buffer(this->metal_context, byteCount);
    copy_metal_buffer(this->metal_context, staging, 0, gpuWeights, 0,
                      byteCount);
    release_metal_buffer(staging);
    this->model_buffer = gpuWeights;
    this->weight_offsets = std::move(offsets);
  } catch (...) {
    release_metal_buffer(staging);
    release_metal_buffer(gpuWeights);
    throw;
  }
}

void Model::release() {
  this->kv_cache.reset();
  release_metal_buffer(this->model_buffer);
  this->model_buffer = nullptr;
  this->weight_offsets.clear();
}

bool Model::loaded() const { return this->model_buffer != nullptr; }

std::size_t Model::weightOffset(const std::string &name) const {
  if (this->model_buffer == nullptr) {
    throw std::runtime_error("Model weights are not loaded into GPU memory");
  }
  auto offset = this->weight_offsets.find(name);
  if (offset == this->weight_offsets.end()) {
    throw std::out_of_range("GPU model tensor not found: " + name);
  }
  return offset->second;
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
                   const std::string &weightName,
                   std::size_t outputOffset,
                   std::size_t sequenceLength, std::size_t inputSize,
                   std::size_t outputSize, MetalCommandBatch *batch,
                   MTL::Buffer *outputBuffer) {
  if (outputBuffer == nullptr) {
    outputBuffer = workspace.buffer;
  }
  if (sequenceLength == 1) {
    if (batch != nullptr) {
      batch->matvecmul(
          workspace.buffer, inputOffset, this->model_buffer,
          this->weightOffset(weightName), outputBuffer, outputOffset,
          inputSize, outputSize);
      return;
    }
    matvecmul_metal(
        this->metal_context, workspace.buffer, inputOffset,
        this->model_buffer, this->weightOffset(weightName), workspace.buffer,
        outputOffset, inputSize, outputSize);
    return;
  }

  matmul_metal(
      this->metal_context, workspace.buffer, inputOffset,
      MatMulFlag::NO_TRANSPOSE, this->model_buffer,
      this->weightOffset(weightName), MatMulFlag::TRANSPOSE,
      workspace.buffer, outputOffset, sequenceLength, inputSize,
      outputSize, true);
}

std::size_t Model::forwardBlock(Workspace &workspace,
                                std::size_t inputOffset,
                                std::size_t sequenceLength,
                                std::size_t blockIndex, KVCache *cache,
                                std::size_t cachePosition,
                                MetalCommandBatch *batch,
                                bool parallelHeads) {
  std::string block = std::format("blocks.{}", blockIndex);
  std::size_t numHeads = std::get<std::uint64_t>(
      this->parameters.metadata("gptscratch.attention.head_count"));
  std::size_t headSize = this->input_size / numHeads;
  std::size_t elementCount = sequenceLength * this->input_size;
  std::size_t elementsPerHead = sequenceLength * headSize;

  if (batch == nullptr) {
    layer_norm_metal(
        this->metal_context, workspace.buffer, inputOffset,
        this->model_buffer, this->weightOffset(block + ".ln1.weight"),
        this->model_buffer, this->weightOffset(block + ".ln1.bias"),
        workspace.buffer, workspace.residualOffset, sequenceLength,
        this->input_size);
  } else {
    batch->layerNorm(
        workspace.buffer, inputOffset, this->model_buffer,
        this->weightOffset(block + ".ln1.weight"), this->model_buffer,
        this->weightOffset(block + ".ln1.bias"), workspace.buffer,
        workspace.residualOffset, sequenceLength, this->input_size);
  }

  if (parallelHeads) {
    if (batch == nullptr || cache == nullptr || sequenceLength != 1) {
      throw std::invalid_argument(
          "Parallel heads require one cached command batch");
    }
    std::vector<std::size_t> queryWeightOffsets;
    std::vector<std::size_t> keyWeightOffsets;
    std::vector<std::size_t> valueWeightOffsets;
    std::vector<std::size_t> queryOutputOffsets;
    std::vector<std::size_t> keyOutputOffsets;
    std::vector<std::size_t> valueOutputOffsets;
    queryWeightOffsets.reserve(numHeads);
    keyWeightOffsets.reserve(numHeads);
    valueWeightOffsets.reserve(numHeads);
    queryOutputOffsets.reserve(numHeads);
    keyOutputOffsets.reserve(numHeads);
    valueOutputOffsets.reserve(numHeads);
    for (std::size_t head = 0; head < numHeads; head++) {
      std::string headPrefix =
          std::format("{}.sa.heads.{}.", block, head);
      queryWeightOffsets.push_back(
          this->weightOffset(headPrefix + "query.weight"));
      keyWeightOffsets.push_back(
          this->weightOffset(headPrefix + "key.weight"));
      valueWeightOffsets.push_back(
          this->weightOffset(headPrefix + "value.weight"));
      queryOutputOffsets.push_back(
          workspace.queryOffset + head * elementsPerHead * sizeof(float));
      std::size_t cacheOffset =
          cache->byteOffset(blockIndex, head, cachePosition);
      keyOutputOffsets.push_back(cacheOffset);
      valueOutputOffsets.push_back(cacheOffset);
    }
    batch->qkvMatvecmulHeads(
        workspace.buffer, workspace.residualOffset, this->model_buffer,
        queryWeightOffsets, keyWeightOffsets, valueWeightOffsets,
        workspace.buffer, queryOutputOffsets, cache->keyBuffer(),
        keyOutputOffsets, cache->valueBuffer(), valueOutputOffsets,
        this->input_size, headSize);
  } else {
    for (std::size_t head = 0; head < numHeads; head++) {
      std::string headPrefix =
          std::format("{}.sa.heads.{}.", block, head);
      std::size_t headOutputOffset =
          head * elementsPerHead * sizeof(float);
      this->linear(workspace, workspace.residualOffset,
                   headPrefix + "query.weight",
                   workspace.queryOffset + headOutputOffset, sequenceLength,
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
                   headPrefix + "key.weight", keyOutputOffset,
                   sequenceLength, this->input_size, headSize, batch,
                   keyOutputBuffer);

      this->linear(workspace, workspace.residualOffset,
                   headPrefix + "value.weight", valueOutputOffset,
                   sequenceLength, this->input_size, headSize, batch,
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
        sequenceLength, headSize, numHeads, true, this->input_size);
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

  this->linear(workspace, readOffset, block + ".sa.proj.weight",
               writeOffset, sequenceLength, this->input_size,
               this->input_size, batch);
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    add_bias_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer, this->weightOffset(block + ".sa.proj.bias"),
        workspace.buffer, writeOffset, elementCount, this->input_size);
  } else {
    batch->addBias(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset(block + ".sa.proj.bias"), workspace.buffer,
        writeOffset, elementCount, this->input_size);
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

  if (batch == nullptr) {
    layer_norm_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer, this->weightOffset(block + ".ln2.weight"),
        this->model_buffer, this->weightOffset(block + ".ln2.bias"),
        workspace.buffer, workspace.residualOffset, sequenceLength,
        this->input_size);
  } else {
    batch->layerNorm(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset(block + ".ln2.weight"), this->model_buffer,
        this->weightOffset(block + ".ln2.bias"), workspace.buffer,
        workspace.residualOffset, sequenceLength, this->input_size);
  }

  std::size_t feedForwardSize = this->input_size * 4;
  std::size_t hiddenElementCount = sequenceLength * feedForwardSize;
  this->linear(workspace, workspace.residualOffset,
               block + ".ffwd.net.0.weight", writeOffset,
               sequenceLength, this->input_size, feedForwardSize, batch);
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    add_bias_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer,
        this->weightOffset(block + ".ffwd.net.0.bias"), workspace.buffer,
        writeOffset, hiddenElementCount, feedForwardSize);
  } else {
    batch->addBias(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset(block + ".ffwd.net.0.bias"), workspace.buffer,
        writeOffset, hiddenElementCount, feedForwardSize);
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

  this->linear(workspace, readOffset, block + ".ffwd.net.2.weight",
               writeOffset, sequenceLength, feedForwardSize,
               this->input_size, batch);
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    add_bias_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer,
        this->weightOffset(block + ".ffwd.net.2.bias"), workspace.buffer,
        writeOffset, elementCount, this->input_size);
  } else {
    batch->addBias(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset(block + ".ffwd.net.2.bias"), workspace.buffer,
        writeOffset, elementCount, this->input_size);
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

  if (batch == nullptr) {
    layer_norm_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer,
        this->weightOffset(finalLayerNorm + ".weight"), this->model_buffer,
        this->weightOffset(finalLayerNorm + ".bias"), workspace.buffer,
        writeOffset, sequenceLength, this->input_size);
  } else {
    batch->layerNorm(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset(finalLayerNorm + ".weight"), this->model_buffer,
        this->weightOffset(finalLayerNorm + ".bias"), workspace.buffer,
        writeOffset, sequenceLength, this->input_size);
  }
  std::swap(readOffset, writeOffset);

  this->linear(workspace, readOffset, "lm_head.weight", writeOffset,
               sequenceLength, this->input_size, this->output_size, batch);
  std::swap(readOffset, writeOffset);

  if (batch == nullptr) {
    add_bias_metal(
        this->metal_context, workspace.buffer, readOffset,
        this->model_buffer, this->weightOffset("lm_head.bias"),
        workspace.buffer, writeOffset, sequenceLength * this->output_size,
        this->output_size);
  } else {
    batch->addBias(
        workspace.buffer, readOffset, this->model_buffer,
        this->weightOffset("lm_head.bias"), workspace.buffer, writeOffset,
        sequenceLength * this->output_size, this->output_size);
  }
  std::swap(readOffset, writeOffset);

  return readOffset;
}

float *Model::execute(const int *tokens, std::size_t sequenceLength,
                      std::size_t startPosition, KVCache *cache,
                      std::size_t cachePosition, bool singleCommand,
                      bool parallelHeads) {
  if (!this->loaded()) {
    throw std::runtime_error("Load model weights before inference");
  }
  Workspace workspace = this->allocateWorkspace(sequenceLength);
  float *embeddings = nullptr;
  try {
    embeddings = this->embedAt(tokens, sequenceLength, startPosition);
    write_metal_buffer(workspace.buffer, workspace.scratchAOffset,
                       embeddings, sequenceLength * this->input_size);
    std::free(embeddings);
    embeddings = nullptr;

    if (singleCommand && (sequenceLength != 1 || cache == nullptr)) {
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
          cachePosition, batch.get(), parallelHeads);
    }

    outputOffset = this->finish(workspace, outputOffset, sequenceLength,
                                batch.get());
    if (batch != nullptr) {
      batch->commitAndWait();
    }
    float *result = read_metal_buffer(
        workspace.buffer, outputOffset,
        sequenceLength * this->output_size);
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
