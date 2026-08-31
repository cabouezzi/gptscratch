#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <mach-o/dyld.h>
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <metal/backend.hpp>
#include <metal/kv_cache.hpp>

namespace inference {

struct MatrixDims {
  std::uint32_t M;
  std::uint32_t K;
  std::uint32_t N;
};

struct AttentionDims {
  std::uint32_t queryLength;
  std::uint32_t keyValueLength;
  std::uint32_t headSize;
  std::uint32_t cacheCapacity;
  std::uint32_t queryStartPosition;
};

struct HeadProjectionOffsets {
  std::uint64_t queryWeight;
  std::uint64_t keyWeight;
  std::uint64_t valueWeight;
  std::uint64_t queryOutput;
  std::uint64_t keyOutput;
  std::uint64_t valueOutput;
};

std::filesystem::path executable_directory() {
  std::uint32_t path_size = 0;
  _NSGetExecutablePath(nullptr, &path_size);

  std::vector<char> executable_path(path_size);
  _NSGetExecutablePath(executable_path.data(), &path_size);
  return std::filesystem::path(executable_path.data()).parent_path();
} // namespace

struct MetalContext::Impl {
  MTL::Device *device = nullptr;
  MTL::CommandQueue *queue = nullptr;
  MTL::Library *library = nullptr;
  std::unordered_map<std::string, MTL::ComputePipelineState *> pipelines;

  Impl() {
    NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
    this->device = MTL::CreateSystemDefaultDevice();
    if (this->device == nullptr) {
      pool->release();
      throw std::runtime_error("Metal is not supported");
    }

    NS::Error *error = nullptr;
    std::filesystem::path metallibPath =
        executable_directory() / "inference.metallib";
    NS::String *metallibPathString =
        NS::String::string(metallibPath.c_str(), NS::UTF8StringEncoding);
    this->library = this->device->newLibrary(metallibPathString, &error);
    if (this->library == nullptr) {
      this->device->release();
      this->device = nullptr;
      pool->release();
      throw std::runtime_error(
          std::format("Could not load {}", metallibPath.string()));
    }

    this->queue = this->device->newCommandQueue();
    if (this->queue == nullptr) {
      this->library->release();
      this->device->release();
      this->library = nullptr;
      this->device = nullptr;
      pool->release();
      throw std::runtime_error("Could not create Metal command queue");
    }
    pool->release();
  }

  ~Impl() {
    for (auto &[name, pipeline] : this->pipelines) {
      pipeline->release();
    }
    this->queue->release();
    this->library->release();
    this->device->release();
  }

  MTL::ComputePipelineState *pipeline(const std::string &name) {
    auto existing = this->pipelines.find(name);
    if (existing != this->pipelines.end()) {
      return existing->second;
    }

    NS::String *functionName =
        NS::String::string(name.c_str(), NS::UTF8StringEncoding);
    MTL::Function *function = this->library->newFunction(functionName);
    if (function == nullptr) {
      throw std::runtime_error("Could not load Metal function " + name);
    }

    NS::Error *error = nullptr;
    MTL::ComputePipelineState *created =
        this->device->newComputePipelineState(function, &error);
    function->release();
    if (created == nullptr) {
      throw std::runtime_error("Could not create Metal pipeline " + name);
    }
    this->pipelines.emplace(name, created);
    return created;
  }
};

struct MetalContextAccess {
  static MetalContext::Impl &get(MetalContext &context) {
    return *context.impl;
  }
};

MetalContext::MetalContext() : impl(std::make_unique<Impl>()) {}

MetalContext::~MetalContext() = default;

MetalContext::MetalContext(MetalContext &&) noexcept = default;

MetalContext &MetalContext::operator=(MetalContext &&) noexcept = default;

namespace {

void validate_metal_buffer_range(MTL::Buffer *buffer, std::size_t byteOffset,
                                 std::size_t byteCount) {
  if (buffer == nullptr) {
    throw std::invalid_argument("Metal buffer cannot be null");
  }
  if (byteOffset > buffer->length() ||
      byteCount > buffer->length() - byteOffset) {
    throw std::out_of_range("Metal buffer range exceeds its allocation");
  }
}

}

MetalCommandBatch::MetalCommandBatch(MetalContext &context)
    : context(&context), commandBuffer(nullptr), encoder(nullptr), pool(nullptr) {
  auto &metal = MetalContextAccess::get(context);
  this->pool = NS::AutoreleasePool::alloc()->init();
  this->commandBuffer = metal.queue->commandBuffer();
  if (this->commandBuffer == nullptr) {
    this->pool->release();
    this->pool = nullptr;
    throw std::runtime_error("Could not create Metal command buffer");
  }
  this->encoder = this->commandBuffer->computeCommandEncoder();
  if (this->encoder == nullptr) {
    this->pool->release();
    this->pool = nullptr;
    this->commandBuffer = nullptr;
    throw std::runtime_error("Could not create Metal compute encoder");
  }
}

MetalCommandBatch::~MetalCommandBatch() {
  if (!this->completed && this->encoder != nullptr) {
    this->encoder->endEncoding();
  }
  if (this->pool != nullptr) {
    this->pool->release();
  }
}

void MetalCommandBatch::matvecmul(
    MTL::Buffer *inputBuffer, std::size_t inputOffset,
    MTL::Buffer *weightBuffer, std::size_t weightOffset,
    MTL::Buffer *outputBuffer, std::size_t outputOffset,
    std::size_t inputSize, std::size_t outputSize) {
  constexpr std::size_t float4Alignment = 4 * sizeof(float);
  if (inputSize == 0 || outputSize == 0 || inputSize % 4 != 0 ||
      inputOffset % float4Alignment != 0 ||
      weightOffset % float4Alignment != 0 ||
      outputOffset % alignof(float) != 0 ||
      inputSize > std::numeric_limits<std::uint32_t>::max() ||
      outputSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal matvecmul dimensions are invalid");
  }
  if (outputSize >
      std::numeric_limits<std::size_t>::max() / inputSize / sizeof(float)) {
    throw std::overflow_error("Metal matvecmul weight size overflows");
  }

  validate_metal_buffer_range(inputBuffer, inputOffset,
                              inputSize * sizeof(float));
  validate_metal_buffer_range(weightBuffer, weightOffset,
                              inputSize * outputSize * sizeof(float));
  validate_metal_buffer_range(outputBuffer, outputOffset,
                              outputSize * sizeof(float));

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalInputSize = static_cast<std::uint32_t>(inputSize);
  std::uint32_t metalOutputSize = static_cast<std::uint32_t>(outputSize);
  this->encoder->setComputePipelineState(metal.pipeline("matvecmul"));
  this->encoder->setBuffer(inputBuffer, inputOffset, 0);
  this->encoder->setBuffer(weightBuffer, weightOffset, 1);
  this->encoder->setBuffer(outputBuffer, outputOffset, 2);
  this->encoder->setBytes(&metalInputSize, sizeof(metalInputSize), 3);
  this->encoder->setBytes(&metalOutputSize, sizeof(metalOutputSize), 4);
  this->encoder->dispatchThreadgroups(
      MTL::Size((outputSize + 3) / 4, 1, 1), MTL::Size(32, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::qkvMatvecmulHeads(
    MTL::Buffer *inputBuffer, std::size_t inputOffset,
    MTL::Buffer *weightBuffer,
    const std::vector<std::size_t> &queryWeightOffsets,
    const std::vector<std::size_t> &keyWeightOffsets,
    const std::vector<std::size_t> &valueWeightOffsets,
    MTL::Buffer *queryOutputBuffer,
    const std::vector<std::size_t> &queryOutputOffsets,
    MTL::Buffer *keyOutputBuffer,
    const std::vector<std::size_t> &keyOutputOffsets,
    MTL::Buffer *valueOutputBuffer,
    const std::vector<std::size_t> &valueOutputOffsets,
    std::size_t inputSize, std::size_t headSize) {
  constexpr std::size_t float4Size = 4 * sizeof(float);
  std::size_t headCount = queryWeightOffsets.size();
  if (headCount == 0 || inputSize == 0 || headSize == 0 ||
      inputSize % 4 != 0 || inputOffset % float4Size != 0 ||
      headCount > std::numeric_limits<std::uint32_t>::max() ||
      inputSize > std::numeric_limits<std::uint32_t>::max() ||
      headSize > std::numeric_limits<std::uint32_t>::max() ||
      keyWeightOffsets.size() != headCount ||
      valueWeightOffsets.size() != headCount ||
      queryOutputOffsets.size() != headCount ||
      keyOutputOffsets.size() != headCount ||
      valueOutputOffsets.size() != headCount) {
    throw std::invalid_argument(
        "Metal head-parallel QKV dimensions are invalid");
  }

  validate_metal_buffer_range(inputBuffer, inputOffset,
                              inputSize * sizeof(float));
  std::size_t weightByteCount = inputSize * headSize * sizeof(float);
  std::size_t outputByteCount = headSize * sizeof(float);
  std::vector<HeadProjectionOffsets> metalOffsets;
  metalOffsets.reserve(headCount);
  for (std::size_t head = 0; head < headCount; head++) {
    if (queryWeightOffsets[head] % float4Size != 0 ||
        keyWeightOffsets[head] % float4Size != 0 ||
        valueWeightOffsets[head] % float4Size != 0 ||
        queryOutputOffsets[head] % sizeof(float) != 0 ||
        keyOutputOffsets[head] % sizeof(float) != 0 ||
        valueOutputOffsets[head] % sizeof(float) != 0) {
      throw std::invalid_argument(
          "Metal head-parallel QKV offsets are misaligned");
    }
    validate_metal_buffer_range(weightBuffer, queryWeightOffsets[head],
                                weightByteCount);
    validate_metal_buffer_range(weightBuffer, keyWeightOffsets[head],
                                weightByteCount);
    validate_metal_buffer_range(weightBuffer, valueWeightOffsets[head],
                                weightByteCount);
    validate_metal_buffer_range(queryOutputBuffer, queryOutputOffsets[head],
                                outputByteCount);
    validate_metal_buffer_range(keyOutputBuffer, keyOutputOffsets[head],
                                outputByteCount);
    validate_metal_buffer_range(valueOutputBuffer, valueOutputOffsets[head],
                                outputByteCount);
    metalOffsets.push_back({
        .queryWeight = queryWeightOffsets[head] / float4Size,
        .keyWeight = keyWeightOffsets[head] / float4Size,
        .valueWeight = valueWeightOffsets[head] / float4Size,
        .queryOutput = queryOutputOffsets[head] / sizeof(float),
        .keyOutput = keyOutputOffsets[head] / sizeof(float),
        .valueOutput = valueOutputOffsets[head] / sizeof(float),
    });
  }

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalInputSize = static_cast<std::uint32_t>(inputSize);
  std::uint32_t metalHeadSize = static_cast<std::uint32_t>(headSize);
  std::uint32_t metalHeadCount = static_cast<std::uint32_t>(headCount);
  this->encoder->setComputePipelineState(
      metal.pipeline("qkv_matvecmul_heads"));
  this->encoder->setBuffer(inputBuffer, inputOffset, 0);
  this->encoder->setBuffer(weightBuffer, 0, 1);
  this->encoder->setBuffer(queryOutputBuffer, 0, 2);
  this->encoder->setBuffer(keyOutputBuffer, 0, 3);
  this->encoder->setBuffer(valueOutputBuffer, 0, 4);
  this->encoder->setBytes(metalOffsets.data(),
                          metalOffsets.size() * sizeof(HeadProjectionOffsets),
                          5);
  this->encoder->setBytes(&metalInputSize, sizeof(metalInputSize), 6);
  this->encoder->setBytes(&metalHeadSize, sizeof(metalHeadSize), 7);
  this->encoder->setBytes(&metalHeadCount, sizeof(metalHeadCount), 8);
  this->encoder->dispatchThreadgroups(
      MTL::Size((headSize + 3) / 4, headCount, 1), MTL::Size(32, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::layerNorm(
    MTL::Buffer *inputBuffer, std::size_t inputOffset,
    MTL::Buffer *gammaBuffer, std::size_t gammaOffset,
    MTL::Buffer *betaBuffer, std::size_t betaOffset,
    MTL::Buffer *outputBuffer, std::size_t outputOffset,
    std::size_t sequenceLength, std::size_t embeddingSize, float epsilon) {
  if (sequenceLength == 0 || embeddingSize == 0 ||
      embeddingSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal LayerNorm dimensions are invalid");
  }
  std::size_t inputSize = sequenceLength * embeddingSize * sizeof(float);
  std::size_t parameterSize = embeddingSize * sizeof(float);
  validate_metal_buffer_range(inputBuffer, inputOffset, inputSize);
  validate_metal_buffer_range(gammaBuffer, gammaOffset, parameterSize);
  validate_metal_buffer_range(betaBuffer, betaOffset, parameterSize);
  validate_metal_buffer_range(outputBuffer, outputOffset, inputSize);

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalEmbeddingSize =
      static_cast<std::uint32_t>(embeddingSize);
  this->encoder->setComputePipelineState(metal.pipeline("layer_norm"));
  this->encoder->setBuffer(inputBuffer, inputOffset, 0);
  this->encoder->setBuffer(gammaBuffer, gammaOffset, 1);
  this->encoder->setBuffer(betaBuffer, betaOffset, 2);
  this->encoder->setBuffer(outputBuffer, outputOffset, 3);
  this->encoder->setBytes(&metalEmbeddingSize, sizeof(metalEmbeddingSize), 4);
  this->encoder->setBytes(&epsilon, sizeof(epsilon), 5);
  this->encoder->dispatchThreadgroups(MTL::Size(sequenceLength, 1, 1),
                                      MTL::Size(32, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::residualAdd(MTL::Buffer *buffer,
                                    std::size_t residualOffset,
                                    std::size_t inputOffset,
                                    std::size_t outputOffset,
                                    std::size_t elementCount) {
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal residual element count is invalid");
  }
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, residualOffset, byteCount);
  validate_metal_buffer_range(buffer, inputOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  this->encoder->setComputePipelineState(metal.pipeline("residual_add"));
  this->encoder->setBuffer(buffer, residualOffset, 0);
  this->encoder->setBuffer(buffer, inputOffset, 1);
  this->encoder->setBuffer(buffer, outputOffset, 2);
  this->encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  this->encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                      MTL::Size(threadsPerThreadgroup, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::relu(MTL::Buffer *buffer, std::size_t inputOffset,
                             std::size_t outputOffset,
                             std::size_t elementCount) {
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal ReLU element count is invalid");
  }
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, inputOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  this->encoder->setComputePipelineState(metal.pipeline("relu"));
  this->encoder->setBuffer(buffer, inputOffset, 0);
  this->encoder->setBuffer(buffer, outputOffset, 1);
  this->encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 2);
  this->encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                      MTL::Size(threadsPerThreadgroup, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::addBias(
    MTL::Buffer *inputBuffer, std::size_t inputOffset,
    MTL::Buffer *biasBuffer, std::size_t biasOffset,
    MTL::Buffer *outputBuffer, std::size_t outputOffset,
    std::size_t elementCount, std::size_t outputWidth) {
  if (elementCount == 0 || outputWidth == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max() ||
      outputWidth > std::numeric_limits<std::uint32_t>::max() ||
      elementCount % outputWidth != 0) {
    throw std::invalid_argument("Metal bias dimensions are invalid");
  }
  std::size_t inputSize = elementCount * sizeof(float);
  std::size_t biasSize = outputWidth * sizeof(float);
  validate_metal_buffer_range(inputBuffer, inputOffset, inputSize);
  validate_metal_buffer_range(biasBuffer, biasOffset, biasSize);
  validate_metal_buffer_range(outputBuffer, outputOffset, inputSize);

  auto &metal = MetalContextAccess::get(*this->context);
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  std::uint32_t metalOutputWidth = static_cast<std::uint32_t>(outputWidth);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
  this->encoder->setComputePipelineState(metal.pipeline("add_bias"));
  this->encoder->setBuffer(inputBuffer, inputOffset, 0);
  this->encoder->setBuffer(biasBuffer, biasOffset, 1);
  this->encoder->setBuffer(outputBuffer, outputOffset, 2);
  this->encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  this->encoder->setBytes(&metalOutputWidth, sizeof(metalOutputWidth), 4);
  this->encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                      MTL::Size(threadsPerThreadgroup, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::cachedAttention(
    KVCache &cache, std::size_t layer, MTL::Buffer *buffer,
    std::size_t queryOffset, std::size_t outputOffset,
    std::size_t queryLength, std::size_t keyValueLength,
    std::size_t queryStartPosition, bool isCausal, std::size_t scaleSize) {
  if (layer >= cache.layerCount || queryLength == 0 || keyValueLength == 0 ||
      keyValueLength > cache.cacheCapacity ||
      queryStartPosition + queryLength > keyValueLength || scaleSize == 0) {
    throw std::invalid_argument("Cached attention dimensions are invalid");
  }
  if (queryLength > std::numeric_limits<std::uint32_t>::max() ||
      keyValueLength > std::numeric_limits<std::uint32_t>::max() ||
      cache.headSize > std::numeric_limits<std::uint32_t>::max() ||
      cache.cacheCapacity > std::numeric_limits<std::uint32_t>::max() ||
      queryStartPosition > std::numeric_limits<std::uint32_t>::max() ||
      cache.headCount > std::numeric_limits<std::uint32_t>::max() ||
      scaleSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Cached attention dimensions exceed uint32_t");
  }
  std::size_t elementCount = queryLength * cache.headCount * cache.headSize;
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, queryOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(*this->context);
  AttentionDims dims = {
      .queryLength = static_cast<std::uint32_t>(queryLength),
      .keyValueLength = static_cast<std::uint32_t>(keyValueLength),
      .headSize = static_cast<std::uint32_t>(cache.headSize),
      .cacheCapacity = static_cast<std::uint32_t>(cache.cacheCapacity),
      .queryStartPosition = static_cast<std::uint32_t>(queryStartPosition),
  };
  std::uint32_t metalHeadCount = static_cast<std::uint32_t>(cache.headCount);
  std::uint32_t metalScaleSize = static_cast<std::uint32_t>(scaleSize);
  std::size_t layerOffset =
      layer * cache.headCount * cache.cacheCapacity * cache.headSize *
      sizeof(float);
  this->encoder->setComputePipelineState(
      metal.pipeline("scaled_dot_product_attention"));
  this->encoder->setBuffer(buffer, queryOffset, 0);
  this->encoder->setBuffer(cache.keys, layerOffset, 1);
  this->encoder->setBuffer(cache.values, layerOffset, 2);
  this->encoder->setBuffer(buffer, outputOffset, 3);
  this->encoder->setBytes(&dims, sizeof(dims), 4);
  this->encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  this->encoder->setBytes(&metalHeadCount, sizeof(metalHeadCount), 6);
  this->encoder->setBytes(&metalScaleSize, sizeof(metalScaleSize), 7);
  this->encoder->dispatchThreadgroups(
      MTL::Size(queryLength, cache.headCount, 1), MTL::Size(32, 1, 1));
  this->encoder->memoryBarrier(MTL::BarrierScopeBuffers);
}

void MetalCommandBatch::commitAndWait() {
  if (this->completed) {
    throw std::logic_error("Metal command batch was already completed");
  }
  this->encoder->endEncoding();
  this->commandBuffer->commit();
  this->commandBuffer->waitUntilCompleted();
  this->completed = true;
  this->encoder = nullptr;
  this->commandBuffer = nullptr;
  this->pool->release();
  this->pool = nullptr;
}

MTL::Buffer *allocate_metal_buffer(MetalContext &context,
                                   std::size_t byteCount) {
  if (byteCount == 0) {
    throw std::invalid_argument("Metal buffer size must be positive");
  }

  auto &metal = MetalContextAccess::get(context);
  MTL::Buffer *buffer = metal.device->newBuffer(
      byteCount, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    throw std::bad_alloc();
  }
  return buffer;
}

MTL::Buffer *allocate_private_metal_buffer(MetalContext &context,
                                           std::size_t byteCount) {
  if (byteCount == 0) {
    throw std::invalid_argument("Metal buffer size must be positive");
  }

  auto &metal = MetalContextAccess::get(context);
  MTL::Buffer *buffer = metal.device->newBuffer(
      byteCount, MTL::ResourceStorageModePrivate);
  if (buffer == nullptr) {
    throw std::bad_alloc();
  }
  return buffer;
}

void write_metal_buffer(MTL::Buffer *buffer, std::size_t byteOffset,
                        const float *source, std::size_t elementCount) {
  if (source == nullptr) {
    throw std::invalid_argument("Metal buffer source cannot be null");
  }
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, byteOffset, byteCount);
  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + byteOffset, source, byteCount);
}

float *read_metal_buffer(MTL::Buffer *buffer, std::size_t byteOffset,
                         std::size_t elementCount) {
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, byteOffset, byteCount);
  float *output = static_cast<float *>(std::malloc(byteCount));
  if (output == nullptr) {
    throw std::bad_alloc();
  }
  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(output, contents + byteOffset, byteCount);
  return output;
}

void copy_metal_buffer(MetalContext &context, MTL::Buffer *source,
                       std::size_t sourceOffset, MTL::Buffer *destination,
                       std::size_t destinationOffset,
                       std::size_t byteCount) {
  validate_metal_buffer_range(source, sourceOffset, byteCount);
  validate_metal_buffer_range(destination, destinationOffset, byteCount);
  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::BlitCommandEncoder *encoder = commandBuffer->blitCommandEncoder();
  encoder->copyFromBuffer(source, sourceOffset, destination,
                          destinationOffset, byteCount);
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

void release_metal_buffer(MTL::Buffer *buffer) {
  if (buffer != nullptr) {
    buffer->release();
  }
}

float *matmul_metal(std::vector<std::vector<float>> const X, MatMulFlag flagX,
                    std::vector<std::vector<float>> const Y, MatMulFlag flagY,
                    bool tile) {

  std::size_t X_rows = X.size();
  std::size_t X_columns = X[0].size();
  std::size_t Y_rows = Y.size();
  std::size_t Y_columns = Y[0].size();

  std::size_t M = flagX == MatMulFlag::TRANSPOSE ? X_columns : X_rows;
  std::size_t K = flagX == MatMulFlag::TRANSPOSE ? X_rows : X_columns;
  std::size_t B_rows =
      flagY == MatMulFlag::TRANSPOSE ? Y_columns : Y_rows;
  std::size_t N = flagY == MatMulFlag::TRANSPOSE ? Y_rows : Y_columns;

  if (K != B_rows)
    throw std::runtime_error(std::format(
        "Invalid shapes: Attempting to multiply ({},{}) and ({},{})", X.size(),
        X[0].size(), Y.size(), Y[0].size()));
  // 1. Memory management scope wrapper for Metal
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  // 2. Initialize GPU Device
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    pool->release();
    throw std::runtime_error("Metal is not supported");
  }

  // 3. Load the named Metal library built by Meson beside this executable.
  NS::Error *error = nullptr;
  std::filesystem::path metallib_path =
      executable_directory() / "inference.metallib";
  NS::String *metallib_path_string =
      NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
  MTL::Library *library = device->newLibrary(metallib_path_string, &error);

  if (!library) {
    device->release();
    pool->release();
    throw std::runtime_error(
        std::format("Could not load {}", metallib_path.string()));
  }

  NS::String *funcName = NS::String::string(
      tile ? "matmul_tile" : "matmul_naive", NS::UTF8StringEncoding);
  MTL::Function *function = library->newFunction(funcName);
  library->release();

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(function, &error);
  function->release();

  // 5. Create one buffer containing A, B, and C in sequence.
  std::size_t aBufferSize = X_rows * X_columns * sizeof(float);
  std::size_t bBufferSize = Y_rows * Y_columns * sizeof(float);
  std::size_t outputBufferSize = M * N * sizeof(float);
  std::size_t aOffset = 0;
  std::size_t bOffset = aOffset + aBufferSize;
  std::size_t outputOffset = bOffset + bBufferSize;
  std::size_t bufferSize = outputOffset + outputBufferSize;
  MTL::Buffer *buffer =
      device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pipeline->release();
    device->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::byte *contents = static_cast<std::byte *>(buffer->contents());

  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(M),
      .K = static_cast<std::uint32_t>(K),
      .N = static_cast<std::uint32_t>(N),
  };

  // Populate arrays using raw float pointers
  float *ptrA = reinterpret_cast<float *>(contents + aOffset);
  for (std::size_t row = 0; row < X_rows; row++) {
    for (std::size_t column = 0; column < X_columns; column++) {
      ptrA[row * X_columns + column] = X[row][column];
    }
  }

  float *ptrB = reinterpret_cast<float *>(contents + bOffset);
  for (std::size_t row = 0; row < Y_rows; row++) {
    for (std::size_t column = 0; column < Y_columns; column++) {
      ptrB[row * Y_columns + column] = Y[row][column];
    }
  }

  float *ptrC = reinterpret_cast<float *>(contents + outputOffset);
  for (std::size_t m = 0; m < M; m++) {
    for (std::size_t n = 0; n < N; n++) {
      ptrC[m * N + n] = 0.0f;
    }
  }

  // 6. Command execution and encoding
  MTL::CommandQueue *queue = device->newCommandQueue();
  MTL::CommandBuffer *commandBuffer = queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, aOffset, 0);
  encoder->setBytes(&flagX, sizeof(flagX), 1);
  encoder->setBuffer(buffer, bOffset, 2);
  encoder->setBytes(&flagY, sizeof(flagY), 3);
  encoder->setBuffer(buffer, outputOffset, 4);
  encoder->setBytes(&dims, sizeof(dims), 5);

  // Define standard execution grid boundaries
  MTL::Size threadsPerThreadgroup =
      tile ? MTL::Size(8, 16, 1) : MTL::Size(32, 32, 1);
  MTL::Size threadgroupCount =
      MTL::Size((N + 31) / 32, (M + 31) / 32, 1);
  encoder->dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup);
  encoder->endEncoding();

  // 7. Run and wait synchronously
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(M * N * sizeof(float)));
  if (output == nullptr) {
    buffer->release();
    pipeline->release();
    queue->release();
    device->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, ptrC, M * N * sizeof(float));

  // 9. Clean up resources manually
  buffer->release();
  pipeline->release();
  queue->release();
  device->release();
  pool->release();

  return output;
}

float *matmul_metal(const float *A, MatMulFlag flagA, const float *B,
                    MatMulFlag flagB, std::size_t M, std::size_t K,
                    std::size_t N, bool tile) {
  MetalContext context;
  return matmul_metal(context, A, flagA, B, flagB, M, K, N, tile);
}

float *matmul_metal(MetalContext &context, const float *A,
                    MatMulFlag flagA, const float *B, MatMulFlag flagB,
                    std::size_t M, std::size_t K, std::size_t N, bool tile) {
  if (A == nullptr || B == nullptr) {
    throw std::invalid_argument("Metal matmul inputs cannot be null");
  }
  if (M == 0 || K == 0 || N == 0) {
    throw std::invalid_argument("Metal matmul dimensions must be positive");
  }
  if (M > std::numeric_limits<std::uint32_t>::max() ||
      K > std::numeric_limits<std::uint32_t>::max() ||
      N > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal matmul dimensions exceed uint32_t");
  }

  std::size_t aBufferSize = M * K * sizeof(float);
  std::size_t bBufferSize = K * N * sizeof(float);
  std::size_t outputBufferSize = M * N * sizeof(float);
  std::size_t aOffset = 0;
  std::size_t bOffset = aOffset + aBufferSize;
  std::size_t outputOffset = bOffset + bBufferSize;
  std::size_t bufferSize = outputOffset + outputBufferSize;

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline =
      metal.pipeline(tile ? "matmul_tile" : "matmul_naive");

  MTL::Buffer *buffer =
      device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }

  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + aOffset, A, aBufferSize);
  std::memcpy(contents + bOffset, B, bBufferSize);

  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(M),
      .K = static_cast<std::uint32_t>(K),
      .N = static_cast<std::uint32_t>(N),
  };

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, aOffset, 0);
  encoder->setBytes(&flagA, sizeof(flagA), 1);
  encoder->setBuffer(buffer, bOffset, 2);
  encoder->setBytes(&flagB, sizeof(flagB), 3);
  encoder->setBuffer(buffer, outputOffset, 4);
  encoder->setBytes(&dims, sizeof(dims), 5);
  encoder->dispatchThreadgroups(MTL::Size((N + 31) / 32, (M + 31) / 32, 1),
                                tile ? MTL::Size(8, 16, 1)
                                     : MTL::Size(32, 32, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(outputBufferSize));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, outputBufferSize);

  buffer->release();
  pool->release();
  return output;
}

void matmul_metal(MetalContext &context, MTL::Buffer *buffer,
                  std::size_t aOffset, MatMulFlag flagA,
                  std::size_t bOffset, MatMulFlag flagB,
                  std::size_t outputOffset, std::size_t M,
                  std::size_t K, std::size_t N, bool tile) {
  matmul_metal(context, buffer, aOffset, flagA, buffer, bOffset, flagB,
               buffer, outputOffset, M, K, N, tile);
}

void matmul_metal(MetalContext &context, MTL::Buffer *aBuffer,
                  std::size_t aOffset, MatMulFlag flagA,
                  MTL::Buffer *bBuffer, std::size_t bOffset,
                  MatMulFlag flagB, MTL::Buffer *outputBuffer,
                  std::size_t outputOffset, std::size_t M,
                  std::size_t K, std::size_t N, bool tile) {
  if (M == 0 || K == 0 || N == 0 ||
      M > std::numeric_limits<std::uint32_t>::max() ||
      K > std::numeric_limits<std::uint32_t>::max() ||
      N > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal matmul dimensions are invalid");
  }

  validate_metal_buffer_range(aBuffer, aOffset, M * K * sizeof(float));
  validate_metal_buffer_range(bBuffer, bOffset, K * N * sizeof(float));
  validate_metal_buffer_range(outputBuffer, outputOffset,
                              M * N * sizeof(float));

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline =
      metal.pipeline(tile ? "matmul_tile" : "matmul_naive");
  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(M),
      .K = static_cast<std::uint32_t>(K),
      .N = static_cast<std::uint32_t>(N),
  };

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(aBuffer, aOffset, 0);
  encoder->setBytes(&flagA, sizeof(flagA), 1);
  encoder->setBuffer(bBuffer, bOffset, 2);
  encoder->setBytes(&flagB, sizeof(flagB), 3);
  encoder->setBuffer(outputBuffer, outputOffset, 4);
  encoder->setBytes(&dims, sizeof(dims), 5);
  encoder->dispatchThreadgroups(MTL::Size((N + 31) / 32, (M + 31) / 32, 1),
                                tile ? MTL::Size(8, 16, 1)
                                     : MTL::Size(32, 32, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

void matvecmul_metal(MetalContext &context, MTL::Buffer *inputBuffer,
                     std::size_t inputOffset, MTL::Buffer *weightBuffer,
                     std::size_t weightOffset, MTL::Buffer *outputBuffer,
                     std::size_t outputOffset, std::size_t inputSize,
                     std::size_t outputSize) {
  constexpr std::size_t float4Alignment = 4 * sizeof(float);
  if (inputSize == 0 || outputSize == 0 || inputSize % 4 != 0 ||
      inputOffset % float4Alignment != 0 ||
      weightOffset % float4Alignment != 0 ||
      inputSize > std::numeric_limits<std::uint32_t>::max() ||
      outputSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal matvecmul dimensions are invalid");
  }
  if (outputSize >
      std::numeric_limits<std::size_t>::max() / inputSize / sizeof(float)) {
    throw std::overflow_error("Metal matvecmul weight size overflows");
  }

  validate_metal_buffer_range(inputBuffer, inputOffset,
                              inputSize * sizeof(float));
  validate_metal_buffer_range(weightBuffer, weightOffset,
                              inputSize * outputSize * sizeof(float));
  validate_metal_buffer_range(outputBuffer, outputOffset,
                              outputSize * sizeof(float));

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline = metal.pipeline("matvecmul");
  std::uint32_t metalInputSize = static_cast<std::uint32_t>(inputSize);
  std::uint32_t metalOutputSize = static_cast<std::uint32_t>(outputSize);

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(inputBuffer, inputOffset, 0);
  encoder->setBuffer(weightBuffer, weightOffset, 1);
  encoder->setBuffer(outputBuffer, outputOffset, 2);
  encoder->setBytes(&metalInputSize, sizeof(metalInputSize), 3);
  encoder->setBytes(&metalOutputSize, sizeof(metalOutputSize), 4);
  encoder->dispatchThreadgroups(MTL::Size((outputSize + 3) / 4, 1, 1),
                                MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *layer_norm_metal(const float *input, const float *gamma,
                        const float *beta, std::size_t sequenceLength,
                        std::size_t embeddingSize, float epsilon) {
  MetalContext context;
  return layer_norm_metal(context, input, gamma, beta, sequenceLength,
                          embeddingSize, epsilon);
}

float *layer_norm_metal(MetalContext &context, const float *input,
                        const float *gamma, const float *beta,
                        std::size_t sequenceLength,
                        std::size_t embeddingSize, float epsilon) {
  if (input == nullptr || gamma == nullptr || beta == nullptr) {
    throw std::invalid_argument("Metal LayerNorm inputs cannot be null");
  }

  std::size_t inputBufferSize =
      sequenceLength * embeddingSize * sizeof(float);
  std::size_t parameterBufferSize = embeddingSize * sizeof(float);
  std::size_t inputOffset = 0;
  std::size_t gammaOffset = inputOffset + inputBufferSize;
  std::size_t betaOffset = gammaOffset + parameterBufferSize;
  std::size_t outputOffset = betaOffset + parameterBufferSize;
  std::size_t bufferSize = outputOffset + inputBufferSize;

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline = metal.pipeline("layer_norm");

  MTL::Buffer *buffer =
      device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }
  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + inputOffset, input, inputBufferSize);
  std::memcpy(contents + gammaOffset, gamma, parameterBufferSize);
  std::memcpy(contents + betaOffset, beta, parameterBufferSize);

  std::uint32_t metalEmbeddingSize =
      static_cast<std::uint32_t>(embeddingSize);
  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, inputOffset, 0);
  encoder->setBuffer(buffer, gammaOffset, 1);
  encoder->setBuffer(buffer, betaOffset, 2);
  encoder->setBuffer(buffer, outputOffset, 3);
  encoder->setBytes(&metalEmbeddingSize, sizeof(metalEmbeddingSize), 4);
  encoder->setBytes(&epsilon, sizeof(epsilon), 5);
  encoder->dispatchThreadgroups(MTL::Size(sequenceLength, 1, 1),
                                MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(inputBufferSize));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, inputBufferSize);

  buffer->release();
  pool->release();
  return output;
}

void layer_norm_metal(MetalContext &context, MTL::Buffer *buffer,
                      std::size_t inputOffset, std::size_t gammaOffset,
                      std::size_t betaOffset, std::size_t outputOffset,
                      std::size_t sequenceLength,
                      std::size_t embeddingSize, float epsilon) {
  layer_norm_metal(context, buffer, inputOffset, buffer, gammaOffset,
                   buffer, betaOffset, buffer, outputOffset,
                   sequenceLength, embeddingSize, epsilon);
}

void layer_norm_metal(MetalContext &context, MTL::Buffer *inputBuffer,
                      std::size_t inputOffset, MTL::Buffer *gammaBuffer,
                      std::size_t gammaOffset, MTL::Buffer *betaBuffer,
                      std::size_t betaOffset, MTL::Buffer *outputBuffer,
                      std::size_t outputOffset,
                      std::size_t sequenceLength,
                      std::size_t embeddingSize, float epsilon) {
  if (sequenceLength == 0 || embeddingSize == 0 ||
      embeddingSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal LayerNorm dimensions are invalid");
  }

  std::size_t inputSize =
      sequenceLength * embeddingSize * sizeof(float);
  std::size_t parameterSize = embeddingSize * sizeof(float);
  validate_metal_buffer_range(inputBuffer, inputOffset, inputSize);
  validate_metal_buffer_range(gammaBuffer, gammaOffset, parameterSize);
  validate_metal_buffer_range(betaBuffer, betaOffset, parameterSize);
  validate_metal_buffer_range(outputBuffer, outputOffset, inputSize);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline = metal.pipeline("layer_norm");
  std::uint32_t metalEmbeddingSize =
      static_cast<std::uint32_t>(embeddingSize);

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(inputBuffer, inputOffset, 0);
  encoder->setBuffer(gammaBuffer, gammaOffset, 1);
  encoder->setBuffer(betaBuffer, betaOffset, 2);
  encoder->setBuffer(outputBuffer, outputOffset, 3);
  encoder->setBytes(&metalEmbeddingSize, sizeof(metalEmbeddingSize), 4);
  encoder->setBytes(&epsilon, sizeof(epsilon), 5);
  encoder->dispatchThreadgroups(MTL::Size(sequenceLength, 1, 1),
                                MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *residual_add_metal(const float *residual, const float *input,
                          std::size_t elementCount) {
  MetalContext context;
  return residual_add_metal(context, residual, input, elementCount);
}

float *residual_add_metal(MetalContext &context, const float *residual,
                          const float *input, std::size_t elementCount) {
  if (residual == nullptr || input == nullptr) {
    throw std::invalid_argument("Metal residual inputs cannot be null");
  }
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal residual element count is invalid");
  }

  std::size_t bufferSize = elementCount * sizeof(float);
  std::size_t residualOffset = 0;
  std::size_t inputOffset = residualOffset + bufferSize;
  std::size_t outputOffset = inputOffset + bufferSize;
  std::size_t combinedBufferSize = outputOffset + bufferSize;
  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline = metal.pipeline("residual_add");

  MTL::Buffer *buffer =
      device->newBuffer(combinedBufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }

  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + residualOffset, residual, bufferSize);
  std::memcpy(contents + inputOffset, input, bufferSize);

  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, residualOffset, 0);
  encoder->setBuffer(buffer, inputOffset, 1);
  encoder->setBuffer(buffer, outputOffset, 2);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(bufferSize));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, bufferSize);

  buffer->release();
  pool->release();
  return output;
}

void residual_add_metal(MetalContext &context, MTL::Buffer *buffer,
                        std::size_t residualOffset,
                        std::size_t inputOffset,
                        std::size_t outputOffset,
                        std::size_t elementCount) {
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal residual element count is invalid");
  }

  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, residualOffset, byteCount);
  validate_metal_buffer_range(buffer, inputOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline = metal.pipeline("residual_add");
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, residualOffset, 0);
  encoder->setBuffer(buffer, inputOffset, 1);
  encoder->setBuffer(buffer, outputOffset, 2);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *relu_metal(const float *input, std::size_t elementCount) {
  MetalContext context;
  return relu_metal(context, input, elementCount);
}

float *relu_metal(MetalContext &context, const float *input,
                  std::size_t elementCount) {
  if (input == nullptr) {
    throw std::invalid_argument("Metal ReLU input cannot be null");
  }
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal ReLU element count is invalid");
  }

  std::size_t bufferSize = elementCount * sizeof(float);
  std::size_t inputOffset = 0;
  std::size_t outputOffset = inputOffset + bufferSize;
  std::size_t combinedBufferSize = outputOffset + bufferSize;
  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline = metal.pipeline("relu");

  MTL::Buffer *buffer =
      device->newBuffer(combinedBufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }

  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + inputOffset, input, bufferSize);

  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, inputOffset, 0);
  encoder->setBuffer(buffer, outputOffset, 1);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 2);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(bufferSize));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, bufferSize);

  buffer->release();
  pool->release();
  return output;
}

void relu_metal(MetalContext &context, MTL::Buffer *buffer,
                std::size_t inputOffset, std::size_t outputOffset,
                std::size_t elementCount) {
  if (elementCount == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal ReLU element count is invalid");
  }

  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, inputOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline = metal.pipeline("relu");
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, inputOffset, 0);
  encoder->setBuffer(buffer, outputOffset, 1);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 2);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *add_bias_metal(const float *input, const float *bias,
                      std::size_t elementCount, std::size_t outputWidth) {
  MetalContext context;
  return add_bias_metal(context, input, bias, elementCount, outputWidth);
}

float *add_bias_metal(MetalContext &context, const float *input,
                      const float *bias, std::size_t elementCount,
                      std::size_t outputWidth) {
  if (input == nullptr || bias == nullptr) {
    throw std::invalid_argument("Metal bias inputs cannot be null");
  }
  if (elementCount == 0 || outputWidth == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max() ||
      outputWidth > std::numeric_limits<std::uint32_t>::max() ||
      elementCount % outputWidth != 0) {
    throw std::invalid_argument("Metal bias dimensions are invalid");
  }

  std::size_t inputBufferSize = elementCount * sizeof(float);
  std::size_t biasBufferSize = outputWidth * sizeof(float);
  std::size_t inputOffset = 0;
  std::size_t biasOffset = inputOffset + inputBufferSize;
  std::size_t outputOffset = biasOffset + biasBufferSize;
  std::size_t bufferSize = outputOffset + inputBufferSize;
  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline = metal.pipeline("add_bias");

  MTL::Buffer *buffer =
      device->newBuffer(bufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }

  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + inputOffset, input, inputBufferSize);
  std::memcpy(contents + biasOffset, bias, biasBufferSize);

  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  std::uint32_t metalOutputWidth = static_cast<std::uint32_t>(outputWidth);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, inputOffset, 0);
  encoder->setBuffer(buffer, biasOffset, 1);
  encoder->setBuffer(buffer, outputOffset, 2);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  encoder->setBytes(&metalOutputWidth, sizeof(metalOutputWidth), 4);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(inputBufferSize));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, inputBufferSize);

  buffer->release();
  pool->release();
  return output;
}

void add_bias_metal(MetalContext &context, MTL::Buffer *buffer,
                    std::size_t inputOffset, std::size_t biasOffset,
                    std::size_t outputOffset,
                    std::size_t elementCount,
                    std::size_t outputWidth) {
  add_bias_metal(context, buffer, inputOffset, buffer, biasOffset,
                 buffer, outputOffset, elementCount, outputWidth);
}

void add_bias_metal(MetalContext &context, MTL::Buffer *inputBuffer,
                    std::size_t inputOffset, MTL::Buffer *biasBuffer,
                    std::size_t biasOffset, MTL::Buffer *outputBuffer,
                    std::size_t outputOffset,
                    std::size_t elementCount,
                    std::size_t outputWidth) {
  if (elementCount == 0 || outputWidth == 0 ||
      elementCount > std::numeric_limits<std::uint32_t>::max() ||
      outputWidth > std::numeric_limits<std::uint32_t>::max() ||
      elementCount % outputWidth != 0) {
    throw std::invalid_argument("Metal bias dimensions are invalid");
  }

  std::size_t inputSize = elementCount * sizeof(float);
  std::size_t biasSize = outputWidth * sizeof(float);
  validate_metal_buffer_range(inputBuffer, inputOffset, inputSize);
  validate_metal_buffer_range(biasBuffer, biasOffset, biasSize);
  validate_metal_buffer_range(outputBuffer, outputOffset, inputSize);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline = metal.pipeline("add_bias");
  std::uint32_t metalElementCount =
      static_cast<std::uint32_t>(elementCount);
  std::uint32_t metalOutputWidth =
      static_cast<std::uint32_t>(outputWidth);
  constexpr std::size_t threadsPerThreadgroup = 256;
  std::size_t threadgroupCount =
      (elementCount + threadsPerThreadgroup - 1) / threadsPerThreadgroup;

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(inputBuffer, inputOffset, 0);
  encoder->setBuffer(biasBuffer, biasOffset, 1);
  encoder->setBuffer(outputBuffer, outputOffset, 2);
  encoder->setBytes(&metalElementCount, sizeof(metalElementCount), 3);
  encoder->setBytes(&metalOutputWidth, sizeof(metalOutputWidth), 4);
  encoder->dispatchThreadgroups(MTL::Size(threadgroupCount, 1, 1),
                                MTL::Size(threadsPerThreadgroup, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *scaled_dot_product_attention_metal(const float *Q, const float *K,
                                          const float *V,
                                          std::size_t seq_len,
                                          std::size_t head_size,
                                          std::size_t num_heads,
                                          bool isCausal,
                                          std::size_t scale_size) {
  MetalContext context;
  return scaled_dot_product_attention_metal(
      context, Q, K, V, seq_len, head_size, num_heads, isCausal,
      scale_size);
}

float *scaled_dot_product_attention_metal(
    MetalContext &context, const float *Q, const float *K, const float *V,
    std::size_t seq_len, std::size_t head_size, std::size_t num_heads,
    bool isCausal, std::size_t scale_size) {
  if (Q == nullptr || K == nullptr || V == nullptr) {
    throw std::invalid_argument("Metal attention inputs cannot be null");
  }
  if (seq_len == 0 || head_size == 0 || num_heads == 0) {
    throw std::invalid_argument("Metal attention dimensions must be positive");
  }
  if (scale_size == 0) {
    scale_size = head_size;
  }
  if (seq_len > std::numeric_limits<std::uint32_t>::max() ||
      head_size > std::numeric_limits<std::uint32_t>::max() ||
      num_heads > std::numeric_limits<std::uint32_t>::max() ||
      scale_size > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal attention dimensions exceed uint32_t");
  }
  if (seq_len > std::numeric_limits<std::size_t>::max() / head_size) {
    throw std::overflow_error("Metal attention buffer size overflow");
  }

  std::size_t elements_per_head = seq_len * head_size;
  if (num_heads >
      std::numeric_limits<std::size_t>::max() / elements_per_head) {
    throw std::overflow_error("Metal attention buffer size overflow");
  }
  std::size_t element_count = num_heads * elements_per_head;
  std::size_t buffer_size = element_count * sizeof(float);
  std::size_t queryOffset = 0;
  std::size_t keyOffset = queryOffset + buffer_size;
  std::size_t valueOffset = keyOffset + buffer_size;
  std::size_t outputOffset = valueOffset + buffer_size;
  std::size_t combinedBufferSize = outputOffset + buffer_size;
  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = metal.device;
  MTL::ComputePipelineState *pipeline =
      metal.pipeline("scaled_dot_product_attention");

  MTL::Buffer *buffer =
      device->newBuffer(combinedBufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }

  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + queryOffset, Q, buffer_size);
  std::memcpy(contents + keyOffset, K, buffer_size);
  std::memcpy(contents + valueOffset, V, buffer_size);
  std::memset(contents + outputOffset, 0, buffer_size);

  AttentionDims dims = {
      .queryLength = static_cast<std::uint32_t>(seq_len),
      .keyValueLength = static_cast<std::uint32_t>(seq_len),
      .headSize = static_cast<std::uint32_t>(head_size),
      .cacheCapacity = static_cast<std::uint32_t>(seq_len),
      .queryStartPosition = 0,
  };
  std::uint32_t metal_num_heads = static_cast<std::uint32_t>(num_heads);
  std::uint32_t metal_scale_size = static_cast<std::uint32_t>(scale_size);

  MTL::CommandBuffer *command_buffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder =
      command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, queryOffset, 0);
  encoder->setBuffer(buffer, keyOffset, 1);
  encoder->setBuffer(buffer, valueOffset, 2);
  encoder->setBuffer(buffer, outputOffset, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  encoder->setBytes(&metal_num_heads, sizeof(metal_num_heads), 6);
  encoder->setBytes(&metal_scale_size, sizeof(metal_scale_size), 7);
  encoder->dispatchThreadgroups(MTL::Size(seq_len, num_heads, 1),
                                MTL::Size(32, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(buffer_size));
  if (!output) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, buffer_size);

  buffer->release();
  pool->release();
  return output;
}

void scaled_dot_product_attention_metal(
    MetalContext &context, MTL::Buffer *buffer,
    std::size_t queryOffset, std::size_t keyOffset,
    std::size_t valueOffset, std::size_t outputOffset,
    std::size_t sequenceLength, std::size_t headSize,
    std::size_t headCount, bool isCausal, std::size_t scaleSize) {
  if (sequenceLength == 0 || headSize == 0 || headCount == 0) {
    throw std::invalid_argument("Metal attention dimensions must be positive");
  }
  if (scaleSize == 0) {
    scaleSize = headSize;
  }
  if (sequenceLength > std::numeric_limits<std::uint32_t>::max() ||
      headSize > std::numeric_limits<std::uint32_t>::max() ||
      headCount > std::numeric_limits<std::uint32_t>::max() ||
      scaleSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal attention dimensions exceed uint32_t");
  }

  std::size_t elementCount = sequenceLength * headSize * headCount;
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, queryOffset, byteCount);
  validate_metal_buffer_range(buffer, keyOffset, byteCount);
  validate_metal_buffer_range(buffer, valueOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline =
      metal.pipeline("scaled_dot_product_attention");
  AttentionDims dims = {
      .queryLength = static_cast<std::uint32_t>(sequenceLength),
      .keyValueLength = static_cast<std::uint32_t>(sequenceLength),
      .headSize = static_cast<std::uint32_t>(headSize),
      .cacheCapacity = static_cast<std::uint32_t>(sequenceLength),
      .queryStartPosition = 0,
  };
  std::uint32_t metalHeadCount =
      static_cast<std::uint32_t>(headCount);
  std::uint32_t metalScaleSize =
      static_cast<std::uint32_t>(scaleSize);

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, queryOffset, 0);
  encoder->setBuffer(buffer, keyOffset, 1);
  encoder->setBuffer(buffer, valueOffset, 2);
  encoder->setBuffer(buffer, outputOffset, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  encoder->setBytes(&metalHeadCount, sizeof(metalHeadCount), 6);
  encoder->setBytes(&metalScaleSize, sizeof(metalScaleSize), 7);
  encoder->dispatchThreadgroups(
      MTL::Size(sequenceLength, headCount, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

float *scaled_dot_product_attention_cached_metal(
    MetalContext &context, KVCache &cache, std::size_t layer,
    const float *Q, std::size_t queryLength, std::size_t keyValueLength,
    std::size_t queryStartPosition, bool isCausal,
    std::size_t scaleSize) {
  if (Q == nullptr) {
    throw std::invalid_argument("Cached attention query cannot be null");
  }

  if (layer >= cache.layerCount || queryLength == 0 ||
      keyValueLength == 0 || keyValueLength > cache.cacheCapacity ||
      queryStartPosition + queryLength > keyValueLength ||
      scaleSize == 0) {
    throw std::invalid_argument("Cached attention dimensions are invalid");
  }
  if (queryLength > std::numeric_limits<std::uint32_t>::max() ||
      keyValueLength > std::numeric_limits<std::uint32_t>::max() ||
      cache.headSize > std::numeric_limits<std::uint32_t>::max() ||
      cache.cacheCapacity > std::numeric_limits<std::uint32_t>::max() ||
      queryStartPosition > std::numeric_limits<std::uint32_t>::max() ||
      cache.headCount > std::numeric_limits<std::uint32_t>::max() ||
      scaleSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Cached attention dimensions exceed uint32_t");
  }

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline =
      metal.pipeline("scaled_dot_product_attention");

  std::size_t elementCount =
      queryLength * cache.headCount * cache.headSize;
  std::size_t bufferSize = elementCount * sizeof(float);
  std::size_t queryOffset = 0;
  std::size_t outputOffset = queryOffset + bufferSize;
  std::size_t combinedBufferSize = outputOffset + bufferSize;
  MTL::Buffer *buffer = metal.device->newBuffer(
      combinedBufferSize, MTL::ResourceStorageModeShared);
  if (buffer == nullptr) {
    pool->release();
    throw std::bad_alloc();
  }
  std::byte *contents = static_cast<std::byte *>(buffer->contents());
  std::memcpy(contents + queryOffset, Q, bufferSize);
  std::memset(contents + outputOffset, 0, bufferSize);

  AttentionDims dims = {
      .queryLength = static_cast<std::uint32_t>(queryLength),
      .keyValueLength = static_cast<std::uint32_t>(keyValueLength),
      .headSize = static_cast<std::uint32_t>(cache.headSize),
      .cacheCapacity = static_cast<std::uint32_t>(cache.cacheCapacity),
      .queryStartPosition =
          static_cast<std::uint32_t>(queryStartPosition),
  };
  std::uint32_t metalHeadCount =
      static_cast<std::uint32_t>(cache.headCount);
  std::uint32_t metalScaleSize = static_cast<std::uint32_t>(scaleSize);
  std::size_t layerOffset =
      layer * cache.headCount * cache.cacheCapacity * cache.headSize *
      sizeof(float);

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, queryOffset, 0);
  encoder->setBuffer(cache.keys, layerOffset, 1);
  encoder->setBuffer(cache.values, layerOffset, 2);
  encoder->setBuffer(buffer, outputOffset, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  encoder->setBytes(&metalHeadCount, sizeof(metalHeadCount), 6);
  encoder->setBytes(&metalScaleSize, sizeof(metalScaleSize), 7);
  encoder->dispatchThreadgroups(
      MTL::Size(queryLength, cache.headCount, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(bufferSize));
  if (output == nullptr) {
    buffer->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, contents + outputOffset, bufferSize);

  buffer->release();
  pool->release();
  return output;
}

void scaled_dot_product_attention_cached_metal(
    MetalContext &context, KVCache &cache, std::size_t layer,
    MTL::Buffer *buffer, std::size_t queryOffset,
    std::size_t outputOffset, std::size_t queryLength,
    std::size_t keyValueLength, std::size_t queryStartPosition,
    bool isCausal, std::size_t scaleSize) {
  if (layer >= cache.layerCount || queryLength == 0 ||
      keyValueLength == 0 || keyValueLength > cache.cacheCapacity ||
      queryStartPosition + queryLength > keyValueLength || scaleSize == 0) {
    throw std::invalid_argument("Cached attention dimensions are invalid");
  }
  if (queryLength > std::numeric_limits<std::uint32_t>::max() ||
      keyValueLength > std::numeric_limits<std::uint32_t>::max() ||
      cache.headSize > std::numeric_limits<std::uint32_t>::max() ||
      cache.cacheCapacity > std::numeric_limits<std::uint32_t>::max() ||
      queryStartPosition > std::numeric_limits<std::uint32_t>::max() ||
      cache.headCount > std::numeric_limits<std::uint32_t>::max() ||
      scaleSize > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Cached attention dimensions exceed uint32_t");
  }

  std::size_t elementCount =
      queryLength * cache.headCount * cache.headSize;
  std::size_t byteCount = elementCount * sizeof(float);
  validate_metal_buffer_range(buffer, queryOffset, byteCount);
  validate_metal_buffer_range(buffer, outputOffset, byteCount);

  auto &metal = MetalContextAccess::get(context);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::ComputePipelineState *pipeline =
      metal.pipeline("scaled_dot_product_attention");
  AttentionDims dims = {
      .queryLength = static_cast<std::uint32_t>(queryLength),
      .keyValueLength = static_cast<std::uint32_t>(keyValueLength),
      .headSize = static_cast<std::uint32_t>(cache.headSize),
      .cacheCapacity = static_cast<std::uint32_t>(cache.cacheCapacity),
      .queryStartPosition =
          static_cast<std::uint32_t>(queryStartPosition),
  };
  std::uint32_t metalHeadCount =
      static_cast<std::uint32_t>(cache.headCount);
  std::uint32_t metalScaleSize =
      static_cast<std::uint32_t>(scaleSize);
  std::size_t layerOffset =
      layer * cache.headCount * cache.cacheCapacity * cache.headSize *
      sizeof(float);

  MTL::CommandBuffer *commandBuffer = metal.queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(buffer, queryOffset, 0);
  encoder->setBuffer(cache.keys, layerOffset, 1);
  encoder->setBuffer(cache.values, layerOffset, 2);
  encoder->setBuffer(buffer, outputOffset, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  encoder->setBytes(&metalHeadCount, sizeof(metalHeadCount), 6);
  encoder->setBytes(&metalScaleSize, sizeof(metalScaleSize), 7);
  encoder->dispatchThreadgroups(
      MTL::Size(queryLength, cache.headCount, 1), MTL::Size(32, 1, 1));
  encoder->endEncoding();
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();
  pool->release();
}

} // namespace inference
