#pragma once

#include "export.hpp"

#include "matmulflags.hpp"
#include <cstddef>
#include <memory>
#include <vector>

namespace MTL {
class Buffer;
class CommandBuffer;
class ComputeCommandEncoder;
}

namespace NS {
class AutoreleasePool;
}

namespace inference {

class KVCache;
class MetalCommandBatch;

class INFERENCE_PUBLIC MetalContext {

public:
  MetalContext();
  ~MetalContext();

  MetalContext(const MetalContext &) = delete;
  MetalContext &operator=(const MetalContext &) = delete;
  MetalContext(MetalContext &&) noexcept;
  MetalContext &operator=(MetalContext &&) noexcept;

private:
  struct Impl;
  std::unique_ptr<Impl> impl;

  friend struct MetalContextAccess;
  friend class MetalCommandBatch;
};

class INFERENCE_PUBLIC MetalCommandBatch {

public:
  explicit MetalCommandBatch(MetalContext &context);
  ~MetalCommandBatch();

  MetalCommandBatch(const MetalCommandBatch &) = delete;
  MetalCommandBatch &operator=(const MetalCommandBatch &) = delete;

  void matvecmul(MTL::Buffer *inputBuffer, std::size_t inputOffset,
                 MTL::Buffer *weightBuffer, std::size_t weightOffset,
                 MTL::Buffer *outputBuffer, std::size_t outputOffset,
                 std::size_t inputSize, std::size_t outputSize);
  void qkvMatvecmulHeads(
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
      std::size_t inputSize, std::size_t headSize);
  void layerNorm(MTL::Buffer *inputBuffer, std::size_t inputOffset,
                 MTL::Buffer *gammaBuffer, std::size_t gammaOffset,
                 MTL::Buffer *betaBuffer, std::size_t betaOffset,
                 MTL::Buffer *outputBuffer, std::size_t outputOffset,
                 std::size_t sequenceLength, std::size_t embeddingSize,
                 float epsilon = 1.0e-5f);
  void residualAdd(MTL::Buffer *buffer, std::size_t residualOffset,
                   std::size_t inputOffset, std::size_t outputOffset,
                   std::size_t elementCount);
  void relu(MTL::Buffer *buffer, std::size_t inputOffset,
            std::size_t outputOffset, std::size_t elementCount);
  void addBias(MTL::Buffer *inputBuffer, std::size_t inputOffset,
               MTL::Buffer *biasBuffer, std::size_t biasOffset,
               MTL::Buffer *outputBuffer, std::size_t outputOffset,
               std::size_t elementCount, std::size_t outputWidth);
  void cachedAttention(KVCache &cache, std::size_t layer,
                       MTL::Buffer *buffer, std::size_t queryOffset,
                       std::size_t outputOffset, std::size_t queryLength,
                       std::size_t keyValueLength,
                       std::size_t queryStartPosition, bool isCausal,
                       std::size_t scaleSize);
  void commitAndWait();

private:
  MetalContext *context;
  MTL::CommandBuffer *commandBuffer;
  MTL::ComputeCommandEncoder *encoder;
  NS::AutoreleasePool *pool;
  bool completed = false;
};

INFERENCE_PUBLIC int run_metal();
INFERENCE_PUBLIC MTL::Buffer *allocate_metal_buffer(
    MetalContext &context, std::size_t byteCount);
INFERENCE_PUBLIC MTL::Buffer *allocate_private_metal_buffer(
    MetalContext &context, std::size_t byteCount);
INFERENCE_PUBLIC void write_metal_buffer(
    MTL::Buffer *buffer, std::size_t byteOffset, const float *source,
    std::size_t elementCount);
INFERENCE_PUBLIC float *read_metal_buffer(
    MTL::Buffer *buffer, std::size_t byteOffset,
    std::size_t elementCount);
INFERENCE_PUBLIC void copy_metal_buffer(
    MetalContext &context, MTL::Buffer *source,
    std::size_t sourceOffset, MTL::Buffer *destination,
    std::size_t destinationOffset, std::size_t byteCount);
INFERENCE_PUBLIC void release_metal_buffer(MTL::Buffer *buffer);
INFERENCE_PUBLIC float *matmul_metal(std::vector<std::vector<float>> const X,
                                     MatMulFlag flagX,
                                     std::vector<std::vector<float>> const Y,
                                     MatMulFlag flagY, bool tile);
INFERENCE_PUBLIC float *matmul_metal(
    const float *A, MatMulFlag flagA, const float *B, MatMulFlag flagB,
    std::size_t M, std::size_t K, std::size_t N, bool tile);
INFERENCE_PUBLIC float *matmul_metal(
    MetalContext &context, const float *A, MatMulFlag flagA, const float *B,
    MatMulFlag flagB, std::size_t M, std::size_t K, std::size_t N, bool tile);
INFERENCE_PUBLIC void matmul_metal(
    MetalContext &context, MTL::Buffer *buffer, std::size_t aOffset,
    MatMulFlag flagA, std::size_t bOffset, MatMulFlag flagB,
    std::size_t outputOffset, std::size_t M, std::size_t K,
    std::size_t N, bool tile);
INFERENCE_PUBLIC void matmul_metal(
    MetalContext &context, MTL::Buffer *aBuffer,
    std::size_t aOffset, MatMulFlag flagA, MTL::Buffer *bBuffer,
    std::size_t bOffset, MatMulFlag flagB, MTL::Buffer *outputBuffer,
    std::size_t outputOffset, std::size_t M, std::size_t K,
    std::size_t N, bool tile);
INFERENCE_PUBLIC void matvecmul_metal(
    MetalContext &context, MTL::Buffer *inputBuffer,
    std::size_t inputOffset, MTL::Buffer *weightBuffer,
    std::size_t weightOffset, MTL::Buffer *outputBuffer,
    std::size_t outputOffset, std::size_t inputSize,
    std::size_t outputSize);
INFERENCE_PUBLIC float *scaled_dot_product_attention_metal(
    const float *Q, const float *K, const float *V, std::size_t seq_len,
    std::size_t head_size, std::size_t num_heads, bool isCausal,
    std::size_t scale_size = 0);
INFERENCE_PUBLIC float *scaled_dot_product_attention_metal(
    MetalContext &context, const float *Q, const float *K, const float *V,
    std::size_t seq_len, std::size_t head_size, std::size_t num_heads,
    bool isCausal, std::size_t scale_size = 0);
INFERENCE_PUBLIC void scaled_dot_product_attention_metal(
    MetalContext &context, MTL::Buffer *buffer, std::size_t queryOffset,
    std::size_t keyOffset, std::size_t valueOffset,
    std::size_t outputOffset, std::size_t sequenceLength,
    std::size_t headSize, std::size_t headCount, bool isCausal,
    std::size_t scaleSize = 0);
INFERENCE_PUBLIC float *scaled_dot_product_attention_cached_metal(
    MetalContext &context, KVCache &cache, std::size_t layer,
    const float *Q, std::size_t queryLength, std::size_t keyValueLength,
    std::size_t queryStartPosition, bool isCausal,
    std::size_t scaleSize);
INFERENCE_PUBLIC void scaled_dot_product_attention_cached_metal(
    MetalContext &context, KVCache &cache, std::size_t layer,
    MTL::Buffer *buffer, std::size_t queryOffset,
    std::size_t outputOffset, std::size_t queryLength,
    std::size_t keyValueLength, std::size_t queryStartPosition,
    bool isCausal, std::size_t scaleSize);
INFERENCE_PUBLIC float *layer_norm_metal(
    const float *input, const float *gamma, const float *beta,
    std::size_t sequenceLength, std::size_t embeddingSize,
    float epsilon = 1.0e-5f);
INFERENCE_PUBLIC float *layer_norm_metal(
    MetalContext &context, const float *input, const float *gamma,
    const float *beta, std::size_t sequenceLength,
    std::size_t embeddingSize, float epsilon = 1.0e-5f);
INFERENCE_PUBLIC void layer_norm_metal(
    MetalContext &context, MTL::Buffer *buffer,
    std::size_t inputOffset, std::size_t gammaOffset,
    std::size_t betaOffset, std::size_t outputOffset,
    std::size_t sequenceLength, std::size_t embeddingSize,
    float epsilon = 1.0e-5f);
INFERENCE_PUBLIC void layer_norm_metal(
    MetalContext &context, MTL::Buffer *inputBuffer,
    std::size_t inputOffset, MTL::Buffer *gammaBuffer,
    std::size_t gammaOffset, MTL::Buffer *betaBuffer,
    std::size_t betaOffset, MTL::Buffer *outputBuffer,
    std::size_t outputOffset, std::size_t sequenceLength,
    std::size_t embeddingSize, float epsilon = 1.0e-5f);
INFERENCE_PUBLIC float *residual_add_metal(
    const float *residual, const float *input, std::size_t elementCount);
INFERENCE_PUBLIC float *residual_add_metal(
    MetalContext &context, const float *residual, const float *input,
    std::size_t elementCount);
INFERENCE_PUBLIC void residual_add_metal(
    MetalContext &context, MTL::Buffer *buffer,
    std::size_t residualOffset, std::size_t inputOffset,
    std::size_t outputOffset, std::size_t elementCount);
INFERENCE_PUBLIC float *relu_metal(
    const float *input, std::size_t elementCount);
INFERENCE_PUBLIC float *relu_metal(
    MetalContext &context, const float *input, std::size_t elementCount);
INFERENCE_PUBLIC void relu_metal(
    MetalContext &context, MTL::Buffer *buffer,
    std::size_t inputOffset, std::size_t outputOffset,
    std::size_t elementCount);
INFERENCE_PUBLIC float *add_bias_metal(
    const float *input, const float *bias, std::size_t elementCount,
    std::size_t outputWidth);
INFERENCE_PUBLIC float *add_bias_metal(
    MetalContext &context, const float *input, const float *bias,
    std::size_t elementCount, std::size_t outputWidth);
INFERENCE_PUBLIC void add_bias_metal(
    MetalContext &context, MTL::Buffer *buffer,
    std::size_t inputOffset, std::size_t biasOffset,
    std::size_t outputOffset, std::size_t elementCount,
    std::size_t outputWidth);
INFERENCE_PUBLIC void add_bias_metal(
    MetalContext &context, MTL::Buffer *inputBuffer,
    std::size_t inputOffset, MTL::Buffer *biasBuffer,
    std::size_t biasOffset, MTL::Buffer *outputBuffer,
    std::size_t outputOffset, std::size_t elementCount,
    std::size_t outputWidth);

} // namespace inference
