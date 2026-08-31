#pragma once

#include "export.hpp"

#include <cstddef>

namespace MTL {
class Buffer;
} // namespace inference

namespace inference {

class MetalContext;

class INFERENCE_PUBLIC KVCache {

public:
  KVCache(MetalContext &context, std::size_t layerCount,
          std::size_t headCount, std::size_t capacity,
          std::size_t headSize);
  ~KVCache();

  KVCache(const KVCache &) = delete;
  KVCache &operator=(const KVCache &) = delete;
  KVCache(KVCache &&) noexcept;
  KVCache &operator=(KVCache &&) noexcept;

  void reset();
  void writeLayer(std::size_t layer, std::size_t position,
                  const float *keys, const float *values,
                  std::size_t tokenCount);
  void writeLayer(std::size_t layer, std::size_t position,
                  MTL::Buffer *source, std::size_t keyOffset,
                  std::size_t valueOffset, std::size_t tokenCount);
  void setLength(std::size_t length);
  std::size_t length() const;
  std::size_t capacity() const;
  MTL::Buffer *keyBuffer() const;
  MTL::Buffer *valueBuffer() const;
  std::size_t byteOffset(std::size_t layer, std::size_t head,
                         std::size_t position) const;

private:
  MTL::Buffer *keys = nullptr;
  MTL::Buffer *values = nullptr;
  std::size_t layerCount;
  std::size_t headCount;
  std::size_t cacheCapacity;
  std::size_t headSize;
  std::size_t currentLength = 0;

  friend float *scaled_dot_product_attention_cached_metal(
      MetalContext &context, KVCache &cache, std::size_t layer,
      const float *Q, std::size_t queryLength,
      std::size_t keyValueLength, std::size_t queryStartPosition,
      bool isCausal, std::size_t scaleSize);
  friend void scaled_dot_product_attention_cached_metal(
      MetalContext &context, KVCache &cache, std::size_t layer,
      MTL::Buffer *buffer, std::size_t queryOffset,
      std::size_t outputOffset, std::size_t queryLength,
      std::size_t keyValueLength, std::size_t queryStartPosition,
      bool isCausal, std::size_t scaleSize);
  friend class MetalCommandBatch;
};

}
