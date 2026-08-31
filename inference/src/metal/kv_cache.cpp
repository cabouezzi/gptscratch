#include <metal/kv_cache.hpp>

#include <metal/backend.hpp>

#include <Metal/Metal.hpp>

#include <cstddef>
#include <cstring>
#include <stdexcept>

namespace inference {

KVCache::KVCache(MetalContext &context, std::size_t layerCount,
                 std::size_t headCount, std::size_t capacity,
                 std::size_t headSize)
    : layerCount(layerCount), headCount(headCount), cacheCapacity(capacity),
      headSize(headSize) {
  if (layerCount == 0 || headCount == 0 || capacity == 0 || headSize == 0) {
    throw std::invalid_argument("KV cache dimensions must be positive");
  }

  std::size_t byteCount =
      layerCount * headCount * capacity * headSize * sizeof(float);
  this->keys = allocate_metal_buffer(context, byteCount);
  try {
    this->values = allocate_metal_buffer(context, byteCount);
  } catch (...) {
    release_metal_buffer(this->keys);
    this->keys = nullptr;
    throw;
  }
  std::memset(this->keys->contents(), 0, byteCount);
  std::memset(this->values->contents(), 0, byteCount);
} // namespace inference

KVCache::~KVCache() {
  release_metal_buffer(this->keys);
  release_metal_buffer(this->values);
}

KVCache::KVCache(KVCache &&other) noexcept
    : keys(other.keys), values(other.values), layerCount(other.layerCount),
      headCount(other.headCount), cacheCapacity(other.cacheCapacity),
      headSize(other.headSize), currentLength(other.currentLength) {
  other.keys = nullptr;
  other.values = nullptr;
  other.currentLength = 0;
}

KVCache &KVCache::operator=(KVCache &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  release_metal_buffer(this->keys);
  release_metal_buffer(this->values);

  this->keys = other.keys;
  this->values = other.values;
  this->layerCount = other.layerCount;
  this->headCount = other.headCount;
  this->cacheCapacity = other.cacheCapacity;
  this->headSize = other.headSize;
  this->currentLength = other.currentLength;
  other.keys = nullptr;
  other.values = nullptr;
  other.currentLength = 0;
  return *this;
}

void KVCache::reset() {
  this->currentLength = 0;
}

void KVCache::writeLayer(std::size_t layer, std::size_t position,
                         const float *keys, const float *values,
                         std::size_t tokenCount) {
  if (keys == nullptr || values == nullptr) {
    throw std::invalid_argument("KV cache inputs cannot be null");
  }
  if (layer >= this->layerCount || position > this->cacheCapacity ||
      tokenCount > this->cacheCapacity - position) {
    throw std::out_of_range("KV cache write exceeds its dimensions");
  }

  float *cachedKeys = static_cast<float *>(this->keys->contents());
  float *cachedValues = static_cast<float *>(this->values->contents());
  std::size_t layerStride =
      this->headCount * this->cacheCapacity * this->headSize;
  std::size_t cacheHeadStride = this->cacheCapacity * this->headSize;
  std::size_t inputHeadStride = tokenCount * this->headSize;

  for (std::size_t head = 0; head < this->headCount; head++) {
    std::size_t destination = layer * layerStride + head * cacheHeadStride +
                              position * this->headSize;
    std::size_t source = head * inputHeadStride;
    std::size_t byteCount = tokenCount * this->headSize * sizeof(float);
    std::memcpy(cachedKeys + destination, keys + source, byteCount);
    std::memcpy(cachedValues + destination, values + source, byteCount);
  }
}

void KVCache::writeLayer(std::size_t layer, std::size_t position,
                         MTL::Buffer *source, std::size_t keyOffset,
                         std::size_t valueOffset, std::size_t tokenCount) {
  if (source == nullptr) {
    throw std::invalid_argument("KV cache source buffer cannot be null");
  }
  std::size_t elementCount = this->headCount * tokenCount * this->headSize;
  std::size_t byteCount = elementCount * sizeof(float);
  if (keyOffset > source->length() ||
      byteCount > source->length() - keyOffset ||
      valueOffset > source->length() ||
      byteCount > source->length() - valueOffset) {
    throw std::out_of_range("KV cache source range exceeds its allocation");
  }

  std::byte *contents = static_cast<std::byte *>(source->contents());
  const float *keys =
      reinterpret_cast<const float *>(contents + keyOffset);
  const float *values =
      reinterpret_cast<const float *>(contents + valueOffset);
  this->writeLayer(layer, position, keys, values, tokenCount);
}

void KVCache::setLength(std::size_t length) {
  if (length > this->cacheCapacity) {
    throw std::out_of_range("KV cache length exceeds its capacity");
  }
  this->currentLength = length;
}

std::size_t KVCache::length() const { return this->currentLength; }

std::size_t KVCache::capacity() const { return this->cacheCapacity; }

MTL::Buffer *KVCache::keyBuffer() const { return this->keys; }

MTL::Buffer *KVCache::valueBuffer() const { return this->values; }

std::size_t KVCache::byteOffset(std::size_t layer, std::size_t head,
                                std::size_t position) const {
  if (layer >= this->layerCount || head >= this->headCount ||
      position >= this->cacheCapacity) {
    throw std::out_of_range("KV cache offset exceeds its dimensions");
  }
  std::size_t elementOffset =
      ((layer * this->headCount + head) * this->cacheCapacity + position) *
      this->headSize;
  return elementOffset * sizeof(float);
}

}
