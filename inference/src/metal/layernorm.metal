#include <metal_stdlib>
using namespace metal;

constant uint LAYERNORM_THREAD_COUNT = 32;

kernel void layer_norm(device const float *input [[buffer(0)]],
                       device const float *gamma [[buffer(1)]],
                       device const float *beta [[buffer(2)]],
                       device float *output [[buffer(3)]],
                       constant uint &embeddingSize [[buffer(4)]],
                       constant float &epsilon [[buffer(5)]],
                       uint tokenId [[threadgroup_position_in_grid]],
                       uint laneId [[thread_index_in_simdgroup]])
{
  uint rowOffset = tokenId * embeddingSize;

  float partialSum = 0.0f;
  for (uint channel = laneId; channel < embeddingSize; channel += LAYERNORM_THREAD_COUNT) {
    partialSum += input[rowOffset + channel];
  }

  float mean = simd_sum(partialSum) / float(embeddingSize);

  float partialVariance = 0.0f;
  for (uint channel = laneId; channel < embeddingSize; channel += LAYERNORM_THREAD_COUNT) {
    // def of variance not E|x^2| - E|x|^2
    float centered = input[rowOffset + channel] - mean;
    partialVariance += centered * centered;
  }

  float variance = simd_sum(partialVariance) / float(embeddingSize);
  // 1 / sqrt(sigma^2 + epsilon)
  // epsilon for stability
  float inverseStandardDeviation = rsqrt(variance + epsilon);

  for (uint channel = laneId; channel < embeddingSize; channel += LAYERNORM_THREAD_COUNT) {
    uint index = rowOffset + channel;
    float normalized = (input[index] - mean) * inverseStandardDeviation;
    output[index] = normalized * gamma[channel] + beta[channel];
  }
}
