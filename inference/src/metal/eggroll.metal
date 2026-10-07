#include <metal_stdlib>
using namespace metal;

struct EGGROLLDims {
  uint T;
  uint N;
  uint r;
  float epsilon;
};

struct EGGROLLUpdateDims {
  uint M;
  uint N;
  uint r;
  uint populationSize;
  float scale;
};

kernel void eggroll_output_add(
    device float *output [[buffer(0)]],
    device const float *XB [[buffer(1)]],
    device const float *A [[buffer(2)]],
    constant EGGROLLDims &dims [[buffer(3)]],
    uint2 id [[thread_position_in_grid]]) {
  uint column = id.x;
  uint row = id.y;

  if (row >= dims.T || column >= dims.N) {
    return;
  }

  float perturbation = 0.0F;
  for (uint rank = 0; rank < dims.r; rank++) {
    perturbation +=
        XB[row * dims.r + rank] * A[column * dims.r + rank];
  }

  float scale = dims.epsilon * rsqrt(float(dims.r));
  output[row * dims.N + column] += scale * perturbation;
}

kernel void eggroll_cross_entropy(
    device const float *logits [[buffer(0)]],
    device const float *targets [[buffer(1)]],
    device float *losses [[buffer(2)]],
    constant uint &vocabularySize [[buffer(3)]],
    uint row [[threadgroup_position_in_grid]],
    uint laneId [[thread_index_in_simdgroup]]) {
  device const float *rowLogits = logits + row * vocabularySize;
  float localMaximum = -INFINITY;
  for (uint token = laneId; token < vocabularySize; token += 32) {
    localMaximum = max(localMaximum, rowLogits[token]);
  }
  float maximum = simd_max(localMaximum);

  float localSum = 0.0F;
  for (uint token = laneId; token < vocabularySize; token += 32) {
    localSum += exp(rowLogits[token] - maximum);
  }
  float exponentialSum = simd_sum(localSum);

  if (laneId == 0) {
    uint target = uint(targets[row]);
    losses[row] = log(exponentialSum) + maximum - rowLogits[target];
  }
}

kernel void eggroll_fitness_reduce(
    device const float *losses [[buffer(0)]],
    device float *fitness [[buffer(1)]],
    constant uint &sequenceLength [[buffer(2)]],
    uint laneId [[thread_index_in_simdgroup]]) {
  float localLoss = 0.0F;
  for (uint row = laneId; row < sequenceLength; row += 32) {
    localLoss += losses[row];
  }
  float totalLoss = simd_sum(localLoss);
  if (laneId == 0) {
    fitness[0] = -totalLoss / float(sequenceLength);
  }
}

kernel void eggroll_weight_update(
    device float *weights [[buffer(0)]],
    device const float *A [[buffer(1)]],
    device const float *B [[buffer(2)]],
    device const float *fitnesses [[buffer(3)]],
    constant EGGROLLUpdateDims &dims [[buffer(4)]],
    uint2 id [[thread_position_in_grid]]) {
  uint input = id.x;
  uint output = id.y;
  if (input >= dims.M || output >= dims.N) {
    return;
  }

  float update = 0.0F;
  for (uint candidate = 0; candidate < dims.populationSize; candidate++) {
    uint aBase = candidate * dims.N * dims.r + output * dims.r;
    uint bBase = candidate * dims.M * dims.r + input * dims.r;
    float direction = 0.0F;
    for (uint rank = 0; rank < dims.r; rank++) {
      direction += A[aBase + rank] * B[bBase + rank];
    }
    update += fitnesses[candidate] * direction;
  }

  weights[output * dims.M + input] +=
      dims.scale * rsqrt(float(dims.r)) * update;
}
