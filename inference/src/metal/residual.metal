#include <metal_stdlib>
using namespace metal;

kernel void residual_add(device const float *residual [[buffer(0)]],
                         device const float *input [[buffer(1)]],
                         device float *output [[buffer(2)]],
                         constant uint &elementCount [[buffer(3)]],
                         uint id [[thread_position_in_grid]]) {
  if (id < elementCount) {
    output[id] = residual[id] + input[id];
  }
}
