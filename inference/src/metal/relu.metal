#include <metal_stdlib>
using namespace metal;

kernel void relu(device const float *input [[buffer(0)]],
                 device float *output [[buffer(1)]],
                 constant uint &elementCount [[buffer(2)]],
                 uint id [[thread_position_in_grid]]) {
  if (id < elementCount) {
    output[id] = max(input[id], 0.0f);
  }
}
