#include <metal_stdlib>
using namespace metal;

kernel void add_bias(device const float *input [[buffer(0)]],
                     device const float *bias [[buffer(1)]],
                     device float *output [[buffer(2)]],
                     constant uint &elementCount [[buffer(3)]],
                     constant uint &outputWidth [[buffer(4)]],
                     uint id [[thread_position_in_grid]]) {
  if (id < elementCount) {
    uint column = id % outputWidth;
    output[id] = input[id] + bias[column];
  }
}
