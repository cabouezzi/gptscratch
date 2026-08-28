#pragma once

#include "export.hpp"

#include "matmulflags.hpp"
#include <cstddef>
#include <vector>

namespace inference {

INFERENCE_PUBLIC int run_metal();
INFERENCE_PUBLIC float *matmul_metal(std::vector<std::vector<float>> const X,
                                     MatMulFlag flagX,
                                     std::vector<std::vector<float>> const Y,
                                     MatMulFlag flagY, bool tile);
INFERENCE_PUBLIC float *scaled_dot_product_attention_metal(
    const float *Q, const float *K, const float *V, std::size_t seq_len,
    std::size_t head_size, bool isCausal);

} // namespace inference
