#pragma once

#include <cstddef>

#include "export.hpp"

namespace inference {

INFERENCE_PUBLIC void scaled_dot_product_attention(
    const float* Q,       // Shape: [seq_len, head_size]
    const float* K,       // Shape: [seq_len, head_size]
    const float* V,       // Shape: [seq_len, head_size]
    float*& output,       // Shape: [seq_len, head_size]
    std::size_t seq_len,
    std::size_t head_size
);

} // namespace inference
