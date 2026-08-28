#pragma once

#include <cstddef>

#include "export.hpp"
#include "matmulflags.hpp"

namespace inference {

INFERENCE_PUBLIC float *matmul(const float *A, MatMulFlag flagA,
                               const float *B, MatMulFlag flagB,
                               std::size_t M, std::size_t K, std::size_t N);
    
} // namespace inference
