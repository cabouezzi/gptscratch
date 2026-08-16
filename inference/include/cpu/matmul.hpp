#pragma once

#include <vector>

namespace inference {

float* matmul(std::vector<std::vector<float>> const X, std::vector<std::vector<float>> const Y);
    
} // namespace inference
