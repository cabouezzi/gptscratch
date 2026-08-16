#pragma once

#include "export.hpp"

#include <vector>

INFERENCE_PUBLIC int run_metal();
INFERENCE_PUBLIC float *matmul_metal(std::vector<std::vector<float>> const X,
                                     std::vector<std::vector<float>> const Y);
