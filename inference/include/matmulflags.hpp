#pragma once

namespace inference
{

enum class MatMulFlag {
    NO_TRANSPOSE = 0,
    TRANSPOSE = 1 << 0
};

} // namespace inference
