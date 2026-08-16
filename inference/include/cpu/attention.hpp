#pragma once

#include <vector>
#include "model.hpp"

namespace inference {

class Head {

public:
  Head(int embed_size, int head_size);
  std::vector<std::vector<float>> forward(std::vector<std::vector<float>> X);

private:
  int embed_size;
  int head_size;

  std::vector<std::vector<float>> Q;  // (embed size, head size)
  std::vector<std::vector<float>> K;  // (embed size, head size)
  std::vector<std::vector<float>> V;  // (embed size, head size)

};

}