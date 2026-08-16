#pragma once

#include "export.hpp"

#include <vector>

namespace inference {

class INFERENCE_PUBLIC Model {

public:
  Model(int input_size, int output_size);
  std::vector<float> forward(std::vector<int> X);
  float* forward(float* X);

private:
  int input_size;
  int output_size;

};

class Head {

public:
  Head(int size, int input_size, int output_size);
  float* forward(float* X);

private:
  int input_size;
  int output_size;

};

class MultiHead {

public:
  MultiHead(int num_heads, int size, int input_size, int output_size);
  float* forward(float* X);

private:
  int num_heads;
  int size;
  int input_size;
  int output_size;
  std::vector<Head> heads;
  std::vector<float> output;

};

class FeedForward {

public:
  FeedForward(int num_embed, int input_size, int output_size);
  float* forward(float* X);

private:
  int input_size;
  int output_size;

};

class Block {

public:
  Block(int n_embed, int n_head, int input_size, int output_size);
  float* forward(float* X);

private:
  int input_size;
  int output_size;

};

} // namespace inference
