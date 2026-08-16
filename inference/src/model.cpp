#include <model.hpp>
#include <stdexcept>

namespace inference {

// === Model ===

Model::Model(int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {}

std::vector<float> Model::forward(std::vector<int> X) { return {}; }

float *Model::forward(float *X) { return X; }

// === Head ===

Head::Head(int size, int input_size, int output_size) {
  (void)size;
  this->input_size = input_size;
  this->output_size = output_size;
}

float *Head::forward(float *X) { return X; }

// === MultiHead ===

MultiHead::MultiHead(int num_heads, int size, int input_size, int output_size) {
  if (num_heads <= 0 || size <= 0 || input_size <= 0 || output_size <= 0) {
    throw std::invalid_argument("MultiHead dimensions must be positive");
  }
  if (output_size != num_heads * size) {
    throw std::invalid_argument(
        "MultiHead output_size must equal num_heads * size");
  }

  this->num_heads = num_heads;
  this->size = size;
  this->input_size = input_size;
  this->output_size = output_size;
  this->output.resize(output_size);

  this->heads.reserve(num_heads);
  for (int i = 0; i < num_heads; i++) {
    this->heads.emplace_back(size, input_size, size);
  }
}

float *MultiHead::forward(float *X) {
  if (X == nullptr) {
    throw std::invalid_argument("MultiHead input cannot be null");
  }

  for (int i = 0; i < this->num_heads; i++) {
    float *head_output = this->heads[i].forward(X);
    std::copy_n(head_output, this->size, this->output.begin() + i * this->size);
  }

  return this->output.data();
}

// === FeedForward ===

FeedForward::FeedForward(int num_embed, int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {}

float *FeedForward::forward(float *X) { return X; }

// === Block ===

Block::Block(int n_embed, int n_head, int input_size, int output_size)
    : input_size(input_size), output_size(output_size) {}

float *Block::forward(float *X) { return X; }

} // namespace inference
