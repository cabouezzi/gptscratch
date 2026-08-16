#include <attention.hpp>
#include <matmul.hpp>
#include <vector>

namespace inference {

Head::Head(int embed_size, int head_size) {
  this->embed_size = embed_size;
  this->head_size = head_size;

  int buffer_size = embed_size * head_size;
  this->Q = std::vector<std::vector<float>>(buffer_size);
  this->K = std::vector<std::vector<float>>(buffer_size);
  this->V = std::vector<std::vector<float>>(buffer_size);
}

std::vector<std::vector<float>>
Head::forward(std::vector<std::vector<float>> X) {
  // attention
  matmul(X, Q);
}

} // namespace inference