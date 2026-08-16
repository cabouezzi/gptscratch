#pragma once

#include "export.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace inference {

class INFERENCE_PUBLIC Tokenizer {

public:
  Tokenizer();
  void load(std::string path);
  std::vector<int> encode(std::string input);
  std::string decode(const std::vector<int>& tokens);

private:
  std::unordered_map<unsigned char, int> char_to_id;
  std::vector<unsigned char> id_to_char;

};

} // namespace inference
