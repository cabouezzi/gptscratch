#include <fstream>
#include <stdexcept>
#include <tokenizer.hpp>
#include <utility>

namespace inference {

Tokenizer::Tokenizer() = default;

std::vector<int> Tokenizer::encode(std::string input) {
  std::vector<int> tokens;
  tokens.reserve(input.size());

  for (unsigned char character : input) {
    if (this->char_to_id.empty()) {
      tokens.push_back(static_cast<int>(character));
      continue;
    }

    auto token = this->char_to_id.find(character);
    if (token == this->char_to_id.end()) {
      throw std::runtime_error("Character is not in the tokenizer vocabulary");
    }
    tokens.push_back(token->second);
  }

  return tokens;
}

std::string Tokenizer::decode(const std::vector<int>& tokens) {
  std::string output;
  output.reserve(tokens.size());

  for (int token : tokens) {
    if (this->id_to_char.empty()) {
      if (token < 0 || token > 255) {
        throw std::runtime_error("Token is outside the byte range");
      }
      output.push_back(static_cast<char>(token));
      continue;
    }

    if (token < 0 || static_cast<std::size_t>(token) >= this->id_to_char.size()) {
      throw std::runtime_error("Token is not in the tokenizer vocabulary");
    }
    output.push_back(static_cast<char>(this->id_to_char[token]));
  }

  return output;
}

void Tokenizer::load(std::string path) {
  std::ifstream file(path);
  if (!file) {
    throw std::runtime_error("Could not open tokenizer file");
  }

  std::unordered_map<unsigned char, int> loaded_char_to_id;
  std::vector<unsigned char> loaded_id_to_char;
  std::string line;
  int id = 0;
  while (std::getline(file, line)) {
    if (line.size() != 1) {
      throw std::runtime_error("Expected exactly one character per line");
    }
    unsigned char character = static_cast<unsigned char>(line[0]);
    loaded_char_to_id[character] = id;
    loaded_id_to_char.push_back(character);
    id++;
  }

  this->char_to_id = std::move(loaded_char_to_id);
  this->id_to_char = std::move(loaded_id_to_char);
}

} // namespace inference
