#pragma once

#include "export.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace inference {

struct GGUFTensor {
  std::vector<std::uint64_t> shape;
  std::uint32_t type;
  std::uint64_t elementCount;
  std::size_t dataOffset;
};

using GGUFMetadataValue =
    std::variant<std::uint64_t, std::int64_t, double, bool, std::string>;

class INFERENCE_PUBLIC ModelParameters {

public:
  static ModelParameters loadGGUF(const std::filesystem::path &path);

  bool contains(const std::string &name) const;
  const GGUFTensor &tensor(const std::string &name) const;
  const float *data(const std::string &name) const;
  const GGUFMetadataValue &metadata(const std::string &key) const;
  const std::unordered_map<std::string, GGUFTensor> &allTensors() const;

private:
  std::vector<std::uint8_t> fileData;
  std::unordered_map<std::string, GGUFTensor> tensors;
  std::unordered_map<std::string, GGUFMetadataValue> metadataValues;
};

} // namespace inference
