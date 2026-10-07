#pragma once

#include "export.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
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
  void saveGGUF(const std::filesystem::path &path, const std::unordered_map<std::string, std::vector<float>> &tensorData) const;
  void saveGGUF(const std::filesystem::path &path, const std::function<std::vector<float>(const std::string &, const GGUFTensor &)> &tensorReader) const;

private:
  struct MappedFileData;
  std::shared_ptr<MappedFileData> fileData;
  std::unordered_map<std::string, GGUFTensor> tensors;
  std::unordered_map<std::string, GGUFMetadataValue> metadataValues;
};

} // namespace inference
