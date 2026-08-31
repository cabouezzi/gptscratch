#include <model_loader.hpp>

#include <bit>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <type_traits>

namespace inference {
namespace {

constexpr std::uint32_t GGUF_VERSION = 3;
constexpr std::uint32_t GGML_TYPE_F32 = 0;
constexpr std::uint32_t DEFAULT_ALIGNMENT = 32;

enum class MetadataType : std::uint32_t {
  UInt8 = 0,
  Int8 = 1,
  UInt16 = 2,
  Int16 = 3,
  UInt32 = 4,
  Int32 = 5,
  Float32 = 6,
  Bool = 7,
  String = 8,
  Array = 9,
  UInt64 = 10,
  Int64 = 11,
  Float64 = 12,
};

class Reader {

public:
  explicit Reader(const std::vector<std::uint8_t> &bytes) : bytes(bytes) {}

  template <typename T> T read() {
    static_assert(std::is_trivially_copyable_v<T>);
    require(sizeof(T));

    T value;
    std::memcpy(&value, bytes.data() + position, sizeof(T));
    position += sizeof(T);

    if constexpr (sizeof(T) > 1) {
      if constexpr (std::endian::native != std::endian::little) {
        throw std::runtime_error("Big-endian hosts are not supported yet");
      }
    }
    return value;
  }

  std::string readString() {
    std::uint64_t length = read<std::uint64_t>();
    if (length > std::numeric_limits<std::size_t>::max()) {
      throw std::runtime_error("GGUF string is too large");
    }
    require(static_cast<std::size_t>(length));

    const char *start =
        reinterpret_cast<const char *>(bytes.data() + position);
    std::string value(start, static_cast<std::size_t>(length));
    position += static_cast<std::size_t>(length);
    return value;
  }

  std::size_t offset() const { return position; }

private:
  void require(std::size_t count) const {
    if (count > bytes.size() - position) {
      throw std::runtime_error("Unexpected end of GGUF file");
    }
  }

  const std::vector<std::uint8_t> &bytes;
  std::size_t position = 0;
};

std::size_t align(std::size_t value, std::size_t alignment) {
  if (alignment == 0) {
    throw std::runtime_error("GGUF alignment cannot be zero");
  }
  std::size_t remainder = value % alignment;
  return remainder == 0 ? value : value + alignment - remainder;
} // namespace

GGUFMetadataValue readMetadataValue(Reader &reader, MetadataType type,
                                    bool retainValue) {
  switch (type) {
  case MetadataType::UInt8: {
    std::uint64_t value = reader.read<std::uint8_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0ULL);
  }
  case MetadataType::Int8: {
    std::int64_t value = reader.read<std::int8_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0LL);
  }
  case MetadataType::UInt16: {
    std::uint64_t value = reader.read<std::uint16_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0ULL);
  }
  case MetadataType::Int16: {
    std::int64_t value = reader.read<std::int16_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0LL);
  }
  case MetadataType::UInt32: {
    std::uint64_t value = reader.read<std::uint32_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0ULL);
  }
  case MetadataType::Int32: {
    std::int64_t value = reader.read<std::int32_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0LL);
  }
  case MetadataType::Float32: {
    double value = reader.read<float>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0.0);
  }
  case MetadataType::Bool: {
    std::uint8_t raw = reader.read<std::uint8_t>();
    if (raw > 1) {
      throw std::runtime_error("Invalid GGUF boolean value");
    }
    return retainValue ? GGUFMetadataValue(raw != 0)
                       : GGUFMetadataValue(false);
  }
  case MetadataType::String: {
    std::string value = reader.readString();
    return retainValue ? GGUFMetadataValue(std::move(value))
                       : GGUFMetadataValue(std::string());
  }
  case MetadataType::Array: {
    MetadataType elementType =
        static_cast<MetadataType>(reader.read<std::uint32_t>());
    std::uint64_t count = reader.read<std::uint64_t>();
    for (std::uint64_t index = 0; index < count; index++) {
      readMetadataValue(reader, elementType, false);
    }
    return std::string("<array>");
  }
  case MetadataType::UInt64: {
    std::uint64_t value = reader.read<std::uint64_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0ULL);
  }
  case MetadataType::Int64: {
    std::int64_t value = reader.read<std::int64_t>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0LL);
  }
  case MetadataType::Float64: {
    double value = reader.read<double>();
    return retainValue ? GGUFMetadataValue(value) : GGUFMetadataValue(0.0);
  }
  }

  throw std::runtime_error("Unsupported GGUF metadata type");
} // namespace inference

std::vector<std::uint8_t> readFile(const std::filesystem::path &path) {
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("Could not open GGUF model: " + path.string());
  }

  std::streamsize size = input.tellg();
  if (size < 0) {
    throw std::runtime_error("Could not determine GGUF model size");
  }

  std::vector<std::uint8_t> bytes(static_cast<std::size_t>(size));
  input.seekg(0);
  if (!bytes.empty() &&
      !input.read(reinterpret_cast<char *>(bytes.data()), size)) {
    throw std::runtime_error("Could not read GGUF model: " + path.string());
  }
  return bytes;
}

}

ModelParameters ModelParameters::loadGGUF(const std::filesystem::path &path) {
  ModelParameters parameters;
  parameters.fileData = readFile(path);
  Reader reader(parameters.fileData);

  const char expectedMagic[]{'G', 'G', 'U', 'F'};
  for (char expected : expectedMagic) {
    if (reader.read<std::uint8_t>() != static_cast<std::uint8_t>(expected)) {
      throw std::runtime_error("File is not a GGUF model");
    }
  }

  std::uint32_t version = reader.read<std::uint32_t>();
  if (version != GGUF_VERSION) {
    throw std::runtime_error("Only GGUF version 3 is supported");
  }

  std::uint64_t tensorCount = reader.read<std::uint64_t>();
  std::uint64_t metadataCount = reader.read<std::uint64_t>();

  for (std::uint64_t index = 0; index < metadataCount; index++) {
    std::string key = reader.readString();
    MetadataType type =
        static_cast<MetadataType>(reader.read<std::uint32_t>());
    GGUFMetadataValue value = readMetadataValue(reader, type, true);
    if (!parameters.metadataValues.emplace(std::move(key), std::move(value))
             .second) {
      throw std::runtime_error("Duplicate GGUF metadata key");
    }
  }

  std::uint64_t alignment = DEFAULT_ALIGNMENT;
  auto alignmentEntry = parameters.metadataValues.find("general.alignment");
  if (alignmentEntry != parameters.metadataValues.end()) {
    if (!std::holds_alternative<std::uint64_t>(alignmentEntry->second)) {
      throw std::runtime_error("general.alignment must be unsigned");
    }
    alignment = std::get<std::uint64_t>(alignmentEntry->second);
  }

  struct PendingTensor {
    std::string name;
    std::vector<std::uint64_t> shape;
    std::uint64_t elementCount;
    std::uint64_t relativeOffset;
  };
  std::vector<PendingTensor> pendingTensors;
  pendingTensors.reserve(static_cast<std::size_t>(tensorCount));

  for (std::uint64_t index = 0; index < tensorCount; index++) {
    std::string name = reader.readString();
    if (name.size() > 64) {
      throw std::runtime_error("GGUF tensor name exceeds 64 bytes");
    }

    std::uint32_t dimensionCount = reader.read<std::uint32_t>();
    if (dimensionCount == 0 || dimensionCount > 4) {
      throw std::runtime_error("Unsupported GGUF tensor rank");
    }

    std::vector<std::uint64_t> ggufDimensions(dimensionCount);
    std::uint64_t elementCount = 1;
    for (std::uint32_t dimension = 0; dimension < dimensionCount;
         dimension++) {
      std::uint64_t value = reader.read<std::uint64_t>();
      if (value == 0 ||
          elementCount > std::numeric_limits<std::uint64_t>::max() / value) {
        throw std::runtime_error("Invalid GGUF tensor dimensions");
      }
      ggufDimensions[dimension] = value;
      elementCount *= value;
    }

    std::uint32_t type = reader.read<std::uint32_t>();
    if (type != GGML_TYPE_F32) {
      throw std::runtime_error("Only F32 GGUF tensors are supported for now");
    }

    std::uint64_t relativeOffset = reader.read<std::uint64_t>();
    if (relativeOffset % alignment != 0) {
      throw std::runtime_error("GGUF tensor offset is not aligned");
    }

    std::vector<std::uint64_t> shape(ggufDimensions.rbegin(),
                                     ggufDimensions.rend());
    pendingTensors.push_back(
        {std::move(name), std::move(shape), elementCount, relativeOffset});
  }

  if (alignment > std::numeric_limits<std::size_t>::max()) {
    throw std::runtime_error("GGUF alignment is too large");
  }
  std::size_t tensorDataOffset =
      align(reader.offset(), static_cast<std::size_t>(alignment));

  for (PendingTensor &pending : pendingTensors) {
    if (pending.relativeOffset >
        std::numeric_limits<std::size_t>::max() - tensorDataOffset) {
      throw std::runtime_error("GGUF tensor offset is too large");
    }
    std::size_t dataOffset =
        tensorDataOffset + static_cast<std::size_t>(pending.relativeOffset);

    if (pending.elementCount >
        std::numeric_limits<std::size_t>::max() / sizeof(float)) {
      throw std::runtime_error("GGUF tensor is too large");
    }
    std::size_t byteCount =
        static_cast<std::size_t>(pending.elementCount) * sizeof(float);
    if (dataOffset > parameters.fileData.size() ||
        byteCount > parameters.fileData.size() - dataOffset) {
      throw std::runtime_error("GGUF tensor data is outside the file");
    }

    GGUFTensor tensor = {
        .shape = std::move(pending.shape),
        .type = GGML_TYPE_F32,
        .elementCount = pending.elementCount,
        .dataOffset = dataOffset,
    };
    if (!parameters.tensors.emplace(std::move(pending.name), std::move(tensor))
             .second) {
      throw std::runtime_error("Duplicate GGUF tensor name");
    }
  }

  return parameters;
}

bool ModelParameters::contains(const std::string &name) const {
  return tensors.contains(name);
}

const GGUFTensor &ModelParameters::tensor(const std::string &name) const {
  auto entry = tensors.find(name);
  if (entry == tensors.end()) {
    throw std::out_of_range("GGUF tensor not found: " + name);
  }
  return entry->second;
}

const float *ModelParameters::data(const std::string &name) const {
  const GGUFTensor &tensorInfo = tensor(name);
  return reinterpret_cast<const float *>(fileData.data() +
                                         tensorInfo.dataOffset);
}

const GGUFMetadataValue &
ModelParameters::metadata(const std::string &key) const {
  auto entry = metadataValues.find(key);
  if (entry == metadataValues.end()) {
    throw std::out_of_range("GGUF metadata not found: " + key);
  }
  return entry->second;
}

const std::unordered_map<std::string, GGUFTensor> &
ModelParameters::allTensors() const {
  return this->tensors;
}

}
