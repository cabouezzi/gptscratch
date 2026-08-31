#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

#include <model_loader.hpp>

namespace {

class TemporaryFile {

public:
  explicit TemporaryFile(const std::string &suffix) {
    auto identifier =
        std::chrono::steady_clock::now().time_since_epoch().count();
    path = std::filesystem::temp_directory_path() /
           ("gptscratch-" + std::to_string(identifier) + suffix);
  }

  ~TemporaryFile() {
    std::error_code error;
    std::filesystem::remove(path, error);
  }

  std::filesystem::path path;
};

template <typename T> void writeValue(std::ofstream &output, T value) {
  output.write(reinterpret_cast<const char *>(&value), sizeof(value));
}

void writeString(std::ofstream &output, const std::string &value) {
  writeValue(output, static_cast<std::uint64_t>(value.size()));
  output.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void padToAlignment(std::ofstream &output, std::size_t alignment) {
  std::size_t position = static_cast<std::size_t>(output.tellp());
  std::size_t padding =
      (alignment - position % alignment) % alignment;
  const std::vector<char> zeros(padding, 0);
  output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

void writeFixture(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary);
  output.write("GGUF", 4);
  writeValue(output, std::uint32_t{3});
  writeValue(output, std::uint64_t{1});
  writeValue(output, std::uint64_t{2});

  writeString(output, "general.architecture");
  writeValue(output, std::uint32_t{8});
  writeString(output, "gptscratch");

  writeString(output, "general.alignment");
  writeValue(output, std::uint32_t{4});
  writeValue(output, std::uint32_t{32});

  writeString(output, "weight");
  writeValue(output, std::uint32_t{2});
  writeValue(output, std::uint64_t{3});
  writeValue(output, std::uint64_t{2});
  writeValue(output, std::uint32_t{0});
  writeValue(output, std::uint64_t{0});

  padToAlignment(output, 32);
  const float values[]{1.0F, -2.0F, 3.5F, 4.0F, 0.25F, -6.0F};
  output.write(reinterpret_cast<const char *>(values), sizeof(values));
}

} // namespace

TEST_CASE("GGUF loader reads F32 tensors and metadata", "[gguf][loader]") {
  TemporaryFile file(".gguf");
  writeFixture(file.path);

  inference::ModelParameters parameters =
      inference::ModelParameters::loadGGUF(file.path);

  REQUIRE(parameters.contains("weight"));
  const inference::GGUFTensor &tensor = parameters.tensor("weight");
  CHECK(tensor.shape == std::vector<std::uint64_t>{2, 3});
  CHECK(tensor.elementCount == 6);

  const float expected[]{1.0F, -2.0F, 3.5F, 4.0F, 0.25F, -6.0F};
  const float *values = parameters.data("weight");
  for (std::size_t index = 0; index < tensor.elementCount; index++) {
    CAPTURE(index);
    CHECK(values[index] == expected[index]);
  }

  CHECK(std::get<std::string>(parameters.metadata("general.architecture")) ==
        "gptscratch");
  CHECK(std::get<std::uint64_t>(parameters.metadata("general.alignment")) ==
        32);
  CHECK_THROWS_AS(parameters.tensor("missing"), std::out_of_range);
}

TEST_CASE("GGUF loader rejects non-GGUF files", "[gguf][loader][validation]") {
  TemporaryFile file(".bin");
  std::ofstream output(file.path, std::ios::binary);
  output.write("NOPE", 4);
  output.close();

  CHECK_THROWS_AS(inference::ModelParameters::loadGGUF(file.path),
                  std::runtime_error);
}
