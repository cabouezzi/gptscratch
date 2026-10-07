#include <catch2/catch_test_macros.hpp>

#include <model.hpp>
#include <tokenizer.hpp>
#include <eggroll/perturbation.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

class TemporaryModelFile {

public:
  TemporaryModelFile() {
    auto identifier =
        std::chrono::steady_clock::now().time_since_epoch().count();
    this->path = std::filesystem::temp_directory_path() /
                 ("gptscratch-model-" + std::to_string(identifier) +
                  ".gguf");
  }

  ~TemporaryModelFile() {
    std::error_code error;
    std::filesystem::remove(this->path, error);
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
  std::size_t padding = (alignment - position % alignment) % alignment;
  std::vector<char> zeros(padding, 0);
  output.write(zeros.data(), static_cast<std::streamsize>(zeros.size()));
}

void writeTensorInfo(std::ofstream &output, const std::string &name,
                     std::uint64_t rows, std::uint64_t columns,
                     std::uint64_t offset) {
  writeString(output, name);
  writeValue(output, std::uint32_t{2});
  writeValue(output, columns);
  writeValue(output, rows);
  writeValue(output, std::uint32_t{0});
  writeValue(output, offset);
}

void writeEmbeddingModel(const std::filesystem::path &path) {
  std::ofstream output(path, std::ios::binary);
  output.write("GGUF", 4);
  writeValue(output, std::uint32_t{3});
  writeValue(output, std::uint64_t{2});
  writeValue(output, std::uint64_t{1});

  writeString(output, "general.alignment");
  writeValue(output, std::uint32_t{4});
  writeValue(output, std::uint32_t{32});

  writeTensorInfo(output, "token_embedding_table.weight", 3, 2, 0);
  writeTensorInfo(output, "position_embedding_table.weight", 2, 2, 32);

  padToAlignment(output, 32);
  const float tokenEmbeddings[]{1.0F, 2.0F, 3.0F,
                                4.0F, 5.0F, 6.0F};
  output.write(reinterpret_cast<const char *>(tokenEmbeddings),
               sizeof(tokenEmbeddings));
  padToAlignment(output, 32);
  const float positionEmbeddings[]{10.0F, 20.0F, 30.0F, 40.0F};
  output.write(reinterpret_cast<const char *>(positionEmbeddings),
               sizeof(positionEmbeddings));
}

} // namespace

TEST_CASE("model combines token and position embeddings", "[model][embedding]") {
  TemporaryModelFile file;
  writeEmbeddingModel(file.path);
  inference::Model model(file.path);
  const int tokens[]{2, 0};

  float *embeddings = model.embed(tokens, 2);
  REQUIRE(embeddings != nullptr);

  const float expected[]{15.0F, 26.0F, 31.0F, 42.0F};
  for (std::size_t index = 0; index < 4; index++) {
    CAPTURE(index);
    CHECK(embeddings[index] == expected[index]);
  }
  std::free(embeddings);
}

TEST_CASE("model embedding validates tokens and context length",
          "[model][embedding][validation]") {
  TemporaryModelFile file;
  writeEmbeddingModel(file.path);
  inference::Model model(file.path);

  const int invalidToken[]{3};
  CHECK_THROWS_AS(model.embed(invalidToken, 1), std::out_of_range);

  const int longSequence[]{0, 1, 2};
  CHECK_THROWS_AS(model.embed(longSequence, 3), std::invalid_argument);
}

TEST_CASE("model weights can span multiple GPU buffers", "[model][weights][sharding]") {
  TemporaryModelFile source;
  TemporaryModelFile saved;
  writeEmbeddingModel(source.path);
  inference::Model model(source.path, 24);

  model.load();
  CHECK(model.loaded());
  CHECK(model.weightShardCount() == 2);
  CHECK(model.weightOffset("token_embedding_table.weight") != model.weightOffset("position_embedding_table.weight"));
  model.save(saved.path);

  inference::ModelParameters savedParameters = inference::ModelParameters::loadGGUF(saved.path);
  const float *savedTokens = savedParameters.data("token_embedding_table.weight");
  const float *savedPositions = savedParameters.data("position_embedding_table.weight");
  const float expectedTokens[]{1.0F, 2.0F, 3.0F, 4.0F, 5.0F, 6.0F};
  const float expectedPositions[]{10.0F, 20.0F, 30.0F, 40.0F};
  for (std::size_t index = 0; index < 6; index++) {
    CHECK(savedTokens[index] == expectedTokens[index]);
  }
  for (std::size_t index = 0; index < 4; index++) {
    CHECK(savedPositions[index] == expectedPositions[index]);
  }

  model.release();
  CHECK_FALSE(model.loaded());
  CHECK(model.weightShardCount() == 0);
}

TEST_CASE("model rejects a tensor larger than its GPU shard limit", "[model][weights][sharding][validation]") {
  TemporaryModelFile file;
  writeEmbeddingModel(file.path);
  inference::Model model(file.path, 23);
  CHECK_THROWS_AS(model.load(), std::runtime_error);
}

TEST_CASE("model applies an EGGROLL perturbation by weight offset",
          "[model][eggroll][.integration]") {
  const std::filesystem::path modelPath =
      std::filesystem::path(INFERENCE_SOURCE_ROOT) / "resources/model.gguf";
  inference::ModelParameters parameters =
      inference::ModelParameters::loadGGUF(modelPath);
  const inference::GGUFTensor &weight = parameters.tensor("lm_head.weight");
  inference::eggroll::EGGROLLPerturbation perturbation =
      inference::eggroll::generatePerturbation(
          static_cast<std::size_t>(weight.shape[1]),
          static_cast<std::size_t>(weight.shape[0]), 1, 17);

  inference::Model model(modelPath);
  model.load();
  inference::Tokenizer tokenizer;
  std::vector<int> tokens = tokenizer.encode("ROMEO:\n");
  std::size_t elementCount = tokens.size() * model.vocabularySize();
  std::size_t offset = model.weightOffset("lm_head.weight");

  float *normal = model.forward(
      tokens.data(), static_cast<unsigned int>(tokens.size()));
  float *zero = model.forwardPerturbed(
      tokens.data(), tokens.size(), offset, perturbation, 0.0F);
  float *positive = model.forwardPerturbed(
      tokens.data(), tokens.size(), offset, perturbation, 0.01F);
  float *negative = model.forwardPerturbed(
      tokens.data(), tokens.size(), offset, perturbation, -0.01F);
  float *normalAfter = model.forward(
      tokens.data(), static_cast<unsigned int>(tokens.size()));

  bool changed = false;
  for (std::size_t index = 0; index < elementCount; index++) {
    CAPTURE(index);
    CHECK(std::fabs(zero[index] - normal[index]) < 0.00001F);
    CHECK(std::fabs(positive[index] + negative[index] -
                    2.0F * normal[index]) < 0.001F);
    CHECK(std::fabs(normalAfter[index] - normal[index]) < 0.00001F);
    if (std::fabs(positive[index] - normal[index]) > 0.00001F) {
      changed = true;
    }
  }
  CHECK(changed);

  std::free(normal);
  std::free(zero);
  std::free(positive);
  std::free(negative);
  std::free(normalAfter);
  model.release();
}

TEST_CASE("batched model execution matches independent sequences", "[model][batch][eggroll][.integration]") {
  const std::filesystem::path modelPath = std::filesystem::path(INFERENCE_SOURCE_ROOT) / "resources/model.gguf";
  inference::Model model(modelPath);
  model.load();
  inference::Tokenizer tokenizer;
  std::vector<int> first = tokenizer.encode("ROMEO:\n");
  std::vector<int> second = tokenizer.encode("JULIET:");
  REQUIRE(first.size() == second.size());
  std::vector<int> batch = first;
  batch.insert(batch.end(), second.begin(), second.end());

  float *firstLogits = model.forward(first.data(), static_cast<unsigned int>(first.size()));
  float *secondLogits = model.forward(second.data(), static_cast<unsigned int>(second.size()));
  float *batchLogits = model.forwardBatch(batch.data(), 2, first.size());
  std::size_t sequenceElements = first.size() * model.vocabularySize();
  for (std::size_t index = 0; index < sequenceElements; index++) {
    CAPTURE(index);
    CHECK(std::fabs(batchLogits[index] - firstLogits[index]) < 0.001F);
    CHECK(std::fabs(batchLogits[sequenceElements + index] - secondLogits[index]) < 0.001F);
  }

  std::free(firstLogits);
  std::free(secondLogits);
  std::free(batchLogits);
  model.release();
}

TEST_CASE("KV cache matches full-context model logits",
          "[model][kv-cache][.integration]") {
  const std::filesystem::path modelPath =
      std::filesystem::path(INFERENCE_SOURCE_ROOT) / "resources/model.gguf";
  inference::Model model(modelPath, 3 * 1024 * 1024);
  REQUIRE_FALSE(model.loaded());
  model.load();
  REQUIRE(model.loaded());
  REQUIRE(model.weightShardCount() > 1);
  inference::Tokenizer tokenizer;
  std::vector<int> prompt = tokenizer.encode("ROMEO:\n");

  float *fullPrompt =
      model.forward(prompt.data(), static_cast<unsigned int>(prompt.size()));
  float *cachedPrompt = model.prefill(prompt.data(), prompt.size());
  REQUIRE(model.cacheLength() == prompt.size());

  std::size_t promptElementCount = prompt.size() * model.vocabularySize();
  for (std::size_t index = 0; index < promptElementCount; index++) {
    CAPTURE(index);
    CHECK(std::fabs(fullPrompt[index] - cachedPrompt[index]) < 0.001F);
  }
  std::free(fullPrompt);
  std::free(cachedPrompt);

  int nextToken = tokenizer.encode("I")[0];
  std::vector<int> extendedPrompt = prompt;
  extendedPrompt.push_back(nextToken);
  float *fullExtended = model.forward(
      extendedPrompt.data(), static_cast<unsigned int>(extendedPrompt.size()));
  float *cachedDecode = model.decode(nextToken);
  REQUIRE(model.cacheLength() == extendedPrompt.size());

  std::size_t lastRow = extendedPrompt.size() - 1;
  for (std::size_t token = 0; token < model.vocabularySize(); token++) {
    CAPTURE(token);
    float expected =
        fullExtended[lastRow * model.vocabularySize() + token];
    CHECK(std::fabs(cachedDecode[token] - expected) < 0.001F);
  }
  std::free(cachedDecode);

  model.resetCache();
  float *singleCommandPrompt = model.prefill(prompt.data(), prompt.size());
  std::free(singleCommandPrompt);
  float *singleCommandDecode = model.decodeSingleCommand(nextToken);
  REQUIRE(model.cacheLength() == extendedPrompt.size());
  for (std::size_t token = 0; token < model.vocabularySize(); token++) {
    CAPTURE(token);
    float expected =
        fullExtended[lastRow * model.vocabularySize() + token];
    CHECK(std::fabs(singleCommandDecode[token] - expected) < 0.001F);
  }
  std::free(singleCommandDecode);

  model.resetCache();
  float *parallelHeadsPrompt = model.prefill(prompt.data(), prompt.size());
  std::free(parallelHeadsPrompt);
  float *parallelHeadsDecode =
      model.decodeSingleCommandParallelHeads(nextToken);
  REQUIRE(model.cacheLength() == extendedPrompt.size());
  for (std::size_t token = 0; token < model.vocabularySize(); token++) {
    CAPTURE(token);
    float expected =
        fullExtended[lastRow * model.vocabularySize() + token];
    CHECK(std::fabs(parallelHeadsDecode[token] - expected) < 0.001F);
  }
  std::free(parallelHeadsDecode);
  std::free(fullExtended);

  model.resetCache();
  CHECK(model.cacheLength() == 0);

  float *capacityPrompt = model.prefill(prompt.data(), prompt.size());
  std::free(capacityPrompt);
  float *freshCacheDecode = nullptr;
  std::size_t decodeCount = model.contextSize() - prompt.size() + 2;
  for (std::size_t index = 0; index < decodeCount; index++) {
    std::free(freshCacheDecode);
    freshCacheDecode = model.decodeSingleCommandParallelHeads(nextToken);
  }
  REQUIRE(model.cacheLength() == 2);
  for (std::size_t token = 0; token < model.vocabularySize(); token++) {
    CAPTURE(token);
    CHECK(std::isfinite(freshCacheDecode[token]));
  }
  std::free(freshCacheDecode);

  model.release();
  CHECK_FALSE(model.loaded());
}
