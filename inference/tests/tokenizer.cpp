#include <catch2/catch_test_macros.hpp>

#include <tokenizer.hpp>

#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr char TRAINING_VOCABULARY[] =
    "\n !$&',-.3:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";

} // namespace

TEST_CASE("tokenizer matches the training vocabulary", "[tokenizer]") {
  inference::Tokenizer tokenizer;
  const std::string vocabulary = TRAINING_VOCABULARY;

  std::vector<int> expected(vocabulary.size());
  std::iota(expected.begin(), expected.end(), 0);

  CHECK(tokenizer.encode(vocabulary) == expected);
  CHECK(tokenizer.decode(expected) == vocabulary);
}

TEST_CASE("tokenizer uses the checkpoint token IDs", "[tokenizer]") {
  inference::Tokenizer tokenizer;
  const std::vector<int> expected{44, 59, 41, 49, 1, 63, 53, 59};

  CHECK(tokenizer.encode("fuck you") == expected);
  CHECK(tokenizer.decode(expected) == "fuck you");
  CHECK(tokenizer.encode("\n") == std::vector<int>{0});
}

TEST_CASE("tokenizer rejects values outside the training vocabulary",
          "[tokenizer][validation]") {
  inference::Tokenizer tokenizer;

  CHECK_THROWS_AS(tokenizer.encode("@"), std::runtime_error);
  CHECK_THROWS_AS(tokenizer.decode({-1}), std::runtime_error);
  CHECK_THROWS_AS(tokenizer.decode({65}), std::runtime_error);
}
