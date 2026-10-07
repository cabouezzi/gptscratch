#include <inference.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <random>
#include <string>
#include <vector>

int main(int argc, char **argv) {
  try {
    std::string prompt = argc > 1 ? argv[1] : "ROMEO:";
    std::optional<std::size_t> tokenCount;
    if (argc > 2) {
      tokenCount = static_cast<std::size_t>(std::stoul(argv[2]));
    }
    bool useCache = argc <= 3 || std::string(argv[3]) != "--no-cache";
    std::string modelPath = argc > 4 ? argv[4] : "resources/model.gguf";

    inference::Tokenizer tokenizer;
    inference::Model model(modelPath);
    model.load();
    std::vector<int> tokens = tokenizer.encode(prompt);
    if (tokens.empty()) {
      throw std::invalid_argument("The prompt cannot be empty");
    }

    std::random_device randomDevice;
    std::mt19937 generator(randomDevice());
    std::cout << prompt << std::flush;

    if (tokens.size() > model.contextSize()) {
      tokens.erase(tokens.begin(), tokens.end() - model.contextSize());
    }
    float *logits =
        useCache
            ? model.prefill(tokens.data(), tokens.size())
            : model.forward(tokens.data(),
                            static_cast<unsigned int>(tokens.size()));
    std::size_t logitsRows = tokens.size();

    for (std::size_t generated = 0;
         !tokenCount.has_value() || generated < *tokenCount; generated++) {
      const float *lastTokenLogits =
          logits + (logitsRows - 1) * model.vocabularySize();
      float maximum = *std::max_element(
          lastTokenLogits, lastTokenLogits + model.vocabularySize());
      std::vector<double> probabilities(model.vocabularySize());
      for (std::size_t token = 0; token < model.vocabularySize(); token++) {
        probabilities[token] =
            std::exp(static_cast<double>(lastTokenLogits[token] - maximum));
      }
      std::free(logits);

      std::discrete_distribution<int> distribution(probabilities.begin(),
                                                   probabilities.end());
      int nextToken = distribution(generator);
      std::cout << tokenizer.decode({nextToken}) << std::flush;

      if (tokenCount.has_value() && generated + 1 == *tokenCount) {
        continue;
      }
      if (!useCache) {
        tokens.push_back(nextToken);
        std::size_t contextStart =
            tokens.size() > model.contextSize()
                ? tokens.size() - model.contextSize()
                : 0;
        std::vector<int> context(tokens.begin() + contextStart, tokens.end());
        logits = model.forward(
            context.data(), static_cast<unsigned int>(context.size()));
        logitsRows = context.size();
      } else {
        logits = model.decodeSingleCommandParallelHeads(nextToken);
        logitsRows = 1;
      }
    }

    std::cout << '\n';
    model.release();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Generation failed: " << error.what() << '\n';
    return 1;
  }
}
