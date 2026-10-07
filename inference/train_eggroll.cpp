#include <inference.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

std::string makeUUID() {
  std::random_device randomDevice;
  std::mt19937_64 generator(randomDevice());
  std::uniform_int_distribution<unsigned int> byteDistribution(0, 255);
  unsigned char bytes[16];
  for (unsigned char &byte : bytes) {
    byte = static_cast<unsigned char>(byteDistribution(generator));
  }
  bytes[6] = static_cast<unsigned char>((bytes[6] & 0x0F) | 0x40);
  bytes[8] = static_cast<unsigned char>((bytes[8] & 0x3F) | 0x80);
  std::ostringstream output;
  output << std::hex << std::setfill('0');
  for (std::size_t index = 0; index < 16; index++) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      output << '-';
    }
    output << std::setw(2) << static_cast<unsigned int>(bytes[index]);
  }
  return output.str();
}

int main() {
  try {
    const std::string textPath = "../training/input.txt";
    const std::size_t stepCount = 8000;
    const std::size_t batchSize = 64;
    const std::size_t populationSize = 4;
    const std::size_t rank = 1;
    const float sigma = 0.0001F;
    const float learningRate = 0.0001F;
    const std::uint64_t seed = 42;

    std::ifstream input(textPath);
    if (!input) {
      throw std::runtime_error("Could not open training text: " + textPath);
    }
    std::string text{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};

    inference::Tokenizer tokenizer;
    std::vector<int> data = tokenizer.encode(text);
    inference::Model model("resources/random_model.gguf");
    model.load();

    if (data.size() < 2) {
      throw std::invalid_argument("Training text must contain at least two tokens");
    }
    std::size_t trainingSize = data.size() * 9 / 10;
    std::size_t sequenceLength = std::min(model.contextSize(), trainingSize - 1);
    std::size_t maximumStart = trainingSize - sequenceLength - 1;
    std::mt19937_64 generator(seed);
    std::uniform_int_distribution<std::size_t> startDistribution(0, maximumStart);

    inference::eggroll::EGGROLLTrainer trainer(model, populationSize, rank, sigma, learningRate, seed);
    std::vector<int> tokens(batchSize * sequenceLength);
    std::vector<int> targets(batchSize * sequenceLength);

    auto trainingStart = std::chrono::steady_clock::now();
    for (std::size_t step = 0; step < stepCount; step++) {
      for (std::size_t batch = 0; batch < batchSize; batch++) {
        std::size_t start = startDistribution(generator);
        for (std::size_t position = 0; position < sequenceLength; position++) {
          tokens[batch * sequenceLength + position] = data[start + position];
          targets[batch * sequenceLength + position] = data[start + position + 1];
        }
      }

      inference::eggroll::EGGROLLStepResult result = trainer.trainBatch(tokens.data(), targets.data(), batchSize, sequenceLength);
      std::cout << "Step " << step << " | fitness " << result.baseFitness << " | loss " << -result.baseFitness << '\n';
    }
    auto trainingEnd = std::chrono::steady_clock::now();
    double trainingSeconds = std::chrono::duration<double>(trainingEnd - trainingStart).count();
    std::string outputPath = "resources/" + makeUUID() + ".gguf";
    model.save(outputPath);
    std::cout << "EGGROLL training: " << trainingSeconds << " seconds\n";
    std::cout << "Saved trained model to " << outputPath << '\n';

    model.release();
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "Training failed: " << error.what() << '\n';
    return 1;
  }
}
