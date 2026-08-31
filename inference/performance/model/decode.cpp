#include <model.hpp>
#include <tokenizer.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t sampleCount = 10;
constexpr std::size_t tokensPerSample = 100;

using Clock = std::chrono::steady_clock;

enum class DecodeMode {
  SeparateCommandBuffers,
  SingleCommandBuffer,
  ParallelHeads,
};

float *decode(inference::Model &model, int token, DecodeMode mode) {
  switch (mode) {
  case DecodeMode::SeparateCommandBuffers:
    return model.decode(token);
  case DecodeMode::SingleCommandBuffer:
    return model.decodeSingleCommand(token);
  case DecodeMode::ParallelHeads:
    return model.decodeSingleCommandParallelHeads(token);
  }
  throw std::logic_error("Unknown decode mode");
}

double runSample(inference::Model &model, const std::vector<int> &prompt,
                 int token, DecodeMode mode) {
  model.resetCache();
  float *prefill = model.prefill(prompt.data(), prompt.size());
  std::free(prefill);

  auto start = Clock::now();
  for (std::size_t index = 0; index < tokensPerSample; index++) {
    float *logits = decode(model, token, mode);
    std::free(logits);
  }
  auto end = Clock::now();
  return std::chrono::duration<double, std::milli>(end - start).count();
}

struct Statistics {
  double mean;
  double median;
  double minimum;
  double maximum;
  double standardDeviation;
};

Statistics statistics(std::vector<double> values) {
  std::sort(values.begin(), values.end());
  double mean =
      std::accumulate(values.begin(), values.end(), 0.0) / values.size();
  double squaredDifference = 0.0;
  for (double value : values) {
    double difference = value - mean;
    squaredDifference += difference * difference;
  }
  return {
      .mean = mean,
      .median = (values[values.size() / 2 - 1] +
                 values[values.size() / 2]) /
                2.0,
      .minimum = values.front(),
      .maximum = values.back(),
      .standardDeviation = std::sqrt(squaredDifference / values.size()),
  };
}

void warmUp(inference::Model &model, const std::vector<int> &prompt,
            int token, DecodeMode mode) {
  model.resetCache();
  float *prefill = model.prefill(prompt.data(), prompt.size());
  std::free(prefill);
  for (std::size_t index = 0; index < 10; index++) {
    float *logits = decode(model, token, mode);
    std::free(logits);
  }
}

} // namespace

int main(int argc, char **argv) {
  const std::filesystem::path modelPath =
      std::filesystem::path(INFERENCE_SOURCE_ROOT) / "resources/model.gguf";
  inference::Model model(modelPath);
  model.load();

  inference::Tokenizer tokenizer;
  std::vector<int> prompt = tokenizer.encode("ROMEO:\n");
  int token = tokenizer.encode(" ").front();

  bool compareParallelHeads =
      argc > 1 && std::string_view(argv[1]) == "--parallel-heads";
  DecodeMode firstMode = compareParallelHeads
                             ? DecodeMode::SingleCommandBuffer
                             : DecodeMode::SeparateCommandBuffers;
  DecodeMode secondMode = compareParallelHeads
                              ? DecodeMode::ParallelHeads
                              : DecodeMode::SingleCommandBuffer;
  std::string_view firstLabel = compareParallelHeads
                                    ? "Sequential heads"
                                    : "Separate command buffers";
  std::string_view secondLabel = compareParallelHeads
                                     ? "Parallel heads"
                                     : "Single command buffer";

  warmUp(model, prompt, token, firstMode);
  warmUp(model, prompt, token, secondMode);

  std::vector<double> firstTimes;
  std::vector<double> secondTimes;
  firstTimes.reserve(sampleCount);
  secondTimes.reserve(sampleCount);

  for (std::size_t sample = 0; sample < sampleCount; sample++) {
    if (sample % 2 == 0) {
      firstTimes.push_back(runSample(model, prompt, token, firstMode));
      secondTimes.push_back(runSample(model, prompt, token, secondMode));
    } else {
      secondTimes.push_back(runSample(model, prompt, token, secondMode));
      firstTimes.push_back(runSample(model, prompt, token, firstMode));
    }
    std::cout << std::format("Completed sample {}/{}\n", sample + 1,
                             sampleCount);
  }

  Statistics first = statistics(firstTimes);
  Statistics second = statistics(secondTimes);

  std::cout << "\nSingle-token decode: 100 tokens per sample\n\n";
  std::cout << std::format("| Sample | {} | {} |\n", firstLabel,
                           secondLabel);
  std::cout << "|---:|---:|---:|\n";
  for (std::size_t sample = 0; sample < sampleCount; sample++) {
    std::cout << std::format("| {} | {:.3f} ms | {:.3f} ms |\n", sample + 1,
                             firstTimes[sample], secondTimes[sample]);
  }

  std::cout << std::format("\n| Statistic | {} | {} |\n", firstLabel,
                           secondLabel);
  std::cout << "|---|---:|---:|\n";
  std::cout << std::format("| Mean | {:.3f} ms | {:.3f} ms |\n",
                           first.mean, second.mean);
  std::cout << std::format("| Median | {:.3f} ms | {:.3f} ms |\n",
                           first.median, second.median);
  std::cout << std::format("| Minimum | {:.3f} ms | {:.3f} ms |\n",
                           first.minimum, second.minimum);
  std::cout << std::format("| Maximum | {:.3f} ms | {:.3f} ms |\n",
                           first.maximum, second.maximum);
  std::cout << std::format("| Std. dev. | {:.3f} ms | {:.3f} ms |\n",
                           first.standardDeviation,
                           second.standardDeviation);
  std::cout << std::format(
      "\nMedian speedup: {:.3f}x ({:.1f}% less time)\n",
      first.median / second.median,
      (1.0 - second.median / first.median) * 100.0);
}
