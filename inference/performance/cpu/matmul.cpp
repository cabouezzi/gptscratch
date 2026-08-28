#include <metal/backend.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace {

using Matrix = std::vector<std::vector<float>>;

struct BenchmarkCase {
  std::string name;
  Matrix X;
  Matrix Y;
  std::size_t iterations;
  std::size_t samples;
};

struct BenchmarkSpec {
  std::string name;
  std::size_t M;
  std::size_t K;
  std::size_t N;
  int x_seed;
  int y_seed;
  std::size_t iterations;
  std::size_t samples;
};

volatile float output_guard = 0.0F;

Matrix make_matrix(std::size_t rows, std::size_t columns, int seed) {
  Matrix matrix(rows, std::vector<float>(columns));
  for (std::size_t row = 0; row < rows; row++) {
    for (std::size_t column = 0; column < columns; column++) {
      int value = static_cast<int>((row * 17 + column * 31 + seed) % 19) - 9;
      matrix[row][column] = static_cast<float>(value) / 8.0F;
    }
  }
  return matrix;
}

void run_once(const BenchmarkCase& benchmark, bool tile,
              const std::string& implementation) {
  float* output = inference::matmul_metal(
      benchmark.X, inference::MatMulFlag::NO_TRANSPOSE, benchmark.Y,
      inference::MatMulFlag::NO_TRANSPOSE, tile);
  if (output == nullptr) {
    std::cerr << implementation << " matmul returned nullptr for "
              << benchmark.name << '\n';
    std::exit(1);
  }

  std::size_t output_size = benchmark.X.size() * benchmark.Y[0].size();
  output_guard = output_guard + output[output_size / 2];
  std::free(output);
}

double median_milliseconds(const BenchmarkCase& benchmark,
                           bool tile,
                           const std::string& implementation) {
  std::vector<double> samples;
  samples.reserve(benchmark.samples);

  if (benchmark.samples > 1) {
    run_once(benchmark, tile, implementation);
  }

  for (std::size_t sample = 0; sample < benchmark.samples; sample++) {
    auto start = std::chrono::steady_clock::now();
    for (std::size_t iteration = 0; iteration < benchmark.iterations;
         iteration++) {
      run_once(benchmark, tile, implementation);
    }
    auto end = std::chrono::steady_clock::now();

    double elapsed =
        std::chrono::duration<double, std::milli>(end - start).count();
    samples.push_back(elapsed / static_cast<double>(benchmark.iterations));
  }

  std::sort(samples.begin(), samples.end());
  return samples[samples.size() / 2];
}

std::vector<BenchmarkSpec> benchmark_cases() {
  return {
      {"64x64 * 64x64", 64, 64, 64, 5, 6, 50, 5},
      {"128x128 * 128x128", 128, 128, 128, 7, 8, 10, 5},
      {"256x256 * 256x256", 256, 256, 256, 9, 10, 2, 5},
      {"512x512 * 512x512", 512, 512, 512, 11, 12, 1, 3},
      {"1024x1024 * 1024x1024", 1024, 1024, 1024, 13, 14, 1, 3},
      {"2048x2048 * 2048x2048", 2048, 2048, 2048, 15, 16, 1, 3},
      {"4096x4096 * 4096x4096", 4096, 4096, 4096, 17, 18, 1, 1},
      {"8192x8192 * 8192x8192", 8192, 8192, 8192, 19, 20, 1, 1},
      {"16384x16384 * 16384x16384", 16384, 16384, 16384, 21, 22, 1, 1},
  };
}

} // namespace

int main() {
  std::cout << std::fixed << std::setprecision(6);
  std::cout << "GPU matmul performance\n\n";
  std::cout << "| Test case | Naive GPU | Tiled GPU |\n";
  std::cout << "|---|---:|---:|\n";

  for (const BenchmarkSpec& spec : benchmark_cases()) {
    BenchmarkCase benchmark{
        spec.name,
        make_matrix(spec.M, spec.K, spec.x_seed),
        make_matrix(spec.K, spec.N, spec.y_seed),
        spec.iterations,
        spec.samples,
    };
    double naive = median_milliseconds(benchmark, false, "naive GPU");
    double tiled = median_milliseconds(benchmark, true, "tiled GPU");
    std::cout << "| " << spec.name << " | " << naive << " ms | " << tiled
              << " ms |" << std::endl;
  }
  return 0;
}
