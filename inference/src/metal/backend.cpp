#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <limits>
#include <mach-o/dyld.h>
#include <new>
#include <stdexcept>
#include <vector>

#include <metal/backend.hpp>

namespace inference {

struct MatrixDims {
  std::uint32_t M;
  std::uint32_t K;
  std::uint32_t N;
};

std::filesystem::path executable_directory() {
  std::uint32_t path_size = 0;
  _NSGetExecutablePath(nullptr, &path_size);

  std::vector<char> executable_path(path_size);
  _NSGetExecutablePath(executable_path.data(), &path_size);
  return std::filesystem::path(executable_path.data()).parent_path();
}

float *matmul_metal(std::vector<std::vector<float>> const X, MatMulFlag flagX,
                    std::vector<std::vector<float>> const Y, MatMulFlag flagY,
                    bool tile) {

  std::size_t X_rows = X.size();
  std::size_t X_columns = X[0].size();
  std::size_t Y_rows = Y.size();
  std::size_t Y_columns = Y[0].size();

  std::size_t M = flagX == MatMulFlag::TRANSPOSE ? X_columns : X_rows;
  std::size_t K = flagX == MatMulFlag::TRANSPOSE ? X_rows : X_columns;
  std::size_t B_rows =
      flagY == MatMulFlag::TRANSPOSE ? Y_columns : Y_rows;
  std::size_t N = flagY == MatMulFlag::TRANSPOSE ? Y_rows : Y_columns;

  if (K != B_rows)
    throw std::runtime_error(std::format(
        "Invalid shapes: Attempting to multiply ({},{}) and ({},{})", X.size(),
        X[0].size(), Y.size(), Y[0].size()));
  // 1. Memory management scope wrapper for Metal
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();

  // 2. Initialize GPU Device
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    pool->release();
    throw std::runtime_error("Metal is not supported");
  }

  // 3. Load the named Metal library built by Meson beside this executable.
  NS::Error *error = nullptr;
  std::filesystem::path metallib_path =
      executable_directory() / "inference.metallib";
  NS::String *metallib_path_string =
      NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
  MTL::Library *library = device->newLibrary(metallib_path_string, &error);

  if (!library) {
    device->release();
    pool->release();
    throw std::runtime_error(
        std::format("Could not load {}", metallib_path.string()));
  }

  NS::String *funcName = NS::String::string(
      tile ? "matmul_tile" : "matmul_naive", NS::UTF8StringEncoding);
  MTL::Function *function = library->newFunction(funcName);
  library->release();

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(function, &error);
  function->release();

  // 5. Create buffers
  MTL::Buffer *bufA =
      device->newBuffer(X_rows * X_columns * sizeof(float),
                        MTL::ResourceStorageModeShared);
  MTL::Buffer *bufB =
      device->newBuffer(Y_rows * Y_columns * sizeof(float),
                        MTL::ResourceStorageModeShared);
  MTL::Buffer *bufC =
      device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);

  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(M),
      .K = static_cast<std::uint32_t>(K),
      .N = static_cast<std::uint32_t>(N),
  };

  // Populate arrays using raw float pointers
  float *ptrA = static_cast<float *>(bufA->contents());
  for (std::size_t row = 0; row < X_rows; row++) {
    for (std::size_t column = 0; column < X_columns; column++) {
      ptrA[row * X_columns + column] = X[row][column];
    }
  }

  float *ptrB = static_cast<float *>(bufB->contents());
  for (std::size_t row = 0; row < Y_rows; row++) {
    for (std::size_t column = 0; column < Y_columns; column++) {
      ptrB[row * Y_columns + column] = Y[row][column];
    }
  }

  float *ptrC = static_cast<float *>(bufC->contents());
  for (std::size_t m = 0; m < M; m++) {
    for (std::size_t n = 0; n < N; n++) {
      ptrC[m * N + n] = 0.0f;
    }
  }

  // 6. Command execution and encoding
  MTL::CommandQueue *queue = device->newCommandQueue();
  MTL::CommandBuffer *commandBuffer = queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder = commandBuffer->computeCommandEncoder();

  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(bufA, 0, 0);
  encoder->setBytes(&flagX, sizeof(flagX), 1);
  encoder->setBuffer(bufB, 0, 2);
  encoder->setBytes(&flagY, sizeof(flagY), 3);
  encoder->setBuffer(bufC, 0, 4);
  encoder->setBytes(&dims, sizeof(dims), 5);

  // Define standard execution grid boundaries
  MTL::Size threadsPerThreadgroup = MTL::Size(32, 32, 1);
  MTL::Size threadgroupCount =
      MTL::Size((N + 31) / 32, (M + 31) / 32, 1);
  encoder->dispatchThreadgroups(threadgroupCount, threadsPerThreadgroup);
  encoder->endEncoding();

  // 7. Run and wait synchronously
  commandBuffer->commit();
  commandBuffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(M * N * sizeof(float)));
  if (output == nullptr) {
    bufA->release();
    bufB->release();
    bufC->release();
    pipeline->release();
    queue->release();
    device->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, ptrC, M * N * sizeof(float));

  // 9. Clean up resources manually
  bufA->release();
  bufB->release();
  bufC->release();
  pipeline->release();
  queue->release();
  device->release();
  pool->release();

  return output;
}

float *scaled_dot_product_attention_metal(const float *Q, const float *K,
                                          const float *V,
                                          std::size_t seq_len,
                                          std::size_t head_size,
                                          std::size_t num_heads,
                                          bool isCausal) {
  if (Q == nullptr || K == nullptr || V == nullptr) {
    throw std::invalid_argument("Metal attention inputs cannot be null");
  }
  if (seq_len == 0 || head_size == 0 || num_heads == 0) {
    throw std::invalid_argument("Metal attention dimensions must be positive");
  }
  if (seq_len % 64 != 0) {
    throw std::invalid_argument(
        "Metal attention currently requires seq_len to be divisible by 64");
  }
  if (seq_len > std::numeric_limits<std::uint32_t>::max() ||
      head_size > std::numeric_limits<std::uint32_t>::max()) {
    throw std::invalid_argument("Metal attention dimensions exceed uint32_t");
  }
  if (seq_len > std::numeric_limits<std::size_t>::max() / head_size) {
    throw std::overflow_error("Metal attention buffer size overflow");
  }

  std::size_t elements_per_head = seq_len * head_size;
  if (num_heads >
      std::numeric_limits<std::size_t>::max() / elements_per_head) {
    throw std::overflow_error("Metal attention buffer size overflow");
  }
  std::size_t element_count = num_heads * elements_per_head;
  std::size_t buffer_size = element_count * sizeof(float);
  NS::AutoreleasePool *pool = NS::AutoreleasePool::alloc()->init();
  MTL::Device *device = MTL::CreateSystemDefaultDevice();
  if (!device) {
    pool->release();
    throw std::runtime_error("Metal is not supported");
  }

  NS::Error *error = nullptr;
  std::filesystem::path metallib_path =
      executable_directory() / "inference.metallib";
  NS::String *metallib_path_string =
      NS::String::string(metallib_path.c_str(), NS::UTF8StringEncoding);
  MTL::Library *library = device->newLibrary(metallib_path_string, &error);
  if (!library) {
    device->release();
    pool->release();
    throw std::runtime_error(
        std::format("Could not load {}", metallib_path.string()));
  }

  NS::String *function_name = NS::String::string(
      "scaled_dot_product_attention", NS::UTF8StringEncoding);
  MTL::Function *function = library->newFunction(function_name);
  library->release();
  if (!function) {
    device->release();
    pool->release();
    throw std::runtime_error(
        "Could not load scaled_dot_product_attention from metallib");
  }

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(function, &error);
  function->release();
  if (!pipeline) {
    std::string error_message = "Could not create Metal attention compute pipeline";
    if (error && error->localizedDescription()) {
      error_message += std::format(": {}", error->localizedDescription()->utf8String());
    }
    device->release();
    pool->release();
    throw std::runtime_error(error_message);
  }

  MTL::Buffer *bufQ =
      device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
  MTL::Buffer *bufK =
      device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
  MTL::Buffer *bufV =
      device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
  MTL::Buffer *bufO =
      device->newBuffer(buffer_size, MTL::ResourceStorageModeShared);
  if (!bufQ || !bufK || !bufV || !bufO) {
    if (bufQ) bufQ->release();
    if (bufK) bufK->release();
    if (bufV) bufV->release();
    if (bufO) bufO->release();
    pipeline->release();
    device->release();
    pool->release();
    throw std::bad_alloc();
  }

  std::memcpy(bufQ->contents(), Q, buffer_size);
  std::memcpy(bufK->contents(), K, buffer_size);
  std::memcpy(bufV->contents(), V, buffer_size);
  std::memset(bufO->contents(), 0, buffer_size);

  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(seq_len),
      .K = static_cast<std::uint32_t>(head_size),
      .N = static_cast<std::uint32_t>(seq_len),
  };

  MTL::CommandQueue *queue = device->newCommandQueue();
  MTL::CommandBuffer *command_buffer = queue->commandBuffer();
  MTL::ComputeCommandEncoder *encoder =
      command_buffer->computeCommandEncoder();
  encoder->setComputePipelineState(pipeline);
  encoder->setBuffer(bufQ, 0, 0);
  encoder->setBuffer(bufK, 0, 1);
  encoder->setBuffer(bufV, 0, 2);
  encoder->setBuffer(bufO, 0, 3);
  encoder->setBytes(&dims, sizeof(dims), 4);
  encoder->setBytes(&isCausal, sizeof(isCausal), 5);
  encoder->dispatchThreadgroups(MTL::Size(seq_len, num_heads, 1),
                                MTL::Size(32, 1, 1));
  encoder->endEncoding();
  command_buffer->commit();
  command_buffer->waitUntilCompleted();

  float *output = static_cast<float *>(std::malloc(buffer_size));
  if (!output) {
    bufQ->release();
    bufK->release();
    bufV->release();
    bufO->release();
    pipeline->release();
    queue->release();
    device->release();
    pool->release();
    throw std::bad_alloc();
  }
  std::memcpy(output, bufO->contents(), buffer_size);

  bufQ->release();
  bufK->release();
  bufV->release();
  bufO->release();
  pipeline->release();
  queue->release();
  device->release();
  pool->release();
  return output;
}

} // namespace inference
