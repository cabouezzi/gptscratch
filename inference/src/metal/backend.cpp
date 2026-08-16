#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION

#include "Metal/Metal.hpp"

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <format>
#include <iostream>
#include <mach-o/dyld.h>
#include <vector>

#include <metal/backend.hpp>

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

float *matmul_metal(std::vector<std::vector<float>> const X,
                    std::vector<std::vector<float>> const Y) {

  if (X[0].size() != Y.size())
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

  NS::String *funcName =
      NS::String::string("matmul", NS::UTF8StringEncoding);
  MTL::Function *function = library->newFunction(funcName);
  library->release();

  MTL::ComputePipelineState *pipeline =
      device->newComputePipelineState(function, &error);
  function->release();

  // 5. Create buffers
  std::size_t M = X.size();
  std::size_t K = Y.size();
  std::size_t N = Y[0].size();

  MTL::Buffer *bufA =
      device->newBuffer(M * K * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufB =
      device->newBuffer(K * N * sizeof(float), MTL::ResourceStorageModeShared);
  MTL::Buffer *bufC =
      device->newBuffer(M * N * sizeof(float), MTL::ResourceStorageModeShared);

  MatrixDims dims = {
      .M = static_cast<std::uint32_t>(M),
      .K = static_cast<std::uint32_t>(K),
      .N = static_cast<std::uint32_t>(N),
  };

  // Populate arrays using raw float pointers
  float *ptrA = static_cast<float *>(bufA->contents());
  for (std::size_t m = 0; m < M; m++) {
    for (std::size_t k = 0; k < K; k++) {
      ptrA[m * K + k] = X[m][k];
    }
  }

  float *ptrB = static_cast<float *>(bufB->contents());
  for (std::size_t k = 0; k < K; k++) {
    for (std::size_t n = 0; n < N; n++) {
      ptrB[k * N + n] = Y[k][n];
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
  encoder->setBuffer(bufB, 0, 1);
  encoder->setBuffer(bufC, 0, 2);
  encoder->setBytes(&dims, sizeof(dims), 3);

  // Define standard execution grid boundaries
  MTL::Size gridSize = MTL::Size(N, M, 1);
  MTL::Size threadGroupSize = MTL::Size(16, 16, 1);
  encoder->dispatchThreads(gridSize, threadGroupSize);
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
