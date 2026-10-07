#pragma once

#include "export.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

namespace inference::eggroll {

class INFERENCE_PUBLIC EGGROLLPerturbation {

public:
  EGGROLLPerturbation(std::size_t M, std::size_t N, std::size_t r);
  ~EGGROLLPerturbation();

  EGGROLLPerturbation(const EGGROLLPerturbation &) = delete;
  EGGROLLPerturbation &operator=(const EGGROLLPerturbation &) = delete;
  EGGROLLPerturbation(EGGROLLPerturbation &&other) noexcept;
  EGGROLLPerturbation &operator=(EGGROLLPerturbation &&other) noexcept;

  std::size_t M;
  std::size_t N;
  std::size_t r;
  float *A;
  float *B;
};

struct INFERENCE_PUBLIC EGGROLLMatrix {
  std::size_t weightOffset;
  std::size_t M;
  std::size_t N;
};

struct INFERENCE_PUBLIC EGGROLLTarget {
  std::size_t weightOffset;
  EGGROLLPerturbation perturbation;
};

using EGGROLLCandidate = std::vector<EGGROLLTarget>;

INFERENCE_PUBLIC EGGROLLPerturbation generatePerturbation(std::size_t M,
                                                          std::size_t N,
                                                          std::size_t r,
                                                          std::uint64_t seed);

INFERENCE_PUBLIC EGGROLLCandidate
generateCandidate(const std::vector<EGGROLLMatrix> &matrices, std::size_t rank,
                  std::uint64_t seed);

} // namespace inference::eggroll
