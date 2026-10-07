#include <eggroll/perturbation.hpp>

#include <limits>
#include <random>
#include <stdexcept>

namespace inference::eggroll {

EGGROLLPerturbation::EGGROLLPerturbation(std::size_t M, std::size_t N,
                                         std::size_t r) {
  if (M == 0 || N == 0 || r == 0) {
    throw std::invalid_argument(
        "EGGROLL perturbation dimensions and rank must be positive");
  }
  if (N > std::numeric_limits<std::size_t>::max() / r ||
      M > std::numeric_limits<std::size_t>::max() / r) {
    throw std::length_error("EGGROLL perturbation dimensions are too large");
  }

  std::size_t aCount = N * r;
  std::size_t bCount = M * r;
  if (aCount > std::numeric_limits<std::size_t>::max() - bCount) {
    throw std::length_error("EGGROLL perturbation dimensions are too large");
  }

  this->M = M;
  this->N = N;
  this->r = r;
  this->A = new float[aCount + bCount];
  this->B = this->A + aCount;
}

EGGROLLPerturbation::~EGGROLLPerturbation() { delete[] this->A; }

EGGROLLPerturbation::EGGROLLPerturbation(EGGROLLPerturbation &&other) noexcept {
  this->M = other.M;
  this->N = other.N;
  this->r = other.r;
  this->A = other.A;
  this->B = other.B;

  other.M = 0;
  other.N = 0;
  other.r = 0;
  other.A = nullptr;
  other.B = nullptr;
}

EGGROLLPerturbation &
EGGROLLPerturbation::operator=(EGGROLLPerturbation &&other) noexcept {
  if (this == &other) {
    return *this;
  }

  delete[] this->A;

  this->M = other.M;
  this->N = other.N;
  this->r = other.r;
  this->A = other.A;
  this->B = other.B;

  other.M = 0;
  other.N = 0;
  other.r = 0;
  other.A = nullptr;
  other.B = nullptr;
  return *this;
}

EGGROLLPerturbation generatePerturbation(std::size_t M, std::size_t N,
                                         std::size_t r, std::uint64_t seed) {
  EGGROLLPerturbation perturbation(M, N, r);
  std::mt19937_64 generator(seed);
  std::normal_distribution<float> distribution(0.0F, 1.0F);

  for (std::size_t index = 0; index < N * r; index++) {
    perturbation.A[index] = distribution(generator);
  }
  for (std::size_t index = 0; index < M * r; index++) {
    perturbation.B[index] = distribution(generator);
  }
  return perturbation;
}

EGGROLLCandidate generateCandidate(const std::vector<EGGROLLMatrix> &matrices,
                                   std::size_t rank, std::uint64_t seed) {
  if (rank == 0) {
    throw std::invalid_argument("EGGROLL rank must be positive");
  }

  std::mt19937_64 generator(seed);
  EGGROLLCandidate candidate;
  candidate.reserve(matrices.size());
  for (const EGGROLLMatrix &matrix : matrices) {
    candidate.push_back({
        .weightOffset = matrix.weightOffset,
        .perturbation =
            generatePerturbation(matrix.M, matrix.N, rank, generator()),
    });
  }
  return candidate;
}

} // namespace inference::eggroll
