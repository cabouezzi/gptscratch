#include <catch2/catch_test_macros.hpp>

#include <eggroll/perturbation.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>

using inference::eggroll::EGGROLLPerturbation;
using inference::eggroll::generateCandidate;
using inference::eggroll::generatePerturbation;

static_assert(!std::is_copy_constructible_v<EGGROLLPerturbation>);
static_assert(!std::is_copy_assignable_v<EGGROLLPerturbation>);
static_assert(std::is_nothrow_move_constructible_v<EGGROLLPerturbation>);
static_assert(std::is_nothrow_move_assignable_v<EGGROLLPerturbation>);

TEST_CASE("EGGROLL perturbation allocates contiguous A and B storage",
          "[eggroll][perturbation]") {
  constexpr std::size_t M = 3;
  constexpr std::size_t N = 2;
  constexpr std::size_t r = 1;

  EGGROLLPerturbation perturbation(M, N, r);

  CHECK(perturbation.M == M);
  CHECK(perturbation.N == N);
  CHECK(perturbation.r == r);
  REQUIRE(perturbation.A != nullptr);
  REQUIRE(perturbation.B != nullptr);
  CHECK(perturbation.B == perturbation.A + N * r);
}

TEST_CASE("EGGROLL perturbation supports ranks greater than one",
          "[eggroll][perturbation]") {
  constexpr std::size_t M = 5;
  constexpr std::size_t N = 3;
  constexpr std::size_t r = 4;

  EGGROLLPerturbation perturbation(M, N, r);

  REQUIRE(perturbation.A != nullptr);
  REQUIRE(perturbation.B != nullptr);
  CHECK(perturbation.B == perturbation.A + N * r);

  perturbation.A[N * r - 1] = 7.0F;
  perturbation.B[M * r - 1] = 11.0F;
  CHECK(perturbation.A[N * r - 1] == 7.0F);
  CHECK(perturbation.B[M * r - 1] == 11.0F);
}

TEST_CASE("EGGROLL perturbation generation is deterministic for a seed",
          "[eggroll][perturbation][generation]") {
  constexpr std::size_t M = 7;
  constexpr std::size_t N = 5;
  constexpr std::size_t r = 2;
  constexpr std::uint64_t seed = 12345;

  EGGROLLPerturbation first = generatePerturbation(M, N, r, seed);
  EGGROLLPerturbation second = generatePerturbation(M, N, r, seed);

  CHECK(std::equal(first.A, first.A + N * r, second.A));
  CHECK(std::equal(first.B, first.B + M * r, second.B));
}

TEST_CASE("EGGROLL perturbation generation changes with the seed",
          "[eggroll][perturbation][generation]") {
  constexpr std::size_t M = 7;
  constexpr std::size_t N = 5;
  constexpr std::size_t r = 2;

  EGGROLLPerturbation first = generatePerturbation(M, N, r, 1);
  EGGROLLPerturbation second = generatePerturbation(M, N, r, 2);

  bool sameA = std::equal(first.A, first.A + N * r, second.A);
  bool sameB = std::equal(first.B, first.B + M * r, second.B);
  CHECK_FALSE((sameA && sameB));
}

TEST_CASE("EGGROLL perturbation generation produces finite values",
          "[eggroll][perturbation][generation]") {
  constexpr std::size_t M = 9;
  constexpr std::size_t N = 11;
  constexpr std::size_t r = 3;

  EGGROLLPerturbation perturbation = generatePerturbation(M, N, r, 42);

  for (std::size_t index = 0; index < N * r; index++) {
    CAPTURE(index);
    CHECK(std::isfinite(perturbation.A[index]));
  }
  for (std::size_t index = 0; index < M * r; index++) {
    CAPTURE(index);
    CHECK(std::isfinite(perturbation.B[index]));
  }
}

TEST_CASE("EGGROLL perturbation rejects zero dimensions and rank",
          "[eggroll][perturbation][validation]") {
  CHECK_THROWS_AS(EGGROLLPerturbation(0, 2, 1), std::invalid_argument);
  CHECK_THROWS_AS(EGGROLLPerturbation(3, 0, 1), std::invalid_argument);
  CHECK_THROWS_AS(EGGROLLPerturbation(3, 2, 0), std::invalid_argument);

  CHECK_THROWS_AS(generatePerturbation(0, 2, 1, 1), std::invalid_argument);
  CHECK_THROWS_AS(generatePerturbation(3, 0, 1, 1), std::invalid_argument);
  CHECK_THROWS_AS(generatePerturbation(3, 2, 0, 1), std::invalid_argument);
}

TEST_CASE("EGGROLL perturbation transfers ownership when moved",
          "[eggroll][perturbation][ownership]") {
  EGGROLLPerturbation source = generatePerturbation(3, 2, 1, 8);
  float *A = source.A;
  float *B = source.B;

  EGGROLLPerturbation destination = std::move(source);

  CHECK(destination.A == A);
  CHECK(destination.B == B);
  CHECK(source.A == nullptr);
  CHECK(source.B == nullptr);
}

TEST_CASE("EGGROLL candidates identify matrices only by offset",
          "[eggroll][perturbation][candidate]") {
  const std::vector<inference::eggroll::EGGROLLMatrix> matrices{
      {.weightOffset = 512, .M = 3, .N = 2},
      {.weightOffset = 2048, .M = 5, .N = 4},
  };

  inference::eggroll::EGGROLLCandidate candidate =
      generateCandidate(matrices, 2, 19);
  REQUIRE(candidate.size() == 2);
  CHECK(candidate[0].weightOffset == 512);
  CHECK(candidate[0].perturbation.M == 3);
  CHECK(candidate[0].perturbation.N == 2);
  CHECK(candidate[0].perturbation.r == 2);
  CHECK(candidate[1].weightOffset == 2048);
  CHECK(candidate[1].perturbation.M == 5);
  CHECK(candidate[1].perturbation.N == 4);
  CHECK(candidate[1].perturbation.r == 2);
}
