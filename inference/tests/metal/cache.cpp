#include <catch2/catch_test_macros.hpp>

#include <metal/backend.hpp>
#include <metal/kv_cache.hpp>

#include <array>
#include <stdexcept>

TEST_CASE("KV cache tracks length and validates writes",
          "[metal][kv-cache]") {
  inference::MetalContext context;
  inference::KVCache cache(context, 2, 2, 3, 4);

  CHECK(cache.length() == 0);
  CHECK(cache.capacity() == 3);

  const std::array<float, 8> keys{1, 2, 3, 4, 5, 6, 7, 8};
  const std::array<float, 8> values{8, 7, 6, 5, 4, 3, 2, 1};
  CHECK_NOTHROW(cache.writeLayer(0, 0, keys.data(), values.data(), 1));
  CHECK_NOTHROW(cache.writeLayer(1, 2, keys.data(), values.data(), 1));
  CHECK_THROWS_AS(cache.writeLayer(2, 0, keys.data(), values.data(), 1),
                  std::out_of_range);
  CHECK_THROWS_AS(cache.writeLayer(0, 3, keys.data(), values.data(), 1),
                  std::out_of_range);

  cache.setLength(3);
  CHECK(cache.length() == 3);
  CHECK_THROWS_AS(cache.setLength(4), std::out_of_range);
  cache.reset();
  CHECK(cache.length() == 0);
}
