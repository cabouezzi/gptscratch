# Performance tests

Build and run the GPU benchmark from the project root:

```sh
meson compile -C builddir matmul_performance
./builddir/performance/matmul_performance
```

It prints the results directly as a Markdown table. The VS Code **Run Matmul
Performance** launch configuration builds and runs this same target.

Each row reports the median GPU time in milliseconds per `matmul` call. The
measurement includes the current API's input copies, output allocation, and
caller-side `free`.

## Generation buffer-layout benchmark

Recorded August 30, 2026 on an M1 Pro using cached generation with the prompt
`ROMEO:`. The active Meson build was `debug` with C++ optimization level `0`;
the Metal compiler was invoked without an explicit optimization flag:

| Generated tokens | Separate Metal buffers, debug | One per-call buffer, debug | Ping-pong, debug | Ping-pong, release | Release vs. debug ping-pong |
|---:|---:|---:|---:|---:|---:|
| 20 | 1.52 s | 1.22 s | 1.26 s | 1.05 s | 1.20x |
| 100 | 5.40 s | 5.26 s | 4.97 s | 4.64 s | 1.07x |
| 200 | 10.59 s | 10.27 s | 9.68 s | 9.32 s | 1.04x |
| 400 | 31.81 s | ~30.00 s | 25.97 s | 25.52 s | 1.02x |

The separate-buffer values were recorded before the contiguous-buffer change;
these were not two freshly built variants run back-to-back. The 20-token row is
particularly sensitive to startup noise. The sustained measured improvement was
approximately 3–6% for the per-call contiguous buffer and another 6% for the
ping-pong workspace at 100–200 tokens. The 400-token case repeatedly rebuilds a
full cache window after reaching the 256-token context limit, so eliminating
per-operation allocations has a larger effect there.

The release ping-pong column uses a separate Meson `release` build with C++
optimization level `3` and debug information disabled. Release optimization has
the largest relative effect on short runs, where CPU-side setup is a larger
share of total execution time.

## C++/Metal versus PyTorch/MPS KV-cache milestone

Recorded August 31, 2026 on an M1 Pro using the same model checkpoint, the
prompt `ROMEO:`, and 200 generated character tokens. Each result is the median
of 10 samples. Every implementation used a KV cache.

| Implementation | Median 200-token time | Relative to C++/Metal |
|---|---:|---:|
| C++/Metal, one command buffer and parallel heads | **0.400 s** | 1.00x |
| PyTorch/MPS, packed QKV + SDPA, FP16 eager | **0.646 s** | 1.62x slower |
| PyTorch/MPS, packed QKV + SDPA, FP32 eager | 0.660 s | 1.65x slower |
| PyTorch/MPS, packed QKV + SDPA, FP32 Inductor | 0.667 s | 1.67x slower |
| PyTorch/MPS, original head-by-head FP32 eager | 3.233 s | 8.08x slower |

The best PyTorch path was approximately **5.00x faster** than the original
module-by-module eager implementation. The important changes were packing the
six heads into one QKV projection per layer, processing all heads together with
scaled-dot-product attention, retaining the KV cache on MPS, using
`torch.inference_mode()`, and avoiding a CPU synchronization for every token.

The packed FP32 implementation matched the original logits exactly for the
verification prompt. FP16's maximum absolute logit difference from the
equivalent unpacked FP16 path was `0.0078125` and generated the same output.

Inductor successfully compiled the packed SDPA model but did not improve this
workload: its median was approximately 1% slower than optimized eager mode. The
earlier `aot_eager` measurement is intentionally omitted because that backend
captures a graph while continuing to dispatch eager MPS operators; it is not an
optimized Metal graph.

PyTorch timings exclude model loading, compilation, and warm-up and include
prefill plus generation. The C++ measurement uses `/usr/bin/time` around the
release executable, so it additionally includes process startup, model loading,
and terminal writes redirected to `/dev/null`. PyTorch used deterministic GPU
argmax selection; the C++ generator used its normal sampling path.

## Parallel attention-head projections

Recorded August 31, 2026 in a release build. Each value below is the median of
10 paired samples, with 100 cached single-token decode operations per sample.
The run order alternated between samples. Both paths use one Metal command
buffer per token; only Q/K/V head projection scheduling differs.

| Q/K/V projection strategy | Median per 100 tokens | Relative performance |
|---|---:|---:|
| 18 sequential dispatches per layer | 220.814 ms | 1.00x |
| One fused, head-parallel dispatch per layer | 128.008 ms | **1.725x faster** |

The head-parallel kernel reduced measured decode time by **42.0%**. It maps the
six attention heads across the dispatch grid and computes Q, K, and V together,
reusing each input-vector load. On the full 200-token generation workload, total
time improved from 0.760 seconds to 0.680 seconds.
