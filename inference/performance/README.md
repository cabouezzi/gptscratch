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
