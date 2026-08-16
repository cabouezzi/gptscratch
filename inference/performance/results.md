# Matmul performance

Median milliseconds per call; includes input copies, output allocation, and free.

| Test case | CPU (-O1) | GPU |
|---|---:|---:|
| 2x2 * 2x2 | — | 1.288329 ms |
| 2x3 * 3x2 | — | 1.313261 ms |
| 64x64 * 64x64 | — | 2.926285 ms |
| 128x128 * 128x128 | — | 2.783050 ms |
| 256x256 * 256x256 | — | 2.766000 ms |
| 512x512 * 512x512 (GPU only) | — | 5.837709 ms |
| 1024x1024 * 1024x1024 (GPU only) | — | 15.367084 ms |
| 2048x2048 * 2048x2048 (GPU only) | — | 87.523000 ms |
| 4096x4096 * 4096x4096 (GPU only) | — | 654.501666 ms |
| 8192x8192 * 8192x8192 (GPU only) | — | 5252.093792 ms |
| 16384x16384 * 16384x16384 (GPU only) | — | 42993.992917 ms |
