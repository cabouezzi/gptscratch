# Optimization TODOs

- [ ] Cache locality: use master buffer instead of separate buffers for Q, K, V.
- [ ] GPU memory: pack matmul inputs, output, and temporary tensors into a reusable master buffer with offsets.
