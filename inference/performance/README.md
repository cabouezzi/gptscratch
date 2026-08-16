# Performance tests

Run the matmul comparison from the project root:

```sh
./performance/run.py
```

The run builds the CPU implementation with `-O1`, compares it with the Metal
implementation, prints a Markdown table, and writes the same table to
`performance/results.md`.

Each normal cell is the median of five samples in milliseconds per `matmul`
call. The dense `4096x4096 * 4096x4096` case uses one sample because it performs
about 68.7 billion multiply-accumulate steps per implementation. The
measurement includes the current API's input copies, output allocation, and
caller-side `free`.
