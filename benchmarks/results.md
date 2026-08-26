# Benchmark results

These are quick production-style numbers from one NVIDIA DGX Spark using the `UD-IQ3_XXS` GGUF quant and llama.cpp `qwen4exp` support.

## Serve profile

```text
ctx-size: 131072
parallel: 2
effective slots: 2 × 65536 tokens
reasoning: off
continuous batching: on
prompt cache: on
batch-size: 1024
ubatch-size: 256
```

## Results

| Concurrency | Successful | Wall time | Aggregate tok/s | Per-stream tok/s | TTFT |
|---:|---:|---:|---:|---:|---:|
| 1 | 1/1 | 9.53s | 30.2 | 30.2 | 0.615s |
| 2 | 2/2 | 11.49s | 52.4 | ~26.3 median | ~0.540s median |
| 4 | 4/4 | 22.53s | 52.6 | ~19.8 median | ~5.96s median |

The C4 result queues because the production profile only has two slots. Use this as a signal that the model works, not as a claim that this is the best four-user Spark model today.

## Server-side timing from a short sanity request

```text
prompt_per_second: ~40.5 tok/s
predicted_per_second: ~27.2 tok/s
```

## Practical interpretation

- Good: one Spark can load and serve a huge Qwen3.8 Flash-Next GGUF.
- Good: 262K single-slot context works.
- Good: 128K total / two 65K slots is usable for two agent sessions.
- Bad: llama.cpp path is slower than mature vLLM NVFP4 + speculative decoding recipes.
- Bad: default thinking mode can burn the whole output budget unless disabled.
