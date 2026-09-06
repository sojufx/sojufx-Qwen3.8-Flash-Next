# Benchmark Notes

## Profile

```text
Native context: 262,144 tokens
MTP: K=3
Draft vocabulary: 65,536 tokens
KV cache: FP8
Recurrent state: BF16
Max sequences: 8
CUDA graph capture: auto
Prefill chunk: 2,048 tokens
```

## Fixed Prompt Suite

One DGX Spark / GB10, warm vLLM server. Four prompts at C1 and C4, two repetitions, `max_tokens=256`, `temperature=0`, and thinking disabled. Values are median aggregate completion throughput.

| Prompt | C1 | C4 |
|---|---:|---:|
| Agent / tool JSON | 37.7 tok/s | 104.9 tok/s |
| Code edit | 38.2 tok/s | 97.1 tok/s |
| Generic coding | 35.7 tok/s | 96.1 tok/s |
| Long-context review | 45.0 tok/s | 104.5 tok/s |

The new profile improved tool and long-review workloads relative to the prior full-draft-vocabulary deployment. The short generic request was slightly slower, so this is a production trade-off, not a universal speed claim.

The server reported 1,120,336 KV tokens after startup, or about 4.27 full 262K-context requests.
