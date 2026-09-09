# Benchmark Notes

## Profile

```text
Native context: 262,144 tokens
MTP: K=3
Draft vocabulary: 47,149 code-tuned tokens
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
| Agent / tool JSON | 37.1 tok/s | 94.4 tok/s |
| Code edit | 42.8 tok/s | 108.4 tok/s |
| Generic coding | 46.9 tok/s | 132.3 tok/s |
| Long-context review | 37.8 tok/s | 95.6 tok/s |

The 47K profile improved every cell in this fixed suite versus our preceding 65K draft-vocabulary deployment. Results remain workload-dependent; benchmark representative traffic before claiming a universal speed gain.

The server reported 1,066,149 KV tokens after this startup, or about 4.07 full 262K-context requests. KV capacity varies slightly by launch.
