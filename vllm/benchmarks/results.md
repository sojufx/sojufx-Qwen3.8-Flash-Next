# Benchmark Notes

## Profile

```text
Native context: 262,144 tokens
MTP: K=3
Draft vocabulary: 24,576 code-tuned tokens
Speculative verifier: TokenV3 cascade, alpha 0.95 for sampled decoding
KV cache: FP8
Recurrent state: BF16
Max sequences: 5
CUDA graph capture: auto
Prefill chunk: 2,048 tokens
```

## Structured Decode Bench

Warm single DGX Spark / GB10. Structured streaming count workload, 400 completion
tokens, `temperature=0`, `top_p=1`, and thinking off. The server was warmed before
measurement. This is one completed run per concurrency level.

| Concurrency | Per-stream decode | Aggregate decode | TTFT |
|---:|---:|---:|---:|
| C1 | 66.17 tok/s | 66.17 tok/s | 447 ms |
| C2 | 55.37 tok/s | 109.70 tok/s | 280 ms |
| C4 | 45.16 tok/s | 179.31 tok/s | 318 ms |

The structured stream is a useful reproducibility check, not a substitute for
agent, code, and long-context evaluation.

## Fixed Prompt Suite

One DGX Spark / GB10, warm vLLM server. Four prompts at C1 and C4, two repetitions, `max_tokens=256`, `temperature=0`, and thinking disabled. Values are median aggregate completion throughput.

| Prompt | C1 | C4 |
|---|---:|---:|
| Agent / tool JSON | 37.8 tok/s | 102.7 tok/s |
| Code edit | 42.8 tok/s | 109.3 tok/s |
| Generic coding | 52.9 tok/s | 123.3 tok/s |
| Long-context review | 41.6 tok/s | 95.7 tok/s |

## Draft Vocabulary Sweep

The live 47K vocabulary was compared with matched-corpus 47K, 40K, 32K, and 24K candidates. The selected 24K vocabulary was the only candidate that improved both agent/tool and code at both C1 and C4. This table shows the live 47K control and selected 24K candidate; every value is the median of two warm 256-token runs at `temperature=0`.

| Prompt | Live 47K C1 | 24K C1 | Live 47K C4 | 24K C4 |
|---|---:|---:|---:|---:|
| Agent / tool JSON | 37.4 | 38.2 | 95.2 | 101.8 |
| Code edit | 42.4 | 43.0 | 108.5 | 111.8 |
| Generic coding | 50.6 | 52.3 | 135.1 | 130.6 |
| Long-context review | 35.6 | 39.2 | 98.8 | 97.5 |

The result is workload-dependent and the sweep has two repetitions per cell. It supports 24K as the default profile, not a universal speed claim.

The server reported 1,066,149 KV tokens after this startup, or about 4.07 full 262K-context requests. KV capacity varies slightly by launch.

## TokenV3 Sampled Comparison

The TokenV3 path is active only when sampling. This matched A/B used the same warm server profile, four prompt classes, two repetitions, `max_tokens=256`, `temperature=0.6`, and C1/C4 concurrency. Values are median aggregate completion throughput.

| Prompt | Exact C1 | TokenV3 C1 | Exact C4 | TokenV3 C4 |
|---|---:|---:|---:|---:|
| Agent / tool JSON | 33.7 | 50.1 | 87.9 | 118.1 |
| Code edit | 40.5 | 46.6 | 108.2 | 120.8 |
| Generic coding | 44.0 | 51.2 | 111.7 | 135.0 |
| Long-context review | 34.6 | 48.3 | 89.9 | 116.1 |

TokenV3 is intentionally lossy: it can accept a sufficiently high-probability draft token instead of requiring exact rejection-sampling equivalence. Treat this table as throughput evidence only. Use exact verification (`VLLM_TOKENV3_ALPHA=0`) when quality equivalence matters, and test tool JSON, coding tasks, and multi-step agent behavior before production adoption.
