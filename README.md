# Qwen3.8 Flash-Next: TokenV3 on One DGX Spark

This is the Sojufx production recipe for running Qwen3.8 Flash-Next NVFP4 on one NVIDIA DGX Spark / GB10. It is a practical operating profile: native 262K context, FP8 KV cache, BF16 recurrent state, native MTP speculative decoding, stable OpenAI-compatible serving, and a measured multi-user decode profile.

The aim is not a synthetic peak. It is a fast, repeatable local server that keeps the model's native context available for coding, tools, agent workloads, and long sessions. The default profile uses an opt-in TokenV3 cascade verifier to improve sampled speculative decode performance.

## Tested Profile

| Component | Setting |
|---|---|
| Model | `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` |
| Runtime image | `vllm/vllm-openai:qwen38-flash-next` |
| Native context | 262,144 tokens (`YaRN=0`) |
| KV cache | FP8 |
| Recurrent state | BF16 |
| Speculative decoding | Native MTP, K=3 |
| Draft vocabulary | 47,149 code-oriented tokens |
| Speculative verifier | TokenV3 cascade, alpha `0.95` |
| Scheduler capacity | 8 sequences |
| Batched prefill limit | 2,048 tokens |
| Decode graphs | `auto` |

The 47K draft vocabulary is the important efficiency lever in this configuration. It makes each MTP draft step much lighter while the target model still verifies the proposed tokens. TokenV3 adds a permissive cascade path for sampled decoding: a drafted token may be retained when its target probability is sufficiently close to the target's best token. It is intentionally not bit-for-bit equivalent to exact speculative decoding.

## Build The TokenV3 Image

Build the local image once before copying `.env.production` into the runtime checkout:

```bash
cd /opt/sojufx-qwen38-flash-next
docker build \
  -t local/vllm-qwen38-tokenv3:0.95 \
  -f docker/Dockerfile.tokenv3 docker
```

The patch is opt-in. `VLLM_TOKENV3_ALPHA=0` restores vLLM's exact rejection sampler without rebuilding the image. Do not combine TokenV3 with synthetic or block verification.

## Measured Decode Benchmark

Warm single DGX Spark / GB10. Structured streaming decode, 400 completion tokens, `temperature=0`, `top_p=1`, thinking off, and the server already warmed up. This follows the public sparkDash Decode Bench protocol. One successful run per concurrency level is shown below; repeat runs before treating a small difference as meaningful.

| Concurrency | Per-stream decode | Aggregate decode | TTFT |
|---:|---:|---:|---:|
| C1 | 66.17 tok/s | 66.17 tok/s | 447 ms |
| C2 | 55.37 tok/s | 109.70 tok/s | 280 ms |
| C4 | 45.16 tok/s | 179.31 tok/s | 318 ms |

This is a measured result for this machine and workload, not a universal hardware or model claim. The same prompt, output length, sampling settings, server build, and warm-up state matter.

### TokenV3 Production Suite

Warm single DGX Spark / GB10, native 262K context, 256 completion-token cap, `temperature=0.6`, C1 and C4, two repetitions. These are median aggregate completion throughput results from the live TokenV3 `alpha=0.95` profile.

| Prompt class | C1 | C4 aggregate |
|---|---:|---:|
| Agent / tool JSON | 50.1 tok/s | 118.1 tok/s |
| Code edit | 46.6 tok/s | 120.8 tok/s |
| Generic coding | 51.2 tok/s | 135.0 tok/s |
| Long-context review | 48.3 tok/s | 116.1 tok/s |

For the same sampled suite with the exact verifier, the matching baseline was `33.7/87.9`, `40.5/108.2`, `44.0/111.7`, and `34.6/89.9 tok/s` respectively. This is a local performance result, not an assertion that permissive verification preserves quality for every task. Validate your coding, tool use, and long-horizon agent workloads before adopting it.

### Production Prompt Suite

The broader local suite uses four workload classes, 256 output tokens, two repetitions, `temperature=0`, and thinking disabled. Values are medians.

| Prompt class | C1 | C4 aggregate |
|---|---:|---:|
| Agent / tool JSON | 36.6 tok/s | 100.9 tok/s |
| Code edit | 45.5 tok/s | 106.9 tok/s |
| Generic coding | 53.0 tok/s | 132.5 tok/s |
| Long-context review | 41.6 tok/s | 93.4 tok/s |

Both benchmark families belong in a production evaluation: a counting stream is a clean decode measurement; realistic tools, code, and long context reveal the cost of actual work.

## Install

The runtime includes model-specific patched layers and a packed PLE table. Fetch the required runtime assets, then apply this repository's tested environment profile.

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark.git \
  /opt/qwen38-flash-runtime
git clone https://github.com/sojufx/sojufx-Qwen3.8-Flash-Next.git \
  /opt/sojufx-qwen38-flash-next

cd /opt/qwen38-flash-runtime
cp /opt/sojufx-qwen38-flash-next/.env.production .env
./download.sh
./start.sh
```

First launch builds the packed PLE table and tunes kernels. Leave about 130 GB free on the SSD and allow 10-12 minutes before the health endpoint is ready. Do not run another large GPU workload while it starts.

## Verify The Server

```bash
BASE_URL=http://127.0.0.1:8001 \
  /opt/sojufx-qwen38-flash-next/scripts/smoke-vllm.sh
```

The server is ready when `/v1/models` responds and the smoke completion returns `OK`.

## Benchmark It The Same Way

Use the Decode Bench built into [sparkDash](https://github.com/MiaAI-Lab/sparkDash), selecting:

```text
Type: Structured
Output tokens: 400
Thinking: off
Temperature: 0
Concurrency: 1, 2, 4
```

It measures post-first-token streaming decode throughput. Warm the server first and record the endpoint, model name, context length, cache dtype, temperature, prompt type, output limit, and repeat count with every result.

## Operational Defaults

- Keep `YARN=0`. This preserves the native 262K context; extended-context YaRN belongs in a separate experiment.
- Keep `MTP_NUM_SPECULATIVE_TOKENS=3`. Higher K is not inherently faster on GB10.
- Keep `VLLM_TOKENV3_ALPHA=0.95` only after validating representative sampled workloads. Set it to `0` for exact speculative verification, including A/B quality work.
- Keep FP8 KV for the balanced production profile. Validate retrieval quality on your own long-context work before changing it.
- Keep `MAX_NUM_SEQS=8` and `MAX_NUM_BATCHED_TOKENS=2048` for responsive concurrent use.
- Keep `HOST_RESERVE_GIB=26` unless you have measured unified-memory stability on your own machine.
- Do not confuse scheduler capacity with full-context concurrency. A launch reporting about 1.07M KV tokens can support roughly four simultaneous 262K-context sessions; the actual number moves slightly with runtime allocation.
- Bind vLLM to loopback port `8001` and put authentication/TLS in a stable gateway in front of it. The public API URL, keys, and client-facing model alias should survive every model reload.
- Thinking should be an explicit client decision. This recipe benchmarks with thinking off because hidden reasoning changes both latency and output budget.

## Public Endpoint Pattern

Keep the runtime model name internal and preserve one stable public alias at the gateway. A client can continue calling `ornith` or another chosen alias while the gateway routes that alias to the current vLLM model. Model experiments then do not require changes in Hermes, Codex, OpenCode, or other clients.

## Required Components And Provenance

This repository documents and benchmarks the Sojufx operating profile. It does not redistribute the model-specific patched runtime layers, model weights, or container image.

- Runtime assets and launcher: [Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark)
- NVFP4 model checkpoint: [Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4)
- Benchmark harness: [sparkDash](https://github.com/MiaAI-Lab/sparkDash)
- TokenV3 cascade-verification research: [MTPLX pull request #485](https://github.com/youssofal/MTPLX/pull/485). This recipe includes a separately maintained vLLM port for the image above.

The recipe documentation and helper scripts in this repository are Apache-2.0. Upstream code, weights, and images retain their respective licenses.
