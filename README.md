# Qwen3.8 Flash-Next NVFP4 on One DGX Spark

A production-minded vLLM recipe for `Mia-AiLab/Qwen3.8-Flash-Next-NVFP4` on a single NVIDIA DGX Spark / GB10.

This supersedes this repository's earlier GGUF / llama.cpp experiment. The deployed configuration uses the NVFP4 checkpoint, PLE offload, FP8 KV cache, native MTP speculative decoding, and vLLM's Qwen3.8 Flash-Next image. It retains the model's native 262,144-token context and exposes an OpenAI-compatible API for text, vision, tools, and agent clients.

## Production Profile

| Component | Setting |
|---|---|
| Runtime | `vllm/vllm-openai:qwen38-flash-next` |
| Context | Native 262,144 tokens, YaRN off |
| Speculative decoding | Native MTP, K=3 |
| MTP draft vocabulary | 47,149 code-tuned tokens |
| KV cache | FP8 |
| Recurrent state | BF16 |
| Scheduler | 8 sequences |
| Decode CUDA graphs | Every MTP verify width (`auto`) |
| Prefill chunk | 2,048 tokens |

The upstream 47K draft vocabulary makes each MTP draft step much lighter. The target model still verifies every drafted token, so it trades acceptance for bandwidth rather than changing the final generated-token distribution.

## Measured Results

One DGX Spark / GB10, warm server. Our fixed production suite used four prompts, 256 output tokens, two runs, `temperature=0`, and thinking disabled. Numbers are medians and should not be compared directly with a different prompt, runtime, or hardware.

| Prompt class | C1 | C4 aggregate |
|---|---:|---:|
| Agent / tool JSON | 37.1 tok/s | 94.4 tok/s |
| Code edit | 42.8 tok/s | 108.4 tok/s |
| Generic coding | 46.9 tok/s | 132.3 tok/s |
| Long-context review | 37.8 tok/s | 95.6 tok/s |

This launch reported 1,066,149 KV tokens available: approximately 4.07 requests at the full 262K context. KV capacity varies slightly by launch. `MAX_NUM_SEQS=8` improves short-context concurrency; it does not make eight 262K requests fit simultaneously.

## Quick Start

This repo layers a validated configuration over the maintained upstream launcher. It does not redistribute its patched PLE-offload implementation.

```bash
git clone https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark.git \
  /opt/Qwen3.8-Flash-Next-Single-DGX-Spark
git clone https://github.com/sojufx/sojufx-Qwen3.8-Flash-Next.git \
  /opt/sojufx-Qwen3.8-Flash-Next

cd /opt/Qwen3.8-Flash-Next-Single-DGX-Spark
cp /opt/sojufx-Qwen3.8-Flash-Next/.env.production .env
./download.sh
./start.sh
```

The first launch creates the packed PLE table and tunes kernels. Allow roughly 130 GB of free SSD space and 10-12 minutes before `/health` is ready. Do not run another large GPU model at the same time.

Smoke test it locally:

```bash
BASE_URL=http://127.0.0.1:8001 /opt/sojufx-Qwen3.8-Flash-Next/scripts/smoke-vllm.sh
```

## Use The Shipped 47K Draft Vocabulary

The production profile uses the vocabulary shipped by the upstream Mia recipe:

```text
files/draft_vocab_en_code_47k.txt
```

It reduces the MTP draft head from 1.18 GiB to 0.22 GiB per draft step. On the benchmarked one-Spark profile, it improved our C1 and C4 results across the fixed agent, code, generic, and long-review suite.

```bash
cd /opt/Qwen3.8-Flash-Next-Single-DGX-Spark
grep '^MTP_DRAFT_VOCAB' .env
# MTP_DRAFT_VOCAB="files/draft_vocab_en_code_47k.txt"
```

For a functional baseline without this optimisation, empty `MTP_DRAFT_VOCAB` in `.env`.

## Operational Notes

- Keep `YARN=0` for native 262K context. Treat longer YaRN serving as a separate experiment.
- FP8 KV enables several deep sessions, but validate retrieval quality against your own workload.
- MTP K=3 was the best fixed depth we measured. Higher depth is not automatically faster.
- `MAX_NUM_SEQS=8` plus `CUDAGRAPH_CAPTURE_SIZES=auto` ensures every MTP verify width is captured for short concurrent requests.
- `MAX_NUM_BATCHED_TOKENS=2048` balances prefill throughput with responsiveness for existing streams.
- Leave `HOST_RESERVE_GIB=26` in place unless you have independently re-measured unified-memory safety.
- Always bind the active production model to local port `8001`. The public gateway remains `https://ai.sojufx.com/v1`; clients and API keys do not change when models change.
- Keep a stable external model alias at the gateway. Our clients continue to request `ornith`, while the gateway rewrites that alias to the currently served vLLM model. This avoids touching Hermes, OpenCode, or other client configuration during a model swap.
- Thinking is disabled by default at the gateway for this production profile, preventing hidden reasoning from consuming an interactive response budget. A client can explicitly opt in per request with `chat_template_kwargs: {"enable_thinking": true}`.

### Stable Gateway Alias

The served model name is intentionally an internal vLLM detail. Keep the public
API base URL and the client-facing model alias stable, then make the gateway
rewrite that alias to the active server model. For example, the active target
can live in a user-owned runtime file:

```bash
mkdir -p ~/.config/vllm
printf '%s\n' qwen3.8-flash-next > ~/.config/vllm/upstream-model-alias
```

On a later model swap, update that file to the new internal served-model name.
Clients continue to use the same endpoint, key, and external model alias.

## Attribution

The vLLM image, PLE-offload implementation, and upstream launcher are maintained by [MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark). This repository records the production configuration and benchmark method validated on our one-Spark system.

Model weights: [Mia-AiLab/Qwen3.8-Flash-Next-NVFP4](https://huggingface.co/Mia-AiLab/Qwen3.8-Flash-Next-NVFP4).

## License

Recipe documentation and helper scripts are Apache-2.0. The model, vLLM image, and upstream launcher retain their own licenses.
