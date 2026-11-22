# Qwen3.8 Flash-Next FP4 on One DGX Spark: 148 tok/s single, 392 tok/s at 4

My production setup for `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one NVIDIA DGX Spark (GB10): the native
[TensorFold](https://github.com/ashhart/TensorFold) v1.2.7 engine with
[MiaAI-Lab's Flash-Next recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold) (patches 0001 to
0005), plus my changes: an 8-bit output layer with 16-bit rescoring, and adaptive draft depth for batches of agreeable drafts.
Full 262K context, FP8 KV cache, images, PDF and video on, 4 concurrent users. One script builds it from public sources: see
[Install](#install).

## sparkDash Decode Bench

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.6 `DecodeBench`, running on the Spark itself (no network hop),
22 November 2026. 400 output tokens per request, temperature 0, thinking off, warmed up first. Aggregate decode tok/s, mean of
three runs, with the lowest and highest run.

**Code**

| Concurrent requests | Aggregate | Range |
| ---: | ---: | ---: |
| 1 | **147.6 tok/s** | 147.4 – 147.8 |
| 2 | **217.7 tok/s** | 212.2 – 220.5 |
| 4 | **391.9 tok/s** | 391.3 – 392.4 |

**Prose**

| Concurrent requests | Aggregate | Range |
| ---: | ---: | ---: |
| 1 | 65.3 tok/s | 62.9 – 66.5 |
| 2 | 96.7 tok/s | 93.6 – 98.3 |
| 4 | 130.7 tok/s | 126.0 – 133.0 |

Code drafts well, so it runs fastest; prose is the harder case. sparkDash's Structured test at 4 requests is left out: it sends
`ignore_eos`, which forces two of its four counting streams to keep generating junk after they have finished, and that tail
drafts badly. It measures the prompt, not the engine.

Against my previous build of this repo (TensorFold v1.1.4, 13 November), code went from 122.6 to 147.6 tok/s at one request (+20%)
and from 239.8 to 391.9 tok/s at four (+63%).

## What Is Running

| Component | Setting |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6` |
| Weight overlay | MiaAI's optional `ABLIT=1`: drowzeys/Keys `house-qsa` overlay (attention output weights of 12 layers). The numbers above were measured with it on. |
| Engine | TensorFold v1.2.7 native (`tensorfold-native serve`), commit `93a2e93e` |
| Patches | MiaAI-Lab 0001 to 0005, then my 0006 to 0008 in [`patches/`](patches/) (below) |
| Context | 262,144 tokens |
| Concurrency | `--parallel 4` |
| KV cache | FP8 |
| Drafting | Built-in MTP head with its 4-bit draft vocabulary head, running-product stop at -0.4, adaptive depth up to 10; copy drafts |
| n-gram / PLE table | Read from the SSD with `pread` (not RAM-resident) |
| Vision | Images (up to 50, 16,384 tokens), PDF pages, and video (up to 4, 16,384 tokens) |

Every engine setting is in [`.env.example`](.env.example); `DRY_RUN=1 ./start.sh` prints the full `docker run` command.

## My Changes On Top Of MiaAI-Lab's Recipe

- **0008, adaptive draft depth (`TF_FLASHNEXT_ADAPTIVE_DEPTH=1`, max 10, cost 0.10).** Each greedy request keeps a running
  estimate of how often its drafts are accepted. When two or more requests are drafting together and one of them accepts at
  least 95% of its drafts (typical for code), it may draft deeper than the normal cap of 8, up to 10, if a simple cost model says
  the longer draft pays for the extra verification. It never drafts shorter than before, so prose and single requests behave
  exactly as without it. Verification is unchanged, so outputs are identical (10 of 10 greedy replies matched token for token).
  Code at two requests went up 16.6% and at four 8.8%; everything else was within run-to-run spread. The idea of choosing depth
  from acceptance comes from the [Hibana](https://github.com/sharar-project/hibana) engine's documentation. My first version,
  which could also shorten drafts, slowed prose by up to 5%, hence raise-only.
- **0007, 8-bit output layer with 16-bit rescoring (`TF_FLASHNEXT_HEAD8=1`).** The 248,320-word output layer is about 5 ms of each
  38 ms decode pass. An FP8 copy picks the top 64 candidates, and those are recomputed from the original BF16 weights. Over 21,357
  decode positions it chose the same token as the full BF16 layer every time. It applies only to plain greedy decoding; sampling,
  JSON or grammar-constrained output and logprobs use the full BF16 layer. It costs 0.66 GB of extra memory.
- **0006, copy-draft diagnostics**: an optional `TF_FLASHNEXT_COPY_LOG=2` log line. Off by default.

Prefix wait and 32-row copy drafts, which I backported to the earlier v1.1.4 build, are now part of upstream TensorFold v1.2.7.

## Install

Needs one DGX Spark (GB10) with Docker and the NVIDIA container runtime, well over 100 GB free on the SSD for weights and build
images, and a Hugging Face token (`HF_TOKEN`) for the optional overlay.

```bash
git clone https://github.com/sojufx/sojufx-Qwen3.8-Flash-Next.git
cd sojufx-Qwen3.8-Flash-Next

cp .env.example .env       # set HF_CACHE and KERNEL_CACHE; ABLIT=1 is the overlay the numbers used
./build.sh                 # builds the image and downloads the pinned weights (IMAGE_ONLY=1: image only)
./start.sh                 # starts the server on 127.0.0.1:8001
```

`build.sh` fetches MiaAI-Lab's recipe at commit `b956dba` into `.recipe/`, applies [`patches/recipe-owner.patch`](patches/recipe-owner.patch)
to it, and runs its `scripts/prepare.sh`. That builds TensorFold v1.2.7 with MiaAI's patches 0001 to 0005 from source on a pinned
NVIDIA PyTorch 26.07 base image, then applies 0006 and 0007, checks every engine source file against the SHA256 list of the build
that produced the numbers above (`patches/production-zig-sources.sha256`), applies 0008, and compiles. No prebuilt engine or
private image is used. The build uses two CPU jobs, so expect it to take a while.

`start.sh` copies your `.env` into `.recipe/` and runs MiaAI's `start.sh`, which also takes `restart` and extra
`tensorfold-native serve` arguments. The server answers on `/v1/models` once the first launch has finished loading.

## Benchmark It The Same Way

Run sparkDash's Decode Bench on the Spark itself against `http://127.0.0.1:8001`, with 400 output tokens, concurrency 1, 2 and 4,
and the Code and Prose prompt types. Warm the server first and repeat each run a few times. From another machine, the network
round trip adds to TTFT and staggers the streams, which lowers the multi-request numbers.

## Earlier vLLM Recipe

The previous vLLM + TokenV3 recipe (`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`, MTP K=3, 24K draft vocabulary) is kept in
[`vllm/`](vllm/README.md).

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) authors: the engine (MIT), including prefix wait (Robert Wojciechowski) and
  32-row copy drafts (outcastofmusic).
- [MiaAI-Lab](https://github.com/MiaAI-Lab): the Flash-Next TensorFold recipe, patches 0001 to 0005, video and PDF support, model
  preparation, and sparkDash.
- [Hibana](https://github.com/sharar-project/hibana): the acceptance-driven draft depth idea behind 0008 (no code reused).
- NVIDIA: the Qwen3.8-Flash-Next FP4-Local checkpoint. drowzeys/Keys: the optional house-qsa overlay.

Full list in [`CREDITS`](CREDITS) and [`NOTICE`](NOTICE). The patches, `build.sh`, `start.sh` and `.env.example` are MIT, like the
TensorFold code they modify; the rest of this repo is Apache-2.0 (see `LICENSE`). Model weights and container images keep their
own terms.
