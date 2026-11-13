# Qwen3.8 Flash-Next FP4 on One DGX Spark: 123 tok/s single, 240 tok/s at 4

The Sojufx production setup for `nvidia/Qwen3.8-Flash-Next-FP4-Local` on one NVIDIA DGX Spark (GB10): the native
[TensorFold](https://github.com/ashhart/TensorFold) engine with [MiaAI-Lab's Flash-Next recipe](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Single-DGX-Spark-TensorFold)
(patches 0001 to 0017), plus my 8-bit output layer with 16-bit rescoring and two backports from upstream TensorFold (32-row
copy drafts and prefix wait). Full 262K context, FP8 KV cache, vision on, 4 concurrent users. One script builds it from public
sources: see [Install](#install).

## sparkDash Decode Bench

Measured with [sparkDash](https://github.com/MiaAI-Lab/sparkDash) 1.8.6 `DecodeBench`, running on the Spark itself (no network hop),
13 November 2026. 400 output tokens per request, temperature 0, top_p 1, thinking off, 32-token warm-up, one wave per row.
Aggregate is total decode tokens over the shared decode window; per request is the mean of the streams.

**Code**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | **122.6 tok/s** | 122.6 tok/s | 92 ms |
| 2 | 168.7 tok/s | 89.3 tok/s | 478 ms |
| 4 | **239.8 tok/s** | 69.7 tok/s | 934 ms |

**Structured** (sparkDash default)

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 107.8 tok/s | 107.8 tok/s | 97 ms |
| 2 | 155.4 tok/s | 82.0 tok/s | 357 ms |

**Prose**

| Concurrent requests | Aggregate | Per request | TTFT |
| ---: | ---: | ---: | ---: |
| 1 | 67.2 tok/s | 67.2 tok/s | 92 ms |
| 2 | 93.6 tok/s | 48.0 tok/s | 363 ms |
| 4 | 126.5 tok/s | 33.2 tok/s | 695 ms |

Structured at 4 requests is left out: in that one wave two of the four streams ran far slower than the rest, and I have not yet
repeated it. Code and structured text draft well, so they run fastest; prose is the harder case. Each figure is a single wave, so
expect a few percent of run-to-run spread.

Against my earlier vLLM recipe (now in [`vllm/`](vllm/README.md)), the same sparkDash structured test went from 66.2 to
107.8 tok/s at one request (+63%) and from 109.7 to 155.4 tok/s at two (+42%).

## What Is Running

| Component | Setting |
|---|---|
| Model | `nvidia/Qwen3.8-Flash-Next-FP4-Local`, revision `b9ce18a6` |
| Weight overlay | MiaAI's optional `ABLIT=1`: drowzeys/Keys `house-qsa` overlay (attention output weights of 12 layers). The numbers above were measured with it on. |
| Engine | TensorFold v1.1.4 native (`tensorfold-native serve`) |
| Patches | MiaAI-Lab 0001 to 0017, then 0018 to 0021 in [`patches/`](patches/) (below) |
| Context | 262,144 tokens |
| Concurrency | `--parallel 4` |
| KV cache | FP8 |
| Drafting | Built-in MTP head, up to 6 drafts, running-product stop at -0.4; copy drafts |
| n-gram / PLE table | Read from the SSD with `pread` (not RAM-resident) |
| Vision | On, up to 50 images and 4 videos, 16,384 image tokens |

Serve command:

```bash
tensorfold-native serve <model-dir> --host 127.0.0.1 --port 8001 --context 262144 --parallel 4 \
  --max-tokens 32768 --temperature 0 --top-p 0.8 --top-k 20 --no-thinking --kv-cache fp8 \
  --vision --vision-max-images 50 --vision-max-videos 4 --vision-image-tokens 16384 --thinking-budget 8192
```

Main engine settings: `TF_FLASHNEXT_DRAFTS=6`, `TF_FLASHNEXT_CONFIDENCE=-0.4`, `TF_FLASHNEXT_PRODUCT_STREAMS=5`,
`TF_FLASHNEXT_DRAFT_AHEAD=1`, `TF_FLASHNEXT_KEEP_LATE=1`, `TF_FLASHNEXT_GATHER_READ=pread`, `TF_FLASHNEXT_HEAD8=1`,
`TENSORFOLD_MEMORY_RESERVE_GIB=10`.

## My Changes On Top Of MiaAI-Lab's Recipe

- **0021, 8-bit output layer with 16-bit rescoring (`TF_FLASHNEXT_HEAD8=1`), my own.** The 248,320-word output layer is about
  5 ms of each 38 ms decode pass. An FP8 copy picks the top 64 candidates, and those are recomputed from the original BF16 weights.
  Over 21,357 decode positions it chose the same token as the full BF16 layer every time. It applies only to plain greedy decoding;
  sampling, JSON or grammar-constrained output and logprobs use the full BF16 layer. Single-user speed went up 4.6%. It costs
  0.66 GB of extra memory.
- **0019, 32-row copy drafts**, backported from upstream TensorFold (outcastofmusic, commits `5bb6459` and `a0b0eb1`). When the
  reply repeats text already in the conversation (file edits, tool output echoes), a lone stream can propose up to 31 copied tokens
  at once. The model still verifies every token. Copy-heavy single-user replies ran about 58% faster in my tests.
- **0018, prefix wait**, backported from upstream TensorFold commit `6588b5a` (Robert Wojciechowski): a request whose prompt
  prefix is already being filled by another request waits for it instead of filling it again. No change to sampling or
  verification.
- **0020, copy-draft diagnostics**, my own: an optional `TF_FLASHNEXT_COPY_LOG=2` log line. Off by default.

## Install

Needs one DGX Spark (GB10) with Docker and the NVIDIA container runtime, well over 100 GB free on the SSD for weights and build images, and a Hugging Face token (`HF_TOKEN`)
for the optional overlay.

```bash
git clone https://github.com/sojufx/sojufx-Qwen3.8-Flash-Next.git
cd sojufx-Qwen3.8-Flash-Next

./build.sh                 # builds the image sojufx-tensorfold:recipe-20261113
cp .env.example .env       # set HF_CACHE and KERNEL_CACHE; ABLIT=1 for the overlay the numbers used
./start.sh --prepare       # downloads the pinned weights
./start.sh                 # starts the server on 127.0.0.1:8001
```

`build.sh` fetches MiaAI-Lab's recipe at commit `20e15e4`, builds TensorFold v1.1.4 with patches 0001 to 0017 from source on a
pinned NVIDIA PyTorch base image, applies 0018 to 0021 from this repo, and checks all 3,301 engine source files against the
SHA256 list of the build that produced the numbers above (`patches/production-zig-sources.sha256`) before compiling. It then runs
TensorFold's host tests. No prebuilt engine or private image is used. The build uses two CPU jobs, so expect it to take a while.

`DRY_RUN=1 ./start.sh` prints the full `docker run` command with every engine setting. The server answers on `/v1/models` once
the first launch has finished loading.

## Benchmark It The Same Way

Run sparkDash's Decode Bench on the Spark itself against `http://127.0.0.1:8001`, with 400 output tokens, concurrency 1, 2 and 4,
and the Code, Structured and Prose prompt types. Warm the server first. From another machine, the network round trip adds to TTFT
and staggers the streams, which lowers the multi-request numbers.

## Earlier vLLM Recipe

The previous vLLM + TokenV3 recipe (`Mia-AiLab/Qwen3.8-Flash-Next-NVFP4`, MTP K=3, 24K draft vocabulary) is kept in
[`vllm/`](vllm/README.md).

## Credits

- [TensorFold](https://github.com/ashhart/TensorFold) authors: the engine (MIT); Robert Wojciechowski for prefix wait;
  outcastofmusic for 32-row copy drafts.
- [MiaAI-Lab](https://github.com/MiaAI-Lab): the Flash-Next TensorFold recipe, patches 0001 to 0017, model preparation, and sparkDash.
- NVIDIA: the Qwen3.8-Flash-Next FP4-Local checkpoint. drowzeys/Keys: the optional house-qsa overlay.

Full list in [`CREDITS`](CREDITS) and [`NOTICE`](NOTICE). The patches, `build.sh`, `start.sh` and `Dockerfile` are MIT, like the
TensorFold code they modify; the rest of this repo is Apache-2.0 (see `LICENSE`). Model weights and container images keep their
own terms.
