#!/usr/bin/env python3
"""Add an opt-in TokenV3 cascade verifier to vLLM's rejection sampler."""

from __future__ import annotations

import sys
from pathlib import Path


def replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError(f"Expected exactly one match for: {old[:80]!r}")
    return source.replace(old, new, 1)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: tokenv3_patch.py SOURCE DESTINATION")

    source = Path(sys.argv[1]).read_text()
    source = replace_once(source, "import torch\n", "import os\n\nimport torch\n")
    helper = '''

@triton.jit
def _compute_global_target_logit_max(
    target_local_max_ptr,
    target_local_max_stride,
    logit_idx,
    vocab_num_blocks,
    PADDED_VOCAB_NUM_BLOCKS: tl.constexpr,
):
    blocks = tl.arange(0, PADDED_VOCAB_NUM_BLOCKS)
    blocks_mask = blocks < vocab_num_blocks
    maxes = tl.load(
        target_local_max_ptr + logit_idx * target_local_max_stride + blocks,
        mask=blocks_mask,
        other=float("-inf"),
    )
    return tl.max(maxes, axis=0)
'''
    source = replace_once(
        source,
        "\n\n@triton.jit\ndef _compute_global_residual_mass(\n",
        helper + "\n\n@triton.jit\ndef _compute_global_residual_mass(\n",
    )
    source = replace_once(
        source,
        "    SYNTHETIC_MODE: tl.constexpr,\n"
        "    USE_BLOCK_VERIFICATION: tl.constexpr,\n):\n",
        "    SYNTHETIC_MODE: tl.constexpr,\n"
        "    USE_BLOCK_VERIFICATION: tl.constexpr,\n"
        "    USE_TOKENV3: tl.constexpr,\n"
        "    TOKENV3_ALPHA: tl.constexpr,\n):\n",
    )
    exact = '''                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
'''
    tokenv3 = '''                if SYNTHETIC_MODE:
                    rate = tl.load(synthetic_conditional_rates_ptr + i)
                    accepted = u < rate
                elif USE_TOKENV3:
                    target_max_logit = _compute_global_target_logit_max(
                        target_local_max_ptr,
                        target_local_max_stride,
                        logit_idx,
                        vocab_num_blocks,
                        PADDED_VOCAB_NUM_BLOCKS,
                    )
                    cascade_keep = target_logprob >= (
                        target_max_logit - target_lse + tl.log(1.0 - TOKENV3_ALPHA)
                    )
                    accepted = cascade_keep | (
                        target_logprob > tl.log(u) + draft_logprob
                    )
                else:
                    # Probability ratio test: p(x) > u * q(x)
                    # Equivalent log form: log_p(x) > log(u) + log_q(x)
                    accepted = target_logprob > tl.log(u) + draft_logprob
'''
    source = replace_once(source, exact, tokenv3)
    setup = '''    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
'''
    setup_tokenv3 = '''    raw_tokenv3_alpha = os.getenv("VLLM_TOKENV3_ALPHA", "").strip()
    tokenv3_alpha = float(raw_tokenv3_alpha) if raw_tokenv3_alpha else 0.0
    use_tokenv3 = tokenv3_alpha > 0.0
    if use_tokenv3 and not 0.0 < tokenv3_alpha < 1.0:
        raise ValueError("VLLM_TOKENV3_ALPHA must be in (0, 1).")
    if use_tokenv3 and (synthetic_conditional_rates is not None or use_block_verification):
        raise ValueError("TokenV3 cannot be combined with synthetic or block verification.")

    num_reqs = cu_num_logits.shape[0] - 1
    num_logits, vocab_size = target_logits.shape
'''
    source = replace_once(source, setup, setup_tokenv3)
    source = replace_once(
        source,
        "        SYNTHETIC_MODE=synthetic_conditional_rates is not None,\n"
        "        USE_BLOCK_VERIFICATION=use_block_verification,\n"
        "        num_warps=1,\n",
        "        SYNTHETIC_MODE=synthetic_conditional_rates is not None,\n"
        "        USE_BLOCK_VERIFICATION=use_block_verification,\n"
        "        USE_TOKENV3=use_tokenv3,\n"
        "        TOKENV3_ALPHA=tokenv3_alpha,\n"
        "        num_warps=1,\n",
    )
    Path(sys.argv[2]).write_text(source)


if __name__ == "__main__":
    main()
