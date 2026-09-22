"""
Phase 3 — Sanity check that MLX-LM can load and run the base model we're
about to fine-tune, BEFORE spending any time on the actual LoRA training run.

Model: mlx-community/Mistral-7B-Instruct-v0.3-4bit — confirmed via the live
Hugging Face API (not from memory) as the most-downloaded pre-converted 4-bit
MLX build of Mistral-7B-Instruct under the mlx-community org (56k+ downloads
vs. ~800 for the next-closest v0.2 build, as of 2026-09-20). load() from
mlx_lm downloads and caches it locally on first use — nothing to fetch by hand.

Deliberately asks 3 prompts with NOTHING to do with last-mile delivery. The
goal here is only "does the base model load and generate coherent text at
all on this machine" — domain quality is what Phase 3's fine-tuning step is
for, so testing domain questions against the un-tuned base model wouldn't
tell us anything useful yet.

Peak memory is read via mlx.core.get_peak_memory(), the same call mlx_lm's
own CLI tools (generate, cache_prompt, perplexity) use for their "Peak
memory: X GB" line — reported once after model load (static weight/cache
footprint) and again after all generation (the run's overall high-water
mark), so a memory-constrained 16GB machine can see both numbers rather than
just a single combined figure.

Run from the repo root:
    python src/mlx_smoke_test.py
"""

import mlx.core as mx
from mlx_lm import generate, load

MODEL_ID = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"

PROMPTS = [
    "Write a haiku about the ocean.",
    "What is the capital of France?",
    "Explain in one sentence why the sky is blue.",
]

MAX_TOKENS = 200


def gb(num_bytes: int) -> float:
    """Match mlx_lm's own convention (see mlx_lm/generate.py): decimal GB,
    i.e. bytes / 1e9, not the binary GiB (bytes / 2**30)."""
    return num_bytes / 1e9


def main() -> None:
    print(f"Loading {MODEL_ID} ...")
    model, tokenizer = load(MODEL_ID)
    print(f"Loaded. Peak memory after load: {gb(mx.get_peak_memory()):.2f} GB\n")

    for i, prompt_text in enumerate(PROMPTS, start=1):
        # No system role: this model's chat template only accepts
        # user/assistant turns (confirmed from its tokenizer_config on
        # Hugging Face) — same constraint our Phase 3 training data will
        # need to account for.
        messages = [{"role": "user", "content": prompt_text}]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True)

        print(f"--- Prompt {i}: {prompt_text!r} ---")
        response = generate(model, tokenizer, prompt, max_tokens=MAX_TOKENS, verbose=False)
        print(response.strip())
        print(f"[Peak memory so far: {gb(mx.get_peak_memory()):.2f} GB]\n")

    print(f"Peak memory for the whole run: {gb(mx.get_peak_memory()):.2f} GB")


if __name__ == "__main__":
    main()
