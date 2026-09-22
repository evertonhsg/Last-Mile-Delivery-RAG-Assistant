"""
Phase 3 — LoRA fine-tune mlx-community/Mistral-7B-Instruct-v0.3-4bit on the
Phase 3 dataset (data/finetune/{train,valid}.jsonl), on a 16GB Apple Silicon
machine.

Uses mlx-lm's Python API rather than its `mlx_lm.lora` CLI, for two reasons:
  1. This file IS the versioned, reproducible config — every hyperparameter
     below is explicit and comitted, instead of living only in a shell
     command someone has to remember to re-type identically next time.
  2. mlx_lm.lora.run() (the function the CLI itself calls) silently
     discards any `training_callback` passed into it — it unconditionally
     overwrites it with `get_reporting_callbacks(args.report_to, ...)` a few
     lines in, which returns None unless you opt into wandb/swanlab. Since
     requirement #4 here is "log every eval step's loss to a file," and we
     don't want a wandb/swanlab dependency for that, we skip run() and call
     mlx_lm.lora.train_model() directly instead — it forwards
     training_callback into the trainer correctly. (Confirmed by reading
     mlx_lm/lora.py and mlx_lm/tuner/trainer.py directly, not assumed.)

Hyperparameter choices, and why:

  num_layers=16 — mlx-lm's OWN default (mlx_lm.lora.CONFIG_DEFAULTS), not a
    guess. For a 32-layer 7B model this LoRA-adapts the top half.

  batch_size=1 — per instruction: smallest possible, minimizes memory
    pressure on a 16GB machine and suits a 79-example training set where
    larger batches would mean very few gradient updates per epoch anyway.

  iters=316 — with batch_size=1, mlx-lm's batching (see
    tuner/trainer.py:iterate_batches) makes each pass through the shuffled
    training set exactly 79 steps (one step per example — no partial
    batches to round away). 316 = 79 * 4, i.e. EXACTLY 4 epochs, landing in
    the middle of the requested 3-5 pass range. Chosen as a clean epoch
    count (rather than an arbitrary iters number) so the loss curve is easy
    to read against "epoch 1 / 2 / 3 / 4" rather than a fraction of a pass.

  steps_per_eval=20 — with only 316 iters total, the CONFIG_DEFAULTS value
    of 200 would give just 3 validation points (iter 1, 200, 316). 20 gives
    ~17 points (iter 1, every 20 after, and the final iter), enough to
    actually see a curve and catch overfitting mid-run instead of only at
    the end.

  val_batches=-1 — use the ENTIRE 15-example validation set on every eval,
    not a sampled subset. With a validation set this small, sampling a
    subset would make each eval point noisy from set composition alone,
    on top of whatever real signal we're trying to read from the curve.

  mask_prompt=True — THE ONE DEVIATION FROM mlx-lm'S DEFAULTS, and worth
    calling out. Each of our training examples has a very long user turn
    (grounding rules + a full retrieved chunk as Context) and a short
    assistant answer. Without prompt masking, the default loss (see
    tuner/trainer.py:default_loss) scores next-token prediction across the
    ENTIRE sequence — meaning most of the training signal would go toward
    "predict the next word of a document chunk that's already sitting
    right there in the input," not toward learning how to answer. That's
    not a useful thing to optimize (the model isn't being asked to
    reproduce the context from memory at inference time — it's handed the
    context directly, every time) and it would dilute the gradient signal
    on the one part of each example that actually needs to be learned: the
    assistant's answer style and citation habits. mask_prompt=True (a
    real, existing mlx-lm option — see tuner/datasets.py:ChatDataset and
    the --mask-prompt CLI flag) restricts the loss to just the assistant
    turn's tokens.

  learning_rate=1e-5, lora_parameters={rank: 8, dropout: 0.0, scale: 20.0},
  optimizer=adam — left at mlx-lm's defaults. A low LR and a rank-8 adapter
  are already conservative choices well-suited to a 94-example dataset;
  there was no specific reason surfaced to move off them.

Run from the repo root:
    python src/train_lora.py
    python src/train_lora.py --iters 160 --adapter-path adapters/lastmile-lora-earlystop
        (e.g. an early-stopped run — see --iters/--adapter-path/--save-every
        below for why these are the only knobs exposed on the command line
        rather than requiring a hand-edit: everything else about a run
        should stay fixed so runs are actually comparable, but "stop
        earlier," "don't clobber the last adapter," and "checkpoint often
        enough to actually land on the optimum" are all normal, expected
        things to vary run-to-run.)
"""

import argparse
import json
import time
import types
from pathlib import Path

import mlx.core as mx
from mlx_lm.lora import CONFIG_DEFAULTS, train_model
from mlx_lm.tuner.callbacks import TrainingCallback
from mlx_lm.tuner.datasets import load_dataset
from mlx_lm.utils import load

DEFAULT_ADAPTER_PATH = "adapters/lastmile-lora"
DEFAULT_ITERS = 316  # 79 train examples * 4 epochs, exactly (batch_size=1)

# Overrides on top of mlx_lm.lora.CONFIG_DEFAULTS — see module docstring for
# the reasoning behind each one that isn't just "left at the mlx-lm default."
# `iters` and `adapter_path` are filled in from CLI args in main() below; the
# values here are just the defaults for a first, full training run.
TRAIN_CONFIG_OVERRIDES = {
    "model": "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
    "train": True,
    "data": "data/finetune",
    "seed": 42,
    "batch_size": 1,
    "iters": DEFAULT_ITERS,
    "val_batches": -1,  # use the full validation set every eval, however large it is
    "steps_per_eval": 20,  # frequent enough to actually see a curve, not just 2-3 points
    "adapter_path": DEFAULT_ADAPTER_PATH,
    "mask_prompt": True,  # only train on the assistant's answer tokens — see docstring
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--iters", type=int, default=DEFAULT_ITERS,
        help=f"Training iterations (default {DEFAULT_ITERS} = 4 epochs over 79 examples "
             "at batch_size=1). Pass e.g. 160 for an early-stop run at ~2 epochs.",
    )
    parser.add_argument(
        "--adapter-path", type=str, default=DEFAULT_ADAPTER_PATH,
        help="Where to save the adapter + training log. Use a fresh path per run "
             "(rather than the default) to keep multiple adapters around for comparison.",
    )
    parser.add_argument(
        "--save-every", type=int, default=CONFIG_DEFAULTS["save_every"],
        help=f"Checkpoint interval (default {CONFIG_DEFAULTS['save_every']}, mlx-lm's own "
             "default). Set this equal to --steps-per-eval's value (20, fixed above) when "
             "you want every validation point to have a matching saved checkpoint — e.g. to "
             "pick out the exact early-stopping optimum afterward without a second run.",
    )
    return parser.parse_args()


class LossLogger(TrainingCallback):
    """Collects every train/val loss report mlx-lm's trainer emits, so the
    full curve (not just the final numbers) can be written out after
    training — mlx-lm's own train() loop only prints these, it doesn't
    persist them anywhere on its own.
    """

    def __init__(self):
        self.train_log: list[dict] = []
        self.val_log: list[dict] = []

    def on_train_loss_report(self, train_info: dict) -> None:
        self.train_log.append(dict(train_info))

    def on_val_loss_report(self, val_info: dict) -> None:
        self.val_log.append(dict(val_info))


def check_overfitting(train_log: list[dict], val_log: list[dict]) -> str:
    """Compare where validation loss bottomed out against where training
    ended, and whether training loss was still falling at that point — the
    textbook overfitting signature (train loss keeps dropping, val loss
    stops and reverses).
    """
    if len(val_log) < 2:
        return "Not enough validation points to assess a trend."

    best = min(val_log, key=lambda e: e["val_loss"])
    final = val_log[-1]

    if final is best:
        return (
            f"No overfitting signal: best val loss ({best['val_loss']:.4f}) was at "
            f"the final eval (iter {best['iteration']})."
        )

    # Training loss at/just before the iteration where val loss bottomed out,
    # vs. at the end of the run.
    train_near_best = max(
        (e for e in train_log if e["iteration"] <= best["iteration"]),
        default=None,
        key=lambda e: e["iteration"],
    )
    train_final = train_log[-1]

    detail = (
        f"Best val loss {best['val_loss']:.4f} at iter {best['iteration']}; "
        f"final val loss {final['val_loss']:.4f} at iter {final['iteration']}."
    )
    if train_near_best is not None:
        detail += (
            f" Train loss was {train_near_best['train_loss']:.4f} near the val-loss "
            f"minimum and {train_final['train_loss']:.4f} at the end."
        )
        if train_final["train_loss"] < train_near_best["train_loss"]:
            return "OVERFITTING SIGNAL: " + detail
    return "Val loss regressed off its minimum, but inconclusively — " + detail


def main() -> None:
    cli_args = parse_args()
    overrides = {
        **TRAIN_CONFIG_OVERRIDES,
        "iters": cli_args.iters,
        "adapter_path": cli_args.adapter_path,
        "save_every": cli_args.save_every,
    }
    config = {**CONFIG_DEFAULTS, **overrides}
    args = types.SimpleNamespace(**config)
    adapter_path = Path(cli_args.adapter_path)

    print(f"Loading base model: {args.model}")
    model, tokenizer = load(args.model, tokenizer_config={"trust_remote_code": True})
    peak_after_load = mx.get_peak_memory() / 1e9
    print(f"Peak memory after load: {peak_after_load:.2f} GB")

    print(f"Loading dataset from {args.data}")
    train_set, valid_set, _test_set = load_dataset(args, tokenizer)
    print(f"Train examples: {len(train_set)}, Val examples: {len(valid_set)}")

    logger = LossLogger()
    adapter_path.mkdir(parents=True, exist_ok=True)

    start = time.perf_counter()
    try:
        train_model(args, model, train_set, valid_set, training_callback=logger)
    finally:
        # Always write whatever was logged, even on a crash/interrupt partway
        # through — a partial curve beats losing the run's history entirely.
        log_path = adapter_path / "training_log.json"
        with open(log_path, "w") as f:
            json.dump(
                {"config": overrides, "train": logger.train_log, "val": logger.val_log},
                f,
                indent=2,
            )
        print(f"Wrote training log to {log_path}")

    elapsed = time.perf_counter() - start
    peak_overall = mx.get_peak_memory() / 1e9

    print("\n" + "=" * 60)
    print("Training summary")
    print("=" * 60)
    if logger.train_log:
        print(f"Final train loss: {logger.train_log[-1]['train_loss']:.4f} "
              f"(iter {logger.train_log[-1]['iteration']})")
    if logger.val_log:
        print(f"Final val loss:   {logger.val_log[-1]['val_loss']:.4f} "
              f"(iter {logger.val_log[-1]['iteration']})")
    print(f"\n{check_overfitting(logger.train_log, logger.val_log)}")
    print(f"\nPeak memory after model load: {peak_after_load:.2f} GB")
    print(f"Peak memory overall (load + training): {peak_overall:.2f} GB")
    print(f"Wall-clock time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")


if __name__ == "__main__":
    main()
