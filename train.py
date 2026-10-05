"""Train the scratch Transformer on a synthetic seq2seq task.

    python train.py --task reverse --steps 1500
"""

import argparse
import json
import os
import time

import numpy as np

from scratchformer import functional as F
from scratchformer.data import BOS, EOS, PAD, SequenceTask, strip
from scratchformer.engine import no_grad
from scratchformer.model import Transformer
from scratchformer.optim import Adam, NoamSchedule, clip_grad_norm
from scratchformer.rng import manual_seed


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--task", default="reverse", choices=["copy", "reverse", "sort"])
    p.add_argument("--num-symbols", type=int, default=10)
    p.add_argument("--min-len", type=int, default=3)
    p.add_argument("--max-len", type=int, default=10)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--d-ff", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--lr-factor", type=float, default=1.0)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--label-smoothing", type=float, default=0.0)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-size", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/latest")
    p.add_argument("--plot", action="store_true", help="save a loss curve (needs matplotlib)")
    return p.parse_args()


def token_accuracy(logits, targets):
    pred = logits.data.argmax(axis=-1)
    mask = targets != PAD
    return float(((pred == targets) & mask).sum() / mask.sum())


def evaluate(model, src, tgt_in, tgt_out, max_len):
    """Teacher-forced loss / token accuracy, plus exact-match of greedy decoding."""
    model.eval()
    with no_grad():
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits, tgt_out, ignore_index=PAD).item()
        tok_acc = token_accuracy(logits, tgt_out)
    generated = model.greedy_decode(src, BOS, EOS, max_len=max_len + 1)
    exact = np.mean([strip(g) == strip(t) for g, t in zip(generated, tgt_out)])
    model.train()
    return loss, tok_acc, float(exact), generated


def build_model(args, vocab_size):
    """Construct the model from training args (also used by predict.py to reload)."""
    return Transformer(
        vocab_size,
        vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_encoder_layers=args.layers,
        n_decoder_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len + 2,
        pad_idx=PAD,
    )


def main():
    args = parse_args()
    manual_seed(args.seed)
    task = SequenceTask(args.task, args.num_symbols, args.min_len, args.max_len, seed=args.seed)
    held_out = SequenceTask(args.task, args.num_symbols, args.min_len, args.max_len, seed=args.seed + 1)
    eval_batch = held_out.sample(args.eval_size)

    model = build_model(args, task.vocab_size)
    optimizer = Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9)
    schedule = NoamSchedule(optimizer, args.d_model, warmup=args.warmup, factor=args.lr_factor)
    print(f"task={args.task} params={model.num_parameters():,}")

    history = {"step": [], "train_loss": [], "eval_step": [], "eval_loss": [], "eval_exact": []}
    start = time.time()
    for step in range(1, args.steps + 1):
        src, tgt_in, tgt_out = task.sample(args.batch_size)
        logits = model(src, tgt_in)
        loss = F.cross_entropy(logits, tgt_out, ignore_index=PAD, label_smoothing=args.label_smoothing)

        optimizer.zero_grad()
        loss.backward()
        grad_norm = clip_grad_norm(optimizer.params, args.clip)
        lr = schedule.step()
        optimizer.step()

        history["step"].append(step)
        history["train_loss"].append(loss.item())

        if step % args.eval_every == 0 or step == args.steps:
            ev_loss, ev_tok, ev_exact, _ = evaluate(model, *eval_batch, args.max_len)
            history["eval_step"].append(step)
            history["eval_loss"].append(ev_loss)
            history["eval_exact"].append(ev_exact)
            print(
                f"step {step:5d} | train loss {loss.item():.4f} | eval loss {ev_loss:.4f} "
                f"| token acc {ev_tok:.3f} | exact match {ev_exact:.3f} "
                f"| lr {lr:.2e} | grad norm {grad_norm:.2f} | {time.time() - start:.0f}s"
            )

    _, _, _, generated = evaluate(model, *eval_batch, args.max_len)
    print("\nsample predictions (held-out):")
    for src, gen in zip(eval_batch[0][:5], generated[:5]):
        print(f"  {strip(src)} -> {strip(gen)}")

    os.makedirs(args.out, exist_ok=True)
    np.savez(os.path.join(args.out, "model.npz"), **model.state_dict())
    with open(os.path.join(args.out, "history.json"), "w") as f:
        json.dump({"args": vars(args), **history}, f)
    if args.plot:
        save_plot(history, os.path.join(args.out, "loss.png"))
    print(f"\nsaved checkpoint and history to {args.out}/")


def save_plot(history, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(history["step"], history["train_loss"], lw=0.8, alpha=0.6, label="train loss")
    ax.plot(history["eval_step"], history["eval_loss"], marker="o", ms=3, label="eval loss")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.plot(history["eval_step"], history["eval_exact"], color="tab:green", ls="--", label="exact match")
    ax2.set_ylabel("exact match")
    ax2.set_ylim(0, 1.02)
    lines = ax.get_lines() + ax2.get_lines()
    ax.legend(lines, [l.get_label() for l in lines], loc="center right")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


if __name__ == "__main__":
    main()
