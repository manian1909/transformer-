"""Load a trained checkpoint and run it.

A checkpoint directory is what train.py writes: model.npz (weights) and
history.json (the args used to build the model).

    python predict.py --ckpt runs/latest                    # accuracy on fresh random data
    python predict.py --ckpt runs/latest --input "3 1 4 1 5"
    python predict.py --ckpt runs/latest --interactive

Symbols are integers 0 .. num_symbols-1 (0-9 with the default settings).
"""

import argparse
import json
import os
from types import SimpleNamespace

import numpy as np

from scratchformer.data import BOS, EOS, NUM_SPECIAL, PAD, TASKS, SequenceTask, strip
from train import build_model


def load(ckpt_dir):
    with open(os.path.join(ckpt_dir, "history.json")) as f:
        args = SimpleNamespace(**json.load(f)["args"])
    model = build_model(args, args.num_symbols + NUM_SPECIAL)
    with np.load(os.path.join(ckpt_dir, "model.npz")) as weights:
        model.load_state_dict({k: weights[k] for k in weights.files})
    return model.eval(), args


def predict(model, args, symbols):
    """symbols: list of ints in [0, num_symbols) -> model output in the same space."""
    if not symbols:
        raise ValueError("enter at least one symbol")
    if len(symbols) > args.max_len:
        raise ValueError(f"model was trained on sequences of at most {args.max_len} symbols")
    if any(not 0 <= s < args.num_symbols for s in symbols):
        raise ValueError(f"symbols must be in 0..{args.num_symbols - 1}")
    src = np.array([[s + NUM_SPECIAL for s in symbols] + [EOS]])
    out = model.greedy_decode(src, BOS, EOS, max_len=args.max_len + 1)[0]
    return [t - NUM_SPECIAL for t in strip(out)]


def parse_symbols(text):
    return [int(tok) for tok in text.replace(",", " ").split()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--ckpt", default="runs/latest", help="directory containing model.npz and history.json")
    p.add_argument("--input", help='space-separated symbols, e.g. "3 1 4 1 5"')
    p.add_argument("--interactive", action="store_true")
    p.add_argument("--samples", type=int, default=500, help="random examples for the accuracy check")
    a = p.parse_args()

    model, args = load(a.ckpt)
    expected_fn = TASKS[args.task]
    print(f"loaded {args.task!r} model ({model.num_parameters():,} params) from {a.ckpt}")

    if a.input:
        symbols = parse_symbols(a.input)
        got = predict(model, args, symbols)
        want = [int(s) for s in expected_fn(np.array(symbols))]
        print(f"input:    {symbols}\noutput:   {got}\nexpected: {want}\n{'correct' if got == want else 'WRONG'}")
        return

    if a.interactive:
        print(f"type up to {args.max_len} symbols in 0..{args.num_symbols - 1} (blank line to quit)")
        while True:
            line = input("> ").strip()
            if not line:
                break
            try:
                symbols = parse_symbols(line)
                got = predict(model, args, symbols)
            except ValueError as e:
                print(f"  error: {e}")
                continue
            want = [int(s) for s in expected_fn(np.array(symbols))]
            print(f"  {got}  {'ok' if got == want else f'(expected {want})'}")
        return

    # default: accuracy on freshly generated sequences the model has never seen
    task = SequenceTask(args.task, args.num_symbols, args.min_len, args.max_len, seed=12345)
    src, _, tgt_out = task.sample(a.samples)
    generated = model.greedy_decode(src, BOS, EOS, max_len=args.max_len + 1)
    correct = [strip(g) == strip(t) for g, t in zip(generated, tgt_out)]
    print(f"exact match on {a.samples} random sequences: {np.mean(correct):.3f}")
    for s, g, ok in list(zip(src, generated, correct))[:5]:
        shown = lambda toks: [t - NUM_SPECIAL for t in strip(toks)]  # noqa: E731
        print(f"  {shown(s)} -> {shown(g)} {'ok' if ok else 'WRONG'}")


if __name__ == "__main__":
    main()
