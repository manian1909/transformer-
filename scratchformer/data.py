"""Synthetic sequence-to-sequence tasks for checking that the model learns.

Each example is a random string of symbols; the target is a deterministic
function of it (copy, reverse, or sort). These are small enough to train on a
CPU in NumPy, yet reversal and sorting cannot be solved without attention.
"""

import numpy as np

PAD, BOS, EOS = 0, 1, 2
NUM_SPECIAL = 3

TASKS = {
    "copy": lambda seq: seq,
    "reverse": lambda seq: seq[::-1],
    "sort": lambda seq: np.sort(seq),
}


class SequenceTask:
    def __init__(self, name="reverse", num_symbols=10, min_len=3, max_len=10, seed=0):
        if name not in TASKS:
            raise ValueError(f"unknown task {name!r}, choose from {sorted(TASKS)}")
        self.name = name
        self.transform = TASKS[name]
        self.num_symbols = num_symbols
        self.min_len = min_len
        self.max_len = max_len
        self.rng = np.random.default_rng(seed)

    @property
    def vocab_size(self):
        return self.num_symbols + NUM_SPECIAL

    def sample(self, batch_size):
        """Return (src, tgt_in, tgt_out) int arrays, right-padded with PAD.

        src     = x_1 .. x_n EOS
        tgt_in  = BOS y_1 .. y_n          (decoder input, teacher forcing)
        tgt_out = y_1 .. y_n EOS          (what the decoder should predict)
        """
        width = self.max_len + 1
        src = np.full((batch_size, width), PAD, dtype=np.int64)
        tgt_in = np.full((batch_size, width), PAD, dtype=np.int64)
        tgt_out = np.full((batch_size, width), PAD, dtype=np.int64)
        for i in range(batch_size):
            n = self.rng.integers(self.min_len, self.max_len + 1)
            seq = self.rng.integers(NUM_SPECIAL, self.vocab_size, size=n)
            out = self.transform(seq)
            src[i, :n], src[i, n] = seq, EOS
            tgt_in[i, 0], tgt_in[i, 1 : n + 1] = BOS, out
            tgt_out[i, :n], tgt_out[i, n] = out, EOS
        return src, tgt_in, tgt_out


def strip(tokens):
    """Cut a generated sequence at the first EOS and drop padding."""
    out = []
    for tok in tokens:
        if tok == EOS:
            break
        if tok != PAD:
            out.append(int(tok))
    return out
