"""Positional encoding and multi-head attention."""

import math

import numpy as np

from . import functional as F
from .nn import Dropout, Linear, Module

NEG_INF = -1e9


def make_padding_mask(tokens, pad_idx):
    """(B, T) token ids -> (B, 1, 1, T) bool mask, True where the key is padding."""
    return (np.asarray(tokens) == pad_idx)[:, None, None, :]


def make_causal_mask(length):
    """(1, 1, T, T) bool mask, True above the diagonal (future positions)."""
    return np.triu(np.ones((length, length), dtype=bool), k=1)[None, None]


def sinusoidal_table(max_len, d_model):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(same)."""
    pos = np.arange(max_len)[:, None]
    div = np.exp(np.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    table = np.zeros((max_len, d_model))
    table[:, 0::2] = np.sin(pos * div)
    table[:, 1::2] = np.cos(pos * div[: d_model // 2])
    return table.astype(np.float32)


class PositionalEncoding(Module):
    """Adds the fixed sinusoidal table to token embeddings (no learned weights)."""

    def __init__(self, d_model, max_len=512, dropout=0.0):
        self.table = sinusoidal_table(max_len, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        length = x.shape[1]
        if length > len(self.table):
            raise ValueError(f"sequence length {length} exceeds max_len {len(self.table)}")
        return self.dropout(x + self.table[:length])


class MultiHeadAttention(Module):
    """softmax(Q K^T / sqrt(d_k)) V computed for ``n_heads`` heads in parallel."""

    def __init__(self, d_model, n_heads, dropout=0.0):
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.w_q = Linear(d_model, d_model)
        self.w_k = Linear(d_model, d_model)
        self.w_v = Linear(d_model, d_model)
        self.w_o = Linear(d_model, d_model)
        self.dropout = Dropout(dropout)
        self.last_attention = None  # (B, H, Tq, Tk) weights, kept for inspection

    def _split_heads(self, x):
        b, t, _ = x.shape
        return x.reshape(b, t, self.n_heads, self.d_k).transpose(0, 2, 1, 3)

    def forward(self, query, key, value, mask=None):
        """query: (B, Tq, D); key/value: (B, Tk, D); mask broadcasts to (B, H, Tq, Tk)."""
        b, tq, d_model = query.shape
        q = self._split_heads(self.w_q(query))
        k = self._split_heads(self.w_k(key))
        v = self._split_heads(self.w_v(value))

        scores = (q @ k.transpose(0, 1, 3, 2)) * (1.0 / math.sqrt(self.d_k))
        if mask is not None:
            scores = scores.masked_fill(mask, NEG_INF)
        weights = F.softmax(scores, axis=-1)
        self.last_attention = weights.data
        weights = self.dropout(weights)

        context = (weights @ v).transpose(0, 2, 1, 3).reshape(b, tq, d_model)
        return self.w_o(context)
