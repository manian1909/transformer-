"""Encoder-decoder Transformer (Vaswani et al., 2017) on the scratch engine.

Sub-layers use pre-norm residuals, x + Sublayer(LayerNorm(x)), with a final
LayerNorm on each stack. Pre-norm trains reliably without a long LR warmup,
which matters when every step runs on NumPy.
"""

import math

import numpy as np

from .attention import (
    MultiHeadAttention,
    PositionalEncoding,
    make_causal_mask,
    make_padding_mask,
)
from .engine import no_grad
from .nn import Dropout, Embedding, LayerNorm, Linear, Module, ModuleList


class FeedForward(Module):
    def __init__(self, d_model, d_ff, dropout=0.0):
        self.fc1 = Linear(d_model, d_ff)
        self.fc2 = Linear(d_ff, d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.fc1(x).relu()))


class EncoderLayer(Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x, src_mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, src_mask))
        x = x + self.dropout(self.ffn(self.norm2(x)))
        return x


class DecoderLayer(Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.0):
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)
        self.dropout = Dropout(dropout)

    def forward(self, x, memory, tgt_mask=None, src_mask=None):
        h = self.norm1(x)
        x = x + self.dropout(self.self_attn(h, h, h, tgt_mask))
        h = self.norm2(x)
        x = x + self.dropout(self.cross_attn(h, memory, memory, src_mask))
        x = x + self.dropout(self.ffn(self.norm3(x)))
        return x


class Transformer(Module):
    def __init__(
        self,
        src_vocab,
        tgt_vocab,
        d_model=64,
        n_heads=4,
        n_encoder_layers=2,
        n_decoder_layers=2,
        d_ff=256,
        dropout=0.1,
        max_len=256,
        pad_idx=0,
    ):
        self.d_model = d_model
        self.pad_idx = pad_idx
        self.src_embed = Embedding(src_vocab, d_model)
        self.tgt_embed = Embedding(tgt_vocab, d_model)
        self.pos = PositionalEncoding(d_model, max_len, dropout)
        self.encoder = ModuleList(
            EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_encoder_layers)
        )
        self.decoder = ModuleList(
            DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_decoder_layers)
        )
        self.encoder_norm = LayerNorm(d_model)
        self.decoder_norm = LayerNorm(d_model)
        self.generator = Linear(d_model, tgt_vocab)

    def encode(self, src):
        src_mask = make_padding_mask(src, self.pad_idx)
        x = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.encoder_norm(x), src_mask

    def decode(self, tgt, memory, src_mask):
        length = tgt.shape[1]
        tgt_mask = make_padding_mask(tgt, self.pad_idx) | make_causal_mask(length)
        x = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(x, memory, tgt_mask, src_mask)
        return self.generator(self.decoder_norm(x))

    def forward(self, src, tgt):
        """src: (B, S) ids, tgt: (B, T) ids (shifted right) -> logits (B, T, V)."""
        memory, src_mask = self.encode(src)
        return self.decode(tgt, memory, src_mask)

    def greedy_decode(self, src, bos_idx, eos_idx, max_len):
        """Autoregressively generate up to ``max_len`` tokens (BOS excluded)."""
        was_training = self.training
        self.eval()
        src = np.asarray(src)
        batch = src.shape[0]
        out = np.full((batch, 1), bos_idx, dtype=np.int64)
        finished = np.zeros(batch, dtype=bool)
        with no_grad():
            memory, src_mask = self.encode(src)
            for _ in range(max_len):
                logits = self.decode(out, memory, src_mask)
                next_tok = logits.data[:, -1].argmax(axis=-1)
                next_tok = np.where(finished, self.pad_idx, next_tok)
                out = np.concatenate([out, next_tok[:, None]], axis=1)
                finished |= next_tok == eos_idx
                if finished.all():
                    break
        self.train(was_training)
        return out[:, 1:]
