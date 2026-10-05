"""Stateless ops built on the engine.

softmax / log_softmax get hand-written backward passes: composing them from
exp/sum/div works too, but the fused versions are faster and numerically safer.
"""

import numpy as np

from .engine import Tensor
from .rng import get_rng


def relu(x):
    return x.relu()


def softmax(x, axis=-1):
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    s = e / e.sum(axis=axis, keepdims=True)
    out = Tensor._from_op(s, (x,), "softmax")
    if out.requires_grad:
        def _backward():
            g = out.grad
            # Jacobian-vector product of softmax: s * (g - <g, s>)
            x._accum(s * (g - (g * s).sum(axis=axis, keepdims=True)))
        out._backward = _backward
    return out


def log_softmax(x, axis=-1):
    shifted = x.data - x.data.max(axis=axis, keepdims=True)
    logsumexp = np.log(np.exp(shifted).sum(axis=axis, keepdims=True))
    out_data = shifted - logsumexp
    out = Tensor._from_op(out_data, (x,), "log_softmax")
    if out.requires_grad:
        def _backward():
            g = out.grad
            x._accum(g - np.exp(out_data) * g.sum(axis=axis, keepdims=True))
        out._backward = _backward
    return out


def embedding(weight, indices):
    """Row lookup ``weight[indices]``; gradients scatter-add back into the rows."""
    return weight[np.asarray(indices, dtype=np.int64)]


def dropout(x, p, training=True):
    if not training or p == 0.0:
        return x
    keep = get_rng().random(x.shape) >= p
    # inverted dropout: rescale at train time so eval needs no change
    return x * (keep.astype(x.dtype) / (1.0 - p))


def cross_entropy(logits, targets, ignore_index=None, label_smoothing=0.0):
    """Mean token-level cross entropy.

    logits: (..., V) Tensor, targets: (...) int array. Positions equal to
    ``ignore_index`` (e.g. padding) do not contribute to the loss.
    """
    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_targets = np.asarray(targets, dtype=np.int64).reshape(-1)
    n = flat_targets.shape[0]

    logp = log_softmax(flat_logits, axis=-1)
    nll = -logp[np.arange(n), flat_targets]
    if label_smoothing > 0.0:
        smooth = -logp.mean(axis=-1)
        nll = nll * (1.0 - label_smoothing) + smooth * label_smoothing

    if ignore_index is None:
        return nll.mean()
    weights = (flat_targets != ignore_index).astype(logits.dtype)
    denom = max(weights.sum(), 1.0)
    return (nll * weights).sum() * (1.0 / denom)
