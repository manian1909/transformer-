"""Cross-check forward values and gradients against PyTorch autograd."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as TF  # noqa: E402

from scratchformer import Tensor  # noqa: E402
from scratchformer import functional as F  # noqa: E402
from scratchformer.attention import MultiHeadAttention, make_causal_mask  # noqa: E402
from scratchformer.nn import LayerNorm  # noqa: E402

rng = np.random.default_rng(3)


def pair(*shape):
    data = rng.standard_normal(shape)
    return (
        Tensor(data.copy(), requires_grad=True, dtype=np.float64),
        torch.tensor(data, requires_grad=True, dtype=torch.float64),
    )


def close(ours, theirs, **kw):
    np.testing.assert_allclose(ours, theirs.detach().numpy(), rtol=1e-7, atol=1e-9, **kw)


def test_composite_expression():
    a, ta = pair(3, 4)
    b, tb = pair(4, 5)
    out = (F.softmax((a @ b).tanh() * 2.0, axis=-1) * (a @ b)).sum()
    tout = (torch.softmax(torch.tanh(ta @ tb) * 2.0, dim=-1) * (ta @ tb)).sum()
    out.backward()
    tout.backward()
    close(out.data, tout)
    close(a.grad, ta.grad)
    close(b.grad, tb.grad)


def test_layer_norm():
    x, tx = pair(2, 5, 8)
    ln = LayerNorm(8).astype(np.float64)
    ln.gamma.data = rng.standard_normal(8)
    ln.beta.data = rng.standard_normal(8)
    tg = torch.tensor(ln.gamma.data, requires_grad=True)
    tb = torch.tensor(ln.beta.data, requires_grad=True)
    w = rng.standard_normal((2, 5, 8))

    (ln(x) * w).sum().backward()
    (TF.layer_norm(tx, (8,), tg, tb, eps=1e-5) * torch.tensor(w)).sum().backward()
    close(x.grad, tx.grad)
    close(ln.gamma.grad, tg.grad)
    close(ln.beta.grad, tb.grad)


@pytest.mark.parametrize("smoothing", [0.0, 0.1])
def test_cross_entropy(smoothing):
    logits, tlogits = pair(2, 6, 9)
    targets = rng.integers(0, 9, size=(2, 6))
    targets[:, -2:] = 0  # treat 0 as padding
    loss = F.cross_entropy(logits, targets, ignore_index=0, label_smoothing=smoothing)
    tloss = TF.cross_entropy(
        tlogits.reshape(-1, 9), torch.tensor(targets.reshape(-1)),
        ignore_index=0, label_smoothing=smoothing,
    )
    loss.backward()
    tloss.backward()
    close(loss.data, tloss)
    close(logits.grad, tlogits.grad)


def test_multi_head_attention_matches_torch_reference():
    d_model, heads = 16, 4
    mha = MultiHeadAttention(d_model, heads).astype(np.float64)
    x, tx = pair(2, 5, d_model)
    mask = make_causal_mask(5)

    out = mha(x, x, x, mask)
    w = rng.standard_normal(out.shape)
    (out * w).sum().backward()

    # same computation written directly in torch with the same weights
    tw = {n: torch.tensor(getattr(mha, n).weight.data, requires_grad=True) for n in ("w_q", "w_k", "w_v", "w_o")}
    tbias = {n: torch.tensor(getattr(mha, n).bias.data) for n in tw}

    def proj(inp, n):
        return inp @ tw[n] + tbias[n]

    def split(t):
        return t.reshape(2, 5, heads, d_model // heads).transpose(1, 2)

    q, k, v = split(proj(tx, "w_q")), split(proj(tx, "w_k")), split(proj(tx, "w_v"))
    scores = q @ k.transpose(-1, -2) / (d_model // heads) ** 0.5
    scores = scores.masked_fill(torch.tensor(mask), float("-inf"))
    ctx = (torch.softmax(scores, -1) @ v).transpose(1, 2).reshape(2, 5, d_model)
    tout = proj(ctx, "w_o")
    (tout * torch.tensor(w)).sum().backward()

    close(out.data, tout)
    close(x.grad, tx.grad)
    for n in tw:
        close(getattr(mha, n).weight.grad, tw[n].grad)
