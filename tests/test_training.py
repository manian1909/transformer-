"""End-to-end check: the model actually learns a seq2seq task."""

import numpy as np

from scratchformer import Tensor
from scratchformer import functional as F
from scratchformer.data import BOS, EOS, PAD, SequenceTask, strip
from scratchformer.model import Transformer
from scratchformer.nn import Linear
from scratchformer.optim import SGD, Adam, NoamSchedule, clip_grad_norm
from scratchformer.rng import manual_seed


def test_optimizers_minimise_a_quadratic():
    target = np.array([1.0, -2.0, 3.0])
    for make in (lambda p: SGD(p, lr=0.1, momentum=0.9), lambda p: Adam(p, lr=0.1)):
        x = Tensor(np.zeros(3), requires_grad=True, dtype=np.float64)
        opt = make([x])
        for _ in range(300):
            opt.zero_grad()
            ((x - target) ** 2).sum().backward()
            opt.step()
        np.testing.assert_allclose(x.data, target, atol=1e-3)


def test_clip_grad_norm():
    layer = Linear(3, 3)
    for p in layer.parameters():
        p.grad = np.full(p.shape, 10.0, dtype=np.float32)
    clip_grad_norm(layer.parameters(), 1.0)
    total = np.sqrt(sum((p.grad ** 2).sum() for p in layer.parameters()))
    assert abs(total - 1.0) < 1e-4


def test_learns_copy_task():
    manual_seed(0)
    task = SequenceTask("copy", num_symbols=6, min_len=2, max_len=5, seed=0)
    model = Transformer(task.vocab_size, task.vocab_size, d_model=32, n_heads=2,
                        n_encoder_layers=1, n_decoder_layers=1, d_ff=64, dropout=0.0)
    opt = Adam(model.parameters())
    sched = NoamSchedule(opt, 32, warmup=50)

    losses = []
    for _ in range(250):
        src, tgt_in, tgt_out = task.sample(32)
        loss = F.cross_entropy(model(src, tgt_in), tgt_out, ignore_index=PAD)
        opt.zero_grad()
        loss.backward()
        clip_grad_norm(opt.params, 1.0)
        sched.step()
        opt.step()
        losses.append(loss.item())

    assert np.mean(losses[-20:]) < 0.1 * np.mean(losses[:20])

    src, _, tgt_out = SequenceTask("copy", 6, 2, 5, seed=99).sample(64)
    generated = model.greedy_decode(src, BOS, EOS, max_len=6)
    exact = np.mean([strip(g) == strip(t) for g, t in zip(generated, tgt_out)])
    assert exact > 0.9
