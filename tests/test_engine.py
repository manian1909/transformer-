import numpy as np
import pytest

from scratchformer import Tensor, no_grad
from scratchformer import functional as F
from scratchformer.gradcheck import gradcheck

rng = np.random.default_rng(42)


def t(*shape, positive=False):
    data = rng.standard_normal(shape)
    if positive:
        data = np.abs(data) + 0.5
    return Tensor(data, requires_grad=True, dtype=np.float64)


@pytest.mark.parametrize(
    "fn, shapes",
    [
        (lambda a, b: a + b, [(3, 4), (4,)]),
        (lambda a, b: a - b, [(2, 3, 4), (3, 1)]),
        (lambda a, b: a * b, [(3, 4), (1, 4)]),
        (lambda a, b: a / (b * b + 1.0), [(3, 4), (3, 4)]),
        (lambda a, b: a @ b, [(3, 4), (4, 5)]),
        (lambda a, b: a @ b, [(2, 3, 4), (4, 5)]),
        (lambda a, b: a @ b, [(2, 1, 3, 4), (5, 4, 2)]),
        (lambda a, b: a @ b, [(4,), (4, 3)]),
        (lambda a, b: a @ b, [(3, 4), (4,)]),
        (lambda a: a ** 3, [(3, 4)]),
        (lambda a: -a, [(3,)]),
        (lambda a: a.exp(), [(3, 4)]),
        (lambda a: a.tanh(), [(3, 4)]),
        (lambda a: a.relu(), [(5, 4)]),
        (lambda a: a.sum(), [(3, 4)]),
        (lambda a: a.sum(axis=0), [(3, 4)]),
        (lambda a: a.sum(axis=(0, 2), keepdims=True), [(2, 3, 4)]),
        (lambda a: a.mean(axis=-1), [(3, 4)]),
        (lambda a: a.max(axis=1), [(3, 4)]),
        (lambda a: a.max(), [(3, 4)]),
        (lambda a: a.reshape(4, 3), [(3, 4)]),
        (lambda a: a.transpose(2, 0, 1), [(2, 3, 4)]),
        (lambda a: a.T, [(3, 4)]),
        (lambda a: a[1:, ::2], [(3, 4)]),
        (lambda a: a[np.array([0, 2, 0])], [(3, 4)]),
        (lambda a: a.masked_fill(np.array([True, False, True, False]), -5.0), [(3, 4)]),
        (lambda a: F.softmax(a, axis=-1), [(3, 5)]),
        (lambda a: F.softmax(a, axis=0), [(3, 5)]),
        (lambda a: F.log_softmax(a, axis=-1), [(2, 3, 5)]),
    ],
)
def test_op_gradients(fn, shapes):
    gradcheck(fn, [t(*s) for s in shapes])


def test_log_and_sqrt_on_positive_inputs():
    gradcheck(lambda a: a.log() + a.sqrt(), [t(3, 4, positive=True)])


def test_reflected_ops():
    gradcheck(lambda a: 2.0 - a + 3.0 * a / 2.0 + 1.0 / (a * a + 1.0), [t(3, 4)])
    w = rng.standard_normal((4, 2))
    gradcheck(lambda a: a @ w, [t(3, 4)])
    gradcheck(lambda a: np.ones((2, 3)) @ a, [t(3, 4)])


def test_cross_entropy_gradients():
    targets = np.array([[1, 2, 0], [3, 0, 0]])
    gradcheck(lambda x: F.cross_entropy(x, targets, ignore_index=0), [t(2, 3, 5)])
    gradcheck(lambda x: F.cross_entropy(x, targets, label_smoothing=0.1), [t(2, 3, 5)])


def test_shared_subexpression_accumulates():
    # a is used on several paths; its gradient must be the sum over all of them
    gradcheck(lambda a: (a * a + a).sum() * a.mean(), [t(3, 3)])


def test_backward_matches_closed_form():
    a = Tensor([1.0, 2.0, 3.0], requires_grad=True, dtype=np.float64)
    (a * a).sum().backward()
    np.testing.assert_allclose(a.grad, [2.0, 4.0, 6.0])


def test_grads_accumulate_across_backward_calls():
    a = Tensor([1.0, 2.0], requires_grad=True, dtype=np.float64)
    (a * 3.0).sum().backward()
    (a * 3.0).sum().backward()
    np.testing.assert_allclose(a.grad, [6.0, 6.0])


def test_no_grad_builds_no_graph():
    a = t(2, 2)
    with no_grad():
        b = (a * 2.0).sum()
    assert not b.requires_grad and b._prev == ()


def test_softmax_is_stable_for_large_logits():
    x = Tensor([[1000.0, 1000.0, -1000.0]], dtype=np.float64)
    np.testing.assert_allclose(F.softmax(x).data, [[0.5, 0.5, 0.0]])
    assert np.all(np.isfinite(F.log_softmax(x).data[:, :2]))


def test_deep_graph_does_not_hit_recursion_limit():
    a = Tensor([1.0], requires_grad=True, dtype=np.float64)
    x = a
    for _ in range(5000):
        x = x + 0.0
    x.sum().backward()
    np.testing.assert_allclose(a.grad, [1.0])
