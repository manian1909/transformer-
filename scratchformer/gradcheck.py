"""Finite-difference gradient checking for the autodiff engine."""

import numpy as np

from .engine import Tensor


def _scalarize(out, projection):
    """Reduce any output to a scalar with a fixed random projection."""
    if out.size == 1:
        return out.sum()
    return (out * projection).sum()


def numerical_grad(fn, inputs, wrt, projection, eps=1e-6, indices=None):
    """Central-difference estimate of d(proj . fn(*inputs)) / d(inputs[wrt])."""
    x = inputs[wrt]
    grad = np.zeros_like(x.data)
    if indices is None:
        indices = list(np.ndindex(x.shape))
    for idx in indices:
        original = x.data[idx]
        x.data[idx] = original + eps
        plus = _scalarize(fn(*inputs), projection).item()
        x.data[idx] = original - eps
        minus = _scalarize(fn(*inputs), projection).item()
        x.data[idx] = original
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def gradcheck(fn, inputs, eps=1e-6, atol=1e-5, rtol=1e-4, max_checks=None, seed=0):
    """Compare autodiff gradients of ``fn`` against finite differences.

    ``inputs`` are float64 Tensors; those with ``requires_grad`` are checked.
    ``max_checks`` limits how many entries per input are probed (sampled at
    random), which keeps checks on whole models affordable.
    Returns the max abs error per input and raises AssertionError on mismatch.
    """
    rng = np.random.default_rng(seed)
    for t in inputs:
        if t.requires_grad and t.dtype != np.float64:
            raise TypeError("gradcheck needs float64 inputs to be meaningful")
        t.grad = None

    out = fn(*inputs)
    projection = rng.standard_normal(out.shape)
    _scalarize(out, projection).backward()
    analytic = [None if t.grad is None else t.grad.copy() for t in inputs]

    errors = []
    for i, t in enumerate(inputs):
        if not t.requires_grad:
            errors.append(None)
            continue
        all_idx = list(np.ndindex(t.shape))
        if max_checks is not None and len(all_idx) > max_checks:
            picks = rng.choice(len(all_idx), size=max_checks, replace=False)
            all_idx = [all_idx[p] for p in picks]
        numeric = numerical_grad(fn, inputs, i, projection, eps=eps, indices=all_idx)
        ana = analytic[i] if analytic[i] is not None else np.zeros_like(t.data)
        a = np.array([ana[idx] for idx in all_idx])
        n = np.array([numeric[idx] for idx in all_idx])
        err = np.abs(a - n)
        if np.any(err > atol + rtol * np.abs(n)):
            worst = int(np.argmax(err))
            raise AssertionError(
                f"gradient mismatch for input {i} at {all_idx[worst]}: "
                f"analytic={a[worst]:.6e} numeric={n[worst]:.6e}"
            )
        errors.append(float(err.max()))
    return errors
