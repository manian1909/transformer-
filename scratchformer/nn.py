"""Minimal module system and basic layers."""

import math

import numpy as np

from . import functional as F
from .engine import Tensor
from .rng import get_rng


class Parameter(Tensor):
    def __init__(self, data):
        super().__init__(np.asarray(data, dtype=np.float32), requires_grad=True)


class Module:
    training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def _children(self):
        for name, value in vars(self).items():
            if isinstance(value, (Parameter, Module)):
                yield name, value

    def named_parameters(self, prefix=""):
        for name, value in self._children():
            full = f"{prefix}{name}"
            if isinstance(value, Parameter):
                yield full, value
            else:
                yield from value.named_parameters(prefix=full + ".")

    def parameters(self):
        return [p for _, p in self.named_parameters()]

    def modules(self):
        yield self
        for _, value in self._children():
            if isinstance(value, Module):
                yield from value.modules()

    def num_parameters(self):
        return sum(p.size for p in self.parameters())

    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self, mode=True):
        for m in self.modules():
            m.training = mode
        return self

    def eval(self):
        return self.train(False)

    def astype(self, dtype):
        """Cast all parameters in place (float64 is used for gradient checks)."""
        for p in self.parameters():
            p.data = p.data.astype(dtype)
            p.grad = None
        return self

    def state_dict(self):
        return {name: p.data.copy() for name, p in self.named_parameters()}

    def load_state_dict(self, state):
        params = dict(self.named_parameters())
        missing = set(params) - set(state)
        unexpected = set(state) - set(params)
        if missing or unexpected:
            raise KeyError(f"missing={sorted(missing)} unexpected={sorted(unexpected)}")
        for name, p in params.items():
            if state[name].shape != p.shape:
                raise ValueError(f"shape mismatch for {name}: {state[name].shape} vs {p.shape}")
            p.data = np.array(state[name], dtype=p.dtype)


class ModuleList(Module):
    def __init__(self, modules):
        self._items = list(modules)

    def _children(self):
        for i, m in enumerate(self._items):
            yield str(i), m

    def __iter__(self):
        return iter(self._items)

    def __getitem__(self, i):
        return self._items[i]

    def __len__(self):
        return len(self._items)


class Linear(Module):
    """y = x W + b with Xavier/Glorot-uniform initialised W of shape (in, out)."""

    def __init__(self, in_features, out_features, bias=True):
        limit = math.sqrt(6.0 / (in_features + out_features))
        self.weight = Parameter(get_rng().uniform(-limit, limit, (in_features, out_features)))
        self.bias = Parameter(np.zeros(out_features)) if bias else None

    def forward(self, x):
        y = x @ self.weight
        return y + self.bias if self.bias is not None else y


class Embedding(Module):
    def __init__(self, num_embeddings, dim):
        self.weight = Parameter(get_rng().normal(0.0, dim ** -0.5, (num_embeddings, dim)))

    def forward(self, indices):
        return F.embedding(self.weight, indices)


class LayerNorm(Module):
    """Normalise over the last axis, then apply a learned scale and shift.

    Written in terms of engine primitives (mean, sub, pow, mul) so the backward
    pass falls out of the autodiff graph rather than being hand-derived.
    """

    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.gamma = Parameter(np.ones(dim))
        self.beta = Parameter(np.zeros(dim))

    def forward(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        centered = x - mu
        var = (centered * centered).mean(axis=-1, keepdims=True)
        x_hat = centered * (var + self.eps) ** -0.5
        return x_hat * self.gamma + self.beta


class Dropout(Module):
    def __init__(self, p=0.1):
        self.p = p

    def forward(self, x):
        return F.dropout(x, self.p, self.training)


class ReLU(Module):
    def forward(self, x):
        return x.relu()
