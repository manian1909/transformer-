"""Optimizers, learning-rate schedule and gradient clipping."""

import math

import numpy as np


class Optimizer:
    def __init__(self, params, lr):
        self.params = [p for p in params if p.requires_grad]
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def step(self):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocity = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        for p, v in zip(self.params, self.velocity):
            if p.grad is None:
                continue
            v *= self.momentum
            v += p.grad
            p.data -= self.lr * v


class Adam(Optimizer):
    """Adam with optional decoupled weight decay (AdamW when weight_decay > 0)."""

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.98), eps=1e-9, weight_decay=0.0):
        super().__init__(params, lr)
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0
        self.m = [np.zeros_like(p.data) for p in self.params]
        self.v = [np.zeros_like(p.data) for p in self.params]

    def step(self):
        self.t += 1
        bc1 = 1.0 - self.beta1 ** self.t
        bc2 = 1.0 - self.beta2 ** self.t
        for p, m, v in zip(self.params, self.m, self.v):
            if p.grad is None:
                continue
            g = p.grad
            m *= self.beta1
            m += (1.0 - self.beta1) * g
            v *= self.beta2
            v += (1.0 - self.beta2) * g * g
            update = (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            if self.weight_decay:
                update = update + self.weight_decay * p.data
            p.data -= (self.lr * update).astype(p.dtype)


class NoamSchedule:
    """lr = factor * d_model^-0.5 * min(step^-0.5, step * warmup^-1.5)."""

    def __init__(self, optimizer, d_model, warmup=400, factor=1.0):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup = warmup
        self.factor = factor
        self.step_num = 0

    def rate(self, step):
        step = max(step, 1)
        return self.factor * self.d_model ** -0.5 * min(step ** -0.5, step * self.warmup ** -1.5)

    def step(self):
        self.step_num += 1
        self.optimizer.lr = self.rate(self.step_num)
        return self.optimizer.lr


def clip_grad_norm(params, max_norm):
    """Rescale gradients so their global L2 norm is at most ``max_norm``."""
    grads = [p.grad for p in params if p.grad is not None]
    total = math.sqrt(sum(float((g.astype(np.float64) ** 2).sum()) for g in grads))
    if total > max_norm:
        scale = max_norm / (total + 1e-6)
        for p in params:
            if p.grad is not None:
                p.grad = p.grad * scale
    return total
