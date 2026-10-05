"""Reverse-mode automatic differentiation on top of NumPy.

Every operation on a ``Tensor`` records its parents and a closure that knows how
to push the output gradient back to them. ``Tensor.backward`` walks the recorded
graph in reverse topological order and calls those closures.
"""

from contextlib import contextmanager

import numpy as np

_grad_enabled = True


@contextmanager
def no_grad():
    """Disable graph construction inside the block (used for inference)."""
    global _grad_enabled
    prev = _grad_enabled
    _grad_enabled = False
    try:
        yield
    finally:
        _grad_enabled = prev


def is_grad_enabled():
    return _grad_enabled


def _unbroadcast(grad, shape):
    """Sum ``grad`` down to ``shape``, undoing NumPy broadcasting."""
    if grad.shape == shape:
        return grad
    # leading axes that broadcasting added
    while grad.ndim > len(shape):
        grad = grad.sum(axis=0)
    # axes that were stretched from size 1
    for axis, size in enumerate(shape):
        if size == 1 and grad.shape[axis] != 1:
            grad = grad.sum(axis=axis, keepdims=True)
    return grad


class Tensor:
    # make ``ndarray <op> Tensor`` defer to the Tensor's reflected operator
    __array_priority__ = 100

    def __init__(self, data, requires_grad=False, dtype=None):
        if isinstance(data, Tensor):
            data = data.data
        arr = np.asarray(data, dtype=dtype)
        if dtype is None and not np.issubdtype(arr.dtype, np.floating):
            arr = arr.astype(np.float32)
        self.data = arr
        self.requires_grad = requires_grad
        self.grad = None
        self._prev = ()
        self._backward = None
        self._op = ""

    # ------------------------------------------------------------------ utils
    @classmethod
    def _from_op(cls, data, parents, op):
        """Create the output of an op, wiring it into the graph if needed."""
        out = cls(data)
        if _grad_enabled and any(p.requires_grad for p in parents):
            out.requires_grad = True
            out._prev = tuple(parents)
            out._op = op
        return out

    def _accum(self, grad):
        """Add ``grad`` into ``self.grad`` (never in place, so aliasing is safe)."""
        if not self.requires_grad:
            return
        grad = np.asarray(grad, dtype=self.data.dtype)
        self.grad = grad if self.grad is None else self.grad + grad

    def _lift(self, other):
        if isinstance(other, Tensor):
            return other
        return Tensor(np.asarray(other, dtype=self.data.dtype))

    @property
    def shape(self):
        return self.data.shape

    @property
    def ndim(self):
        return self.data.ndim

    @property
    def size(self):
        return self.data.size

    @property
    def dtype(self):
        return self.data.dtype

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        grad = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor({self.data!r}{grad})"

    def numpy(self):
        return self.data

    def item(self):
        return self.data.item()

    def detach(self):
        return Tensor(self.data)

    def astype(self, dtype):
        return Tensor(self.data.astype(dtype), requires_grad=self.requires_grad)

    # ------------------------------------------------------------ arithmetic
    def __add__(self, other):
        other = self._lift(other)
        out = Tensor._from_op(self.data + other.data, (self, other), "add")
        if out.requires_grad:
            def _backward():
                self._accum(_unbroadcast(out.grad, self.shape))
                other._accum(_unbroadcast(out.grad, other.shape))
            out._backward = _backward
        return out

    def __mul__(self, other):
        other = self._lift(other)
        out = Tensor._from_op(self.data * other.data, (self, other), "mul")
        if out.requires_grad:
            def _backward():
                self._accum(_unbroadcast(out.grad * other.data, self.shape))
                other._accum(_unbroadcast(out.grad * self.data, other.shape))
            out._backward = _backward
        return out

    def __pow__(self, exponent):
        if isinstance(exponent, Tensor):
            raise TypeError("only scalar exponents are supported")
        out = Tensor._from_op(self.data ** exponent, (self,), "pow")
        if out.requires_grad:
            def _backward():
                self._accum(out.grad * exponent * self.data ** (exponent - 1))
            out._backward = _backward
        return out

    def __matmul__(self, other):
        other = self._lift(other)
        out = Tensor._from_op(self.data @ other.data, (self, other), "matmul")
        if out.requires_grad:
            def _backward():
                g = out.grad
                a, b = self.data, other.data
                # promote 1-D operands so the batched formulas below apply
                a2 = a[None, :] if a.ndim == 1 else a
                b2 = b[:, None] if b.ndim == 1 else b
                g2 = g
                if a.ndim == 1:
                    g2 = np.expand_dims(g2, -2)
                if b.ndim == 1:
                    g2 = np.expand_dims(g2, -1)
                ga = g2 @ np.swapaxes(b2, -1, -2)
                gb = np.swapaxes(a2, -1, -2) @ g2
                if a.ndim == 1:
                    ga = ga.squeeze(-2)
                if b.ndim == 1:
                    gb = gb.squeeze(-1)
                self._accum(_unbroadcast(ga, a.shape))
                other._accum(_unbroadcast(gb, b.shape))
            out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-self._lift(other))

    def __truediv__(self, other):
        other = self._lift(other)
        return self * other ** -1.0

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return self._lift(other) - self

    def __rtruediv__(self, other):
        return self._lift(other) / self

    def __rmatmul__(self, other):
        return self._lift(other) @ self

    # --------------------------------------------------------------- backprop
    def _topo_order(self):
        """Nodes reachable from ``self``, parents before children (iterative DFS)."""
        order, visited = [], set()
        stack = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in node._prev:
                if id(parent) not in visited:
                    stack.append((parent, False))
        return order

    def backward(self, grad=None):
        if not self.requires_grad:
            raise RuntimeError("backward() called on a tensor that does not require grad")
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError("grad must be given for non-scalar outputs")
            grad = np.ones_like(self.data)
        order = self._topo_order()
        self._accum(grad)
        for node in reversed(order):
            if node._backward is not None and node.grad is not None:
                node._backward()
        # like PyTorch, only leaves keep their .grad
        for node in order:
            if node._prev:
                node.grad = None
