"""A single seedable RNG shared by weight init and dropout, for reproducible runs."""

import numpy as np

_rng = np.random.default_rng(0)


def manual_seed(seed):
    global _rng
    _rng = np.random.default_rng(seed)


def get_rng():
    return _rng
