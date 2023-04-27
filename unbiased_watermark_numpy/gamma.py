#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Gamma_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: np.ndarray):
        assert np.issubdtype(shuffle.dtype, np.integer)
        assert np.all(np.sort(shuffle) == np.arange(len(shuffle)))
        self.shuffle = shuffle

    @classmethod
    def from_random(cls, rng: random.Random, n: int):
        indices = np.arange(n)
        rng.shuffle(indices)
        return cls(indices)


class Gamma_Reweight(AbstractReweight):
    def __init__(self, gamma: float):
        self.gamma = gamma

    def reweight(self, code: Gamma_WatermarkCode, p: np.ndarray) -> np.ndarray:
        shuffled_p = p[code.shuffle]
        cumsum = np.cumsum(shuffled_p)
        reweighted_cumsum = np.where(
            cumsum < 1 / 2,
            (1 - self.gamma) * cumsum,
            -self.gamma + (1 + self.gamma) * cumsum,
        )
        rewighted_p = np.diff(np.insert(reweighted_cumsum, 0, 0))
        return np.take(rewighted_p, np.argsort(code.shuffle))
