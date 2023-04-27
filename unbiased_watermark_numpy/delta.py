#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Delta_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, u: float):
        assert u >= 0 and u <= 1
        self.u = u

    @classmethod
    def from_random(cls, rng: random.Random, *args, **kwargs):
        return cls(rng.random())


class Delta_Reweight(AbstractReweight):
    def reweight(self, code: Delta_WatermarkCode, p: np.ndarray) -> np.ndarray:
        cumsum = np.cumsum(p)
        return np.diff(np.insert(cumsum, 0, 0) < code.u).astype(p.dtype)
