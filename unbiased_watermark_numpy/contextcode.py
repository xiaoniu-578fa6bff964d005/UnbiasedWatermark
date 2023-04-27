#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random

from .base import AbstractContextCodeExtractor


class All_ContextCodeExtractor(AbstractContextCodeExtractor):
    def extract(self, context: np.ndarray) -> any:
        return context.tobytes()


class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    def __init__(self, n: int):
        """Extracts the last n tokens in the context"""
        self.n = n

    def extract(self, context: np.ndarray) -> any:
        return context[-self.n :].tobytes()
