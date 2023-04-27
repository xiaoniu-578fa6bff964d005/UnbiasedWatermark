#!/usr/bin/env python
# -*- coding: utf-8 -*-

from torch import FloatTensor, LongTensor
from .base import AbstractContextCodeExtractor


class All_ContextCodeExtractor(AbstractContextCodeExtractor):
    def extract(self, context: LongTensor) -> any:
        return context.detach().cpu().numpy().tobytes()


class PrevN_ContextCodeExtractor(AbstractContextCodeExtractor):
    def __init__(self, n: int):
        """Extracts the last n tokens in the context"""
        self.n = n

    def extract(self, context: LongTensor) -> any:
        return context[-self.n :].detach().cpu().numpy().tobytes()
