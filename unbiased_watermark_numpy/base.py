#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABC, abstractmethod
import numpy as np
import random


class AbstractWatermarkCode(ABC):
    @classmethod
    @abstractmethod
    def from_random(cls, rng: random.Random, n: int):
        """n is the totla number of symbols/tokens."""
        pass


class AbstractReweight(ABC):
    @abstractmethod
    def reweight(self, code: AbstractWatermarkCode, p: np.ndarray) -> np.ndarray:
        pass


class AbstractScore(ABC):
    @abstractmethod
    def score(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        """p is the original distribution, q is the distribution after reweighting."""
        pass


class LLR_Score(AbstractScore):
    def score(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            lr = np.divide(
                q, p, out=np.ones_like(p), where=np.logical_or(p != 0, q != 0)
            )
            return np.log(lr)


class AbstractContextCodeExtractor(ABC):
    @abstractmethod
    def extract(self, context: np.ndarray) -> any:
        """Should return a context code `c` which will be used to initialize a random.Generator."""
        pass
