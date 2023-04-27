#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from torch.nn import functional as F

from . import AbstractWatermarkCode, AbstractReweight, AbstractScore


class Gamma_WatermarkCode(AbstractWatermarkCode):
    def __init__(self, shuffle: LongTensor):
        self.shuffle = shuffle
        self.unshuffle = torch.argsort(shuffle, dim=-1)

    @classmethod
    def from_random(
        cls,
        rng: torch.Generator | list[torch.Generator],
        vocab_size: int,
    ):
        if isinstance(rng, list):
            batch_size = len(rng)
            shuffle = torch.stack(
                [
                    torch.randperm(vocab_size, generator=rng[i], device=rng[i].device)
                    for i in range(batch_size)
                ]
            )
        else:
            shuffle = torch.randperm(vocab_size, generator=rng, device=rng.device)
        return cls(shuffle)


class Gamma_Reweight(AbstractReweight):
    watermark_code_type = Gamma_WatermarkCode

    def __init__(self, gamma: float):
        self.gamma = gamma

    def reweight_logits(
        self, code: AbstractWatermarkCode, p_logits: FloatTensor
    ) -> FloatTensor:
        shuffled_p_logits = torch.gather(p_logits, -1, code.shuffle)
        cumsum = torch.cumsum(F.softmax(shuffled_p_logits, dim=-1), dim=-1)
        reweighted_cumsum = torch.where(
            cumsum < 1 / 2,
            (1 - self.gamma) * cumsum,
            -self.gamma + (1 + self.gamma) * cumsum,
        )
        suffled_rewighted_p = torch.diff(
            reweighted_cumsum,
            dim=-1,
            prepend=torch.zeros_like(reweighted_cumsum[..., :1]),
        )
        rewighted_p = torch.gather(suffled_rewighted_p, -1, code.unshuffle)
        reweighted_p_logits = torch.log(rewighted_p)
        return reweighted_p_logits
