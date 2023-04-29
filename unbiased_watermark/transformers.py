#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
from torch import FloatTensor, LongTensor
from transformers import LogitsProcessor

from .base import AbstractReweight, AbstractContextCodeExtractor, AbstractScore


class WatermarkLogitsProcessor(LogitsProcessor):
    def __init__(
        self,
        private_key: any,
        reweight: AbstractReweight,
        context_code_extractor: AbstractContextCodeExtractor,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.private_key = private_key
        self.reweight = reweight
        self.context_code_extractor = context_code_extractor
        self.cc_history = set()

    def get_rng_seed(self, context_code: any) -> any:
        self.cc_history.add(context_code)
        import hashlib

        m = hashlib.sha256()
        m.update(context_code)
        m.update(self.private_key)
        full_hash = m.digest()
        truncated_hash = full_hash[:8]
        seed = int.from_bytes(truncated_hash, byteorder="big")
        return seed

    def reset_history(self):
        self.cc_history = set()

    def __call__(self, input_ids: LongTensor, scores: FloatTensor) -> FloatTensor:
        batch_size = input_ids.size(0)
        context_codes = [
            self.context_code_extractor.extract(input_ids[i]) for i in range(batch_size)
        ]

        mask, rng = zip(
            *[
                (
                    context_code in self.cc_history,
                    torch.Generator(device=scores.device).manual_seed(
                        self.get_rng_seed(context_code)
                    ),
                )
                for context_code in context_codes
            ]
        )
        rng = list(rng)
        mask = torch.tensor(mask, device=scores.device)
        watermark_code = self.reweight.watermark_code_type.from_random(
            rng, scores.size(1)
        )
        reweighted_scores = self.reweight.reweight_logits(watermark_code, scores)
        return torch.where(mask[:, None], scores, reweighted_scores)


def get_score(
    text: str,
    watermark_processor: WatermarkLogitsProcessor,
    score: AbstractScore,
    model,
    tokenizer,
    temperature=0.2,
    prompt: str = "",
    **kwargs
) -> list[float]:
    input_ids = tokenizer.encode(text)
    prompt_len = len(tokenizer.encode(prompt))
    input_ids = torch.tensor(input_ids).unsqueeze(0).to(model.device)
    outputs = model(input_ids)
    logits = outputs.logits[:, :-1, :] / temperature
    new_logits = torch.zeros_like(logits)
    for i in range(logits.size(1)):
        if i == prompt_len:
            watermark_processor.reset_history()
        new_logits[:, i] = watermark_processor(input_ids[:, : i + 1], logits[:, i])
    all_scores = score.score(logits, new_logits)
    scores = torch.gather(all_scores, -1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return [None] + scores[0].tolist(), prompt_len
