#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import numpy.typing as npt
import torch
from torch import FloatTensor
from torch.nn import functional as F

from . import AbstractScore


class RobustLLR_Score_Batch(AbstractScore):
    def __init__(self, batch_query):
        self.batch_query = batch_query

    @classmethod
    def from_grid(cls, dist_ps: npt.ArrayLike, dist_qs: npt.ArrayLike):
        dist_ps = np.array(dist_ps)
        dist_qs = np.array(dist_qs)
        assert dist_ps.ndim == 1
        assert dist_qs.ndim == 1
        assert np.all(dist_ps >= 0) and np.all(dist_qs >= 0)
        with np.errstate(divide="ignore"):
            dist_p_logs = np.log(dist_ps)
            dist_q_logs = np.log(dist_qs)
        dist_p_logs.sort()
        dist_q_logs.sort()
        batch_query = [(d_p_l, d_q_l) for d_p_l in dist_p_logs for d_q_l in dist_q_logs]
        return cls(batch_query)

    def score(
        self, p_logits: FloatTensor, q_logits: FloatTensor, n_workers=None
    ) -> FloatTensor:
        """dim -2 in result is for different queries, dim -1 is for different tokens"""
        q_logits = F.log_softmax(q_logits, dim=-1).unsqueeze(-2)
        p_logits = F.log_softmax(p_logits, dim=-1).unsqueeze(-2)
        dist_p_logs = torch.tensor(
            [dist_p_log for dist_p_log, dist_q_log in self.batch_query],
            device=p_logits.device,
            dtype=p_logits.dtype,
        ).unsqueeze(-1)
        dist_q_logs = torch.tensor(
            [dist_q_log for dist_p_log, dist_q_log in self.batch_query],
            device=p_logits.device,
            dtype=p_logits.dtype,
        ).unsqueeze(-1)

        max_llr = get_max_llr(p_logits, q_logits, dist_p_logs, dist_q_logs)
        min_llr = -get_max_llr(q_logits, p_logits, dist_q_logs, dist_p_logs)
        trivial_pos = max_llr < min_llr
        llr = q_logits - p_logits
        r_llr = torch.where(
            trivial_pos, torch.tensor(0.0), torch.clamp(llr, min_llr, max_llr)
        )
        return r_llr


def get_max_llr(
    # shape = (..., 1, vocab_size)
    p_logits: FloatTensor,
    q_logits: FloatTensor,
    # shape = (query_size, 1)
    dist_p_logs: FloatTensor,
    dist_q_logs: FloatTensor,
):
    # shape = (..., 1, vocab_size)
    llr = q_logits - p_logits
    # shape = (..., 1, vocab_size)
    sort_index = torch.argsort(llr, dim=-1, descending=True)

    p_logits = p_logits.gather(-1, sort_index)
    q_logits = q_logits.gather(-1, sort_index)

    # shape = (..., 1, vocab_size)
    sum_q_logits = torch.logcumsumexp(q_logits, dim=-1)
    sum_p_logits = torch.logcumsumexp(p_logits, dim=-1)

    # shape = (..., query_size, vocab_size)
    modified_q_logits = torch.where(
        sum_q_logits <= dist_q_logs,
        torch.tensor(float("-inf"), device=q_logits.device, dtype=q_logits.dtype),
        sum_q_logits + torch.log(-torch.expm1(dist_q_logs - sum_q_logits)),
    )
    modified_p_logits = torch.logaddexp(sum_p_logits, dist_p_logs)

    # shape = (..., 1, vocab_size)
    llr = q_logits - p_logits
    # shape = (..., query_size, vocab_size)
    modified_llr = modified_q_logits - modified_p_logits

    # pad left modified_llr with -inf
    # shape = (..., query_size, vocab_size)
    modified_llr = F.pad(modified_llr[..., :-1], (1, 0), value=float("-inf"))
    #  get index by argmax
    # shape = (..., query_size)
    cut_index = torch.argmax((llr < modified_llr).to(torch.int), dim=-1)
    # shape = (..., query_size, 1)
    return modified_llr.gather(-1, cut_index.unsqueeze(-1))
