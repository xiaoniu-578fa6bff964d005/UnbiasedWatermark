#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random

from . import AbstractScore


def get_max_lr(
    p: np.ndarray, q: np.ndarray, dist_p: float, dist_q: float
) -> tuple[float, set]:
    arctan_lr = [
        (i, np.arctan2(q[i], p[i])) for i in range(len(p))
    ]  # arctan of likelihood ratio with index
    arctan_lr.sort(key=lambda x: x[1], reverse=True)
    max_set = set()

    def lowest_lr():
        sum_q = max(sum([q[j] for j in max_set]) - dist_q, 0)
        sum_p = sum([p[j] for j in max_set]) + dist_p
        if sum_q == 0:
            return 0
        else:
            return sum_q / sum_p

    for i in range(len(p)):
        if max_set:
            if arctan_lr[i][1] < np.arctan(lowest_lr()):
                break
        max_set.add(arctan_lr[i][0])
    return lowest_lr(), max_set


def get_min_lr(
    p: np.ndarray, q: np.ndarray, dist_p: float, dist_q: float
) -> tuple[float, set]:
    lowest_inv_lr, min_set = get_max_lr(q, p, dist_q, dist_p)
    return 1 / lowest_inv_lr, min_set


#  def get_max_lr(
#      p: np.ndarray, q: np.ndarray, dist_p: float, dist_q: float
#  ) -> tuple[float, set]:
#      lr = [(i, q[i] / p[i]) for i in range(len(p))]  # likelihood ratio with index
#      lr.sort(key=lambda x: x[1], reverse=True)
#      max_set = set()
#
#      def lowest_lr():
#          sum_q = max(sum([q[j] for j in max_set]) - dist_q, 0)
#          sum_p = sum([p[j] for j in max_set]) + dist_p
#          with np.errstate(divide="ignore"):
#              return sum_q / sum_p
#
#      for i in range(len(p)):
#          if max_set:
#              if lr[i][1] < lowest_lr():
#                  break
#          max_set.add(lr[i][0])
#      return lowest_lr(), max_set
#
#
#  def get_min_lr(
#      p: np.ndarray, q: np.ndarray, dist_p: float, dist_q: float
#  ) -> tuple[float, set]:
#      lr = [(i, q[i] / p[i]) for i in range(len(p))]  # likelihood ratio with index
#      lr.sort(key=lambda x: x[1])
#      min_set = set()
#
#      def highest_lr():
#          sum_q = sum([q[j] for j in min_set]) + dist_q
#          sum_p = max(sum([p[j] for j in min_set]) - dist_p, 0)
#          with np.errstate(divide="ignore"):
#              print(sum_q, sum_p)
#              print(lr)
#              return sum_q / sum_p
#
#      for i in range(len(p)):
#          if min_set:
#              if lr[i][1] > highest_lr():
#                  break
#          min_set.add(lr[i][0])
#      return highest_lr(), min_set


class RobustLLR_Score(AbstractScore):
    def __init__(self, dist_p: float, dist_q: float):
        self.dist_p = dist_p
        self.dist_q = dist_q

    def score(self, p: np.ndarray, q: np.ndarray) -> np.ndarray:
        with np.errstate(divide="ignore"):
            lr = np.divide(
                q, p, out=np.ones_like(p), where=np.logical_or(p != 0, q != 0)
            )
            max_lr, max_set = get_max_lr(p, q, self.dist_p, self.dist_q)
            min_lr, min_set = get_min_lr(p, q, self.dist_p, self.dist_q)
            if max_set.intersection(min_set) or max_lr <= min_lr:
                return np.zeros_like(p)
            lr = np.clip(lr, min_lr, max_lr)
            return np.log(lr)
