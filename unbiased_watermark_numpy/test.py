#!/usr/bin/env python
# -*- coding: utf-8 -*-


import numpy as np
import random

from . import *

import unittest


class Delta_Test(unittest.TestCase):
    def test_1(self):
        from . import LLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = LLR_Score()
        self.assertTrue(
            np.allclose(
                score.score(p, q), np.array([-np.inf, -np.inf, -np.inf, np.log(2.5)])
            )
        )

    def test_2(self):
        from . import RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            np.allclose(
                score.score(p, q), np.array([-np.inf, -np.inf, -np.inf, np.log(2.5)])
            )
        )

    def test_2_2(self):
        from . import LLR_Score, RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.0, 0.3, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            np.allclose(score.score(p, q), np.array([0, -np.inf, -np.inf, np.log(2.5)]))
        )
        score = LLR_Score()
        self.assertTrue(
            np.allclose(score.score(p, q), np.array([0, -np.inf, -np.inf, np.log(2.5)]))
        )

    def test_3(self):
        from . import RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0, 0.1)
        self.assertTrue(
            np.allclose(
                score.score(p, q),
                np.array([np.log(1 / 6), np.log(1 / 6), np.log(1 / 6), np.log(9 / 4)]),
            )
        )

    def test_4(self):
        from . import RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0.1, 0.0)
        self.assertTrue(
            np.allclose(
                score.score(p, q), np.array([-np.inf, -np.inf, -np.inf, np.log(2)])
            )
        )

    def test_5(self):
        from . import RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0.1, 0.1)
        self.assertTrue(
            np.allclose(
                score.score(p, q),
                np.array([np.log(1 / 5), np.log(1 / 5), np.log(1 / 5), np.log(9 / 5)]),
            )
        )

    def test_6(self):
        from . import RobustLLR_Score

        rng = random.Random(0)
        code = Delta_WatermarkCode.from_random(rng)
        self.assertEqual(code.u, 0.8444218515250481)
        reweight = Delta_Reweight()
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, np.array([0, 0, 0, 1])))
        score = RobustLLR_Score(0.5, 0.5)
        self.assertTrue(
            np.allclose(
                score.score(p, q),
                np.array([0, 0, 0, 0]),
            )
        )


class Gamma_Test(unittest.TestCase):
    def test_1(self):
        from . import LLR_Score

        rng = random.Random(0)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [2, 0, 1, 3])
        reweight = Gamma_Reweight(0)
        p = np.array([0.1, 0.2, 0.3, 0.4])
        q = reweight.reweight(code, p)
        self.assertTrue(np.allclose(q, p))
        score = LLR_Score()
        self.assertTrue(np.allclose(score.score(p, q), np.array([0, 0, 0, 0])))

    def test_2(self):
        from . import LLR_Score

        rng = random.Random(0)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [2, 0, 1, 3])
        reweight = Gamma_Reweight(0.5)
        p = np.array([0.1, 0.2, 0.4, 0.3])
        q = reweight.reweight(code, p)
        score = LLR_Score()
        self.assertTrue(
            np.allclose(score.score(p, q), np.log([1 / 2, 3 / 2, 1 / 2, 3 / 2]))
        )
