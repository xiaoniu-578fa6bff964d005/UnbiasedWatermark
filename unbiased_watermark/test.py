#!/usr/bin/env python
# -*- coding: utf-8 -*-


import torch
from torch.nn import functional as F

from . import *

import unittest


class Delta_Test(unittest.TestCase):
    def test_1(self):
        from . import LLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [float("-inf"), float("-inf"), float("-inf"), math.log(2.5)]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_1_batch(self):
        from . import LLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                float("-inf"),
                                float("-inf"),
                                float("-inf"),
                                math.log(2.5),
                            ],
                            [
                                float("-inf"),
                                math.log(5),
                                float("-inf"),
                                float("-inf"),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_2(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [float("-inf"), float("-inf"), float("-inf"), math.log(2.5)]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_2_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                float("-inf"),
                                float("-inf"),
                                float("-inf"),
                                math.log(2.5),
                            ],
                            [
                                float("-inf"),
                                math.log(5),
                                float("-inf"),
                                float("-inf"),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_2_2(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.tensor([-1e3, math.log(0.3), math.log(0.3), math.log(0.4)])
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [float("-inf"), float("-inf"), float("-inf"), math.log(2.5)]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_2_2_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        p_logits = torch.tensor(
            [
                [-1e3, math.log(0.3), math.log(0.3), math.log(0.4)],
                [-1e3, math.log(0.3), math.log(0.3), math.log(0.4)],
            ]
        )
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0, 0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                float("-inf"),
                                float("-inf"),
                                float("-inf"),
                                math.log(2.5),
                            ],
                            [
                                float("-inf"),
                                math.log(10 / 3),
                                float("-inf"),
                                float("-inf"),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_3(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0, 0.1)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            math.log(1 / 6),
                            math.log(1 / 6),
                            math.log(1 / 6),
                            math.log(9 / 4),
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_3_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0, 0.1)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                math.log(1 / 6),
                                math.log(1 / 6),
                                math.log(1 / 6),
                                math.log(9 / 4),
                            ],
                            [
                                math.log(1 / 8),
                                math.log(9 / 2),
                                math.log(1 / 8),
                                math.log(1 / 8),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_4(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0.1, 0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            float("-inf"),
                            float("-inf"),
                            float("-inf"),
                            math.log(2),
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_4_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0.1, 0.0)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                float("-inf"),
                                float("-inf"),
                                float("-inf"),
                                math.log(2),
                            ],
                            [
                                float("-inf"),
                                math.log(10 / 3),
                                float("-inf"),
                                float("-inf"),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_5(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0.1, 0.1)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            math.log(1 / 5),
                            math.log(1 / 5),
                            math.log(1 / 5),
                            math.log(9 / 5),
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_5_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0.1, 0.1)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                math.log(1 / 5),
                                math.log(1 / 5),
                                math.log(1 / 5),
                                math.log(9 / 5),
                            ],
                            [
                                math.log(1 / 7),
                                math.log(3),
                                math.log(1 / 7),
                                math.log(1 / 7),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )

    def test_6(self):
        from . import RobustLLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.u, 0.8302518725395203)
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor([0, 0, 0, 1], dtype=torch.float32),
            )
        )
        score = RobustLLR_Score(0.5, 0.5)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor([0.0, 0, 0, 0]),
                    -100,
                    100,
                ),
            )
        )

    def test_6_batch(self):
        from . import RobustLLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(11)
        code = Delta_WatermarkCode.from_random(rng, 4)
        self.assertTrue(
            torch.allclose(
                code.u, torch.tensor([0.8302518725395203, 0.14904171228408813])
            )
        )
        reweight = Delta_Reweight()
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.softmax(q_logits, dim=-1),
                torch.tensor(
                    [[0, 0, 0, 1], [0, 1, 0, 0]],
                    dtype=torch.float32,
                ),
            )
        )
        score = RobustLLR_Score(0.5, 0.5)
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [0.0, 0, 0, 0],
                            [0.0, 0, 0, 0],
                        ]
                    ),
                    -100,
                    100,
                ),
            )
        )


class Gamma_Test(unittest.TestCase):
    def test_1(self):
        from . import LLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [3, 1, 0, 2])
        reweight = Gamma_Reweight(0)
        p_logits = torch.log(torch.tensor([0.1, 0.2, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.log_softmax(q_logits, dim=-1), F.log_softmax(p_logits, dim=-1)
            )
        )
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [0.0, 0, 0, 0],
                    ),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )

    def test_1_batch(self):
        from . import LLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(1)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [[3, 1, 0, 2], [1, 3, 2, 0]])
        reweight = Gamma_Reweight(0)
        p_logits = torch.log(torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        self.assertTrue(
            torch.allclose(
                F.log_softmax(q_logits, dim=-1), F.log_softmax(p_logits, dim=-1)
            )
        )
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor([[0.0, 0, 0, 0], [0.0, 0, 0, 0]]),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )

    #  def test_2(self):
    #      from . import LLR_Score
    #
    #      rng = random.Random(0)
    #      code = Gamma_WatermarkCode.from_random(rng, 4)
    #      self.assertEqual(code.shuffle.tolist(), [2, 0, 1, 3])
    #      reweight = Gamma_Reweight(0.5)
    #      p = np.array([0.1, 0.2, 0.4, 0.3])
    #      q = reweight.reweight(code, p)
    #      score = LLR_Score()
    #      self.assertTrue(
    #          np.allclose(score.score(p, q), np.log([1 / 2, 3 / 2, 1 / 2, 3 / 2]))
    #      )
    def test_2(self):
        from . import LLR_Score
        import math

        rng = torch.Generator()
        rng.manual_seed(5)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [3, 1, 0, 2])
        reweight = Gamma_Reweight(0.5)
        p_logits = torch.log(torch.tensor([0.2, 0.1, 0.3, 0.4]))
        q_logits = reweight.reweight_logits(code, p_logits)
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            math.log(3 / 2),
                            math.log(1 / 2),
                            math.log(3 / 2),
                            math.log(1 / 2),
                        ],
                    ),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )

    def test_2_batch(self):
        from . import LLR_Score
        import math

        rng = [torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(1)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(code.shuffle.tolist(), [[3, 1, 0, 2], [1, 3, 2, 0]])
        reweight = Gamma_Reweight(0.5)
        p_logits = torch.log(torch.tensor([[0.2, 0.1, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4]]))
        q_logits = reweight.reweight_logits(code, p_logits)
        score = LLR_Score()
        self.assertTrue(
            torch.allclose(
                torch.clamp(score.score(p_logits, q_logits), -100, 100),
                torch.clamp(
                    torch.tensor(
                        [
                            [
                                math.log(3 / 2),
                                math.log(1 / 2),
                                math.log(3 / 2),
                                math.log(1 / 2),
                            ],
                            [
                                math.log(3 / 2),
                                math.log(1 / 2),
                                math.log(3 / 2),
                                math.log(1 / 2),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )


class GPT2_Test(unittest.TestCase):
    def test_1(self):
        from transformers import pipeline, set_seed, LogitsProcessorList

        generator = pipeline("text-generation", model="gpt2", do_sample=True)
        prompt = "Count to 100: 1,2,3,4,5,6"

        set_seed(42)
        from . import (
            WatermarkLogitsProcessor,
            Delta_Reweight,
            PrevN_ContextCodeExtractor,
        )

        #  cs = generator(prompt, max_length=30, num_return_sequences=5, do_sample=True)
        cs = generator(prompt, max_length=30, num_return_sequences=5)
        for c in cs:
            print(c["generated_text"])
        print("=====")
        watermark_processor = WatermarkLogitsProcessor(
            b"private key",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
        set_seed(42)
        cs = generator(
            prompt,
            max_length=30,
            num_return_sequences=5,
            do_sample=True,
            logits_processor=LogitsProcessorList([watermark_processor]),
        )
        for c in cs:
            print(c["generated_text"])
        print("=====")
        watermark_processor = WatermarkLogitsProcessor(
            b"private key",
            Gamma_Reweight(0.1),
            PrevN_ContextCodeExtractor(5),
        )
        set_seed(42)
        cs = generator(
            prompt,
            max_length=30,
            num_return_sequences=5,
            do_sample=True,
            logits_processor=LogitsProcessorList([watermark_processor]),
        )
        for c in cs:
            print(c["generated_text"])


class OPT_Test(unittest.TestCase):
    def test_1_3b(self):
        from transformers import pipeline, set_seed, LogitsProcessorList

        generator = pipeline(
            "text-generation", model="facebook/opt-1.3b", do_sample=True
        )
        prompt = "Count to 100: 1,2,3,4,5,6"

        set_seed(42)
        from . import (
            WatermarkLogitsProcessor,
            Delta_Reweight,
            PrevN_ContextCodeExtractor,
        )

        cs = generator(prompt, max_length=30, num_return_sequences=5, do_sample=True)
        for c in cs:
            print(c["generated_text"])
        print("=====")
        watermark_processor = WatermarkLogitsProcessor(
            b"private key",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
        set_seed(42)
        cs = generator(
            prompt,
            max_length=30,
            num_return_sequences=5,
            do_sample=True,
            logits_processor=LogitsProcessorList([watermark_processor]),
        )
        for c in cs:
            print(c["generated_text"])
        print("=====")
        watermark_processor = WatermarkLogitsProcessor(
            b"private key",
            Gamma_Reweight(0.1),
            PrevN_ContextCodeExtractor(5),
        )
        set_seed(42)
        cs = generator(
            prompt,
            max_length=30,
            num_return_sequences=5,
            do_sample=True,
            logits_processor=LogitsProcessorList([watermark_processor]),
        )
        for c in cs:
            print(c["generated_text"])
