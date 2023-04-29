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

        rng = [torch.Generator(), torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(1)
        rng[2].manual_seed(2)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(
            code.shuffle.tolist(), [[3, 1, 0, 2], [1, 3, 2, 0], [0, 1, 3, 2]]
        )
        reweight = Gamma_Reweight(0)
        p_logits = torch.log(
            torch.tensor(
                [[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
            )
        )
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
                    torch.tensor([[0.0, 0, 0, 0], [0.0, 0, 0, 0], [0.0, 0, 0, 0]]),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )

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

        rng = [torch.Generator(), torch.Generator(), torch.Generator()]
        rng[0].manual_seed(5)
        rng[1].manual_seed(1)
        rng[2].manual_seed(2)
        code = Gamma_WatermarkCode.from_random(rng, 4)
        self.assertEqual(
            code.shuffle.tolist(), [[3, 1, 0, 2], [1, 3, 2, 0], [0, 1, 3, 2]]
        )
        reweight = Gamma_Reweight(0.5)
        p_logits = torch.log(
            torch.tensor(
                [[0.2, 0.1, 0.3, 0.4], [0.2, 0.1, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]]
            )
        )
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
                            [
                                math.log(1 / 2),
                                math.log(1 / 2),
                                math.log(3 / 2),
                                math.log(1),
                            ],
                        ]
                    ),
                    -100,
                    100,
                ),
                atol=1e-5,
            )
        )


class LLM_Test(unittest.TestCase):
    def generation(
        self,
        model="gpt2",
        seed=42,
        prompt="list(range(10))=[0,1,2",
        temperature=0.2,
        max_length=20,
        num_return_sequences=5,
        **kwargs,
    ):
        from transformers import (
            pipeline,
            set_seed,
            LogitsProcessorList,
            AutoTokenizer,
            TemperatureLogitsWarper,
        )
        from . import (
            WatermarkLogitsProcessor,
            Delta_Reweight,
            PrevN_ContextCodeExtractor,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        generator = pipeline(
            "text-generation",
            model=model,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            **kwargs,
        )

        def run(**kwargs):
            set_seed(seed)
            results = generator(
                prompt,
                **kwargs,
            )
            return [r["generated_text"] for r in results]

        delta_wp = WatermarkLogitsProcessor(
            b"private key",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
        gamma_wp = WatermarkLogitsProcessor(
            b"private key",
            Gamma_Reweight(1),
            PrevN_ContextCodeExtractor(5),
        )
        result = {
            "no": run(
                logits_processor=LogitsProcessorList(
                    [TemperatureLogitsWarper(temperature)]
                )
            ),
            "delta": run(
                logits_processor=LogitsProcessorList(
                    [TemperatureLogitsWarper(temperature), delta_wp]
                )
            ),
            "gamma": run(
                logits_processor=LogitsProcessorList(
                    [TemperatureLogitsWarper(temperature), gamma_wp]
                )
            ),
        }
        return result

    def test_opt_1(self):
        if not torch.cuda.is_available():
            return
        result = self.generation(
            model="facebook/opt-1.3b",
            prompt="Hello, I'm",
            max_length=40,
            device=0,
            num_return_sequences=1,
            temperature=0.4,
        )
        print(repr(result))

    def test_gpt2_1(self):
        result = self.generation(
            model="gpt2",
            prompt="Hello, I'm",
            max_length=40,
            num_return_sequences=1,
            temperature=0.4,
        )
        print(repr(result))

    def test_code_gpt2_1(self):
        result = self.generation(
            model="shibing624/code-autocomplete-gpt2-base",
            prompt="list(range(10))=[0,1,2",
        )
        print(repr(result))

    def show_score(self, model="gpt2", texts={}, temperature=0.2, prompt="", **kwargs):
        from transformers import (
            pipeline,
            LogitsProcessorList,
            AutoTokenizer,
            TemperatureLogitsWarper,
        )
        from . import (
            WatermarkLogitsProcessor,
            Delta_Reweight,
            PrevN_ContextCodeExtractor,
            get_score,
        )

        tokenizer = AutoTokenizer.from_pretrained(model)
        generator = pipeline(
            "text-generation",
            model=model,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            max_length=40,
            num_return_sequences=1,
            **kwargs,
        )

        delta_wp = WatermarkLogitsProcessor(
            b"private key",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
        gamma_wp = WatermarkLogitsProcessor(
            b"private key",
            Gamma_Reweight(1),
            PrevN_ContextCodeExtractor(5),
        )
        llr_score = LLR_Score()
        robust_llr_score = RobustLLR_Score(0.1, 0.1)
        for k in texts:
            print(f"==={k}===")
            for text in texts[k]:
                print("Text: ", text)
                print(f"re_wt\tdetect\tscore")
                for wp_name, wp in [("delta", delta_wp), ("gamma", gamma_wp)]:
                    for score_name, score in [
                        ("llr", llr_score),
                        ("r_llr", robust_llr_score),
                    ]:
                        scores, prompt_len = get_score(
                            text,
                            wp,
                            score,
                            generator.model,
                            generator.tokenizer,
                            temperature=temperature,
                            prompt=prompt,
                        )
                        sum_score = sum(scores[max(1, prompt_len) :])
                        print(f"{wp_name}\t{score_name}\t{sum_score}")

    def generation_test(
        self,
        model="gpt2",
        generation_config={},
        detect_config={},
    ):
        result = self.generation(
            model=model,
            **generation_config,
        )
        print(repr(result))
        self.show_score(model=model, texts=result, **detect_config)

    def test_gpt2_2(self):
        self.generation_test(
            "gpt2",
            generation_config={
                "temperature": 0.4,
                "max_length": 100,
                "num_return_sequences": 1,
                "prompt": "Hello, I'm",
            },
            detect_config={
                "temperature": 0.4,
                "prompt": "Hello, I'm",
            },
        )

    def test_opt_2(self):
        if not torch.cuda.is_available():
            return
        self.generation_test(
            "facebook/opt-1.3b",
            generation_config={
                "temperature": 0.4,
                "max_length": 100,
                "num_return_sequences": 1,
                "prompt": "Hello, I'm",
                "device": 0,
            },
            detect_config={
                "temperature": 0.4,
                "prompt": "Hello, I'm",
                "device": 0,
            },
        )

    def test_gpt2_3(self):
        self.generation_test(
            "gpt2",
            generation_config={
                "temperature": 0.4,
                "max_length": 100,
                "num_return_sequences": 1,
                "prompt": "Hello, I'm",
            },
            detect_config={
                "temperature": 0.3,
                "prompt": "",
            },
        )

    def test_opt_3(self):
        if not torch.cuda.is_available():
            return
        self.generation_test(
            "facebook/opt-1.3b",
            generation_config={
                "temperature": 0.4,
                "max_length": 100,
                "num_return_sequences": 1,
                "prompt": "Hello, I'm",
                "device": 0,
            },
            detect_config={
                "temperature": 0.3,
                "prompt": "",
                "device": 0,
            },
        )

    def test_opt_3_2(self):
        if not torch.cuda.is_available():
            return
        self.generation_test(
            "facebook/opt-1.3b",
            generation_config={
                "temperature": 0.4,
                "max_length": 100,
                "num_return_sequences": 2,
                "prompt": "To maximize parallelism",
                "device": 0,
            },
            detect_config={
                "temperature": 0.3,
                "prompt": "",
                "device": 0,
            },
        )
