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
    def generation_test(
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
            set_seed(42)
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
        print(
            f"""{{
            "no": {repr(run(logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature)])))},
            "delta": {repr(run(logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature), delta_wp])))},
            "gamma": {repr(run(logits_processor=LogitsProcessorList([TemperatureLogitsWarper(temperature), gamma_wp])))},
        }}"""
        )

    def test_opt_1(self):
        if not torch.cuda.is_available():
            return
        self.generation_test(
            model="facebook/opt-1.3b",
            prompt="Hello, I'm",
            max_length=40,
            device=0,
            temperature=0.4,
        )

    def test_gpt2_1(self):
        self.generation_test(model="gpt2", prompt="list(range(10))=[0,1,2")

    def test_code_gpt2_1(self):
        self.generation_test(
            model="shibing624/code-autocomplete-gpt2-base",
            prompt="list(range(10))=[0,1,2",
        )

    def score_test(self, model="gpt2", texts={}, temperature=0.2, prompt="", **kwargs):
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
            get_score,
        )

        generator = pipeline(
            "text-generation",
            model=model,
            **kwargs,
        )
        prompt_len = len(generator.tokenizer.encode(prompt))

        delta_wp = WatermarkLogitsProcessor(
            b"private key",
            Delta_Reweight(),
            PrevN_ContextCodeExtractor(5),
        )
        gamma_wp = WatermarkLogitsProcessor(
            b"private key",
            Gamma_Reweight(0.1),
            PrevN_ContextCodeExtractor(5),
        )
        llr_score = LLR_Score()
        robust_llr_score = RobustLLR_Score(0.1, 0.1)
        for k in texts:
            print(f"==={k}===")
            for text in texts[k]:
                print("Text: ", text)
                for wp_name, wp in [("delta", delta_wp), ("gamma", gamma_wp)]:
                    for score_name, score in [
                        ("llr", llr_score),
                        ("r_llr", robust_llr_score),
                    ]:
                        scores = get_score(
                            text,
                            wp,
                            score,
                            generator.model,
                            generator.tokenizer,
                            temperature=temperature,
                        )
                        wp.reset_history()
                        sum_score = sum(scores[max(1, prompt_len) :])
                        print(f"{wp_name}\t{score_name}\t{sum_score}")
                        #  if k == "delta" and wp_name == "delta":
                        #      print("scores: ", scores)

    def test_gpt2_2(self):
        texts = {"no": ["list(range(10))=[0,1,2,3]"]}
        self.score_test(model="gpt2", texts=texts)

    def test_opt_2(self):
        texts = {
            "no": [
                "Hello, I'm interested in the following:  * Moon HA Mudkip * Moon HA Totodile * Moon HA Larvitar * Moon HA Mudkip * Moon HA Timid HA",
                "Hello, I'm interested in the following:  - Lure Ball HA Chimchar (Bold, 4EMs) - Lure Ball HA Gible (Bold, 4EMs)",
                "Hello, I'm a new player and I've been playing for about a week. I've been playing on the same server for a while but I'm just starting to get into it. I'm",
                "Hello, I'm new to this sub. I think I'm going to post a lot of my work here. I'm a graphic designer and I'm looking for a job. I'm not sure",
                "Hello, I'm interested in the following:  * Moon Ball HA Dratini * Moon Ball HA Dratini * Moon Ball HA Tepig * Moon Ball HA Lileep * Moon",
            ],
            "delta": [
                "Hello, I'm a new player, I'm currently in the process of making a character, I have a level 11 wizard, I'm trying to decide between a wizard and a cleric, I've",
                "Hello, I'm interested in your HA Torchic and Friend Ball Scyther. I have a DBHA Dratini, DBHA Porygon, DBHA Omanyte, DBHA O",
                "Hello, I'm a new player and I have a question. I am currently farming the last stage of the event, and I have a question. I'm trying to get the golden ticket, but",
                "Hello, I'm new here. I'm a bit confused. I've been on this sub for a while now, but I don't understand what's going on.\nThis is a sub for",
                "Hello, I'm interested in the Dior lipsticks. How much are you asking for them?\nHi! I'm not sure what the current price is on them, but I'd be happy",
            ],
            "gamma": [
                "Hello, I'm interested in the following:  * HA Scatterbug * HA Treecko * HA Vulpix * HA Turtwig * HA Mudkip * HA Mudkip *",
                "Hello, I'm interested in the following:  -  * DBHA Lileep  - DBHA Snivy  - DBHA Tangela  - DBHA Vulpix ",
                "Hello, I'm a new player to the game and I'm looking to join a guild. I have a level 28 priest, and I'm looking to join a guild that is active and has a",
                "Hello, I'm new to the game and I think I'm going to buy the game for the first time. I'm looking for a guild to join. I'm a healer and I'm looking",
                "Hello, I'm interested in the following:  * Moon Ball HA Vulpix * Moon Ball HA Dratini * Moon Ball HA Corphish * Moon Ball HA Lileep * Moon",
            ],
        }
        self.score_test(
            model="facebook/opt-1.3b",
            texts=texts,
            prompt="Hello, I'm",
            device=0,
        )
