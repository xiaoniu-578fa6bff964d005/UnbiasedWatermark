#!/usr/bin/env python
# -*- coding: utf-8 -*-

from unbiased_watermark import (
    Delta_Reweight,
    Gamma_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)
from .lm_watermarking.watermark_processor import (
    WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
)


def get_wps():

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

    john_wps = [
        WatermarkLogitsProcessor_John(
            vocab=list(range(10)),  # placeholder
            gamma=0.5,
            delta=delta,
            seeding_scheme="simple_1",
        )
        for delta in [0.0, 1.0, 2.0]
    ]
    return [delta_wp, gamma_wp, *john_wps]


import torch


def get_num_gpus():
    num_gpus = torch.cuda.device_count()
    return num_gpus
