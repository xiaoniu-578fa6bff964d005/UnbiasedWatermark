# Unbiased Watermark - Adding watermark to language model without sacrificing performance

## Library Usage

```
from unbiased_watermark import (
    Delta_Reweight,
    Gamma_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)

delta_wp = WatermarkLogitsProcessor(
    b"private key",
    Delta_Reweight(),
    PrevN_ContextCodeExtractor(5),
)
gamma_wp = WatermarkLogitsProcessor(
    b"private key",
    Gamma_Reweight(),
    PrevN_ContextCodeExtractor(5),
)

from unbiased_watermark import patch_model

# current generation() doesn't accept logits_warper parameter.
# let's fix it
patch_model(model)
output_ids = model.generate(
    input_ids,
    max_length=128,
    do_sample=True,
    num_beams=1,
    top_k=0,
    temperature=temperature,
    logits_warper=LogitsProcessorList([delta_wp]), # or gamma_wp
)
```

# Run experiment

```
pip install -e .
python experiments.py
```
