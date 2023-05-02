# The metric is BLEU score
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import evaluate

metric = evaluate.load("sacrebleu")
wmt = load_dataset("wmt17", "de-en")
wmt = wmt["test"]

tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-de")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-de").cuda()

en_column = []
de_column = []

for i in range(len(wmt)):
    en = wmt[i]["translation"]["en"]
    de = wmt[i]["translation"]["de"]
    en_column.append(en)
    de_column.append(de)

wmt = wmt.add_column("en", en_column)
wmt = wmt.add_column("de", de_column)


def preprocess_function(example):
    source = example["en"]
    # Encode the source and target text
    source_encoding = tokenizer(
        source,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    return {"inputs": source_encoding["input_ids"]}


tokenized_wmt = wmt.map(preprocess_function, batched=True)

batch_size = 8
num_batches = len(tokenized_wmt) // batch_size + 1
generated_translations = []

from tqdm import tqdm
from unbiased_watermark import (
    Delta_Reweight,
    Gamma_Reweight,
    WatermarkLogitsProcessor,
    PrevN_ContextCodeExtractor,
)
from transformers import (
    LogitsProcessorList,
    TemperatureLogitsWarper,
    set_seed,
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
temperature = 1.0

set_seed(42)
for i in tqdm(range(num_batches)):
    batch_start = i * batch_size
    batch_end = (i + 1) * batch_size
    if batch_end < len(tokenized_wmt):
        batch = tokenized_wmt[batch_start:batch_end]
    else:
        batch = tokenized_wmt[batch_start:]
    batch_translation = model.generate(
        torch.Tensor(batch["inputs"]).to(device=model.device).long(),
        max_length=512,
        do_sample=True,
        num_beams=1,
        logits_processor=LogitsProcessorList(
            #  [TemperatureLogitsWarper(temperature)]
            [TemperatureLogitsWarper(temperature), delta_wp]
            #  [TemperatureLogitsWarper(temperature), gamma_wp]
        ),
    )
    decodes = tokenizer.batch_decode(batch_translation, skip_special_tokens=True)
    # The result is for debugging use
    # result = metric.compute(predictions=decodes, references=batch['de'])
    generated_translations.extend(decodes)

# Calculate the ROUGE scores
bleu_scores = metric.compute(
    predictions=generated_translations, references=tokenized_wmt["de"]
)

print("BLEU: ")
print(bleu_scores["score"])