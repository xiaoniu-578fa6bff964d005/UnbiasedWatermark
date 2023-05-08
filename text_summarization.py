from datasets import load_dataset, load_metric
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM
import torch

device = "cuda"
# Prepare and tokenize dataset
cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)

# load dataset and model
model = AutoModelForSeq2SeqLM.from_pretrained("philschmid/bart-large-cnn-samsum").cuda()
tokenizer = AutoTokenizer.from_pretrained("philschmid/bart-large-cnn-samsum")

# Process the test dataset in batches to generate summaries
def preprocess_data(example):
    source = example["article"]
    target = example["highlights"]

    # Encode the source and target text
    source_encoding = tokenizer(
        source,
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )
    target_encoding = tokenizer(
        target,
        max_length=128,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "inputs": source_encoding["input_ids"],
        "labels": target_encoding["input_ids"],
    }


# Apply preprocessing to the dataset
tokenized_cnn_daily = cnn_daily["test"].map(preprocess_data, batched=True)

# Setup evaluation
#  rouge = load_metric("rouge")
import evaluate

rouge = evaluate.load("rouge")


batch_size = 8
num_batches = len(tokenized_cnn_daily) // batch_size
num_batches = min(num_batches, 100)
generated_summaries = []

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
    batch = tokenized_cnn_daily[batch_start:batch_end]
    batch_summaries = model.generate(
        torch.Tensor(batch["inputs"]).to(device=model.device).long(),
        max_length=128,
        do_sample=True,
        num_beams=1,
        logits_processor=LogitsProcessorList(
            #  [TemperatureLogitsWarper(temperature)]
            [TemperatureLogitsWarper(temperature), delta_wp]
            #  [TemperatureLogitsWarper(temperature), gamma_wp]
        ),
    )
    decodes = tokenizer.batch_decode(batch_summaries, skip_special_tokens=True)
    # The result is for debugging use
    # result = rouge.compute(predictions=batch['highlights'], references=decodes, use_stemmer=True)
    # import pdb; pdb.set_trace()
    generated_summaries.extend(decodes)

# Calculate the ROUGE scores
rouge_scores = rouge.compute(
    predictions=generated_summaries,
    references=tokenized_cnn_daily["highlights"][: len(generated_summaries)],
    rouge_types=["rouge1", "rouge2", "rougeL"],
    use_stemmer=True,
    use_aggregator=False,
)

print("ROUGE-1: ")
print(np.mean(rouge_scores["rouge1"]))
print("ROUGE-2: ")
print(np.mean(rouge_scores["rouge2"]))
print("ROUGE-L: ")
print(np.mean(rouge_scores["rougeL"]))
