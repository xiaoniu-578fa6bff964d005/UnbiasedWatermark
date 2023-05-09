#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
from queue import Empty
from torch.multiprocessing import set_start_method

try:
    set_start_method("spawn")
except RuntimeError:
    pass

import torch

import sys
sys.path.append("lm-watermarking/")
from watermark_processor import WatermarkLogitsProcessor as WatermarkLogitsProcessor_John

#  central version:
#  load dataset, preprocess
#  load model
#  create watermark
#  for loop
#      generate
#      decode
#  load metric
#  compute metric

# mp version:

#  exp worker:
#  load dataset
#  create watermark
#  put string list and task param to tq

#  store worker:
#  for result in rq:
#      store result

#  gpu workers:
#  load tokenizer
#  load model
#  for item in tq
#      preprocess
#      generate
#      decode


def text_summarization_exp_worker(tq, rq, batch_size=8,model_str="philschmid/bart-large-cnn-samsum"):
    from datasets import load_dataset
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper
    
    model = AutoModelForSeq2SeqLM.from_pretrained(model_str)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)

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
        Gamma_Reweight(1),
        PrevN_ContextCodeExtractor(5),
    )
    
    john_wp = WatermarkLogitsProcessor_John(vocab=list(tokenizer.get_vocab().values()),
                                               gamma=0.25,
                                               delta=2.0,
                                               seeding_scheme="simple_1")

    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    ds = cnn_daily["test"]
    #  ds = ds.shard(num_shards=200, index=0)
    #  print("ds len:", len(ds))
    ds = ds.map(group_batch, batched=True, batch_size=batch_size)
    from tqdm import tqdm

    for batch in tqdm(ds):
        tq.put({"batch": batch, "watermark_processor": john_wp})
        tq.put({"batch": batch, "watermark_processor": None})
        tq.put({"batch": batch, "watermark_processor": delta_wp})
        tq.put({"batch": batch, "watermark_processor": gamma_wp})
        


def text_summarization_store_worker(rq, rqe):
    import json

    with open("data/text_summarization_john.txt", "w") as f:
        while True:
            if rqe.is_set():
                if rq.empty():
                    break
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            f.write(json.dumps(result))
            f.write("\n")
            f.flush()


# Process the test dataset in batches to generate summaries
def preprocess_data(example, tokenizer):
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


def text_summarization_gpu_worker(
    tq, tqe, rq, gpu_id=0, model_str="philschmid/bart-large-cnn-samsum", temperature=1.0
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from unbiased_watermark import patch_model

    model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    while True:
        if tqe.is_set():
            if tq.empty():
                break
        try:
            task = tq.get(timeout=1)
        except Empty as e:
            continue
        batch = task["batch"]
        tbatch = preprocess_data(batch, tokenizer)
        wp = task["watermark_processor"]
        lps = []
        if wp is not None:
            wp.reset_history()
            lps.append(wp)

        set_seed(42)
#         batch_summaries = model.generate(
#             torch.Tensor(tbatch["inputs"]).to(device=model.device).long(),
#             max_length=128,
#             do_sample=True,
#             num_beams=1,
#             top_k=0,
#             temperature=temperature,
#             logits_processor=LogitsProcessorList(lps),
#         )
        batch_summaries = model.generate(
            torch.Tensor(tbatch["inputs"]).to(device=model.device).long(),
            max_length=128,
            do_sample=True,
            num_beams=1,
            top_k=0,
            temperature=temperature,
            logits_warper=LogitsProcessorList(lps),
        )
        decodes = tokenizer.batch_decode(batch_summaries, skip_special_tokens=True)
        wp_str = repr(wp)
        for decode, i in zip(decodes, batch["id"]):
            rq.put({"decode": decode, "id": i, "watermark_processor": wp_str})


def text_summarization_map():
    num_gpus = torch.cuda.device_count()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()

    exp_worker = Process(
        target=text_summarization_exp_worker,
        args=(tq, rq),
        kwargs={
            "batch_size": 128,
            "model_str":"philschmid/bart-large-cnn-samsum"
            #  "batch_size": 1
        },
    )
    gpu_workers = [
        Process(target=text_summarization_gpu_worker, args=(tq, tqe, rq, i))
        for i in range(num_gpus)
    ]
    store_worker = Process(target=text_summarization_store_worker, args=(rq, rqe))

    exp_worker.start()
    for w in gpu_workers:
        w.start()
    store_worker.start()

    exp_worker.join()
    tqe.set()
    for w in gpu_workers:
        w.join()
    rqe.set()
    store_worker.join()


def text_summarization_evaluate():
    import numpy as np

    from datasets import load_dataset

    in_ds = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)["test"]
    in_ds = in_ds.sort("id")
    out_ds = load_dataset("json", data_files={"test": "data/text_summarization_john.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")
    wp_types = set(out_ds["watermark_processor"])
    print(wp_types)

    import evaluate

    rouge = evaluate.load("rouge")
    result = {}
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        rouge_scores = rouge.compute(
            predictions=s_out_ds["decode"],
            references=in_ds["highlights"],
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
            use_aggregator=False,
        )
        result[wp_type] = rouge_scores
    import json

    json.dump(result, open("data/text_summarization_rouge_john.json", "w"))


if __name__ == "__main__":
    text_summarization_map()
    text_summarization_evaluate()