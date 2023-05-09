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


def process_in_ds(ds):
    ds = ds.flatten()

    def add_id(example):
        import hashlib

        if isinstance(example["translation.en"], list):
            example["id"] = [
                hashlib.sha1(x.encode("utf-8")).hexdigest()
                for x in example["translation.en"]
            ]
        elif isinstance(example["translation.en"], str):
            example["id"] = hashlib.sha1(
                example["transslation.en"].encode("utf-8")
            ).hexdigest()
        return example

    ds = ds.map(add_id, batched=True)
    return ds


def machine_translation_exp_worker(tq, rq, batch_size=8):
    from datasets import load_dataset

    wmt17 = load_dataset("wmt17", "de-en").shuffle(seed=42)

    from .common import get_wps

    wps = get_wps()

    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    ds = wmt17["test"]
    ds = process_in_ds(ds)

    #  ds = ds.shard(num_shards=200, index=0)
    #  print("ds len:", len(ds))
    ds = ds.map(group_batch, batched=True, batch_size=batch_size)
    from tqdm import tqdm

    for batch in tqdm(ds):
        tq.put({"batch": batch, "watermark_processor": None})
        for wp in wps:
            tq.put({"batch": batch, "watermark_processor": wp})


def machine_translation_store_worker(rq, rqe):
    import json
    import os

    if not os.path.exists("data"):
        os.makedirs("data")
    with open("data/machine_translation.txt", "w") as f:
        while not (rqe.is_set() and rq.empty()):
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            f.write(json.dumps(result))
            f.write("\n")
            f.flush()


# Process the test dataset in batches to generate summaries
def preprocess_data(example, tokenizer):
    source = example["translation.en"]
    target = example["translation.de"]

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
        max_length=512,
        truncation=True,
        padding="max_length",
        return_tensors="pt",
    )

    return {
        "inputs": source_encoding["input_ids"],
        "labels": target_encoding["input_ids"],
    }


def machine_translation_gpu_worker(
    tq, tqe, rq, gpu_id=0, model_str="Helsinki-NLP/opus-mt-en-de", temperature=1.0
):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from unbiased_watermark import patch_model

    model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    while not (tqe.is_set() and tq.empty()):
        try:
            task = tq.get(timeout=1)
        except Empty as e:
            continue
        batch = task["batch"]
        tbatch = preprocess_data(batch, tokenizer)
        wp = task["watermark_processor"]
        lps = []
        if wp is not None:
            if "reset_history" in dir(wp):
                wp.reset_history()
            if "vocab_size" in dir(wp):
                wp.vocab_size = model.config.vocab_size
            lps.append(wp)

        set_seed(42)
        batch_summaries = model.generate(
            torch.Tensor(tbatch["inputs"]).to(device=model.device).long(),
            max_length=512,
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


def machine_translation_map():
    from .common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()

    exp_worker = Process(
        target=machine_translation_exp_worker,
        args=(tq, rq),
        kwargs={
            "batch_size": 64
            #  "batch_size": 1
        },
    )
    gpu_workers = [
        Process(target=machine_translation_gpu_worker, args=(tq, tqe, rq, i))
        for i in range(num_gpus)
    ]
    store_worker = Process(target=machine_translation_store_worker, args=(rq, rqe))

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


def machine_translation_evaluate():
    import numpy as np

    from datasets import load_dataset

    in_ds = load_dataset("wmt17", "de-en").shuffle(seed=42)["test"]
    in_ds = process_in_ds(in_ds)
    in_ds = in_ds.sort("id")

    out_ds = load_dataset("json", data_files={"test": "data/machine_translation.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")
    wp_types = set(out_ds["watermark_processor"])

    import evaluate

    bertscore = evaluate.load("bertscore")
    result = {}
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        bertscores = bertscore.compute(
            predictions=s_out_ds["decode"],
            references=in_ds["translation.de"],
            lang="de",
            rescale_with_baseline=True,
        )
        result[wp_type] = bertscores
    import json

    json.dump(result, open("data/machine_translation_bertscore.json", "w"))
