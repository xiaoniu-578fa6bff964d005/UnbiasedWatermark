#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.multiprocessing as mp
from torch.multiprocessing import Process, Queue, Event
from queue import Empty
from torch.multiprocessing import set_start_method
import json
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
from transformers import LogitsProcessorList, TemperatureLogitsWarper
from unbiased_watermark import patch_model

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


def machine_translation_exp_worker(tq, rq, batch_size=8):
    from datasets import load_dataset

    cnn_daily = load_dataset("wmt17", "de-en").shuffle(seed=42)

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

    def group_batch(batch):
        return {k: [v] for k, v in batch.items()}

    ds = cnn_daily["test"]
    en_column = []
    de_column = []
    ids = []

    for i in range(len(ds)):
        en = ds[i]["translation"]["en"]
        de = ds[i]["translation"]["de"]
        ids.append(i)
        en_column.append(en)
        de_column.append(de)
    
    ds = ds.add_column("en", en_column)
    ds = ds.add_column("de", de_column)
    ds = ds.add_column("id", ids)
    
    #  ds = ds.shard(num_shards=200, index=0)
    #  print("ds len:", len(ds))
    ds = ds.map(group_batch, batched=True, batch_size=batch_size)
    from tqdm import tqdm

    for batch in tqdm(ds):
        tq.put({"batch": batch, "watermark_processor": None})
        tq.put({"batch": batch, "watermark_processor": delta_wp})
        tq.put({"batch": batch, "watermark_processor": gamma_wp})


def machine_translation_store_worker(rq, rqe):
    import json

    with open("data/machine_translation.txt", "w") as f:
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
    source = example["en"]
    target = example["de"]

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
        # import pdb; pdb.set_trace()
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
    num_gpus = torch.cuda.device_count()

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
    en_column = []
    de_column = []
    ids = []

    for i in range(len(in_ds)):
        en = in_ds[i]["translation"]["en"]
        de = in_ds[i]["translation"]["de"]
        ids.append(i)
        en_column.append(en)
        de_column.append(de)
    
    in_ds = in_ds.add_column("en", en_column)
    in_ds = in_ds.add_column("de", de_column)
    in_ds = in_ds.add_column("id", ids)

    out_ds = load_dataset("json", data_files={"test": "data/machine_translation.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")
    wp_types = set(out_ds["watermark_processor"])
    print(wp_types)

    bleu = evaluate.load("sacrebleu")
    result = {}
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        import pdb; pdb.set_trace()
        bleu_scores = bleu.compute(
            predictions=s_out_ds["decode"],
            references=in_ds["de"],
        )
        result[wp_type] = bleu_scores['score']

    json.dump(result, open("data/machine_translation_bleu.json", "w"))


if __name__ == "__main__":
    # machine_translation_map()
    machine_translation_evaluate()
