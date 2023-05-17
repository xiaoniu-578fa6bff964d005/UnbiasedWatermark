def get_wps():
    from unbiased_watermark import (
        Delta_Reweight,
        Gamma_Reweight,
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
    )

    import random

    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")
    delta_wp = WatermarkLogitsProcessor(
        private_key,
        Delta_Reweight(),
        PrevN_ContextCodeExtractor(5),
    )
    gamma_wp = WatermarkLogitsProcessor(
        private_key,
        Gamma_Reweight(),
        PrevN_ContextCodeExtractor(5),
    )
    import copy

    delta_wp_woh = copy.deepcopy(delta_wp)
    delta_wp_woh.ignore_history = True
    gamma_wp_woh = copy.deepcopy(gamma_wp)
    gamma_wp_woh.ignore_history = True

    from .lm_watermarking.watermark_processor import (
        WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
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
    return [None, delta_wp, gamma_wp, delta_wp_woh, gamma_wp_woh, *john_wps]


def get_num_gpus():
    import torch

    num_gpus = torch.cuda.device_count()
    return num_gpus


def batched_wp_task_worker(tq, get_in_ds, batch_size=8):
    ds = get_in_ds()

    from experiments.common import get_wps

    wps = get_wps()

    from tqdm import tqdm

    for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
        for wp in wps:
            tq.put({"batch": batch, "watermark_processor": wp})


def merged_task_worker(
    get_in_ds, output_filepath, tq, batch_size=8, watermark_only=False
):
    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": output_filepath})["test"]
    out_ds = out_ds.sort("id")

    dss, wps = add_reference(in_ds, out_ds)

    from tqdm import tqdm

    for ds, wp_str in zip(dss, wps):
        if watermark_only:
            if "John" in wp_str or "None" == wp_str:
                continue
        for batch in tqdm(ds.iter(batch_size=batch_size), total=len(ds) // batch_size):
            tq.put(batch)


def log(line: dict, f):
    import json

    f.write(json.dumps(line))
    f.write("\n")
    f.flush()


def simple_store_worker(path, rq, rqe):
    import os

    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))

    from queue import Empty

    with open(path, "w") as f:
        while not (rqe.is_set() and rq.empty()):
            try:
                result = rq.get(timeout=1)
            except Empty as e:
                continue
            assert isinstance(result, dict)
            if result == {}:
                continue
            if isinstance(next(iter(result.values())), list):
                assert all([isinstance(v, list) for v in result.values()])
                lens = [len(v) for v in result.values()]
                assert all([l == lens[0] for l in lens])
                for i in range(lens[0]):
                    log({k: v[i] for k, v in result.items()}, f)
            else:
                log(result, f)


def group_batch(batch):
    return {k: [v] for k, v in batch.items()}


def tokenize_batch(example, tokenizer, fields=["input"], max_length: int | dict = 512):
    result = {}

    if tokenizer.name_or_path == "facebook/mbart-large-en-ro":
        tokenizer.tgt_lang = "ro_RO"

    for field in fields:
        if field in example:
            kwargs = {}
            if isinstance(max_length, dict):
                kwargs["max_length"] = max_length[field]
            else:
                kwargs["max_length"] = max_length
            if field in ["output", "reference"]:
                kwargs["text_target"] = example[field]
            else:
                kwargs["text"] = example[field]
            if field == "output":
                kwargs["add_special_tokens"] = False
            result[field] = tokenizer(
                **kwargs,
                truncation=True,
                padding=True,
                return_tensors="pt",
            )

    return result


def set_spawn():
    import torch.multiprocessing as mp

    try:
        mp.set_start_method("spawn")
    except RuntimeError:
        pass


def remove_tailing_pad_s(s: str):
    index = s.find("<pad>")
    if index == -1:
        return s
    else:
        return s[:index]


def remove_tailing_pad(strs: list[str]):
    return [remove_tailing_pad_s(s) for s in strs]


def transformer_worker(tq, tqe, rq, gpu_id, model_str, generation_kwargs={}):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from unbiased_watermark import patch_model

    model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            task = tq.get(timeout=1)
        except Empty as e:
            continue
        batch = task["batch"]
        tbatch = tokenize_batch(batch, tokenizer)
        wp = task["watermark_processor"]
        lps = []
        if wp is not None:
            if "reset_history" in dir(wp):
                wp.reset_history()
            if "vocab_size" in dir(wp):
                wp.vocab_size = model.config.vocab_size
            lps.append(wp)

        # for reproducibility and sufficient randomness
        import hashlib

        hash = hashlib.sha256()
        hash.update(str(batch["id"]).encode("utf-8"))
        seed = hash.digest()
        seed = int.from_bytes(seed, "big") % (2**32 - 1)

        set_seed(seed)
        outputs_ids = model.generate(
            tbatch["input"]["input_ids"].to(device=model.device),
            attention_mask=tbatch["input"]["attention_mask"].to(device=model.device),
            do_sample=True,
            num_beams=1,
            #  top_k=50,   # default
            logits_warper=LogitsProcessorList(lps),
            **generation_kwargs,
        )
        outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=False)
        outputs = remove_tailing_pad(outputs)
        display_outputs = tokenizer.batch_decode(outputs_ids, skip_special_tokens=True)
        wp_str = repr(wp)
        rq.put(
            {
                "output": outputs,
                "display_output": display_outputs,
                "id": batch["id"],
                "watermark_processor": [wp_str] * len(outputs),
            }
        )


def add_reference(in_ds, out_ds):
    """assuming ordered by ids"""
    wp_types = set(out_ds["watermark_processor"])

    s_out_dss = []
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        s_out_ds = s_out_ds.add_column("input", in_ds["input"])
        s_out_ds = s_out_ds.add_column("reference", in_ds["reference"])
        s_out_dss.append(s_out_ds)
    from datasets import concatenate_datasets

    return s_out_dss, wp_types


def bertscore_worker(tq, tqe, rq, gpu_id=0):
    import bert_score

    scorer = bert_score.BERTScorer(
        lang="de",
        rescale_with_baseline=True,
        device=f"cuda:{gpu_id}",
        use_fast_tokenizer=True,
    )

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        (P, R, F) = scorer.score(batch["display_output"], batch["reference"])
        rq.put(
            {
                **batch,
                "bertscore.precision": P.tolist(),
                "bertscore.recall": R.tolist(),
                "bertscore.f1": F.tolist(),
            }
        )


def rouge_worker(tq, tqe, rq):
    import evaluate

    rouge = evaluate.load("rouge")

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        rouge_scores = rouge.compute(
            predictions=batch["display_output"],
            references=batch["reference"],
            rouge_types=["rouge1", "rouge2", "rougeL"],
            use_stemmer=True,
            use_aggregator=False,
        )
        rq.put({**rouge_scores, **batch})


def remove_text_worker(tq, tqe, rq):
    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        for f in ["input", "output", "reference", "display_output"]:
            if f in batch:
                del batch[f]
        rq.put(batch)


import torch


@torch.no_grad()
def get_ppl(model, tbatch):
    input_ids = tbatch["input"]["input_ids"].to(model.device)
    attention_mask = tbatch["input"]["attention_mask"].to(model.device)
    labels = tbatch["output"]["input_ids"][..., 1:].to(model.device)
    outputs = model(input_ids=input_ids, labels=labels, attention_mask=attention_mask)

    from torch.nn import CrossEntropyLoss

    loss_fct = CrossEntropyLoss(reduction="none")

    #  output.logits: [batch_size, sequence_length, vocab_size]
    #  labels: [batch_size, sequence_length]
    shape = labels.shape
    #  loss: [batch_size, sequence_length]
    losses = loss_fct(
        outputs.logits.reshape(-1, outputs.logits.shape[-1]),
        labels.view(-1),
    ).reshape(shape)
    label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(model.device)
    #  loss: [batch_size]
    losses = (losses * label_attention_mask.float()).sum(
        dim=-1
    ) / label_attention_mask.sum(dim=-1)
    ppl = torch.exp(losses).cpu().tolist()
    return ppl


def ppl_worker(tq, tqe, rq, gpu_id, oracle_model_str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    model = AutoModelForSeq2SeqLM.from_pretrained(oracle_model_str).to(f"cuda:{gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(oracle_model_str)

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
        )
        ppl = get_ppl(model, tbatch)
        with torch.cuda.device(model.device):
            torch.cuda.empty_cache()

        rq.put(
            {
                **batch,
                "ppl": ppl,
            }
        )


@torch.no_grad()
def get_score(model, tbatch, wp, score):
    input_ids = tbatch["input"]["input_ids"].to(model.device)
    attention_mask = tbatch["input"]["attention_mask"].to(model.device)
    #  labels : [batch_size, output_sequence_length-1]
    labels = tbatch["output"]["input_ids"][..., 1:].to(model.device)
    label_attention_mask = tbatch["output"]["attention_mask"][..., 1:].to(model.device)
    #  decoder_input_ids : [batch_size, output_sequence_length-1]
    decoder_input_ids = tbatch["output"]["input_ids"][..., :-1].to(model.device)
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
    )

    from transformers import GenerationConfig

    generation_config = GenerationConfig.from_model_config(model.config)
    logits_processor = model._get_logits_processor(
        generation_config,
        input_ids_seq_length=labels.shape[-1],
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
    )
    logits_warper = model._get_logits_warper(generation_config)

    #  logits: [batch_size, output_sequence_length-1, vocab_size]
    logits = outputs.logits
    del outputs
    del input_ids
    del attention_mask
    with torch.cuda.device(model.device):
        torch.cuda.empty_cache()
    old_logits = torch.clone(logits)
    new_logits = torch.clone(logits)
    for i in range(logits.size(1)):
        pre = decoder_input_ids[:, : i + 1]
        t = logits[:, i]
        t = logits_processor(pre, t)
        t = logits_warper(pre, t)
        old_logits[:, i] = t
        new_logits[:, i] = wp(pre, t)
    del logits

    # compute entropy
    import torch.nn.functional as F

    _logits = F.log_softmax(old_logits, dim=-1)
    _logits.nan_to_num_()
    entropy = -(torch.exp(_logits) * _logits).sum(dim=-1)

    # compute score
    from unbiased_watermark import RobustLLR_Score_Batch_v1, RobustLLR_Score_Batch_v2

    if isinstance(score, RobustLLR_Score_Batch_v1):
        #  all_scores: [batch_size, output_sequence_length-1, query_size, vocab_size]
        all_scores = score.score(old_logits, new_logits)
        #  query_ids: [batch_size, output_sequence_length-1, query_size]
        query_ids = labels.unsqueeze(-1).expand(
            tuple(-1 for _ in range(decoder_input_ids.ndim)) + (all_scores.size(-2),)
        )
        #  scores: [batch_size, output_sequence_length-1, query_size]
        scores = torch.gather(all_scores, -1, query_ids.unsqueeze(-1)).squeeze(-1)
    elif isinstance(score, RobustLLR_Score_Batch_v2):
        #  llr: [batch_size, output_sequence_length-1, vocab_size]
        #  max_llr: [batch_size, output_sequence_length-1, query_size]
        llr, max_llr, min_llr = score.score(old_logits, new_logits)
        #  query_ids: [batch_size, output_sequence_length-1]
        query_ids = labels
        #  unclipped_scores: [batch_size, output_sequence_length-1]
        unclipped_scores = torch.gather(llr, -1, query_ids.unsqueeze(-1)).squeeze(-1)
        #  scores: [batch_size, output_sequence_length-1, query_size]
        scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
    return (
        scores * label_attention_mask.unsqueeze(-1),
        entropy * label_attention_mask,
        label_attention_mask,
    )


def score_worker(tq, tqe, rq, gpu_id, model_str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    model = AutoModelForSeq2SeqLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    from unbiased_watermark import RobustLLR_Score_Batch_v1, RobustLLR_Score_Batch_v2

    grid_size = 10
    dist_qs = [i / grid_size for i in range(0, grid_size + 1)]
    scorer = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)

    from queue import Empty

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        assert len(set(batch["watermark_processor"])) == 1

        from unbiased_watermark import (
            Delta_Reweight,
            Gamma_Reweight,
            WatermarkLogitsProcessor,
            PrevN_ContextCodeExtractor,
        )

        wp_str = batch["watermark_processor"][0]
        wp = eval(wp_str)
        wp.ignore_history = True

        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
        )
        # score: [batch_size, sequence_length, query_size]
        # entropy: [batch_size, sequence_length]
        # label_attention_mask: [batch_size, sequence_length]
        score, entropy, label_attention_mask = get_score(model, tbatch, wp, scorer)

        import gc

        gc.collect()
        # score: [batch_size, query_size]
        sum_score = score.sum(-2)
        # best_index: [batch_size]
        best_index = torch.argmax(sum_score, dim=-1)
        best_dist_q = [dist_qs[i] for i in best_index.cpu().tolist()]
        best_sum_score = (
            torch.gather(
                sum_score,
                -1,
                best_index.unsqueeze(-1),
            )
            .squeeze(-1)
            .cpu()
            .tolist()
        )

        # best_score: [batch_size, sequence_length]
        best_score = (
            torch.gather(
                score,
                -1,
                best_index.unsqueeze(-1).unsqueeze(-1).expand(-1, score.size(-2), -1),
            )
            .squeeze(-1)
            .cpu()
            .tolist()
        )
        #  lens: [batch_size]
        cum_label_attention_mask = torch.cumsum(label_attention_mask, dim=-1)
        lens = cum_label_attention_mask[:, -1]
        #  assert attention is like 11110000
        _lens_m_1 = torch.argmax(
            cum_label_attention_mask,
            dim=-1,
        )
        assert torch.all(lens == _lens_m_1 + 1)

        lens = lens.cpu().tolist()
        best_score = [best_score[i][: lens[i]] for i in range(len(best_score))]
        entropy = entropy.cpu().tolist()
        entropy = [entropy[i][: lens[i]] for i in range(len(entropy))]

        rq.put(
            {
                **batch,
                "best_dist_q": best_dist_q,
                "best_sum_score": best_sum_score,
                "best_score": best_score,
                "lens": lens,
                "entropy": entropy,
            }
        )
