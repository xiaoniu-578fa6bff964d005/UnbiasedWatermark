#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch

cache = {
    "model_str": None,
    "generator": None,
    "tokenizer": None,
}


def load_model(model_str):
    if model_str == cache["model_str"]:
        return cache
    else:
        from transformers import pipeline

        generator = pipeline(
            "text-generation",
            model=model_str,
            do_sample=True,
            num_beams=1,
            device=0,
        )

        cache["model_str"] = model_str
        cache["generator"] = generator
        cache["tokenizer"] = generator.tokenizer
        return cache


def get_wp(watermark_type, key):
    from unbiased_watermark import (
        WatermarkLogitsProcessor,
        Delta_Reweight,
        Gamma_Reweight,
        PrevN_ContextCodeExtractor,
    )

    if watermark_type == "delta":
        rw = Delta_Reweight()
    elif watermark_type == "gamma":
        rw = Gamma_Reweight()
    else:
        raise ValueError(f"Unknown watermark type: {watermark_type}")
    wp = WatermarkLogitsProcessor(key, rw, PrevN_ContextCodeExtractor(5))
    return wp


def generate(model_str, prompt, watermark_type, key, **kwargs):
    cache = load_model(model_str)
    generator = cache["generator"]
    from unbiased_watermark import patch_model

    patch_model(generator.model)
    if watermark_type is None:
        lws = []
    else:
        lws = [get_wp(watermark_type, key)]
    outputs = generator(prompt, logits_warper=lws, **kwargs)
    generator.model._clear_patch_context()
    return [r[0]["generated_text"] for r in outputs]


def get_prompt_length(model_str, prompt):
    cache = load_model(model_str)
    tokenizer = cache["tokenizer"]
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    return inputs["input_ids"].shape[1]


def r_llr_score(model_str, texts, dist_qs, watermark_type, key, **kwargs):
    from unbiased_watermark import RobustLLR_Score_Batch_v2

    score = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)

    wp = get_wp(watermark_type, key)
    wp.ignore_history = True

    cache = load_model(model_str)

    inputs = cache["tokenizer"](texts, return_tensors="pt", padding=True)

    from transformers import GenerationConfig

    model = cache["generator"].model
    input_ids = inputs["input_ids"][..., :-1].to(model.device)
    attention_mask = inputs["attention_mask"][..., :-1].to(model.device)
    labels = inputs["input_ids"][..., 1:].to(model.device)
    labels_mask = inputs["attention_mask"][..., 1:].to(model.device)
    generation_config = GenerationConfig.from_model_config(model.config)
    logits_processor = model._get_logits_processor(
        generation_config,
        input_ids_seq_length=input_ids.shape[-1],
        encoder_input_ids=input_ids,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
    )
    logits_warper = model._get_logits_warper(generation_config)

    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits
    old_logits = torch.clone(logits)
    new_logits = torch.clone(logits)
    for i in range(logits.size(1)):
        pre = input_ids[:, : i + 1]
        t = logits[:, i]
        t = logits_processor(pre, t)
        t = logits_warper(pre, t)
        old_logits[:, i] = t
        new_logits[:, i] = wp(pre, t)
    llr, max_llr, min_llr = score.score(old_logits, new_logits)
    query_ids = labels
    unclipped_scores = torch.gather(llr, -1, query_ids.unsqueeze(-1)).squeeze(-1)
    # scores : [batch_size, input_ids_len, query_size]
    scores = torch.clamp(unclipped_scores.unsqueeze(-1), min_llr, max_llr)
    return labels, labels_mask, scores * labels_mask.unsqueeze(-1)


def is_displayable(s):
    import unicodedata

    non_displayable_categories = {"Cc", "Cf", "Cs", "Co", "Cn"}
    for c in s:
        if c == "\n":
            continue
        if unicodedata.category(c) in non_displayable_categories:
            return False
        elif c == "\ufffd":  # Unicode Replacement Character
            return False
    return True


def print_latex(result: list[tuple[int, str, float]]):
    from pylatexenc.latexencode import unicode_to_latex

    for token_id, token, score in result:
        if token == "\n":
            token = r"\textbackslash n"
        else:
            token = unicode_to_latex(token)
        if score is None:
            print(token, end="")
        else:
            print(rf"\tws{{{token}}}{{{score:.3f}}}", end="")
    print()


def show_r_llr_score(
    model_str,
    text,
    compute_range=(None, None),
    show_latex=False,
    merge_till_displayable=True,
    **kwargs,
):
    n = 10
    dist_qs = [float(i) / n for i in range(n + 1)]

    labels, _, scores = r_llr_score(model_str, [text], dist_qs=dist_qs, **kwargs)
    import numpy as np

    labels = np.array(labels[0].cpu())
    scores = np.array(scores[0].cpu())
    if compute_range[0] is None:
        compute_range = (0, compute_range[1])
    if compute_range[1] is None:
        compute_range = (compute_range[0], len(labels))
    scores[: compute_range[0], :] = 0
    scores[compute_range[1] :, :] = 0
    sum_scores = np.sum(scores, axis=0)
    best_index = np.argmax(sum_scores)
    best_dist_q = dist_qs[best_index]
    print("best_dist_q:", best_dist_q)
    print("best_sum_score:", sum_scores[best_index])

    cache = load_model(model_str)
    tokenizer = cache["tokenizer"]

    result = []
    i = 0
    while i < len(labels):
        for j in range(i + 1, len(labels) + 1):
            token_id = labels[i:j]
            token = tokenizer.decode(token_id, skip_special_tokens=False)
            if merge_till_displayable and not is_displayable(token):
                continue
            break
        if j < compute_range[0] or i >= compute_range[1]:
            result.append((token_id, token, None))
        else:
            result.append((token_id, token, scores[i:j, best_index].sum()))
        i = j

    if not show_latex:
        print(result)
    else:
        print_latex(result)


def run_demo(prompt, model_str, seed, max_length, key):
    prompt_len = get_prompt_length(args.model, prompt)
    set_seed(seed)
    output1 = generate(
        args.model, [prompt], watermark_type=None, key=key, max_length=100
    )[0]
    set_seed(seed)
    output2 = generate(
        args.model, [prompt], watermark_type="delta", key=key, max_length=100
    )[0]
    set_seed(seed)
    output3 = generate(
        args.model, [prompt], watermark_type="gamma", key=key, max_length=100
    )[0]

    print("\n")
    set_seed(seed)
    show_r_llr_score(
        args.model,
        output1,
        compute_range=(prompt_len, None),
        show_latex=True,
        watermark_type="delta",
        key=key,
    )
    print("\n")
    set_seed(seed)
    show_r_llr_score(
        args.model,
        output2,
        compute_range=(prompt_len, None),
        show_latex=True,
        watermark_type="delta",
        key=key,
    )
    print("\n")
    set_seed(seed)
    show_r_llr_score(
        args.model,
        output3,
        compute_range=(prompt_len, None),
        show_latex=True,
        watermark_type="gamma",
        key=key,
    )


import argparse

parser = argparse.ArgumentParser()
#  parser.add_argument("--model", type=str, default="facebook/opt-1.3b")
#  parser.add_argument("--model", type=str, default="facebook/opt-6.7b")
parser.add_argument("--model", type=str, default="gpt2")
#  parser.add_argument("--model", type=str, default="facebook/opt-13b")
parser.add_argument("--seed", type=int)
parser.add_argument("--prompt", type=str, default="To maximize parallelism")
parser.add_argument("--watermark_type", choices=["delta", "gamma"], default=None)
parser.add_argument("--max_length", type=int, default=100)
parser.add_argument("--key", type=str, default="private key")
parser.add_argument("--text", type=str)
parser.add_argument("--test", action="store_true")
parser.add_argument("--compute_range", type=str, default="(None, None)")
parser.add_argument("--show_latex", action="store_true")
parser.add_argument("--demo", action="store_true")
args = parser.parse_args()


if not torch.cuda.is_available():
    print("CUDA is not available. Exiting...")
    exit()

from transformers import set_seed

if args.seed is not None:
    set_seed(args.seed)
if args.demo:
    prompt = "What is a watermark? What's the purpose of it?"
    if args.seed is not None:
        seed = args.seed
    else:
        seed = 42
    key = args.key.encode("utf-8")

    run_demo(prompt, args.model, seed, args.max_length, key)
elif args.test:
    assert args.watermark_type is not None
    assert args.text is not None
    compute_range = eval(args.compute_range)
    show_r_llr_score(
        args.model,
        args.text,
        compute_range=compute_range,
        show_latex=args.show_latex,
        watermark_type=args.watermark_type,
        key=args.key.encode("utf-8"),
    )
else:
    print(
        generate(
            args.model,
            [args.prompt],
            watermark_type=args.watermark_type,
            key=args.key.encode("utf-8"),
            max_length=args.max_length,
        )[0]
    )
