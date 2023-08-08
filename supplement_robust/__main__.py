from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer, set_seed
import torch

from unbiased_watermark import patch_model


def prepare_inputs(tokenizer, total_size, input_length, batch_id=0):
    # return an input_ids tensor (total_size, input_length)
    # truncate from cnn_dailymail dataset
    from datasets import load_dataset

    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = ds.select(range(batch_id * total_size, (batch_id + 1) * total_size))
    texts = list(ds["article"])

    if tokenizer.name_or_path in ["daryl149/llama-2-7b-chat-hf", "gpt2"]:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=input_length,
    )
    assert torch.all(inputs["attention_mask"] == 1)
    return inputs["input_ids"]


def get_wps():
    from unbiased_watermark import (
        Delta_Reweight,
        Gamma_Reweight,
        DeltaGumbel_Reweight,
        WatermarkLogitsProcessor,
        PrevN_ContextCodeExtractor,
    )

    import random

    random.seed(42)
    private_key = random.getrandbits(1024).to_bytes(128, "big")
    delta_wp = WatermarkLogitsProcessor(
        private_key,
        Delta_Reweight(),
        #  PrevN_ContextCodeExtractor(1),  # keep same with john paper
        PrevN_ContextCodeExtractor(3),
    )
    gamma_wp = WatermarkLogitsProcessor(
        private_key,
        Gamma_Reweight(),
        PrevN_ContextCodeExtractor(3),
    )

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
        for delta in [1.0, 2.0]
    ]
    return [
        None,
        delta_wp,
        gamma_wp,
        *john_wps,
    ]


def generate_output_ids(model, input_ids, wp, output_length):
    from transformers import LogitsProcessorList, MinNewTokensLengthLogitsProcessor

    lps = [
        MinNewTokensLengthLogitsProcessor(
            input_ids.shape[1], output_length, model.config.eos_token_id
        )
    ]
    if wp is not None:
        if "reset_history" in dir(wp):
            wp.reset_history()
        if "vocab_size" in dir(wp):
            wp.vocab_size = model.config.vocab_size
        lps.append(wp)

    set_seed(42)
    outputs_ids = model.generate(
        input_ids,
        do_sample=True,
        num_beams=1,
        logits_warper=LogitsProcessorList(lps),
        max_new_tokens=output_length,
        temperature=1.0,
    )
    model._clear_patch_context()
    return outputs_ids[:, input_ids.shape[1] :]


@torch.no_grad()
def get_score(model, input_ids, wp, output_ids, grid_size=10):
    from unbiased_watermark import RobustLLR_Score_Batch_v2

    dist_qs = [i / grid_size for i in range(0, grid_size + 1)]
    #  dist_qs = [0.0]
    scorer = RobustLLR_Score_Batch_v2.from_grid([0.0], dist_qs)

    wp.ignore_history = True
    wp.reset_history()

    com_input_ids = torch.cat([input_ids, output_ids[:, :-1]], dim=-1).to(model.device)
    outputs = model(input_ids=com_input_ids)
    labels = output_ids.to(model.device)
    logits = outputs.logits[:, input_ids.shape[1] - 1 :]

    from transformers import GenerationConfig
    from transformers import MinNewTokensLengthLogitsProcessor

    generation_config = GenerationConfig.from_model_config(model.config)
    logits_processor = model._get_logits_processor(
        generation_config,
        input_ids_seq_length=input_ids.shape[-1],
        encoder_input_ids=None,
        prefix_allowed_tokens_fn=None,
        logits_processor=[],
    )
    logits_warper = model._get_logits_warper(generation_config)
    logits_warper.append(
        MinNewTokensLengthLogitsProcessor(
            input_ids.shape[1], output_ids.shape[1], model.config.eos_token_id
        )
    )

    with torch.cuda.device(model.device):
        torch.cuda.empty_cache()
    old_logits = torch.clone(logits)
    scores = torch.zeros(
        logits.shape[:-1] + (scorer.query_size(),), device=logits.device
    )
    for i in range(logits.size(1)):
        pre = com_input_ids[:, : input_ids.shape[1] + i]
        t = logits[:, i]
        t = logits_processor(pre, t)
        t = logits_warper(pre, t)
        old_logits[:, i] = t
        new_logits = wp(pre, t)
        scores[:, i] = wp.get_score(labels[:, i], old_logits[:, i], new_logits, scorer)
    del logits

    sum_score = scores.sum(-2)
    best_index = torch.argmax(sum_score, dim=-1)
    best_dist_q = [dist_qs[i] for i in best_index.cpu().tolist()]
    best_sum_score = torch.gather(sum_score, -1, best_index.unsqueeze(-1)).squeeze(-1)
    import math

    final_score = best_sum_score - math.log(grid_size + 1)

    wp.ignore_history = False  # set it back
    return final_score.cpu().tolist()


def get_z_score(tokenizer, device, input_ids, output_ids):
    # don't support batch
    from .lm_watermarking.watermark_processor import (
        WatermarkLogitsProcessor as WatermarkLogitsProcessor_John,
        WatermarkDetector as WatermarkDetector_John,
    )

    wp = WatermarkDetector_John(
        vocab=list(tokenizer.get_vocab().values()),
        gamma=0.5,
        tokenizer=tokenizer,
        device=device,
        normalizers=[],
    )
    wp.min_prefix_len = input_ids.shape[-1]
    r = wp._score_sequence(torch.cat([input_ids, output_ids], dim=-1))
    return r["z_score"]


def randomly_substitute(ids, max_id, portion=0.0):
    #  ids: [batch_size, seq_len]
    #  for each seq in batch, let about {portion} of ids are substituted with random ids
    #  return: [batch_size, seq_len]
    subs_mask = torch.rand_like(ids, dtype=torch.float32) < portion
    subs_ids = torch.randint_like(ids, high=max_id)
    return torch.where(subs_mask, subs_ids, ids)


def get_auc(score_vanilla, score_positive):
    import numpy as np
    import sklearn.metrics

    score_vanilla = np.clip(score_vanilla, -1e6, 1e6)
    score_positive = np.clip(score_positive, -1e6, 1e6)
    fpr, tpr, thresholds = sklearn.metrics.roc_curve(
        np.concatenate([np.zeros_like(score_vanilla), np.ones_like(score_positive)]),
        np.concatenate([score_vanilla, score_positive]),
    )
    return sklearn.metrics.auc(fpr, tpr)


model_str = "gpt2"
gpu_id = 0
#  total_size = 64
#  total_size = 512
num_batch = 4
total_size = 512 * num_batch
#  input_length = 32
#  output_length = 32
input_length = 16
output_length = 16


def get_show_name(wp_str):
    if wp_str == "None":
        return "None"
    if "Gamma_Reweight" in wp_str:
        return "Gamma_Reweight"
    if "Delta_Reweight" in wp_str:
        return "Delta_Reweight"
    if "John" in wp_str:
        delta = wp_str.split("delta=")[1].split(",")[0]
        return f"Soft(delta={delta})"


def main():
    model = AutoModelForCausalLM.from_pretrained(model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(model_str)

    aucs = []
    for batch_id in range(num_batch):
        input_ids = prepare_inputs(
            tokenizer, int(total_size / num_batch), input_length, batch_id
        ).to(f"cuda:{gpu_id}")

        wps = get_wps()
        wp_strs = [repr(wp) for wp in wps]

        outputs = {}
        for wp_str, wp in zip(wp_strs, wps):
            print(wp_str)
            output_ids = generate_output_ids(model, input_ids, wp, output_length)
            outputs[wp_str] = output_ids

        interference_strength = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
        auc = {}
        for wp_str, wp in zip(wp_strs, wps):
            if wp_str not in outputs:
                continue
            if "None" == wp_str:
                continue
            auc[wp_str] = {}
            for strength in interference_strength:
                set_seed(42)
                perturnbed_ids = randomly_substitute(
                    outputs[wp_str], tokenizer.vocab_size, strength
                )

                if "Reweight" in wp_str:
                    score = get_score(model, input_ids, wp, perturnbed_ids)
                    vanilla_score = get_score(model, input_ids, wp, outputs["None"])
                elif "John" in wp_str:
                    score = [
                        get_z_score(tokenizer, model.device, i, o)
                        for i, o in zip(input_ids, perturnbed_ids)
                    ]
                    vanilla_score = [
                        get_z_score(tokenizer, model.device, i, o)
                        for i, o in zip(input_ids, outputs["None"])
                    ]
                auc[wp_str][strength] = get_auc(vanilla_score, score)
            print(get_show_name(wp_str))
            print("\t", auc[wp_str])
        aucs.append(auc)

    from prettytable import PrettyTable
    import numpy as np

    x = PrettyTable()
    x.field_names = [""] + [str(strength) for strength in interference_strength]
    for wp_str in aucs[0]:
        data = []
        for strength in interference_strength:
            l = [auc[wp_str][strength] for auc in aucs]
            data.append(f"{np.mean(l):.4f}±{np.std(l):.4f}")
        x.add_row([get_show_name(wp_str)] + data)
    print(x)


if __name__ == "__main__":
    main()
