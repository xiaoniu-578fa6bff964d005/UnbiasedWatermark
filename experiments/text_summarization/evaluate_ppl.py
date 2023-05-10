import torch


@torch.no_grad()
def get_ppl(model, tbatch):
    input_ids = tbatch["input"]["input_ids"].to(model.device)
    attention_mask = tbatch["input"]["attention_mask"].to(model.device)
    labels = tbatch["output"]["input_ids"].to(model.device)
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
    label_attention_mask = tbatch["output"]["attention_mask"].to(model.device)
    #  loss: [batch_size]
    losses = (losses * label_attention_mask.float()).sum(
        dim=-1
    ) / label_attention_mask.sum(dim=-1)
    ppl = torch.exp(losses).cpu().tolist()
    return ppl


def ppl_worker(tq, tqe, rq, gpu_id, oracle_model_str):
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    from transformers import LogitsProcessorList, TemperatureLogitsWarper

    from unbiased_watermark import patch_model

    model = AutoModelForSeq2SeqLM.from_pretrained(oracle_model_str).to(f"cuda:{gpu_id}")
    patch_model(model)
    tokenizer = AutoTokenizer.from_pretrained(oracle_model_str)

    from queue import Empty

    from experiments.common import tokenize_batch

    while not (tqe.is_set() and tq.empty()):
        try:
            batch = tq.get(timeout=1)
        except Empty as e:
            continue
        tbatch = tokenize_batch(
            batch,
            tokenizer,
            ["input", "output"],
            max_length={"input": 512, "output": 128},
        )
        ppl = get_ppl(model, tbatch)

        rq.put(
            {
                **batch,
                "ppl": ppl,
            }
        )


def pipeline():
    from experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()
    r2q = Queue()
    r2qe = Event()

    from experiments.common import (
        merged_task_worker,
        bertscore_worker,
        rouge_worker,
        simple_store_worker,
        remove_text_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, "data/text_summarization.txt", tq, rq),
        #  kwargs={"batch_size": 32},
        #  kwargs={"batch_size": 1},
        kwargs={"batch_size": 128},
    )

    ppl_worker_ = [
        Process(
            target=ppl_worker,
            args=(
                tq,
                tqe,
                rq,
                i,
                #  "facebook/bart-large-cnn"
                "philschmid/bart-large-cnn-samsum",
            ),
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=("data/text_summarization_ppl.txt", r2q, r2qe),
    )

    task_worker_.start()
    for w in ppl_worker_:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    tqe.set()
    for w in ppl_worker_:
        w.join()
    rqe.set()
    rt_worker.join()
    r2qe.set()
    store_worker.join()
