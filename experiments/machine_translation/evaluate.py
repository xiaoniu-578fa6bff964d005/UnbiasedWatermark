def task_worker(tq, rq, batch_size=8):
    from . import get_in_ds

    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": "data/machine_translation.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")

    from experiments.common import add_reference, group_batch

    ds = add_reference(in_ds, out_ds)
    ds = ds.map(group_batch, batched=True, batch_size=batch_size)

    from tqdm import tqdm

    for batch in tqdm(ds):
        tq.put(batch)


def bleu_task(t):
    lds, i, wp_str, bleu_scorer = t

    import random

    a = random.Random(i).sample(lds["output"], 2000)
    b = random.Random(i).sample(lds["reference"], 2000)

    #  lds = s_out_ds.shuffle(seed=i)
    bleu_score = bleu_scorer.compute(
        predictions=a,
        references=b,
    )
    return {"watermark_processor": wp_str, "bleu": bleu_score["score"]}


def compute_bleu():
    from . import get_in_ds

    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": "data/machine_translation.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")

    wp_types = set(out_ds["watermark_processor"])

    s_out_dss = {}
    for wp_type in wp_types:
        s_out_ds = out_ds.filter(lambda x: x["watermark_processor"] == wp_type)
        assert len(s_out_ds) == len(in_ds)
        s_out_ds = s_out_ds.add_column("reference", in_ds["reference"])
        s_out_dss[wp_type] = s_out_ds

    import evaluate

    bleu_scorer = evaluate.load("sacrebleu")

    from concurrent.futures import ProcessPoolExecutor
    import json

    with open("data/machine_translation_bleu.txt", "w") as f:
        with ProcessPoolExecutor() as executor:
            from tqdm import tqdm

            for r in tqdm(
                executor.map(
                    bleu_task,
                    [
                        (s_out_ds, i, wp_str, bleu_scorer)
                        for (wp_str, s_out_ds) in s_out_dss.items()
                        for i in range(100)
                    ],
                ),
                total=len(wp_types) * 100,
            ):
                f.write(json.dumps(r))
                f.write("\n")
                f.flush()


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

    task_worker_ = Process(
        target=task_worker,
        args=(tq, rq),
        kwargs={"batch_size": 256},
    )
    from experiments.common import (
        bertscore_worker,
        simple_store_worker,
        remove_text_worker,
    )

    bertscore_workers = [
        Process(target=bertscore_worker, args=(tq, tqe, rq, i)) for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=("data/machine_translation_result.txt", r2q, r2qe),
    )

    task_worker_.start()
    for w in bertscore_workers:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    tqe.set()
    for w in bertscore_workers:
        w.join()
    rqe.set()
    rt_worker.join()
    r2qe.set()
    store_worker.join()
