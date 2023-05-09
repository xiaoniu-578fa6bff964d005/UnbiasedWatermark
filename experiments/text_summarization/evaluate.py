def task_worker(tq, rq, batch_size=8):
    from . import get_in_ds

    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": "data/text_summarization.txt"})[
        "test"
    ]
    out_ds = out_ds.sort("id")

    from experiments.common import add_reference, group_batch

    ds = add_reference(in_ds, out_ds)
    ds = ds.map(group_batch, batched=True, batch_size=batch_size)

    from tqdm import tqdm

    for batch in tqdm(ds):
        tq.put(batch)


def pipeline():
    from experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    t2q = Queue(maxsize=num_gpus)
    t2qe = Event()
    rq = Queue()
    rqe = Event()

    task_worker_ = Process(
        target=task_worker,
        args=(tq, rq),
        kwargs={"batch_size": 256},
    )
    from experiments.common import bertscore_worker, rouge_worker

    bertscore_workers = [
        Process(target=bertscore_worker, args=(tq, tqe, t2q, i))
        for i in range(num_gpus)
    ]
    rouge_workers = [
        Process(target=rouge_worker, args=(t2q, t2qe, rq)) for i in range(num_gpus)
    ]
    from experiments.common import simple_store_worker

    store_worker = Process(
        target=simple_store_worker,
        args=("data/text_summarization_result.txt", rq, rqe),
    )

    task_worker_.start()
    for w in bertscore_workers:
        w.start()
    for w in rouge_workers:
        w.start()
    store_worker.start()

    task_worker_.join()
    tqe.set()
    for w in bertscore_workers:
        w.join()
    t2qe.set()
    for w in rouge_workers:
        w.join()
    rqe.set()
    store_worker.join()
