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
        simple_store_worker,
        remove_text_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, "data/machine_translation.txt", tq),
        kwargs={"batch_size": 256},
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
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in bertscore_workers:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0
