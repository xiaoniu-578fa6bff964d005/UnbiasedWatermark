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
        ppl_worker,
        remove_text_worker,
        simple_store_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, "data/machine_translation.txt", tq, rq),
        kwargs={"batch_size": 16},
    )

    ppl_worker_ = [
        Process(
            target=ppl_worker,
            args=(tq, tqe, rq, i, "facebook/mbart-large-en-ro"),
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=("data/machine_translation_ppl.txt", r2q, r2qe),
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