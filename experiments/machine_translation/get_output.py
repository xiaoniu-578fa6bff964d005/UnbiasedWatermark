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

    from experiments.common import batched_wp_task_worker, transformer_worker
    from . import get_in_ds

    task_worker_ = Process(
        target=batched_wp_task_worker,
        args=(tq, rq),
        kwargs={"get_in_ds": get_in_ds, "batch_size": 256},
    )
    gpu_workers = [
        Process(
            target=transformer_worker,
            args=(tq, tqe, rq, i),
            kwargs={
                "model_str": "Helsinki-NLP/opus-mt-en-de",
                "generation_kwargs": {
                    "max_length": 512,
                    "temperature": 1.0,
                },
            },
        )
        for i in range(num_gpus)
    ]
    from experiments.common import simple_store_worker

    store_worker = Process(
        target=simple_store_worker, args=("data/machine_translation.txt", rq, rqe)
    )

    task_worker_.start()
    for w in gpu_workers:
        w.start()
    store_worker.start()

    task_worker_.join()
    tqe.set()
    for w in gpu_workers:
        w.join()
    rqe.set()
    store_worker.join()
