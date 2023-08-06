def pipeline():
    from supplement_experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    from supplement_experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()
    rq = Queue()
    rqe = Event()

    from supplement_experiments.common import batched_wp_task_worker, transformer_worker
    from . import get_in_ds

    task_worker_ = Process(
        target=batched_wp_task_worker,
        args=(tq,),
        kwargs={"get_in_ds": get_in_ds, "batch_size": 128},
    )
    gpu_workers = [
        Process(
            target=transformer_worker,
            args=(tq, tqe, rq, i),
            kwargs={
                #  "model_str": "meta-llama/Llama-2-7b-chat-hf",
                "model_str": "daryl149/llama-2-7b-chat-hf",
                "decoder_only": True,
                "generation_kwargs": {
                    "max_new_tokens": 128,
                    "temperature": 1.0,
                },
                "tokenization_kwargs": {
                    "task_template": "{input}\nTLDR:",
                    "max_length": 3072,
                },
            },
        )
        for i in range(num_gpus)
    ]
    from supplement_experiments.common import simple_store_worker

    store_worker = Process(
        target=simple_store_worker, args=("data/text_summarization.txt", rq, rqe)
    )

    task_worker_.start()
    for w in gpu_workers:
        w.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in gpu_workers:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    store_worker.join()
    assert store_worker.exitcode == 0
