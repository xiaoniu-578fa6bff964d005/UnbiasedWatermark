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
    r2q = Queue()
    r2qe = Event()

    from supplement_experiments.common import (
        merged_task_worker,
        score_worker,
        remove_text_worker,
        simple_store_worker,
    )

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, "data/machine_translation.txt", tq),
        kwargs={"batch_size": 8, "watermark_only": True},
    )

    score_worker_ = [
        Process(
            target=score_worker,
            args=(tq, tqe, rq, i, "facebook/mbart-large-en-ro"),
        )
        for i in range(num_gpus)
    ]
    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=("data/machine_translation_score.txt", r2q, r2qe),
    )

    task_worker_.start()
    for w in score_worker_:
        w.start()
    rt_worker.start()
    store_worker.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in score_worker_:
        w.join()
        assert w.exitcode == 0
    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0


def _pipeline2(test_config, rq):
    from torch.multiprocessing import Process, Queue, Event

    from supplement_experiments.common import get_num_gpus

    num_gpus = get_num_gpus()

    tq = Queue(maxsize=num_gpus)
    tqe = Event()

    from supplement_experiments.common import merged_task_worker, score_worker2

    from . import get_in_ds

    task_worker_ = Process(
        target=merged_task_worker,
        args=(get_in_ds, "data/machine_translation.txt", tq),
        kwargs={
            "batch_size": 8,
            "watermark_only": True,
            "wh_only": True,
            "no_gumbel": True,
        },
    )

    score_worker_ = [
        Process(
            target=score_worker2,
            args=(tq, tqe, rq, i, test_config),
        )
        for i in range(num_gpus)
    ]

    task_worker_.start()
    for w in score_worker_:
        w.start()

    task_worker_.join()
    assert task_worker_.exitcode == 0
    tqe.set()
    for w in score_worker_:
        w.join()
        assert w.exitcode == 0


def pipeline2():
    from supplement_experiments.common import set_spawn

    set_spawn()

    from torch.multiprocessing import Process, Queue, Event

    rq = Queue()
    rqe = Event()
    r2q = Queue()
    r2qe = Event()

    from supplement_experiments.common import remove_text_worker, simple_store_worker

    rt_worker = Process(target=remove_text_worker, args=(rq, rqe, r2q))
    store_worker = Process(
        target=simple_store_worker,
        args=("data/machine_translation_ablation.txt", r2q, r2qe),
    )
    rt_worker.start()
    store_worker.start()

    default_test_config = {
        "temperature": 1.0,
        "top_k": 50,
        "no_input": False,
        "model_str": "facebook/mbart-large-en-ro",
    }

    from supplement_experiments.common import get_wps

    wps = get_wps()[3:5]
    for wp in wps:
        default_test_config["wp_str"] = repr(wp)
        _pipeline2(default_test_config, rq)
        #  for temperature in [0.5, 1.5]:
        #      print("sensitivity to temperature", temperature)
        #      _pipeline2({**default_test_config, "temperature": temperature}, rq)
        #
        #  for top_k in [20, 100, 0]:
        #      print("sensitivity to top_k", top_k)
        #      _pipeline2({**default_test_config, "top_k": top_k}, rq)
        #
        #  for no_input in [True]:
        #      print("sensitivity to no_input", no_input)
        #      _pipeline2({**default_test_config, "no_input": no_input}, rq)
        #
        #  for model_str in []:  # don't know alternative model so far
        #      print("sensitivity to model", model_str)
        #      _pipeline2({**default_test_config, "model_str": model_str}, rq)

    rqe.set()
    rt_worker.join()
    assert rt_worker.exitcode == 0
    r2qe.set()
    store_worker.join()
    assert store_worker.exitcode == 0
