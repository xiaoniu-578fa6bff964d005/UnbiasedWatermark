def bleu_task(t):
    i, wp_str = t
    global sval
    lds = sval[wp_str]

    import random

    a = random.Random(i).sample(lds["display_output"], len(lds["display_output"]) // 2)
    b = random.Random(i).sample(lds["reference"], len(lds["reference"]) // 2)

    global bleu_scorer
    bleu_score = bleu_scorer.compute(
        predictions=a,
        references=b,
    )
    return {"watermark_processor": wp_str, "bleu": bleu_score["score"]}


def set_global(args):
    global sval
    sval = {
        k: {k2: [s for s in v2] for (k2, v2) in v.items()} for (k, v) in args.items()
    }

    global bleu_scorer
    import evaluate

    bleu_scorer = evaluate.load("sacrebleu")


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

    from concurrent.futures import ProcessPoolExecutor
    import json

    import multiprocessing as mp

    sval = {
        wp_str: {
            "display_output": list(s_out_ds["display_output"]),
            "reference": list(s_out_ds["reference"]),
        }
        for (wp_str, s_out_ds) in s_out_dss.items()
    }

    with open("data/machine_translation_bleu.txt", "w") as f:
        with ProcessPoolExecutor(initializer=set_global, initargs=(sval,)) as executor:
            from tqdm import tqdm

            num = 10000
            for r in tqdm(
                executor.map(
                    bleu_task,
                    [
                        (i, wp_str)
                        for (wp_str, s_out_ds) in s_out_dss.items()
                        for i in range(num)
                    ],
                ),
                total=len(wp_types) * num,
            ):
                f.write(json.dumps(r))
                f.write("\n")
                f.flush()
