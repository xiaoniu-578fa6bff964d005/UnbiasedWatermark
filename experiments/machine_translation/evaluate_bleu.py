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
