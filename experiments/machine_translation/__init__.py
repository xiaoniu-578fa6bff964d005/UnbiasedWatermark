def process_in_ds(ds):
    ds = ds.flatten()
    ds = ds.rename_column("translation.en", "input")
    ds = ds.rename_column("translation.ro", "reference")

    def _to_id(s: str):
        import hashlib

        return hashlib.sha1(s.encode("utf-8")).hexdigest()

    def add_id(example):
        if isinstance(example["input"], list):
            example["id"] = [_to_id(s) for s in example["input"]]
        elif isinstance(example["input"], str):
            example["id"] = _to_id(example["input"])
        return example

    ds = ds.map(add_id, batched=True)
    return ds


def get_in_ds():
    from datasets import load_dataset

    wmt17 = load_dataset("wmt16", "ro-en").shuffle(seed=42)
    ds = wmt17["test"]

    import os

    if os.environ.get("EXP_DEBUG", None) == "1":
        ds = ds.select(range(0, 2))
    if os.environ.get("EXP_DEBUG", None) == "2":
        ds = ds.shard(num_shards=100, index=0)

    ds = process_in_ds(ds)
    ds = ds.sort("id")

    return ds


from . import get_output
from . import evaluate
from . import evaluate_bleu
from . import evaluate_ppl
from . import evaluate_score
