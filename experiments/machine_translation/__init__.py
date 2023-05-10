def process_in_ds(ds):
    ds = ds.flatten()
    ds = ds.rename_column("translation.en", "input")
    ds = ds.rename_column("translation.de", "reference")

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

    wmt17 = load_dataset("wmt17", "de-en").shuffle(seed=42)
    ds = wmt17["train"]

    ds = ds.shard(num_shards=100, index=0)
    #  print("ds len:", len(ds))

    ds = process_in_ds(ds)
    ds = ds.sort("id")

    return ds


from . import get_output
from . import evaluate