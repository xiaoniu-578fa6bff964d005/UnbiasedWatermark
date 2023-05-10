def process_in_ds(ds):
    ds = ds.rename_column("article", "input")
    ds = ds.rename_column("highlights", "reference")
    return ds


def get_in_ds():
    from datasets import load_dataset

    cnn_daily = load_dataset("cnn_dailymail", "3.0.0").shuffle(seed=42)
    ds = cnn_daily["test"]
    ds = process_in_ds(ds)
    ds = ds.sort("id")

    #  ds = ds.shard(num_shards=200, index=0)
    #  print("ds len:", len(ds))

    return ds


def get_merged_ds(path):
    in_ds = get_in_ds()

    from datasets import load_dataset

    out_ds = load_dataset("json", data_files={"test": path})["test"]
    out_ds = out_ds.sort("id")

    from experiments.common import add_reference, group_batch

    ds = add_reference(in_ds, out_ds)
    return ds


from . import get_output
from . import evaluate
from . import evaluate_ppl
