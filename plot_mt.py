import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_json(open("/home/bobchen/data/machine_translation_result.txt", "r", encoding="utf8"), lines=True)

show_wp = ["No Watermark", "$\delta$-reweight", "$\gamma$-reweight"]
john_wps_set = set()


def map_wp_str(wp_str):
    if "Delta" in wp_str:
        return show_wp[1]
    elif "Gamma" in wp_str:
        return show_wp[2]
    elif "John" in wp_str:
        import re

        delta = re.findall(r"delta=(\d+\.?\d*)", wp_str)[0]
        n = "Soft Red List" + f"($\delta$={delta})"
        john_wps_set.add(n)
        return n
    if wp_str == "None":
        return show_wp[0]
    else:
        raise ValueError("Unknown watermark: {}".format(wp_str))

sdf=df.melt(
    id_vars=["wp_str"],
    value_vars=[c for c in df.columns if df[c].dtype == np.float64],
    var_name="score",
    value_name="value",
)
ndf = pd.read_json(open("/home/bobchen/data/machine_translation_bleu.txt", "r", encoding="utf8"), lines=True)
ndf = ndf.assign(wp_str=ndf["watermark_processor"].apply(map_wp_str))
ndf=ndf.melt(
    id_vars=["wp_str"],
    value_vars=[c for c in ndf.columns if ndf[c].dtype == np.float64],
    var_name="score",
    value_name="value",
)
sdf = pd.concat([sdf, ndf])
sdf = sdf.groupby(["wp_str", "score"]).agg(["mean", "std", "count"])
sdf = (
    sdf["value"]
    .apply(
        lambda x: "{:.4f}±{:.4f}".format(x["mean"], x["std"] / np.sqrt(x["count"])),
        axis=1,
    )
    .unstack()
)
sdf = sdf.loc[show_wp + john_wps]