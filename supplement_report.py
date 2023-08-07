import numpy as np
import pandas as pd
import polars as pl
import os

#  data_path = "data.05_24_11_31"
data_path = "data"


def read_json_lines_to_df(file_path):
    df = pd.read_json(open(file_path, "r", encoding="utf8"), lines=True)
    return df


def get_sample_df(data_path=data_path, task="text_summarization"):
    df = read_json_lines_to_df(f"{data_path}/{task}.txt")
    if os.path.exists(f"{data_path}/{task}_result.txt"):
        df2 = read_json_lines_to_df(f"{data_path}/{task}_result.txt")
        df = pd.merge(df, df2, on=["id", "watermark_processor"])
    if os.path.exists(f"{data_path}/{task}_ppl.txt"):
        df3 = read_json_lines_to_df(f"{data_path}/{task}_ppl.txt")
        df = pd.merge(df, df3, on=["id", "watermark_processor"])
    if os.path.exists(f"{data_path}/{task}_score.txt"):
        df4 = read_json_lines_to_df(f"{data_path}/{task}_score.txt")
        df = pd.merge(df, df4, on=["id", "watermark_processor"], how="left")
    return df


def get_show_wp_name(wp_str):
    show_wp = [
        "No Watermark",
        "$\delta$-reweight",
        "$\gamma$-reweight",
        "$\delta$-reweight (woh)",
        "$\gamma$-reweight (woh)",
    ]
    if "Delta" in wp_str or "Gamma" in wp_str:
        woh = ", True)" in wp_str
        if "Delta" in wp_str and not woh:
            return show_wp[1]
        elif "Delta" in wp_str and woh:
            return show_wp[3]
        elif "Gamma" in wp_str and not woh:
            return show_wp[2]
        elif "Gamma" in wp_str and woh:
            return show_wp[4]
    elif "John" in wp_str:
        import re

        delta = re.findall(r"delta=(\d+\.?\d*)", wp_str)[0]
        n = "Soft" + f"($\delta$={delta})"
        return n
    if wp_str == "None":
        return show_wp[0]
    else:
        raise ValueError("Unknown watermark: {}".format(wp_str))


def sample_df_2_stat(df, bootstrap=False, show_wp=None):
    sdf = df.melt(
        id_vars=["show_wp_name"],
        value_vars=[c for c in df.columns if df[c].dtype == np.float64],
        var_name="score",
        value_name="value",
    )
    sdf = sdf.groupby(["show_wp_name", "score"]).agg(["mean", "std", "count"])

    def format_fn(x):
        mean = x["mean"]
        if not bootstrap:
            std = x["std"] / np.sqrt(x["count"])
        else:
            std = x["std"]
        if not np.isfinite(std):
            return f"{mean:.2f}±{std:.2f}"
        useful_digits = np.max(-int(np.floor(np.log10(std / 3))), 0)
        fmt_str = f"{{:.{useful_digits}f}}±{{:.{useful_digits}f}}"
        return fmt_str.format(mean, std)

    sdf = sdf["value"].apply(format_fn, axis=1).unstack()
    if show_wp:
        sdf = sdf.loc[show_wp]
    return sdf


def compute_poem():
    task = "poem_generation"
    df = get_sample_df(data_path, task)
    df = df.assign(show_wp_name=df["watermark_processor"].apply(get_show_wp_name))

    stat = sample_df_2_stat(df)
    #  print(sample_df_2_stat(tsdf[['show_wp_name','bertscore.precision','bertscore.recall','rouge2','rougeL']], show_wp=show_wp).to_latex())
    print(stat)
    pass


def main():
    compute_poem()


if __name__ == "__main__":
    main()
