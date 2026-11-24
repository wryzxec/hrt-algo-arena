import pandas as pd, numpy as np, re
SUFFIX = {"Group", "Co"}
def load(kind="ins"):
    bars = pd.read_parquet(f"data/{kind}_bars.parquet")
    hl = pd.read_parquet(f"data/{kind}_headlines.parquet")
    sess = np.sort(bars.session.unique())
    close = bars.pivot(index="session", columns="bar_ix", values="close").reindex(sess)
    t = hl.headline.str.split()
    hl["company"] = [" ".join(x[:3]) if x[2] in SUFFIX else " ".join(x[:2]) for x in t]
    hl["rest"] = [" ".join(x[3:]) if x[2] in SUFFIX else " ".join(x[2:]) for x in t]
    hl["verb"] = hl.rest.str.split().str[0]
    hl["tmpl"] = hl.rest.str.replace(r"\$[\d.]+[MBK]?", "$X", regex=True).str.replace(r"\d+(\.\d+)?%", "N%", regex=True)
    ret = None
    if kind == "ins":
        ret = pd.read_parquet("data/ins_returns.parquet").set_index("session").returns.reindex(sess)
    return bars, hl, close, ret
