import sys; sys.path.insert(0, "data")
import pandas as pd, numpy as np
bars = pd.read_parquet("data/ins_bars.parquet"); hl = pd.read_parquet("data/ins_headlines.parquet"); ret = pd.read_parquet("data/ins_returns.parquet")
print(bars.head(), bars.shape, bars.columns.tolist())
print(hl.head(), hl.shape, hl.columns.tolist())
print(ret.describe())
print("headlines per session", hl.groupby("session").size().describe())
# template the headlines: replace company names
import re
h = hl["headline"]
print(h.sample(40, random_state=0).to_string())
print("unique headlines", h.nunique())
