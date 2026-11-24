import pandas as pd, numpy as np, re, collections
hl = pd.read_parquet("data/ins_headlines.parquet")
toks = hl.headline.str.split()
c2 = collections.Counter(" ".join(t[:2]) for t in toks)
c3 = collections.Counter(" ".join(t[:3]) for t in toks)
print(len(c2), c2.most_common(5))
third = collections.Counter(t[2] for t in toks)
print(third.most_common(40))
