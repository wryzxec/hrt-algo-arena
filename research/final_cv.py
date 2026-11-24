exec(open("research/final_pipeline.py").read())
from sklearn.model_selection import KFold
N = len(ins["S"]); allo = np.arange(len(oos["S"]))
sh = lambda p, i=slice(None): (p * y)[i].mean() / (p * y)[i].std() * np.sqrt(252)
for seed in (0, 1, 2):
    folds = list(KFold(5, shuffle=True, random_state=seed).split(np.arange(N))); p = np.zeros(N)
    for a, b in folds: p[b] = train_predict(a, ins, b, [(ins, b, 49), (oos, allo, 49)])
    print(f"seed {seed}: 5fold={sh(p):.3f} folds={[round(sh(p, b), 2) for _, b in folds]}", flush=True)
# repo partitions: train on 70% only, score the 15% selection and 15% audit splits
sys.path.insert(0, "src"); from arena.data import partitions
tr, va, au = (pd.Index(ins["S"]).get_indexer(x) for x in partitions(ins["S"]))
for name, te in (("selection", va), ("audit", au)):
    p = train_predict(tr, ins, te, [(ins, te, 49), (oos, allo, 49)]); print(f"repo {name} split (n={len(te)}): {sh(p * 0 + p, slice(None)) if False else ((p*y[te]).mean()/(p*y[te]).std()*np.sqrt(252)):.3f}", flush=True)
