import argparse
import importlib
import json
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from arena.data import load_sessions, partitions

def score(y, pred):
    y, pred = np.asarray(y,dtype=float), np.asarray(pred,dtype=float)
    if y.ndim != 1 or pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("Expected aligned finite 1D arrays")
    pnl = y*pred
    sd = pnl.std(ddof=0)
    return {"sharpe": float(np.sqrt(252)*pnl.mean()/sd) if sd else 0., "mean_pnl":float(pnl.mean()), "std_pnl":float(sd), "n":len(y)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    X,y=load_sessions()
    train,valid,audit=partitions(X.index)
    module=importlib.import_module(args.model)
    model=module.build()
    model.fit(X.loc[train],y.loc[train])
    pred=np.asarray(model.predict(X.loc[valid]),dtype=float)
    metrics=score(y.loc[valid],pred)
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    pd.DataFrame({"session":valid,"pred":pred}).to_csv(out/"validation.csv",index=False)
    metrics.update(model=args.model,seed=42,split="70/15/15",ddof=0,git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2)+"\n")
    print(json.dumps(metrics,indent=2))
    model=module.build()
    dev=np.concatenate([train,valid])
    model.fit(X.loc[dev],y.loc[dev])
    oos,_=load_sessions("oos")
    predictions=np.asarray(model.predict(oos),dtype=float)
    if predictions.shape != (len(oos),) or not np.isfinite(predictions).all(): raise ValueError("Invalid submission")
    pd.DataFrame({"headline_id":oos.index,"pred":predictions}).to_csv(out/"predictions.csv",index=False)

if __name__ == "__main__": main()
