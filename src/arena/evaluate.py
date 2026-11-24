import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import platform
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from arena.data import load_sessions, partitions
from arena.validate import file_sha256, validate_predictions

def imported_source_provenance(model_module):
    """Identify imported model and harness files independently of the working directory."""
    modules = {name: module for name, module in tuple(sys.modules.items())
               if name == "arena" or name.startswith("arena.")}
    modules["arena.evaluate"] = sys.modules[__name__]
    modules[model_module.__name__] = model_module
    sources = {}
    for name, module in sorted(modules.items()):
        filename = getattr(module, "__file__", None)
        if filename:
            path = Path(filename).resolve()
            if path.is_file():
                sources[name] = {"path": str(path), "sha256": file_sha256(path)}
    return sources

def score(y, pred):
    y, pred = np.asarray(y,dtype=float), np.asarray(pred,dtype=float)
    if y.ndim != 1 or not len(y) or pred.shape != y.shape or not np.isfinite(pred).all() or not np.isfinite(y).all():
        raise ValueError("Expected aligned finite 1D arrays")
    with np.errstate(over="ignore", invalid="ignore"):
        pnl = y*pred
    if not np.isfinite(pnl).all():
        raise ValueError("Prediction times return must be finite")
    sd = pnl.std(ddof=0)
    if not np.isfinite(sd):
        raise ValueError("PnL standard deviation overflowed")
    return {"sharpe": float(np.sqrt(252)*pnl.mean()/sd) if sd else 0., "mean_pnl":float(pnl.mean()), "std_pnl":float(sd), "n":len(y)}

def main():
    started = time.perf_counter()
    p=argparse.ArgumentParser()
    p.add_argument("--model",required=True)
    p.add_argument("--out",required=True)
    args=p.parse_args()
    out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    # New output directories preserve the history of trials and avoid mixed runs.
    if any((out / name).exists() for name in ("validation.csv", "predictions.csv", "metrics.json")):
        p.error("Report files already exist; use a new --out directory for each run")
    X,y=load_sessions()
    train,valid,audit=partitions(X.index)
    module=importlib.import_module(args.model)
    model=module.build()
    model.fit(X.loc[train],y.loc[train])
    pred=np.asarray(model.predict(X.loc[valid]),dtype=float)
    metrics=score(y.loc[valid],pred)
    selection_seconds = time.perf_counter() - started
    validation = pd.DataFrame({"session":valid,"pred":pred})
    validate_predictions(validation, valid, "session")
    metrics.update(model=args.model,seed=42,split="70/15/15",ddof=0,git_sha=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip())
    model=module.build()
    dev=np.concatenate([train,valid])
    model.fit(X.loc[dev],y.loc[dev])
    oos,_=load_sessions("oos")
    predictions=np.asarray(model.predict(oos),dtype=float)
    if predictions.shape != (len(oos),) or not np.isfinite(predictions).all(): raise ValueError("Invalid submission")
    submission = pd.DataFrame({"headline_id":oos.index,"pred":predictions})
    validate_predictions(submission, oos.index)
    validation.to_csv(out/"validation.csv",index=False)
    submission.to_csv(out/"predictions.csv",index=False)
    # Validate serialized bytes, not just the in-memory frame.
    validate_predictions(pd.read_csv(out/"predictions.csv"), oos.index)
    pnl = y.loc[valid].to_numpy() * pred
    absolute_pnl = np.abs(pnl)
    top_count = max(1, int(np.ceil(.01 * len(pnl))))
    metrics.update(
        utc=datetime.now(timezone.utc).isoformat(),
        selection_seconds=selection_seconds, runtime_seconds=time.perf_counter()-started,
        device=platform.machine(), python=platform.python_version(),
        git_dirty=bool(subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=normal", "--", "src", "pyproject.toml"], text=True).strip()),
        packages={name: importlib.metadata.version(name) for name in ("numpy", "pandas", "scipy", "scikit-learn", "pyarrow")},
        source_sha256={str(path): file_sha256(path) for path in sorted(Path("src/arena").rglob("*.py"))},
        git_context_directory=str(Path.cwd().resolve()),
        imported_source=imported_source_provenance(module),
        artifact_sha256={name: file_sha256(out/name) for name in ("validation.csv", "predictions.csv")},
        train_n=len(train), selection_n=len(valid), refit_n=len(dev), oos_n=len(oos),
        audit_scored=False,
        pnl_quantiles={str(q):float(np.quantile(pnl,q)) for q in (0,.01,.5,.99,1)},
        top_1pct_absolute_pnl_share=float(np.sort(absolute_pnl)[-top_count:].sum()/absolute_pnl.sum()) if absolute_pnl.sum() else 0.,
    )
    (out/"metrics.json").write_text(json.dumps(metrics,indent=2,allow_nan=False)+"\n")
    print(json.dumps(metrics,indent=2,allow_nan=False))

if __name__ == "__main__": main()
