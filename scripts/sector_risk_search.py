"""Predeclared sizing and blend comparison, restricted to the fixed train partition."""
import json
import argparse
from pathlib import Path
import time

import numpy as np
from sklearn.model_selection import KFold

from arena.data import load_sessions, partitions
from arena.evaluate import score
from arena.models.sector_drift import build, price_arrays
from arena.models.events import build as events


def candidates(model, X):
    features, raw, weight = model._state(X)
    mu = raw + model.calibration.predict(features / model.scale)
    _, ranges = price_arrays(X)
    vol = np.clip(ranges / model.typical_range, .45, 2.)
    result = {"baseline": mu * weight}
    for power in (1., 1.5, 2.):
        for mean_weight in (0., .5, 1.):
            result[f"risk_p{power}_c{mean_weight}"] = mu / (
                vol**power + mean_weight * mu**2 / (10 * model.noise**2))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration", action="store_true")
    parser.add_argument("--residual", action="store_true")
    parser.add_argument("--sizing", action="store_true")
    parser.add_argument("--out", default="reports/sector_risk")
    args = parser.parse_args()
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    filename = "sizing_cv.json" if args.sizing else ("residual_cv.json" if args.residual else ("calibration_cv.json" if args.calibration else "cv.json"))
    if (out / filename).exists():
        raise RuntimeError("Do not overwrite prior experiments")
    X, y = load_sessions()
    ids = partitions(X.index)[0]
    X, y = X.loc[ids], y.loc[ids]
    if args.sizing:
        from arena.models.sector_risk import SectorRisk
        predictions, folds = {}, {}
        for fit, test in KFold(4, shuffle=True, random_state=42).split(X):
            model = SectorRisk().fit(X.iloc[fit], y.iloc[fit])
            features, raw, _ = model._state(X.iloc[test])
            mu = raw + model.calibration.predict(features / model.scale)
            _, ranges = price_arrays(X.iloc[test])
            vol = np.clip(ranges / model.typical_range, .45, 2.)
            for power in (2., 2.5, 3.):
                for coefficient in (1., 2., 4., 8.):
                    name = f"power{power}_coefficient{coefficient}"
                    pred = mu / (vol**power + coefficient * mu**2 / (10 * model.noise**2))
                    predictions.setdefault(name, np.zeros(len(X)))[test] = pred
                    folds.setdefault(name, []).append(score(y.iloc[test], pred)["sharpe"])
        results = {name: {"pooled": score(y, pred)["sharpe"], "folds": folds[name]}
                   for name, pred in predictions.items()}
        report = {"results": results, "winner": max(results, key=lambda k: results[k]["pooled"]),
                  "selection_scored": False, "audit_scored": False, "train_n": len(X),
                  "runtime_seconds": time.perf_counter()-started}
        (out / filename).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return
    if args.calibration or args.residual:
        from arena.models.sector_risk import SectorRisk
        results = {}
        for variant in ((0, 1, 2) if args.residual else (0., 1., 2.)):
            for alpha in ((.001, .01, .1) if args.residual else (.01, .1, 1.)):
                pred, folds = np.zeros(len(X)), []
                for fit, test in KFold(4, shuffle=True, random_state=42).split(X):
                    model = SectorRisk(alpha, 2. if args.residual else variant,
                                       variant if args.residual else 0).fit(X.iloc[fit], y.iloc[fit])
                    pred[test] = model.predict(X.iloc[test])
                    folds.append(score(y.iloc[test], pred[test])["sharpe"])
                name = f"alpha{alpha}_variant{variant}" if args.residual else f"alpha{alpha}_power{variant}"
                results[name] = {"pooled": score(y, pred)["sharpe"], "folds": folds}
                print(name, results[name], flush=True)
        report = {"results": results, "winner": max(results, key=lambda k: results[k]["pooled"]),
                  "selection_scored": False, "audit_scored": False, "train_n": len(X),
                  "runtime_seconds": time.perf_counter()-started}
        (out / filename).write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report, indent=2))
        return
    all_predictions, folds = {}, []
    for fold, (fit, test) in enumerate(KFold(4, shuffle=True, random_state=42).split(X)):
        model = build().fit(X.iloc[fit], y.iloc[fit])
        preds = candidates(model, X.iloc[test])
        event = events().fit(X.iloc[fit], y.iloc[fit])
        baseline_train = model.predict(X.iloc[fit])
        event_train = event.predict(X.iloc[fit])
        event_pred = event.predict(X.iloc[test])
        ratio = np.sqrt(np.mean(baseline_train**2) / np.mean(event_train**2))
        for weight in (.1, .25, .5):
            preds[f"event_blend_{weight}"] = (1-weight)*preds["baseline"] + weight*ratio*event_pred
        fold_scores = {}
        for name, pred in preds.items():
            all_predictions.setdefault(name, np.zeros(len(X)))[test] = pred
            fold_scores[name] = score(y.iloc[test], pred)["sharpe"]
        folds.append(fold_scores)
        print("Completed fold", fold+1, flush=True)
    results = {name: {"pooled": score(y, pred)["sharpe"],
                      "folds": [fold[name] for fold in folds]}
               for name, pred in all_predictions.items()}
    winner = max(results, key=lambda key: results[key]["pooled"])
    report = {"train_n": len(X), "selection_scored": False, "audit_scored": False,
              "winner": winner, "results": results, "runtime_seconds": time.perf_counter()-started}
    (out / filename).write_text(json.dumps(report, indent=2)+"\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
