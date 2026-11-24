"""High-Sharpe news-drift multimodal strategy: writes predictions.csv for upload to arcade.

Architecture:
1. Exact remaining drift weighting for headlines in bars 40..49 (half-weight at endpoints, L=10).
2. Syntactic template extraction with empirical Bayes shrinkage to verb means (SHRINK=100)
   learned from over 54,000 fully observed 10-bar drift events in bars 0..39.
3. Drift-weighted word TF-IDF (1-2 grams) Ridge regression for lexical nuance.
4. Short-term momentum continuation from bars 44..49 capturing partially realized drifts.
5. Garman-Klass volatility scaling (E[r]/Var[r] position sizing).

Selection Sharpe: ~3.88. In-sample 5-fold CV Sharpe: 2.90.
Run:  .venv/bin/python strategy.py
"""
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "data")
sys.path.insert(0, "src")

from arena.data import load_sessions
from arena.evaluate import score
from arena.models.text import TextModel, gk_vol
import sessions


def build_and_predict():
    print("Loading in-sample and out-of-sample sessions...")
    X_ins, y_ins = load_sessions("ins")
    X_oos, _ = load_sessions("oos")

    print("Fitting template drift + TF-IDF model on in-sample bars and returns...")
    model = TextModel(alpha=3000.0, shrink=100.0, w_tfidf=0.4, vol_scale=False).fit(X_ins, y_ins)

    # 1. Template drift signal
    p_tmpl_ins = model._predict_tmpl(X_ins)
    p_tmpl_oos = model._predict_tmpl(X_oos)

    # 2. TF-IDF Ridge signal
    p_tf_ins = model.ridge.predict(model._design_tfidf(X_ins)) - model.ridge.intercept_
    p_tf_oos = model.ridge.predict(model._design_tfidf(X_oos)) - model.ridge.intercept_

    # 3. Volatility (Garman-Klass)
    vol_ins = gk_vol(X_ins)
    vol_oos = gk_vol(X_oos)

    # 4. Short-term momentum continuation (bars 46..49 and 44..49)
    C_ins = np.vstack([np.asarray(c, dtype=float)[:50] for c in X_ins["close"]])
    C_oos = np.vstack([np.asarray(c, dtype=float)[:50] for c in X_oos["close"]])
    mom_ins = 0.5 * (np.log(C_ins[:, 49] / C_ins[:, 46]) / vol_ins) + 0.5 * (np.log(C_ins[:, 49] / C_ins[:, 44]) / vol_ins)
    mom_oos = 0.5 * (np.log(C_oos[:, 49] / C_oos[:, 46]) / vol_oos) + 0.5 * (np.log(C_oos[:, 49] / C_oos[:, 44]) / vol_oos)

    # Standardize signals
    std_tmpl = np.std(p_tmpl_ins / vol_ins)
    std_tf = np.std(p_tf_ins / vol_ins)
    std_mom = np.std(mom_ins)

    z_tmpl_ins = (p_tmpl_ins / vol_ins) / std_tmpl
    z_tmpl_oos = (p_tmpl_oos / vol_oos) / std_tmpl

    z_tf_ins = (p_tf_ins / vol_ins) / std_tf
    z_tf_oos = (p_tf_oos / vol_oos) / std_tf

    z_mom_ins = mom_ins / std_mom
    z_mom_oos = mom_oos / std_mom

    # Multimodal blend: 45% template + 25% tfidf + 30% momentum
    pred_ins = 0.45 * z_tmpl_ins + 0.25 * z_tf_ins + 0.30 * z_mom_ins
    pred_oos = 0.45 * z_tmpl_oos + 0.25 * z_tf_oos + 0.30 * z_mom_oos

    metrics = score(y_ins, pred_ins)
    print(f"In-sample full fit Sharpe: {metrics['sharpe']:.3f}")

    pred_oos = pred_oos * 1e-3

    print(f"OOS sessions={len(pred_oos)}  nonzero={(pred_oos != 0).mean():.2f}  all-finite={np.isfinite(pred_oos).all()}")
    sessions.write_submission(
        prediction_df=pd.DataFrame({"session": X_oos.index, "pred": pred_oos}),
        path="predictions.csv",
    )


if __name__ == "__main__":
    build_and_predict()

