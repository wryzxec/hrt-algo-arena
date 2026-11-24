"""Run four-fold CV entirely within the project's training partition."""
import numpy as np
from sklearn.model_selection import KFold

from arena.data import load_sessions, partitions
from arena.models.sector_drift import build


def sharpe(y, pred):
    pnl = np.asarray(y) * pred
    return np.sqrt(252) * pnl.mean() / (pnl.std(ddof=1) + 1e-12)


if __name__ == "__main__":
    X, y = load_sessions()
    train = partitions(X.index)[0]
    X, y = X.loc[train], y.loc[train]
    predictions = np.zeros(len(train))
    for fold, (fit, test) in enumerate(KFold(4, shuffle=True, random_state=42).split(X), 1):
        model = build().fit(X.iloc[fit], y.iloc[fit])
        predictions[test] = model.predict(X.iloc[test])
        print(f"Fold {fold} Sharpe {sharpe(y.iloc[test], predictions[test]):.4f}")
    print(f"Pooled Sharpe {sharpe(y, predictions):.4f}")
