"""Final pipeline as functions so the same code runs in CV and for the submission."""
import sys; sys.path.insert(0, "research")
exec(open("research/sector_cv2.py").read().split("N = len(ins")[0])   # prep(), fit(), signal(), wk, remaining, tm/tix, ins, oos, y
import numpy as np
from sklearn.linear_model import Ridge
HL = (1, 2, 3, 5, 10, 25, 1000); ALPHA = 50.0; LAM = 200 / (np.median(ins["gk"]) ** 2)
def resid49(d, idx, coef):
    cm = dict(zip(tm, coef)); h = d["hl"][d["hl"].i.isin(idx) & (d["hl"].bar_ix <= 49)]; rm = {v: k for k, v in enumerate(idx)}
    c = h.tmpl.map(cm).fillna(0).to_numpy(); b = h.bar_ix.to_numpy(); i = h.i.map(rm).to_numpy()
    R = np.diff(d["C"][idx][:, :50], axis=1); R = (R - R.mean(0)) / d["gk"][idx][:, None]; E = np.zeros_like(R)
    for k in range(11):
        j = b + k - 1; ok = (j >= 0) & (j < 49); np.add.at(E, (i[ok], j[ok]), wk(k) * c[ok])
    return R - E
def feats(d, idx, coef):
    Z = resid49(d, idx, coef); t = np.arange(49)[::-1]
    return np.column_stack([signal(d, idx, coef)] + [(Z * np.exp(-t / h)).sum(1) for h in HL])
def train_predict(tr_idx, te, te_idx, extra):
    """tr_idx: in-sample rows with labels; te: dataset dict to predict; extra: label-free (dataset, idx, maxbar) parts"""
    coef = fit([(ins, tr_idx, 99)] + extra, True, LAM)
    Xa, Xb = feats(ins, tr_idx, coef), feats(te, te_idx, coef)
    ya = y[tr_idx] / ins["gk"][tr_idx]; mu, sd = Xa.mean(0), Xa.std(0) + 1e-12
    r = Ridge(alpha=ALPHA).fit((Xa - mu) / sd, ya)
    m = r.predict((Xb - mu) / sd) - r.intercept_                    # E[y/vol | x]
    return m / (te["gk"][te_idx] * (ya.var() + m ** 2))             # Sharpe-optimal: E[r]/E[r^2]
