# Deep / feature-engineering results (common 70/15/15 split, selection Sharpe, n=1500)

One split of 1500 sessions: luck alone moves a single score by about ±0.4, so treat gaps of less than ~0.3 as unconfirmed.

| Model | Sel. Sharpe | Notes |
|---|---|---|
| Text TF-IDF × remaining drift, Ridge α=3000 (PR #8) | 3.34 | |
| … ÷ Garman–Klass vol (PR #8 final) | 3.48 | vol sizing, theory-backed, +0.04 in 5-fold CV |
| … trained on 41× shifted windows (train sessions' bars 50–99) | 3.59–3.66 | augmentation; needs bars >49 of TRAIN sessions → outside harness contract |
| Structured NN (template embedding × learned age curve) + aug, 3 seeds | 3.63 | early-stopped on inner 10% of train |
| MLP (template×age bag) + aug, 3 seeds | 3.61 | |
| **Blend: text+aug + Struct NN + MLP NN** | **3.80** | component corr ~0.8 |
| GRU + aug (epoch peak) | 3.31 | overfits after 2 epochs |
| GRU / CNN / Transformer / MLP without aug | 0.7–1.7 (honest stop) | 7k sessions too few |
| One-hot template × drift Ridge | 2.70 (2.84 ÷vol) | TF-IDF shares strength across templates |
| Ridge with ALL engineered features (≈1.3k cols) | 1.6–2.1 | overfits |
| HGB on signal + surprise + price | 2.08–2.33 | |
| Momentum/reversal/vol price features | ≤ +0.05 | |

Generator facts found: each headline gives a ~linear 10-bar drift, then stops; drift ∝ stock volatility
(0.72→1.30 across vol quintiles); no company, per-session or repeat-mention effects; dollar/percent sizes irrelevant;
leftover price paths are near random walk (best linear corr 0.02).
